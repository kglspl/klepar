import argparse
from collections import deque
import gc
import glob
import itertools
import json
import math
import os
import sys
import struct
import tempfile
import threading
import time

from geometer import Plane, Point, Line
import h5py
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import pyperclip
import scipy
from scipy.spatial import Delaunay
from skimage.draw import line_aa
import tifffile
import tkinter as tk
from tkinter import filedialog, PhotoImage, ttk
import zarr


Image.MAX_IMAGE_PIXELS = None

class Klepar:
    def __init__(self):
        self.overlay_alpha = 255
        self.barrier_mask = None  # New mask to act as a barrier for flood fill
        self.editing_barrier = False  # False for editing label, True for editing barrier
        self.max_propagation_steps = 100  # Default maximum propagation steps
        self.show_barrier = True
        self.voxel_data = None
        self.prediction_data = None
        self.photo_img = None
        self.th_layer = 0
        self.resized_img = None
        self.z_index = 0
        self.pencil_size = 10
        self.click_coordinates = None
        self.threshold = [10]
        self.log_text = None
        self.zoom_level = 2
        self.max_zoom_level = 15
        self.drag_start_x = None
        self.drag_start_y = None
        self.image_position_x = 0
        self.image_position_y = 0
        self.pencil_cursor = None  # Reference to the circle representing the pencil size
        self.flood_fill_active = False  # Flag to control flood fill
        self.history = []  # List to store a limited history of image states
        self.max_history_size = 3  # Maximum number of states to store
        self.mask_data = None
        self.show_mask = True  # Default to showing the mask
        self.flatten_mask = True  # Display mask flattened (but still remember 3D)
        self.show_image = True
        self.show_prediction = True
        self.show_surface_offsets = True
        self.initial_load = True
        self.mat_affine = np.eye(3)
        self.mat_affine[0, 0] = self.mat_affine[1, 1] = self.zoom_level
        self.slice_cache = {}
        self.format = None
        self.canvas = None
        self.stride = 1
        self.surface_adjuster_offsets = None
        self.surface_adjuster_nodes = []
        self.surface_adjuster_tri = None
        self.roi = {}
        self._threaded_update_surface_adjuster_offsets_running = False
        arg_parser = self.init_argparse()
        arguments = arg_parser.parse_args()
        self.ppm = PPMParser(arguments.ppm, skip=4).open()
        self.default_masks_directory = '/src/kgl/assets/'
        self.init_ui(arguments)

    @staticmethod
    def init_argparse():
        parser = argparse.ArgumentParser(usage="%(prog)s [OPTION] [FILE]...", description="Visualize and help annotate Vesuvius Challenge data.")
        parser.add_argument("--h5fs-file", help="full path to H5FS (.h5) file; the first dataset there will be used")
        parser.add_argument("--axes", help="axes sequence in H5FS dataset", choices=['xyz', 'yxz', 'xzy', 'zxy', 'yzx', 'zyx'], default="xyz")
        parser.add_argument("--roi", help="region of interest (in dataset coords and axes!) to be loaded into memory and used, in x0-x1,y0-y1,z0-y1 notation (e.g. '0-1000,0-700,0-50')", default="0-1000,0-700,0-50")
        parser.add_argument("--surface-adjust-file", help="file from which to load surface adjustment nodes and save them on each change")
        parser.add_argument("--stride", help="stride to help interpred roi, adjust coordinates and save surface adjuster offsets resized to correct dimensions")
        parser.add_argument("--h5fs-scroll", help="full path to scroll H5FS (.h5) file; the first dataset there will be used")
        parser.add_argument("--ppm", help="full path to surface (PPM) file")
        return parser

    def parse_h5_roi_argument(self, roi, h5_axes_seq, stride):
        axes_rois = roi.split(',')
        for i, axis_roi in enumerate(axes_rois):
            start, end = axis_roi.split('-')
            # we only apply stride to x and y axis, not to z:
            axis = h5_axes_seq[i]
            stride_for_this_axis = 1 if axis == 'z' else stride
            yield int(start) // stride_for_this_axis
            yield int(end) // stride_for_this_axis
            # save data for displaying on info screen:
            self.roi[axis] = (int(start), int(end))

    def prepare_image_slice(self, z_index):
        """Prepare the image slice for display."""
        if self.format == 'zarr':
            if self.voxel_data.dtype == np.uint16:
                # Convert to float for normalization, then scale and convert to uint8
                img_data = self.voxel_data[z_index, :, :].astype('float32')
                img_data = (img_data / img_data.max() * 255).astype('uint8')
            else:
                img_data = self.voxel_data[z_index, :, :].astype('uint8')

        elif self.format == 'tiff':
            if self.voxel_data[0].dtype == np.uint16:
                # Convert to float for normalization, then scale and convert to uint8
                img_data = self.voxel_data[z_index][:, :].astype('float32')
                img_data = (img_data / img_data.max() * 255).astype('uint8')
            else:
                img_data = self.voxel_data[z_index][:, :].astype('uint8')

        elif self.format == 'h5fs':
            # debug surface_adjuster_offsets:
            # for i in range(self.dimx):
            #     self.surface_adjuster_offsets[:, i] = i % 10
            if self.surface_adjuster_offsets is not None and self.show_surface_offsets:
                indexes = (self.surface_adjuster_offsets[:, :].round().astype(np.int8) + self.z_index)[np.newaxis, :, :]
                img_data = np.take_along_axis(self.voxel_data, indexes, axis=0)[0, :, :]
            else:
                img_data = self.voxel_data[z_index, :, :]

        img = Image.fromarray(img_data).convert('RGBA')
        self.slice_cache[z_index] = img
        return img

    def load_data(self, h5_filename=None, h5_axes_seq=None, h5_roi=None, h5_scroll_filename=None):
        if not h5_filename:
            selected_path = filedialog.askdirectory(title="Select Directory")
            if not selected_path:
                return

        try:
            if h5_filename:
                self.h5_data_file = h5py.File(h5_filename, 'r')
                dataset_name, dataset_shape, dataset_type, dataset_chunks = self._h5_get_first_dataset_info(self.h5_data_file['/'])
                print(f"Opening {h5_filename}, dataset:", dataset_name, dataset_shape, dataset_type, dataset_chunks)
                if dataset_type != np.uint16:
                    raise Exception(f"Don't know how to display this dataset dtype ({dataset_type}), sorry")
                self.dataset = self.h5_data_file.require_dataset(dataset_name, shape=dataset_shape, dtype=dataset_type, chunks=dataset_chunks)
                self.format = 'h5fs'
                x0, x1, y0, y1, z0, z1 = list(self.parse_h5_roi_argument(h5_roi, h5_axes_seq, self.stride))
                self.voxel_data = (self.dataset[x0:x1, y0:y1, z0:z1] / 256).astype(np.uint8)

                self.dataset_shape_xyz = (dataset_shape[h5_axes_seq.index('x')], dataset_shape[h5_axes_seq.index('y')], dataset_shape[h5_axes_seq.index('z')])

                # we want to get zyx, so we perform swapaxes() until that happens: (kind of a bubblesort of axes)
                h5_axes_seq = [*h5_axes_seq]  # convert to list of characters
                if h5_axes_seq[0] != 'z':
                    swap_with = h5_axes_seq.index('z')
                    self.voxel_data = self.voxel_data.swapaxes(0, swap_with)
                    h5_axes_seq[swap_with] = h5_axes_seq[0]
                    h5_axes_seq[0] = 'z'
                if h5_axes_seq[1] != 'y':
                    swap_with = h5_axes_seq.index('y')
                    self.voxel_data = self.voxel_data.swapaxes(1, swap_with)
                    h5_axes_seq[swap_with] = h5_axes_seq[1]
                    h5_axes_seq[1] = 'y'

                self.dimz, self.dimy, self.dimx = self.voxel_data.shape
                self.file_name = os.path.basename(h5_filename)

                self.h5_scroll_data_file = None
                if h5_scroll_filename:
                    self.h5_scroll_data_file = h5py.File(h5_scroll_filename, 'r')
                    dataset_name, dataset_shape, dataset_type, dataset_chunks = self._h5_get_first_dataset_info(self.h5_scroll_data_file['/'])
                    print("Opening scroll dataset:", dataset_name, dataset_shape, dataset_type, dataset_chunks)
                    self.scroll_dataset = self.h5_scroll_data_file.require_dataset(dataset_name, shape=dataset_shape, dtype=dataset_type, chunks=dataset_chunks)

            else:
                # Check if the directory contains Zarr or TIFF files
                if os.path.exists(os.path.join(selected_path, '.zarray')):
                    # Load the Zarr data into the voxel_data attribute
                    self.voxel_data = zarr.open(selected_path, mode='r')
                    self.format = 'zarr'
                elif glob.glob(os.path.join(selected_path, '*.tif')):
                    # Load TIFF slices into a 3D numpy array using memory-mapped files
                    tiff_files = sorted(glob.glob(os.path.join(selected_path, '*.tif')), key=lambda x: int(os.path.basename(x).split('.')[0]))
                    self.voxel_data = [tifffile.memmap(f) for f in tiff_files]
                    self.format = 'tiff'
                    #self.voxel_data = np.stack(slices, axis=0)
                else:
                    self.update_log("Directory does not contain recognizable Zarr or TIFF files.")
                    print(selected_path)
                    return
                self.dimz = len(self.voxel_data)
                self.dimy, self.dimx = self.voxel_data[0].shape
                self.file_name = os.path.basename(selected_path)

            self.update_log(f"Data loaded successfully.")
            self.clear_slice_cache()
            self.mask_data = np.zeros((self.dimz,self.dimy,self.dimx), dtype=np.uint8)
            self.barrier_mask = np.zeros_like(self.mask_data, dtype=np.uint8)
            self.surface_adjuster_offsets = np.zeros(shape=(self.dimy, self.dimx), dtype=np.float32)
            self.z_index = self.dimz // 2
            if self.voxel_data is not None:
                self.threshold = [10 for _ in range(self.dimz)]
            self.initial_load = True
            self.update_display_slice()
            self.root.title(f"Klepar - {self.file_name}")
            self.bucket_layer_slider.configure(from_=0, to=self.dimz - 1)
            self.bucket_layer_slider.set(0)
            self.update_log(f"Data loaded successfully.")
        except Exception as e:
            self.update_log(f"Error loading data: {e}")

    def clear_slice_cache(self):
        self.slice_cache = {}
        gc.collect()

    def load_prediction(self):
        if self.voxel_data is None:
            self.update_log("No voxel data loaded. Load voxel data first.")
            return
        self.prediction_loaded = False
        # File dialog to select prediction PNG file
        pred_file_path = filedialog.askopenfilename(title="Select Prediction PNG", filetypes=[("PNG files", "*.png")])

        if pred_file_path:
            try:
                # Load the prediction PNG file
                loaded_prediction = Image.open(pred_file_path)

                # Convert the image to a NumPy array
                prediction_data_np = np.array(loaded_prediction)

                # Calculate padding and remove it
                '''
                pad0 = (64 - self.voxel_data.shape[1] % 64) # 64 tile size
                pad1 = (64 - self.voxel_data.shape[2] % 64)
                if pad0 or pad1:
                    prediction_data_np = prediction_data_np[:-pad0, :-pad1]
                '''
                self.prediction_data = prediction_data_np
                # Check if the dimensions match
                if self.prediction_data.shape[:2] == (self.dimy, self.dimx):
                    self.update_display_slice()
                    self.prediction_loaded = True
                    self.update_log("Prediction loaded successfully.")
                else:
                    self.update_log("Error: Prediction dimensions do not match the voxel data dimensions.")
            except Exception as e:
                self.update_log(f"Error loading prediction: {e}")

    def load_mask(self):
        if self.voxel_data is None:
            print("LOG: No voxel data loaded. Load voxel data first.")
            return

        # Prompt to save changes if there are any unsaved changes
        # if self.history:
        #     if not tk.messagebox.askyesno("Unsaved Changes", "You have unsaved changes. Do you want to continue without saving?"):
        #         return

        # kglspl: specific changes - masks are named mask_*.tif, they are made with stride=8

        # File dialog to select mask file
        mask_filename = filedialog.askopenfilename(
            initialdir=self.default_masks_directory,
            title="Select Masks TIFF File",
            filetypes=[("Mask TIFF files", "mask_*.tif")]
        )

        if not mask_filename:
            print("LOG: Not loading mask data, cancelled.")
            return

        try:
            im = Image.open(mask_filename)
            # self.mask_data = np.zeros(shape=(self.dataset.shape[0], self.dataset.shape[1]))  # axes: x, y
            d = 4 // self.stride
            if d != 1:
                im = im.resize((im.size[0] * d, im.size[1] * d), Image.NEAREST)
            mask = np.array(im, dtype=np.uint8).T / 255
            print('mask', mask.shape, mask.dtype, mask.min(), mask.max())
            assert len(mask.shape) == 2
            print('roi', self.roi)
            x0 = self.roi['x'][0] // self.stride
            y0 = self.roi['y'][0] // self.stride
            part = mask[x0:x0 + self.mask_data.shape[2], y0:y0 + self.mask_data.shape[1]].T
            print('part', part.shape, part.dtype, part.min(), part.max())

            mask_data = np.stack([part for _ in range(self.mask_data.shape[0])], axis=0)
            self.mask_data = mask_data

        except Exception as e:
            # print(f"LOG: Error loading mask: {e}")
            raise


    def save_image(self):
        if self.mask_data is not None:
            # Construct the default file name for saving
            base_name = os.path.splitext(os.path.basename(self.file_name))[0]
            default_save_file_name = f"{base_name}_label.zarr"
            parent_directory = os.path.join(self.file_name, os.pardir)
            # Open the file dialog with the proposed file name
            save_file_path = filedialog.asksaveasfilename(
                initialdir=parent_directory,
                title="Select Directory to Save Mask Zarr",
                initialfile=default_save_file_name,
                filetypes=[("Zarr files", "*.zarr")]
            )

            if save_file_path:
                try:
                    # Save the Zarr array to the chosen file path
                    zarr.save_array(save_file_path, self.mask_data)
                    self.update_log(f"Mask saved as Zarr in {save_file_path}")
                except Exception as e:
                    self.update_log(f"Error saving mask as Zarr: {e}")
        else:
            self.update_log("No mask data to save.")

    def save_mask_3d(self):
        if self.mask_data is None:
            self.update_log("No mask data to save.")
            return

        # Construct the default file name for saving
        default_save_file_name = f"{self.default_masks_directory}mask_xxx.h5"
        # Open the file dialog with the proposed file name
        save_file_path = filedialog.asksaveasfilename(
            initialdir=self.default_masks_directory,
            title="Select Directory to Save Mask (H5FS)",
            initialfile=default_save_file_name,
            filetypes=[("H5 mask files", "mask_*.h5")]
        )

        if not save_file_path:
            return

        try:
            with h5py.File(save_file_path, 'a') as f:
                shape = (self.dataset_shape_xyz[2], self.dataset_shape_xyz[1] * self.stride, self.dataset_shape_xyz[0] * self.stride)
                dset = f.require_dataset('maskzyx', shape, dtype=bool)
                if self.stride != 1:
                    mask = scipy.ndimage.zoom(self.mask_data, (1, self.stride, self.stride), order=0, grid_mode=True)
                else:
                    mask = self.mask_data
                dset[:, self.roi['y'][0]:self.roi['y'][0]+mask.shape[1], self.roi['x'][0]:self.roi['x'][0]+mask.shape[2]] = mask
            self.update_log(f"Mask saved as H5FS in {save_file_path}")
        except Exception as e:
            self.update_log(f"Error saving mask as H5FS: {e}")

    def save_flattened_mask(self):
        if self.mask_data is None:
            print("LOG: No mask data to save.")
            return

        # Construct the default file name for saving
        # base_name = os.path.splitext(os.path.basename(self.file_name))[0]
        # default_save_file_name = f"mask_{base_name}.tif"
        # Open the file dialog with the proposed file name
        save_file_path = filedialog.asksaveasfilename(
            initialdir=self.default_masks_directory,
            title="Select File to Save Mask to",
            initialfile='mask_xxx.tif',
            filetypes=[("Mask TIFF files", "mask_*.tif")]
        )

        if not save_file_path:
            print("LOG: Not saving mask data, cancelled.")
            return

        full_data = None
        expected_shape = ((self.dataset_shape_xyz[1] * self.stride) // 4, (self.dataset_shape_xyz[0] * self.stride) // 4)
        if os.path.exists(save_file_path):
            full_im = Image.open(save_file_path)
            full_data = np.array(full_im, dtype=np.uint8)
            if full_data.shape != expected_shape:
                raise Exception(f"Sorry, looks like this mask is a wrong shape for this dataset with this stride ({self.stride})")
        else:
            full_data = np.zeros(expected_shape, dtype=np.uint8)

        try:
            # Save the flattened mask as TIFF to the chosen file path
            data = np.max(self.mask_data, axis=0).astype(bool).astype(np.uint8).T * 255

            # Resize data:
            im = Image.fromarray(data, 'L')
            im = im.resize(((im.size[0] * self.stride) // 4, (im.size[1] * self.stride) // 4), Image.NEAREST)
            data = np.array(im, dtype=np.uint8)

            # Apply it to full_data (from the mask file)
            full_data[self.roi['y'][0] // 4:self.roi['y'][1] // 4, self.roi['x'][0] // 4:self.roi['x'][1] // 4] = data.T

            # Depending on the file we used, stride could be 1, 2, 4,... However our masks are saved always with stride 4, so let's resize as needed:
            im = Image.fromarray(full_data, 'L')
            # im.thumbnail((data.shape[1] // shrink_factor, data.shape[0] // shrink_factor), Image.Resampling.LANCZOS)
            im.save(save_file_path, 'TIFF')
            print(f"LOG: Mask saved as TIFF in {save_file_path}, size {im.size}")
        except Exception as e:
            print(f"LOG: Error saving mask: {e}")
            raise

    def update_threshold_layer(self, layer):
        try:
            self.th_layer = int(float(layer))
            self.bucket_layer_var.set(f"{self.th_layer}")

            # Update the Bucket Threshold Slider to the current layer's threshold value
            current_threshold = self.threshold[self.th_layer]
            self.bucket_threshold_var.set(f"{current_threshold}")
            # You may need to adjust this line depending on how the slider is named in your code
            self.bucket_threshold_slider.set(current_threshold)

            self.update_log(f"Layer {self.th_layer} selected, current threshold is {current_threshold}.")
        except ValueError:
            self.update_log("Invalid layer value.")

    def update_threshold_value(self, val):
        try:
            self.threshold[self.th_layer] = int(float(val))
            self.bucket_threshold_var.set(f"{int(float(val))}")
            self.update_log(f"Layer {self.th_layer} threshold set to {self.threshold[self.th_layer]}.")
        except ValueError:
            self.update_log("Invalid threshold value.")

    def threaded_flood_fill(self):
        if self.click_coordinates and self.voxel_data is not None:
            # Run flood_fill_3d in a separate thread
            thread = threading.Thread(target=self.flood_fill_3d, args=(self.click_coordinates,))
            thread.start()
        else:
            self.update_log("No starting point or data for flood fill.")

    def flood_fill_3d(self, start_coord):
        self.flood_fill_active = True
        if self.format in ['zarr', 'h5fs']:
            target_color = int(self.voxel_data[start_coord])
        elif self.format == 'tiff':
            z, y, x = start_coord
            target_color = int(self.voxel_data[z][y,x])
        queue = deque([start_coord])
        visited = set()

        counter = 0
        while self.flood_fill_active and queue and counter < self.max_propagation_steps:
            cz, cy, cx = queue.popleft()

            if (cz, cy, cx) in visited or not (0 <= cz < self.dimz and 0 <= cy < self.dimy and 0 <= cx < self.dimx):
                continue

            visited.add((cz, cy, cx))

            if self.barrier_mask[cz, cy, cx] != 0:
                continue

            if self.format in ['zarr', 'h5fs']:
                voxel_value = int(self.voxel_data[cz, cy, cx])
                print(voxel_value)

            elif self.format == 'tiff':
                voxel_value = int(self.voxel_data[cz][cy, cx])
            if abs(voxel_value - target_color) <= self.threshold[cz]:
                self.mask_data[cz, cy, cx] = 1
                counter += 1
                for dz in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dz == 0 and dx == 0 and dy == 0:
                                continue
                            queue.append((cz + dz, cy + dy, cx + dx))

            if counter % 10 == 0:
                self.root.after(1, self.update_display_slice)
        if self.flood_fill_active == True:
            self.flood_fill_active = False
            self.update_log("Flood fill ended.")
            self.update_display_slice()

    def stop_flood_fill(self):
        self.flood_fill_active = False
        self.update_log("Flood fill stopped.")

    def save_state(self):
        # Save the current state of the image before modifying it
        if self.mask_data is not None:
            if len(self.history) == self.max_history_size:
                self.history.pop(0)  # Remove the oldest state
            self.history.append(self.mask_data.copy())

    def undo_last_action(self):
        if self.history:
            self.mask_data = self.history.pop()
            self.update_display_slice()
            self.update_log("Last action undone.")
        else:
            self.update_log("No more actions to undo.")

    def on_canvas_press(self, event):
        self.drag_start_x = event.x
        self.drag_start_y = event.y
    '''
    def on_canvas_drag(self, event):
        if self.drag_start_x is not None and self.drag_start_y is not None:
            dx = event.x - self.drag_start_x
            dy = event.y - self.drag_start_y
            self.image_position_x += dx
            self.image_position_y += dy
            self.drag_start_x, self.drag_start_y = event.x, event.y
    '''

    def on_canvas_drag(self, event):
        if self.drag_start_x is None or self.drag_start_y is None:
            return

        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y
        self.translate(dx, dy)
        self.update_display_slice()
        self.drag_start_x, self.drag_start_y = event.x, event.y
        self.update_nav3d_display()

    def on_canvas_pencil_drag(self, event):
        if self.mode.get() == "pencil" or self.mode.get() == "eraser":
            self.save_state()
            self.color_pixel(self.calculate_image_coordinates(event))

    def on_canvas_release(self, event):
        self.drag_start_x = None
        self.drag_start_y = None
        self.update_display_slice()
        self.update_nav3d_display()

    def resize_with_aspect(self, image, target_width, target_height, zoom=1):
        original_width, original_height = image.size
        zoomed_width, zoomed_height = int(original_width * zoom), int(original_height * zoom)
        aspect_ratio = original_height / original_width
        new_height = int(target_width * aspect_ratio)
        new_height = min(new_height, target_height)
        return image.resize((zoomed_width, zoomed_height), Image.Resampling.NEAREST)

    def resize_to_fit_canvas(self, image, canvas_width, canvas_height):
        """Resize image to fit the canvas while maintaining aspect ratio."""
        original_width, original_height = image.size
        aspect_ratio = original_width / original_height

        if canvas_width / canvas_height > aspect_ratio:
            new_width = int(aspect_ratio * canvas_height)
            new_height = canvas_height
        else:
            new_width = canvas_width
            new_height = int(canvas_width / aspect_ratio)

        self.zoom_level = min(new_width / original_width, new_height / original_height)

        return image.resize((new_width, new_height), Image.Resampling.NEAREST)

    def update_display_slice(self):
        if self.voxel_data is not None and self.canvas is not None:
            target_width_xy = self.canvas.winfo_width()
            target_height_xy = self.canvas.winfo_height()

            # Convert the current slice to an RGBA image
            if self.show_image:
                if self.z_index in self.slice_cache:
                    img = self.slice_cache[self.z_index]
                else:
                    img = self.prepare_image_slice(self.z_index)
            else:
                img = Image.new('RGBA', (target_width_xy, target_height_xy))

            # Only overlay the mask if show_mask is True
            if self.mask_data is not None and self.show_mask:
                if self.flatten_mask:
                    mask = np.uint8(np.amax(self.mask_data, axis=0) * self.overlay_alpha)
                else:
                    mask = np.uint8(self.mask_data[self.z_index, :, :] * self.overlay_alpha)
                yellow = np.zeros_like(mask, dtype=np.uint8)
                yellow[:, :] = 255  # Yellow color
                mask_img = Image.fromarray(np.stack([yellow, yellow, np.zeros_like(mask), mask], axis=-1), 'RGBA')

                # Overlay the mask on the original image
                img = Image.alpha_composite(img, mask_img)

            if self.barrier_mask is not None and self.show_barrier:
                barrier = np.uint8(self.barrier_mask[self.z_index, :, :] * self.overlay_alpha)
                red = np.zeros_like(barrier, dtype=np.uint8)
                red[:, :] = 255  # Red color
                barrier_img = Image.fromarray(np.stack([red, np.zeros_like(barrier), np.zeros_like(barrier), barrier], axis=-1), 'RGBA')

                # Overlay the barrier mask on the original image
                img = Image.alpha_composite(img, barrier_img)

            if self.prediction_data is not None and self.show_prediction:
                if self.prediction_loaded == False:
                    pred = np.uint8(self.prediction_data[:, :] * self.overlay_alpha)
                    blue = np.zeros_like(pred, dtype=np.uint8)
                    blue[:, :] = 255  # Red color
                    self.pred_img = Image.fromarray(np.stack([np.zeros_like(pred), np.zeros_like(pred), blue, pred], axis=-1), 'RGBA')

                # Overlay the barrier mask on the original image
                img = Image.alpha_composite(img, self.pred_img)

            if self.surface_adjuster_nodes:
                # we want to draw nodes in different color depending on z
                red = np.zeros((self.dimy, self.dimx), dtype=np.uint8)
                green = np.zeros((self.dimy, self.dimx), dtype=np.uint8)
                blue = np.zeros((self.dimy, self.dimx), dtype=np.uint8)
                mask = np.zeros((self.dimy, self.dimx), dtype=np.uint8)
                rect_size = math.ceil(5. / self.zoom_level)
                for z, y, x in self.surface_adjuster_nodes:
                    if z <= self.z_index:
                        luminosity = round((z / max(self.z_index, 1)) * 255)
                        red[y-rect_size:y+rect_size, x-rect_size:x+rect_size] = luminosity
                    if z >= self.z_index:
                        luminosity = round((self.z_index / max(z, 1)) * 255)
                        green[y-rect_size:y+rect_size, x-rect_size:x+rect_size] = luminosity
                    blue[y-rect_size:y+rect_size, x-rect_size:x+rect_size] = 255 if z == self.z_index else luminosity // 2
                    mask[y-rect_size:y+rect_size, x-rect_size:x+rect_size] = 255

                nodes_img = Image.fromarray(np.stack([red, green, blue, mask], axis=-1), 'RGBA')
                img = Image.alpha_composite(img, nodes_img)

                # draw triangulation lines between nodes:
                mask = np.zeros((self.dimy, self.dimx), dtype=np.uint8)
                if self.surface_adjuster_tri is not None:
                    for corners in self.surface_adjuster_tri.simplices:
                        for k in range(3):
                            _, n0y, n0x = self.surface_adjuster_nodes[corners[k]]
                            _, n1y, n1x = self.surface_adjuster_nodes[corners[(k+1) % 3]]
                            rr, cc, val = line_aa(n0y, n0x, n1y, n1x)
                            mask[rr, cc] = val * 255
                lines_img = Image.fromarray(np.stack([mask, mask, np.zeros_like(mask), mask], axis=-1), 'RGBA')
                img = Image.alpha_composite(img, lines_img)

                    # Resize the image with aspect ratio
            '''
            if self.initial_load:
                img = self.resize_to_fit_canvas(img, target_width_xy, target_height_xy)
                self.initial_load = False
            else:
                img = self.resize_with_aspect(img, target_width_xy, target_height_xy, zoom=self.zoom_level)

            # Convert back to a format that can be displayed in Tkinter
            self.resized_img = img.convert('RGB')
            self.photo_img = ImageTk.PhotoImage(image=self.resized_img)
            self.canvas.create_image(self.image_position_x, self.image_position_y, anchor=tk.NW, image=self.photo_img)
            self.canvas.tag_raise(self.z_slice_text)
            self.canvas.tag_raise(self.cursor_pos_text)
            '''

            # Apply the affine transformation
            mat_inv = np.linalg.inv(self.mat_affine)
            affine_inv = (
                mat_inv[0, 0], mat_inv[0, 1], mat_inv[0, 2],
                mat_inv[1, 0], mat_inv[1, 1], mat_inv[1, 2]
            )

            # Transform the image using the affine matrix
            self.resized_img = img.transform(
                (target_width_xy, target_height_xy),
                Image.AFFINE,
                affine_inv,
                Image.Resampling.NEAREST
            )

            # Convert back to a format that can be displayed in Tkinter
            self.photo_img = ImageTk.PhotoImage(image=self.resized_img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_img)

            self.canvas.tag_raise(self.z_slice_text)
            self.canvas.tag_raise(self.zoom_text)
            self.canvas.tag_raise(self.cursor_pos_text)

    def update_center_coordinates(self):
        # Get canvas coordinates and create crosshair on the right (surface) canvas:
        pw, ph = self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2
        self.canvas.create_line((pw-10, ph), (pw+10, ph), width=1, fill='red')
        self.canvas.create_line((pw, ph-10), (pw, ph+10), width=1, fill='red')

        # Get surface coordinates:
        _, surface_y_px, surface_x_px = self.calculate_image_coordinates((None, ph, pw))
        surface_x, surface_y = surface_x_px * self.stride + self.roi['x'][0], surface_y_px * self.stride + self.roi['y'][0]

        # Get corresponding 3D coordinates:
        scroll_x, scroll_y, scroll_z, scroll_nx, scroll_ny, scroll_nz = self.ppm.get_3d_coords(surface_x, surface_y, rounded_xyz=True)
        self.center_coordinates = surface_x, surface_y, scroll_x, scroll_y, scroll_z, scroll_nx, scroll_ny, scroll_nz

    def update_nav3d_display(self):
        if self.voxel_data is None:
            return

        self.update_center_coordinates()
        surface_x, surface_y, scroll_x, scroll_y, scroll_z, scroll_nx, scroll_ny, scroll_nz = self.center_coordinates

        # Load images from scroll 3D data and show it on nav3d canvases:
        imgs = []
        for i, c in enumerate([self.canvas_z, self.canvas_x, self.canvas_y]):
            pw, ph = c.winfo_width() // 2, c.winfo_height() // 2
            if i == 0:
                img_data = (self.scroll_dataset[scroll_y-ph:scroll_y+ph, scroll_x-pw:scroll_x+pw, scroll_z] // 256).astype(np.uint8)
            elif i == 1:
                img_data = (self.scroll_dataset[scroll_y-ph:scroll_y+ph, scroll_x, scroll_z-pw:scroll_z+pw] // 256).astype(np.uint8)
            else:
                img_data = (self.scroll_dataset[scroll_y, scroll_x-ph:scroll_x+ph, scroll_z-pw:scroll_z+pw] // 256).astype(np.uint8)
            img = Image.fromarray(img_data).convert('RGBA')
            imgs.append(img)

        # Draw vicinity points on images while still in Image format:
        imgs = self.draw_vicinity_points_on_nav3d(imgs, pw, ph, surface_x, surface_y, scroll_x, scroll_y, scroll_z)
        imgs = self.draw_normals_on_nav3d(imgs, pw, ph, scroll_nx, scroll_ny, scroll_nz)

        # Render photoimages on nav3d canvases:
        self.canvas_3d_photoimgs = []  # PhotoImage's must be saved on instance or they will be garbage collected before displayed
        for i, c in enumerate([self.canvas_z, self.canvas_x, self.canvas_y]):
            self.canvas_3d_photoimgs.append(ImageTk.PhotoImage(image=imgs[i]))  # must be on instance or it will be garbage collected before it is displayed
            c.create_image(5, 5, anchor=tk.NW, image=self.canvas_3d_photoimgs[i])

        # Draw center navigation lines on canvases:
        b = 5  # border offset
        for i, c in enumerate([self.canvas_z, self.canvas_x, self.canvas_y]):
            if i == 0:  # 0 == z
                c.create_line((pw+b, 0), (pw+b, ph//2), width=2, fill='red')
                c.create_line((pw+b, round(1.5*ph)), (pw+b, 2 * ph + 1), width=2, fill='red')
                c.create_line((0, ph+b), (pw//2, ph+b), width=2, fill='blue')
                c.create_line((round(1.5*pw), ph+b), (2*pw + 1, ph+b), width=2, fill='blue')
            elif i == 1:  # 1 == x
                c.create_line((pw+b, 0), (pw+b, ph//2), width=2, fill='green')
                c.create_line((pw+b, round(1.5*ph)), (pw+b, 2 * ph + 1), width=2, fill='green')
                c.create_line((0, ph+b), (pw//2, ph+b), width=2, fill='blue')
                c.create_line((round(1.5*pw), ph+b), (2*pw + 1, ph+b), width=2, fill='blue')
            else:  # 2 == y
                c.create_line((pw+b, 0), (pw+b, ph//2), width=2, fill='green')
                c.create_line((pw+b, round(1.5*ph)), (pw+b, 2 * ph + 1), width=2, fill='green')
                c.create_line((0, ph+b), (pw//2, ph+b), width=2, fill='red')
                c.create_line((round(1.5*pw), ph+b), (2*pw + 1, ph+b), width=2, fill='red')

        # Update labels with 3D coordinates:
        self.canvas_z.itemconfigure(self.canvas_z_text, text=f"Z: {scroll_z}")
        self.canvas_x.itemconfigure(self.canvas_x_text, text=f"X: {scroll_x}")
        self.canvas_y.itemconfigure(self.canvas_y_text, text=f"Y: {scroll_y}")
        self.canvas_z.tag_raise(self.canvas_z_text)
        self.canvas_x.tag_raise(self.canvas_x_text)
        self.canvas_y.tag_raise(self.canvas_y_text)

    def draw_vicinity_points_on_nav3d(self,
                                        imgs, pw, ph,
                                        center_surface_x, center_surface_y,
                                        center_scroll_x, center_scroll_y, center_scroll_z,
                                        padding=50, stride=1):
        # get 3D points for the vicinity of surface x/y, if they match 3D x/y/z slice that we are showing, draw a marker there
        for sx in range(center_surface_x - padding, center_surface_x + padding + 1, stride):
            for sy in range(center_surface_y - padding, center_surface_y + padding + 1, stride):
                scroll_x, scroll_y, scroll_z, _, _, _ = self.ppm.get_3d_coords(sx, sy, rounded_xyz=True)

                if sx == center_surface_x and sy == center_surface_y:
                    color = (0xff, 0xff, 0x00)
                else:
                    color = (0xff, 0x66, 0x00)
                if scroll_z == center_scroll_z:
                    imgs[0].putpixel((pw + scroll_x - center_scroll_x, ph + scroll_y - center_scroll_y), color)
                if scroll_x == center_scroll_x:
                    imgs[1].putpixel((pw + scroll_z - center_scroll_z, ph + scroll_y - center_scroll_y), color)
                if scroll_y == center_scroll_y:
                    imgs[2].putpixel((pw + scroll_z - center_scroll_z, ph + scroll_x - center_scroll_x), color)

        return imgs

    def draw_normals_on_nav3d(self, imgs, pw, ph, scroll_nx, scroll_ny, scroll_nz, color=(0xff, 0xff, 0x00), color_zindex=(0xff, 0x00, 0xff), length=20):
        zindex_diff = self.z_index - self.dimz // 2
        draw = ImageDraw.Draw(imgs[0])
        draw.line((pw, ph, pw+round(scroll_nx * length), ph+round(scroll_ny * length)), fill=color)
        draw.line((pw, ph, pw+round(scroll_nx * zindex_diff), ph+round(scroll_ny * zindex_diff)), fill=color_zindex)
        draw = ImageDraw.Draw(imgs[1])
        draw.line((pw, ph, pw+round(scroll_nz * length), ph+round(scroll_ny * length)), fill=color)
        draw.line((pw, ph, pw+round(scroll_nz * zindex_diff), ph+round(scroll_ny * zindex_diff)), fill=color_zindex)
        draw = ImageDraw.Draw(imgs[2])
        draw.line((pw, ph, pw+round(scroll_nz * length), ph+round(scroll_nx * length)), fill=color)
        draw.line((pw, ph, pw+round(scroll_nz * zindex_diff), ph+round(scroll_nx * zindex_diff)), fill=color_zindex)
        return imgs

    def update_info_display(self):
        if self.voxel_data is None:
            return

        self.canvas.itemconfigure(self.z_slice_text, text=f"Z-Slice: {self.z_index}")
        self.canvas.itemconfigure(self.zoom_text, text=f"Zoom: {self.zoom_level:.2f}")
        if self.click_coordinates:
            try:
                _, cursor_y, cursor_x = self.calculate_image_coordinates(self.click_coordinates)
            except:
                cursor_x, cursor_y = 0, 0
            offset = self.surface_adjuster_offsets[cursor_y, cursor_x]

            surface_x = cursor_x * self.stride + self.roi['x'][0]
            surface_y = cursor_y * self.stride + self.roi['y'][0]
            self.canvas.itemconfigure(self.cursor_pos_text, text=f"Cursor Surface Position: ({surface_x}, {surface_y}, offset {offset:.2f})")

        self.update_nav3d_display()

    def on_canvas_click(self, event):
        self.save_state()
        img_coords = self.calculate_image_coordinates(event)
        mode = self.mode.get()
        if mode == "bucket":
            if self.flood_fill_active == True:
                self.update_log("Last flood fill hasn't finished yet.")
            else:
                # Assuming the flood fill functionality
                self.click_coordinates = img_coords
                self.update_log("Starting flood fill...")
                self.threaded_flood_fill()  # Assuming threaded_flood_fill is implemented for non-blocking UI
        elif mode in ["pencil", "eraser"]:
            # Assuming the pencil (pixel editing) functionality
            self.color_pixel(img_coords)  # Assuming color_pixel is implemented
        elif mode == "surface-adjuster":
            if self.show_surface_offsets:
                # User was looking at the image which was modified ('Show Offsets' is on), so we must take into account any offsets already applied:
                _, y, x = self.click_coordinates
                self.click_coordinates = (self.z_index + self.surface_adjuster_offsets[y, x], y, x)

            bounds_affected = None
            # Toggle node:
            existing_index = self.near_existing_surface_adjuster_node(img_coords)
            if existing_index is not None:
                bounds_affected = self.remove_surface_adjuster_node(existing_index)
            else:
                self.add_surface_adjuster_node(img_coords)

            self.save_surface_adjust_file()

            if len(self.surface_adjuster_nodes) > 2:
                # triangulation: https://docs.scipy.org/doc/scipy/tutorial/spatial.html
                points = np.array([(y, x) for _, y, x in self.surface_adjuster_nodes])
                self.surface_adjuster_tri = Delaunay(points)

                if existing_index is None:  # if we added a new node, we need to get the affected bounds here, when we have the new triangulation
                    new_node_index = len(self.surface_adjuster_nodes) - 1
                    bounds_affected = self.get_surface_adjuster_bounds_affected_on_toggle(new_node_index)
            else:
                self.surface_adjuster_tri = None

            # Updating canvas in real-time is too slow (even with bounds affected) so it is disabled here - we have a button for updating everything now.
            # if bounds_affected is not None:
            #     self.update_surface_adjuster_offsets(bounds_affected)


    def near_existing_surface_adjuster_node(self, img_coords):
        z, y, x = img_coords
        for i, (nz, ny, nx) in enumerate(self.surface_adjuster_nodes):
            if abs(nx - x) < 5 and abs(ny - y) < 5:
                return i

        return None

    def remove_surface_adjuster_node(self, existing_index):
        # before deleting node, find its past neighbors, then extract affected bounds so that we can refresh just that part of offsets array:
        bounds_affected = self.get_surface_adjuster_bounds_affected_on_toggle(existing_index) if self.surface_adjuster_tri else None

        del self.surface_adjuster_nodes[existing_index]
        return bounds_affected

    def add_surface_adjuster_node(self, img_coords):
        z, y, x = img_coords
        self.surface_adjuster_nodes.append((z, y, x))

    def get_surface_adjuster_bounds_affected_on_toggle(self, node_index):
        # As an optimization, we only update offsets where they changed, that is, with bounds from its neighbors and itself:
        # - if we remove an existing node: before deleting it
        # - if we add a new node: after adding it
        affected_triangles = [t for t in self.surface_adjuster_tri.simplices if node_index in t]
        their_nodes_indexes = set(itertools.chain.from_iterable(affected_triangles))
        nodes_affected = np.array([self.surface_adjuster_nodes[i] for i in their_nodes_indexes])
        min_y, max_y, min_x, max_x = nodes_affected[:, 1].min(), nodes_affected[:, 1].max(), nodes_affected[:, 2].min(), nodes_affected[:, 2].max()
        return min_y, max_y, min_x, max_x

    def threaded_update_surface_adjuster_offsets(self):
        if self.surface_adjuster_tri is None:
            self.update_log("No surface adjuster triangulation available")
            self.surface_adjuster_offsets[:, :] = 0  # reset offsets (they might have been set from before, when triangulation still existed)
            self.clear_slice_cache()
            return

        if self._threaded_update_surface_adjuster_offsets_running:
            self.update_log("Surface adjuster offsets already updating, ignoring")
            return

        self._threaded_update_surface_adjuster_offsets_running = True
        thread = threading.Thread(target=self.update_surface_adjuster_offsets, args=(None,))
        thread.start()

    def print_3d_bounding_box(self):
        resolution = 10
        # initial values:
        max_scroll_x, max_scroll_y, max_scroll_z, _, _, _ = self.ppm.get_3d_coords(self.roi['x'][0], self.roi['y'][0], rounded_xyz=True)
        min_scroll_x, min_scroll_y, min_scroll_z = max_scroll_x, max_scroll_y, max_scroll_z

        for sx in range(self.roi['x'][0], self.roi['x'][1], resolution):
            for sy in range(self.roi['y'][0], self.roi['y'][1], resolution):
                scroll_x, scroll_y, scroll_z, _, _, _ = self.ppm.get_3d_coords(sx, sy, rounded_xyz=True)
                min_scroll_x, max_scroll_x = min(min_scroll_x, scroll_x), max(max_scroll_x, scroll_x)
                min_scroll_y, max_scroll_y = min(min_scroll_y, scroll_y), max(max_scroll_y, scroll_y)
                min_scroll_z, max_scroll_z = min(min_scroll_z, scroll_z), max(max_scroll_z, scroll_z)

        print('3d bbox:', (min_scroll_x, max_scroll_x, min_scroll_y, max_scroll_y, min_scroll_z, max_scroll_z))
        self.update_log('3d bbox:' + str((min_scroll_x, max_scroll_x, min_scroll_y, max_scroll_y, min_scroll_z, max_scroll_z)))

    def update_surface_adjuster_offsets(self, bounds_affected):
        # given the triangulation (in self.surface_adjuster_tri) we need to update offsets
        # this is probably not the most efficient implementation...
        tri = self.surface_adjuster_tri
        if tri is None:
            return
        nodes = np.array(self.surface_adjuster_nodes)
        if bounds_affected is None:
            min_y, max_y, min_x, max_x = nodes[:, 1].min(), nodes[:, 1].max(), nodes[:, 2].min(), nodes[:, 2].max()
        else:
            min_y, max_y, min_x, max_x = bounds_affected
        # print('bounds:', min_y, max_y, min_x, max_x)

        planes = [Plane(Point(*nodes[t[0]]), Point(*nodes[t[1]]), Point(*nodes[t[2]])) for t in tri.simplices]

        progress = ProgressPrinter(min_y, max_y, 'Calculating surface adjuster offset', print_=self.update_log)
        for y in range(min_y, max_y + 1):
            progress.progress(y)
            for x in range(min_x, max_x + 1):
                s = tri.find_simplex([y, x])
                if s < 0:
                    continue
                isect = planes[s].meet(Line(Point(0, y, x), Point(self.dimz, y, x)))
                self.surface_adjuster_offsets[y, x] = isect.normalized_array[0] - self.dimz // 2

        self.clear_slice_cache()
        self._threaded_update_surface_adjuster_offsets_running = False
        self.save_surface_adjust_offsets()
        self.update_log("Surface adjuster offsets updated and saved.")

    def calculate_image_coordinates(self, input):
        if input is None:
            return 0, 0, 0  # Default values
        if isinstance(input, tuple):
            _, y, x = input
        elif hasattr(input, 'x') and hasattr(input, 'y'):
            x, y = input.x, input.y
        else:
            # Handle unexpected input types
            raise ValueError("Input must be a tuple or an event object")

        if self.voxel_data is None:
            return 0, 0, 0

        # Apply the inverse of the affine transformation to the clicked coordinates
        mat_inv = np.linalg.inv(self.mat_affine)
        transformed_point = np.dot(mat_inv, [x, y, 1])

        # Extract the image coordinates from the transformed point
        img_x = int(transformed_point[0])
        img_y = int(transformed_point[1])

        # Ensure the coordinates are within the bounds of the image
        img_x = max(0, min(img_x, self.voxel_data.shape[2] - 1))
        img_y = max(0, min(img_y, self.voxel_data.shape[1] - 1))

        return self.z_index, img_y, img_x

    def color_pixel(self, img_coords):
        z_index, center_y, center_x = img_coords
        if self.voxel_data is not None:
            # Calculate the square bounds of the circle
            min_x = max(0, center_x - self.pencil_size)
            max_x = min(self.dimx - 1, center_x + self.pencil_size)
            min_y = max(0, center_y - self.pencil_size)
            max_y = min(self.dimx - 1, center_y + self.pencil_size)

        if self.mode.get() in ["pencil", "eraser"]:
            # Decide which mask to edit based on editing_barrier flag
            target_mask = self.barrier_mask if self.editing_barrier else self.mask_data
            mask_value = 1 if self.mode.get() == "pencil" else 0
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    # Check if the pixel is within the circle's radius
                    if math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) <= self.pencil_size:
                            target_mask[z_index, y, x] = mask_value
            self.update_display_slice()


    def update_pencil_size(self, val):
        self.pencil_size = int(float(val))
        self.pencil_size_var.set(f"{self.pencil_size}")
        self.update_log(f"Pencil size set to {self.pencil_size}")

    def update_pencil_cursor(self, event):
        # Remove the old cursor representation
        if self.pencil_cursor:
            self.canvas.delete(self.pencil_cursor)
            self.update_display_slice()

        if self.mode.get() == "pencil":
            color = "yellow" if not self.editing_barrier else "red"
        if self.mode.get() == "eraser":
            color = "white"
        if self.mode.get() == "eraser" or self.mode.get() == "pencil":
            radius = self.pencil_size * self.zoom_level  # Adjust radius based on zoom level
            self.pencil_cursor = self.canvas.create_oval(event.x - radius, event.y - radius, event.x + radius, event.y + radius, outline=color, width=2)
        self.click_coordinates = (self.z_index, event.y, event.x)
        self.update_info_display()

    def scroll_or_zoom(self, event):
        # Adjust for different platforms
        ctrl_pressed = False
        if sys.platform.startswith('win'):
            # Windows
            ctrl_pressed = event.state & 0x0004
            delta = event.delta
        elif sys.platform.startswith('linux') or sys.platform.startswith('darwin'):
            # Linux or macOS
            ctrl_pressed = event.state & 4
            delta = 1 if event.num == 4 else -1

        if ctrl_pressed:
            self.zoom(delta)
        else:
            self.scroll(delta)

        self.update_info_display()

    def scroll(self, delta):
        if self.voxel_data is not None:
            # Update the z_index based on scroll direction
            delta = 1 if delta > 0 else -1
            self.z_index = max(0, min(self.z_index + delta, self.dimz - 1))
            self.update_display_slice()

    '''
    def zoom(self, delta):
        zoom_amount = 0.1  # Adjust the zoom sensitivity as needed
        if delta > 0:
            self.zoom_level = min(self.max_zoom_level, self.zoom_level + zoom_amount)
        else:
            self.zoom_level = max(1, self.zoom_level - zoom_amount)
        self.update_display_slice()
    '''

    def translate(self, offset_x, offset_y):
        mat = np.eye(3)
        mat[0, 2] = float(offset_x)
        mat[1, 2] = float(offset_y)
        self.mat_affine = np.dot(mat, self.mat_affine)

    def scale(self, scale_factor, cx, cy):
        self.translate(-cx, -cy)
        mat = np.eye(3)
        mat[0, 0] = mat[1, 1] = scale_factor
        self.mat_affine = np.dot(mat, self.mat_affine)
        self.zoom_level = self.mat_affine[0, 0]
        self.translate(cx, cy)

    def zoom(self, delta):
        zoom_amount = 1.1 if delta > 0 else 0.9
        canvas_center_x = self.canvas.winfo_width() / 2
        canvas_center_y = self.canvas.winfo_height() / 2
        self.scale(zoom_amount, canvas_center_x, canvas_center_y)
        self.update_display_slice()

    def key_handler(self, ev):
        # print(repr(ev.keysym), ev.state)

        # Ctrl+C or Ctrl+Insert (copy to clipboard)
        if ev.state == 20 and ev.keysym in ['c', 'Insert']:
            try:
                _, _, scroll_x, scroll_y, scroll_z, _, _, _ = self.center_coordinates
            except Exception as ex:
                print(f'Copying 3D x/y/z coordinates to clipboard failed! {str(ex)}')
                return
            print(f'Copying 3D x/y/z coordinates to clipboard: {scroll_x}, {scroll_y}, {scroll_z}')
            pyperclip.copy(f'{scroll_x}, {scroll_y}, {scroll_z}')
            return

        # Ctrl+. (reset z-index)
        if ev.state == 20 and ev.keysym == 'period':
            print(f'Resetting z-index')
            self.z_index = self.dimz // 2
            self.update_display_slice()
            self.update_info_display()
            return

        # Ctrl+0 (reset zoom)
        if ev.state == 20 and ev.keysym == '0':
            print(f'Resetting zoom')
            self.zoom_level = self.mat_affine[0, 0] = self.mat_affine[1, 1] = 1.
            self.update_display_slice()
            self.update_info_display()
            return

        # Ctrl+S (save flattened mask)
        if ev.state == 20 and ev.keysym == 's':
            print(f'Saving flattened mask')
            self.save_flattened_mask()
            return

    def toggle_mask(self):
        # Toggle the state
        self.show_mask = not self.show_mask
        # Update the variable for the Checkbutton
        self.show_mask_var.set(self.show_mask)
        # Update the display to reflect the new state
        self.update_display_slice()
        self.update_log(f"Mask {'shown' if self.show_mask else 'hidden'}.\n")

    def toggle_flatten_mask(self):
        self.flatten_mask = not self.flatten_mask
        self.flatten_mask_var.set(self.flatten_mask)
        self.update_display_slice()

    def toggle_barrier(self):
        # Toggle the state
        self.show_barrier = not self.show_barrier
        # Update the variable for the Checkbutton
        self.show_barrier_var.set(self.show_barrier)
        # Update the display to reflect the new state
        self.update_display_slice()
        self.update_log(f"Barrier {'shown' if self.show_barrier else 'hidden'}.\n")

    def toggle_image(self):
        # Toggle the state
        self.show_image = not self.show_image
        # Update the variable for the Checkbutton
        self.show_image_var.set(self.show_image)
        # Update the display to reflect the new state
        self.update_display_slice()
        self.update_log(f"Image {'shown' if self.show_image else 'hidden'}.\n")

    def toggle_prediction(self):
        # Toggle the state
        self.show_prediction = not self.show_prediction
        # Update the variable for the Checkbutton
        self.show_prediction_var.set(self.show_prediction)
        # Update the display to reflect the new state
        self.update_display_slice()
        self.update_log(f"Ink predicton {'shown' if self.show_prediction else 'hidden'}.\n")

    def toggle_surface_offsets(self):
        self.show_surface_offsets = not self.show_surface_offsets
        # Update the variable for the Checkbutton
        self.show_surface_offsets_var.set(self.show_surface_offsets)
        # Update the display to reflect the new state
        self.clear_slice_cache()
        self.update_display_slice()
        self.update_log(f"Surface offsets {'shown' if self.show_surface_offsets else 'hidden'}.\n")

    def toggle_editing_mode(self):
        # Toggle between editing label and barrier
        self.editing_barrier = not self.editing_barrier
        self.update_log(f"Editing {'Barrier' if self.editing_barrier else 'Label'}")

    def update_alpha(self, val):
        self.overlay_alpha = int(float(val))
        self.update_display_slice()

    def show_help(self):
        help_window = tk.Toplevel(self.root)
        help_window.title("Info")
        help_window.geometry("800x700")  # Adjust size as necessary
        help_window.resizable(True, True)

        # Text widget with a vertical scrollbar
        help_text_widget = tk.Text(help_window, wrap="word", width=40, height=30)  # Adjust width and height as needed
        help_text_scrollbar = tk.Scrollbar(help_window, command=help_text_widget.yview)
        help_text_widget.configure(yscrollcommand=help_text_scrollbar.set)

        # Pack the scrollbar and text widget
        help_text_scrollbar.pack(side="right", fill="y")
        help_text_widget.pack(side="left", fill="both", expand=True)


        info_text = """Klepar: A tool for exploring and adjusting PPM surfaces for the Vesuvius Challenge (scrollprize.org).
Based on Vesuvian Kintsugi (created by Dr. Giorgio Angelotti, Vesuvius Kintsugi is designed for efficient 3D voxel image labeling: https://github.com/giorgioangel/vesuvius-kintsugi).
Released under the MIT license.
"""
        # Insert the help text into the text widget and disable editing
        help_text_widget.insert("1.0", info_text)

    def update_max_propagation(self, val):
        self.max_propagation_steps = int(float(val))
        self.max_propagation_var.set(f"{self.max_propagation_steps}")
        self.update_log(f"Max Propagation Steps set to {self.max_propagation_steps}")

    def update_log(self, message):
        if self.log_text is not None:
            self.log_text.insert(tk.END, message + "\n")
            self.log_text.see(tk.END)
        else:
            print(f"Log not ready: {message}")

    @staticmethod
    def create_tooltip(widget, text):
        # Implement a simple tooltip
        tooltip = tk.Toplevel(widget)
        tooltip.wm_overrideredirect(True)
        tooltip.wm_geometry("+0+0")
        tooltip.withdraw()

        label = tk.Label(tooltip, text=text, background="#FFFFE0", relief='solid', borderwidth=1, padx=1, pady=1)
        label.pack(ipadx=1)

        def enter(event):
            x = y = 0
            x, y, cx, cy = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 20
            tooltip.wm_geometry(f"+{x}+{y}")
            tooltip.deiconify()

        def leave(event):
            tooltip.withdraw()

        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)

    def on_exit(self):
        if self.format == 'h5fs':
            print("Closing H5 file.")
            self.h5_data_file.close()
            if self.h5_scroll_data_file:
                self.h5_scroll_data_file.close()
        self.ppm.close()

    # butchered from: https://stackoverflow.com/a/53340677
    def _h5_get_first_dataset_info(self, obj):
        if type(obj) in [h5py._hl.group.Group,h5py._hl.files.File]:
            for key in obj.keys():
                return self._h5_get_first_dataset_info(obj[key])
        elif type(obj)==h5py._hl.dataset.Dataset:
            return obj.name, obj.shape, obj.dtype, obj.chunks

    def save_surface_adjust_file(self):
        if not self.surface_adjust_filename:
            f = tempfile.NamedTemporaryFile(mode="w",suffix=".json",prefix="surface-")
            self.surface_adjust_filename = f.name
            self.update_log(f"File not specified, saving to {f.name}")
            json.dump(self.surface_adjuster_nodes, f)
            f.close()
            return

        with open(self.surface_adjust_filename, 'w') as f:
            json.dump(self.surface_adjuster_nodes, f)

    def load_surface_adjust_file(self):
        if not self.surface_adjust_filename:
            return
        try:
            with open(self.surface_adjust_filename, 'r') as f:
                self.surface_adjuster_nodes = json.load(f)
                # triangulation: https://docs.scipy.org/doc/scipy/tutorial/spatial.html
                points = np.array([(y, x) for _, y, x in self.surface_adjuster_nodes])
                self.surface_adjuster_tri = Delaunay(points)
        except FileNotFoundError:
            print(f"File not found: {self.surface_adjust_filename}, will create one for saving if needed.")

    def save_surface_adjust_offsets(self):
        filename = f'{self.surface_adjust_filename}.offsets.h5'
        with h5py.File(filename, 'a') as f:
            shape = (self.dataset_shape_xyz[0], self.dataset_shape_xyz[1])
            dset = f.require_dataset("offsets", shape=shape, dtype=np.float32)
            # print('dset.shape', dset.shape)
            # print('self.surface_adjuster_offsets.shape', self.surface_adjuster_offsets.shape)
            x0 = self.roi['x'][0] // self.stride
            y0 = self.roi['y'][0] // self.stride
            dset[x0:x0 + self.surface_adjuster_offsets.shape[1], y0:y0 + self.surface_adjuster_offsets.shape[0]] = self.surface_adjuster_offsets.T  # output: x, y
        self.update_log(f"Saved offsets to {filename}")
        print(f"Saved offsets to {filename}")

    def init_ui(self, arguments):
        self.root = tk.Tk()
        self.root.attributes('-zoomed', True)
        #self.root.iconbitmap("./icons/favicon.ico")
        self.root.title("Vesuvius Klepar")

        # Use a ttk.Style object to configure style aspects of the application
        style = ttk.Style()
        style.configure('TButton', padding=5)  # Add padding around buttons
        style.configure('TFrame', padding=5)  # Add padding around frames

        # Create a toolbar frame at the top with some padding
        self.toolbar_frame = ttk.Frame(self.root, padding="5 5 5 5")
        self.toolbar_frame.pack(side=tk.TOP, fill=tk.X)

        # Create a drawing tools frame
        drawing_tools_frame = tk.Frame(self.toolbar_frame)
        drawing_tools_frame.pack(side=tk.LEFT, padx=5)

        # Load and set icons for buttons (icons need to be added)
        load_icon = PhotoImage(file='./icons/open-64.png')
        save_icon = PhotoImage(file='./icons/save-64.png')
        prediction_icon = PhotoImage(file='./icons/prediction-64.png')
        # undo_icon = PhotoImage(file='./icons/undo-64.png')
        brush_icon = PhotoImage(file='./icons/brush-64.png')
        eraser_icon = PhotoImage(file='./icons/eraser-64.png')
        surface_adjuster_icon = PhotoImage(file='./icons/surface-adjuster-64.png')
        surface_adjuster_offsets_icon = PhotoImage(file='./icons/surface-update-offsets-64.png')
        bucket_icon = PhotoImage(file='./icons/bucket-64.png')
        stop_icon = PhotoImage(file='./icons/stop-60.png')
        help_icon = PhotoImage(file='./icons/help-48.png')
        load_mask_icon = PhotoImage(file='./icons/ink-64.png')

        self.mode = tk.StringVar(value="pencil")

        # Add buttons with icons and tooltips to the toolbar frame
        load_button = ttk.Button(self.toolbar_frame, image=load_icon, command=self.load_data)
        load_button.image = load_icon
        load_button.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(load_button, "Open Zarr 3D Image")

        load_mask_button = ttk.Button(self.toolbar_frame, image=load_mask_icon, command=self.load_mask)
        load_mask_button.image = load_mask_icon
        load_mask_button.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(load_mask_button, "Load Ink Mask")

        save_button = ttk.Button(self.toolbar_frame, image=save_icon, command=self.save_mask_3d)
        save_button.image = save_icon
        save_button.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(save_button, "Save H5FS 3D Mask")

        save_button = ttk.Button(self.toolbar_frame, image=save_icon, command=self.save_flattened_mask)
        save_button.image = save_icon
        save_button.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(save_button, "Save Flattened 2D Mask as TIFF")

        load_prediction = ttk.Button(self.toolbar_frame, image=prediction_icon, command=self.load_prediction)
        load_prediction.image = load_icon
        load_prediction.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(load_prediction, "Load Ink Prediction")

        # undo_button = ttk.Button(self.toolbar_frame, image=undo_icon, command=self.undo_last_action)
        # undo_button.image = undo_icon
        # undo_button.pack(side=tk.LEFT, padx=2)
        # self.create_tooltip(undo_button, "Undo Last Action")

        surface_adjuster_offsets_button = ttk.Button(self.toolbar_frame, image=surface_adjuster_offsets_icon, command=self.threaded_update_surface_adjuster_offsets)
        surface_adjuster_offsets_button.image = surface_adjuster_offsets_icon
        surface_adjuster_offsets_button.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(surface_adjuster_offsets_button, "Update Surface Adjuster Offsets")

        # surface_adjuster_offsets_button = ttk.Button(self.toolbar_frame, image=surface_adjuster_offsets_icon, command=)
        # surface_adjuster_offsets_button.image = surface_adjuster_offsets_icon
        # surface_adjuster_offsets_button.pack(side=tk.LEFT, padx=2)
        # self.create_tooltip(surface_adjuster_offsets_button, "Print 3D bounding box")

        # Brush tool button
        brush_button = ttk.Radiobutton(self.toolbar_frame, image=brush_icon, variable=self.mode, value="pencil")
        brush_button.image = brush_icon
        brush_button.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(brush_button, "Brush Tool")

        # Eraser tool button
        eraser_button = ttk.Radiobutton(self.toolbar_frame, image=eraser_icon, variable=self.mode, value="eraser")
        eraser_button.image = eraser_icon
        eraser_button.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(eraser_button, "Eraser Tool")

        surface_adjuster_button = ttk.Radiobutton(self.toolbar_frame, image=surface_adjuster_icon, variable=self.mode, value="surface-adjuster")
        surface_adjuster_button.image = surface_adjuster_icon
        surface_adjuster_button.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(surface_adjuster_button, "Surface Adjuster Tool")

        self.editing_barrier_var = tk.BooleanVar(value=self.editing_barrier)
        toggle_editing_button = ttk.Checkbutton(self.toolbar_frame, text="Edit Barrier", command=self.toggle_editing_mode, variable=self.editing_barrier_var)
        toggle_editing_button.pack(side=tk.LEFT, padx=5)

        self.pencil_size_var = tk.StringVar(value=f"{self.pencil_size}")  # Default pencil size
        pencil_size_label = ttk.Label(self.toolbar_frame, text="Pencil Size:")
        pencil_size_label.pack(side=tk.LEFT, padx=(10, 2))  # Add some padding for spacing

        pencil_size_slider = ttk.Scale(self.toolbar_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.update_pencil_size)
        pencil_size_slider.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(pencil_size_slider, "Adjust Pencil Size")

        pencil_size_value_label = ttk.Label(self.toolbar_frame, textvariable=self.pencil_size_var)
        pencil_size_value_label.pack(side=tk.LEFT, padx=(0, 10))

        # Bucket tool button
        bucket_button = ttk.Radiobutton(self.toolbar_frame, image=bucket_icon, variable=self.mode, value="bucket")
        bucket_button.image = bucket_icon
        bucket_button.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(bucket_button, "Flood Fill Tool")

        # Stop tool button
        stop_button = ttk.Button(self.toolbar_frame, image=stop_icon, command=self.stop_flood_fill)
        stop_button.image = stop_icon
        stop_button.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(stop_button, "Stop Flood Fill")

        # Help button
        help_button = ttk.Button(self.toolbar_frame, image=help_icon, command=self.show_help)
        help_button.image = help_icon
        help_button.pack(side=tk.RIGHT, padx=2)
        self.create_tooltip(help_button, "Info")

        # Bucket Threshold Slider
        '''
        self.bucket_threshold_var = tk.StringVar(value="4")  # Default threshold
        bucket_threshold_label = ttk.Label(self.toolbar_frame, text="Bucket Threshold:")
        bucket_threshold_label.pack(side=tk.LEFT, padx=(10, 2))  # Add some padding for spacing

        self.bucket_threshold_slider = ttk.Scale(self.toolbar_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.update_threshold_value)
        self.bucket_threshold_slider.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(self.bucket_threshold_slider, "Adjust Bucket Threshold")

        bucket_threshold_value_label = ttk.Label(self.toolbar_frame, textvariable=self.bucket_threshold_var)
        bucket_threshold_value_label.pack(side=tk.LEFT, padx=(0, 10))
        '''
        # The canvas itself remains in the center
        self.center_frame = tk.Frame(self.root, bg="red")
        self.center_frame.pack(side=tk.TOP, fill='both', expand=True)

        self.nav3d_frame = tk.Frame(self.center_frame, width="201", bg="white")
        self.nav3d_frame.pack(side=tk.LEFT, fill=tk.Y, expand=False)

        self.canvas_z = tk.Canvas(self.nav3d_frame, bg='white', highlightthickness=5, highlightbackground="green")
        self.canvas_z.pack(fill='both', expand=True)
        self.canvas_x = tk.Canvas(self.nav3d_frame, bg='white', highlightthickness=5, highlightbackground="red")
        self.canvas_x.pack(fill='both', expand=True)
        self.canvas_y = tk.Canvas(self.nav3d_frame, bg='white', highlightthickness=5, highlightbackground="blue")
        self.canvas_y.pack(fill='both', expand=True)
        self.canvas_z_text = self.canvas_z.create_text(10, 10, anchor=tk.NW, text="Z: /", fill="red", font=('Helvetica', 15, 'bold'))
        self.canvas_x_text = self.canvas_x.create_text(10, 10, anchor=tk.NW, text="X: /", fill="red", font=('Helvetica', 15, 'bold'))
        self.canvas_y_text = self.canvas_y.create_text(10, 10, anchor=tk.NW, text="Y: /", fill="red", font=('Helvetica', 15, 'bold'))


        self.canvas = tk.Canvas(self.center_frame, bg='white')
        self.canvas.pack(fill='both', expand=True)

        self.z_slice_text = self.canvas.create_text(10, 10, anchor=tk.NW, text=f"Z-Slice: {self.z_index}", fill="red", font=('Helvetica', 12, 'bold'))
        self.zoom_text = self.canvas.create_text(10, 30, anchor=tk.NW, text=f"Zoom: {self.zoom_level:.2f}", fill="red", font=('Helvetica', 12, 'bold'))
        self.cursor_pos_text = self.canvas.create_text(10, 50, anchor=tk.NW, text="Cursor Position: (0, 0)", fill="red", font=('Helvetica', 12, 'bold'))


        # Bind event handlers
        self.canvas.bind("<Motion>", self.update_pencil_cursor)
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<ButtonPress-3>", self.on_canvas_press)
        self.canvas.bind("<B3-Motion>", self.on_canvas_pencil_drag)
        self.canvas.bind("<ButtonRelease-3>", self.on_canvas_release)
        self.canvas.bind("<Button-3>", self.on_canvas_click)  # Assuming on_canvas_click is implemented
        self.canvas.bind("<MouseWheel>", self.scroll_or_zoom)  # Assuming scroll_or_zoom is implemented
        # On Linux, Button-4 is scroll up and Button-5 is scroll down
        self.canvas.bind("<Button-4>", self.scroll_or_zoom)
        self.canvas.bind("<Button-5>", self.scroll_or_zoom)
        self.root.bind("<Key>", self.key_handler)

        # Variables for toggling states
        self.show_mask_var = tk.BooleanVar(value=self.show_mask)
        self.flatten_mask_var = tk.BooleanVar(value=self.flatten_mask)
        self.show_barrier_var = tk.BooleanVar(value=self.show_barrier)
        self.show_image_var = tk.BooleanVar(value=self.show_image)
        self.show_prediction_var = tk.BooleanVar(value=self.show_prediction)
        self.show_surface_offsets_var = tk.BooleanVar(value=self.show_surface_offsets)

        # Create a frame to hold the toggle buttons
        toggle_frame = tk.Frame(self.root)
        toggle_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=2)

        # Create toggle buttons for mask and image visibility
        toggle_mask_button = ttk.Checkbutton(toggle_frame, text="Mask", command=self.toggle_mask, variable=self.show_mask_var)
        toggle_mask_button.pack(side=tk.LEFT, padx=5, anchor='s')

        toggle_mask_button = ttk.Checkbutton(toggle_frame, text="Flatten mask", command=self.toggle_flatten_mask, variable=self.flatten_mask_var)
        toggle_mask_button.pack(side=tk.LEFT, padx=5, anchor='s')

        toggle_barrier_button = ttk.Checkbutton(toggle_frame, text="Barrier", command=self.toggle_barrier, variable=self.show_barrier_var)
        toggle_barrier_button.pack(side=tk.LEFT, padx=5, anchor='s')

        toggle_prediction_button = ttk.Checkbutton(toggle_frame, text="Prediction", command=self.toggle_prediction, variable=self.show_prediction_var)
        toggle_prediction_button.pack(side=tk.LEFT, padx=5, anchor='s')

        toggle_prediction_button = ttk.Checkbutton(toggle_frame, text="Surface offsets", command=self.toggle_surface_offsets, variable=self.show_surface_offsets_var)
        toggle_prediction_button.pack(side=tk.LEFT, padx=5, anchor='s')

        # Slider for adjusting the alpha (opacity)
        self.alpha_var = tk.IntVar(value=self.overlay_alpha)
        alpha_label = ttk.Label(toggle_frame, text="Opacity:")
        alpha_label.pack(side=tk.LEFT, padx=5, anchor='s')
        alpha_slider = ttk.Scale(toggle_frame, from_=0, to=255, orient=tk.HORIZONTAL, command=self.update_alpha)
        alpha_slider.set(self.overlay_alpha)  # Set the default position of the slider
        alpha_slider.pack(side=tk.LEFT, padx=5, anchor='s')
        self.create_tooltip(alpha_slider, "Adjust Overlay Opacity")

        toggle_image_button = ttk.Checkbutton(toggle_frame, text="Toggle Image", command=self.toggle_image, variable=self.show_image_var)
        toggle_image_button.pack(side=tk.LEFT, padx=5, anchor='s')

        # Create a frame specifically for the sliders
        slider_frame = ttk.Frame(toggle_frame)
        slider_frame.pack(side=tk.RIGHT, padx=5)

        # Bucket Layer Slider
        self.bucket_layer_var = tk.StringVar(value="0")
        bucket_layer_label = ttk.Label(slider_frame, text="Bucket Layer:")
        bucket_layer_label.pack(side=tk.LEFT, padx=(10, 2))

        self.bucket_layer_slider = ttk.Scale(slider_frame, from_=0, to=0, orient=tk.HORIZONTAL, command=self.update_threshold_layer)
        self.bucket_layer_slider.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(self.bucket_layer_slider, "Adjust Bucket Layer")

        bucket_layer_value_label = ttk.Label(slider_frame, textvariable=self.bucket_layer_var)
        bucket_layer_value_label.pack(side=tk.LEFT, padx=(0, 10))

        # Bucket Threshold Slider
        self.bucket_threshold_var = tk.StringVar(value="4")
        bucket_threshold_label = ttk.Label(slider_frame, text="Bucket Threshold:")
        bucket_threshold_label.pack(side=tk.LEFT, padx=(10, 2))

        self.bucket_threshold_slider = ttk.Scale(slider_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.update_threshold_value)
        self.bucket_threshold_slider.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(self.bucket_threshold_slider, "Adjust Bucket Threshold")

        bucket_threshold_value_label = ttk.Label(slider_frame, textvariable=self.bucket_threshold_var)
        bucket_threshold_value_label.pack(side=tk.LEFT, padx=(0, 10))

        # Max Propagation Slider
        self.max_propagation_var = tk.IntVar(value=self.max_propagation_steps)
        max_propagation_label = ttk.Label(slider_frame, text="Max Propagation:")
        max_propagation_label.pack(side=tk.LEFT, padx=(10, 2))

        max_propagation_slider = ttk.Scale(slider_frame, from_=1, to=500, orient=tk.HORIZONTAL, command=self.update_max_propagation)
        max_propagation_slider.set(self.max_propagation_steps)
        max_propagation_slider.pack(side=tk.LEFT, padx=2)
        self.create_tooltip(max_propagation_slider, "Adjust Max Propagation Steps for Flood Fill")

        max_propagation_value_label = ttk.Label(slider_frame, textvariable=self.max_propagation_var)
        max_propagation_value_label.pack(side=tk.LEFT, padx=(0, 10))

        # Create a frame for the log text area and scrollbar
        log_frame = tk.Frame(self.root)
        log_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Create the log text widget
        self.log_text = tk.Text(log_frame, height=4, width=50)
        self.log_text.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Create the scrollbar and associate it with the log text widget
        log_scrollbar = tk.Scrollbar(log_frame, command=self.log_text.yview)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text['yscrollcommand'] = log_scrollbar.set

        self.stride = max(1, int(arguments.stride)) if arguments.stride else 1
        print(f"Stride: {self.stride}")

        for c in [self.canvas, self.canvas_x, self.canvas_y, self.canvas_z]:
            c.pack()
            c.update()

        if arguments.h5fs_file:
            self.load_data(h5_filename=arguments.h5fs_file, h5_axes_seq=arguments.axes, h5_roi=arguments.roi, h5_scroll_filename=arguments.h5fs_scroll)

        self.surface_adjust_filename = arguments.surface_adjust_file if arguments.surface_adjust_file else None
        self.load_surface_adjust_file()

        self.update_info_display()

        self.root.mainloop()
        self.on_exit()


class ProgressPrinter:
    def __init__(self, min_n, max_n, label, interval=3, print_=None):
        self.start_time = time.time()
        self.last_print_time = None
        self.min_n = min_n
        self.max_n = max_n
        self.label = label
        self.interval = interval
        self.print_ = print if print_ is None else print_

    def progress(self, n):
        now = time.time()
        if not self.last_print_time or now - self.last_print_time > self.interval:
            if not self.last_print_time:
                eta = .0
                percent_done = 0
            else:
                eta = (self.max_n - n) * (self.last_print_time - self.start_time) / (n - self.min_n)
                percent_done = 100. * ((n - self.min_n) / (self.max_n - self.min_n))
            self.print_(f'{self.label}: {n} ({self.min_n} -> {self.max_n}), progress: {percent_done:.2f}%, ETA: {eta:.2f}s')
            self.last_print_time = now


class PPMParser(object):
    def __init__(self, filename, skip=None):
        self.filename = filename
        self.skip = skip

    def open(self):
        print('Opening PPM file {}'.format(self.filename))
        self.f = open(self.filename, 'rb')
        self.info, self.header_size, self.header_content = PPMParser.vcps_parse_header(self.f)
        return self

    def close (self):
        print('Closing file.')
        self.f.close()

    def __enter__ (self):
        return self

    def __exit__ (self, exc_type, exc_value, traceback):
        self.close()

    def im_zeros(self, dtype):
        # allocates exactly the size that is needed for resulting image, taking skip into account:
        if self.skip is None:
            a = np.zeros((self.info['width'], self.info['height']), dtype=dtype)
        else:
            a = np.zeros((self.info['width'] // self.skip + 1, self.info['height'] // self.skip + 1), dtype=dtype)
        return a

    @staticmethod
    def vcps_parse_header(f):
        info = {}
        header_size = 0
        header_content = b''
        while True:
            l_bytes = f.readline()
            header_size += len(l_bytes)
            header_content += l_bytes
            l = l_bytes.decode('utf-8').rstrip("\n")
            if l == '<>':
                break
            k, v = l.split(': ', 1)
            if v.isnumeric():
                v = int(v)
            info[k] = v
        return info, header_size, header_content

    def read_next_coords(self, skip_empty=True):
        f = self.f
        im_width = self.info['width']
        skip = self.skip
        n = -1
        while True:
            n += 1

            buf = f.read(6*8)
            if not buf:
                break
            x, y, z, nx, ny, nz = struct.unpack('<dddddd', buf)

            if skip_empty and int(x) == 0:
                continue

            imx, imy = n % im_width, n // im_width

            if skip is not None:
                # skip most of the data and adjust image coordinates if we are using skip:
                if imx % skip or imy % skip:
                    continue

                yield imx // skip, imy // skip, x, y, z, nx, ny, nz
            else:
                yield imx, imy, x, y, z, nx, ny, nz

    def get_3d_coords(self, imx, imy, rounded_xyz=False):
        f = self.f
        im_width = self.info['width']
        pos = self.header_size + (imy * im_width + imx) * 6 * 8
        f.seek(pos, os.SEEK_SET)

        buf = f.read(6*8)
        if not buf:
            return None, None, None, None, None, None
        x, y, z, nx, ny, nz = struct.unpack('<dddddd', buf)
        if rounded_xyz:
            return round(x), round(y), round(z), nx, ny, nz
        else:
            return x, y, z, nx, ny, nz


if __name__ == "__main__":
    editor = Klepar()