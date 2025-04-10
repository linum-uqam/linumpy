import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, RangeSlider
from scipy.interpolate import RegularGridInterpolator

PREV_REF_LABEL = 'Previous slice as reference'
NEXT_REF_LABEL = 'Next slice as reference'
NO_REF_LABEL = 'No reference slice'


class ManualImageCorrection():
    """
    Manual image correction using a graphical user interface. Corrections
    include independent translation and rotation of each z-slice as well
    as image intensities rescaling per z-slice.

    Parameters
    ----------
    data: ndarray of shape (nz, ny, nx)
        Stack of images, where images are stacked along the first axis (z).
    resolution: 3-tuple
        Resolution of the dataset (rz, ry, rx).
    downsample_factor: int
        Factor by which the full resolution images are downscaled
        prior to rendering. Tradeoff between image quality and interactivity
        of the GUI. Does not influence the resolution of the corrected image.
    transforms: ndarray (nz, 3), optional
        Transform for each slice where each array (3,) contains a translation
        (ty, tx) and a rotation (theta).
    custom_ranges: ndarray (nz, 2), optional
        Intensities for rescaling each slice. One (vmin, vmax) per slice.
    """
    def __init__(self, data, resolution, downsample_factor,
                 transforms=None, custom_ranges=None):
        # We will work on a dataset rescaled between [0, 1]
        data = data - data.min()
        data = data / data.max()

        self.downsample = downsample_factor
        z = np.arange(data.shape[0])
        self.max_z = np.max(z)

        y = np.arange(data.shape[1])
        x = np.arange(data.shape[2])
        self.image_interpolator = RegularGridInterpolator((z, y, x), data,
                                                          bounds_error=False,
                                                          fill_value=0)
        self.grid_coordinates = np.stack(np.meshgrid(z, y, x, indexing='ij'), axis=-1)

        # Transforms array contains translation and rotation
        # for each slice in the order (ty, tx, theta)
        self.transforms = transforms
        if transforms is None:
            self.transforms = np.zeros((len(z), 3))
        if self.transforms.shape != (len(z), 3):
            raise ValueError("Invalid shape for transforms file: "
                             f"expected ({len(z)}, 3), got {self.transforms.shape}.")

        # Base intensity normalization will rescale each slice
        # between its min and max values to the range [0, 1]
        self.custom_ranges = custom_ranges
        if custom_ranges is None:
            self.custom_ranges = np.array([np.min(data, axis=(1, 2)),
                                           np.max(data, axis=(1, 2))]).T
        if self.custom_ranges.shape != (len(z), 2):
            raise ValueError("Invalid shape for custom ranges file: "
                             f"expected ({len(z)}, 3), got {self.custom_ranges.shape}.")

        self.ref_z_mode = NO_REF_LABEL
        self.current_x = len(x) // 2
        self.current_y = len(y) // 2
        self.current_z = 0

        self.fig, axs = plt.subplots(1, 3, figsize=(16, 8))
        self.fig.subplots_adjust(bottom=0.38, top=0.95, left=0.05, right=0.9)

        # intensities will always be displayed between (0, 1)
        aspect_a = resolution[0] / resolution[2]
        self.axim_a = axs[0].imshow(self.get_view_a(), aspect=aspect_a,
                                    vmin=0.0, vmax=1.0,
                                    interpolation='nearest', cmap='magma')
        aspect_b = resolution[0] / resolution[1]
        self.axim_b = axs[1].imshow(self.get_view_b(), aspect=1.0/aspect_b,
                                    vmin=0.0, vmax=1.0,
                                    interpolation='nearest', cmap='magma')
        aspect_c = resolution[1] / resolution[2]
        self.axim_c = axs[2].imshow(self.get_view_c(), aspect=aspect_c,
                                    vmin=0.0, vmax=1.0,
                                    interpolation='nearest', cmap='magma')
        axs[0].set_axis_off()
        axs[1].set_axis_off()
        axs[2].set_axis_off()

        ax_current_z = self.fig.add_axes([0.15, 0.30, 0.45, 0.03])
        ax_ref_z = self.fig.add_axes([0.65, 0.30, 0.25, 0.05])
        ax_offset_a = self.fig.add_axes([0.15, 0.25, 0.75, 0.03])
        ax_offset_b = self.fig.add_axes([0.15, 0.20, 0.75, 0.03])
        ax_theta = self.fig.add_axes([0.15, 0.15, 0.75, 0.03])
        ax_current_y = self.fig.add_axes([0.15, 0.10, 0.75, 0.03])
        ax_current_x = self.fig.add_axes([0.15, 0.05, 0.75, 0.03])
        ax_scalebar = self.fig.add_axes([0.91, 0.40, 0.01, 0.55])

        self.scalebar = RangeSlider(ax_scalebar, "Scalebar",
                                    valmin=0.0, valmax=1.0,
                                    valinit=(self.custom_ranges[self.current_z, 0],
                                             self.custom_ranges[self.current_z, 1]),
                                    orientation='vertical')

        self.s_offset_a = Slider(ax_offset_a, "Offset left image",
                                 valmin=-data.shape[2]/2,
                                 valmax=data.shape[2]/2,
                                 valinit=self.transforms[self.current_z, 0])
        self.s_offset_b = Slider(ax_offset_b, "Offset right image",
                                 valmin=-data.shape[1]/2,
                                 valmax=data.shape[1]/2,
                                 valinit=self.transforms[self.current_z, 1])
        self.s_current_z = Slider(ax_current_z, "Current slice z",
                                  valmin=0, valmax=data.shape[0],
                                  valinit=0, valstep=np.arange(data.shape[0]))
        self.s_current_y = Slider(ax_current_y, "Current slice y",
                                  valmin=0, valmax=data.shape[1],
                                  valinit=self.current_y, valstep=np.arange(data.shape[1]))
        self.s_current_x = Slider(ax_current_x, "Current slice x",
                                  valmin=0, valmax=data.shape[2],
                                  valinit=self.current_x, valstep=np.arange(data.shape[2]))
        self.s_theta = Slider(ax_theta, 'Rotation',
                              valmin=-np.pi/6.0, valmax=np.pi/6.0,
                              valinit=self.transforms[self.current_z, 2])
        self.radio_buttons = RadioButtons(ax_ref_z, [NO_REF_LABEL, PREV_REF_LABEL, NEXT_REF_LABEL], 0)

        # register callbacks
        self.s_current_z.on_changed(self.on_change_z)
        self.s_current_y.on_changed(self.on_change_y)
        self.s_current_x.on_changed(self.on_change_x)
        self.s_offset_a.on_changed(self.on_change_offset_a)
        self.s_offset_b.on_changed(self.on_change_offset_b)
        self.s_theta.on_changed(self.on_change_theta)
        self.radio_buttons.on_clicked(self.on_change_ref_z)
        self.scalebar.on_changed(self.on_change_scaling)

    def start(self):
        """
        Start GUI.

        Returns
        -------
        return: bool
            True when the window closes.
        """
        plt.show(block=True)
        return True

    def on_change_scaling(self, scaling_range):
        self.custom_ranges[self.current_z] = scaling_range
        self.axim_a.set(data=self.get_view_a())
        self.axim_b.set(data=self.get_view_b())
        self.axim_c.set(data=self.get_view_c())
        self.fig.canvas.draw_idle()

    def on_change_z(self, val):
        self.current_z = int(val)
        self.s_offset_a.set_val(self.transforms[self.current_z, 0])
        self.s_offset_b.set_val(self.transforms[self.current_z, 1])
        self.s_theta.set_val(self.transforms[self.current_z, 2])
        self.scalebar.set_val(self.custom_ranges[self.current_z, :])
        self.axim_c.set(data=self.get_view_c())
        self.fig.canvas.draw_idle()

    def on_change_y(self, val):
        self.current_y = int(val)
        self.axim_b.set(data=self.get_view_b())
        self.fig.canvas.draw_idle()

    def on_change_x(self, val):
        self.current_x = int(val)
        self.axim_a.set(data=self.get_view_a())
        self.fig.canvas.draw_idle()

    def on_change_offset_a(self, val):
        self.transforms[self.current_z, 0] = val
        self.axim_a.set(data=self.get_view_a())
        self.axim_b.set(data=self.get_view_b())
        self.axim_c.set(data=self.get_view_c())
        self.fig.canvas.draw_idle()

    def on_change_offset_b(self, val):
        self.transforms[self.current_z, 1] = val
        self.axim_a.set(data=self.get_view_a())
        self.axim_b.set(data=self.get_view_b())
        self.axim_c.set(data=self.get_view_c())
        self.fig.canvas.draw_idle()

    def on_change_theta(self, val):
        self.transforms[self.current_z, 2] = val
        self.axim_a.set(data=self.get_view_a())
        self.axim_b.set(data=self.get_view_b())
        self.axim_c.set(data=self.get_view_c())
        self.fig.canvas.draw_idle()

    def on_change_ref_z(self, label):
        self.ref_z_mode = label
        self.axim_c.set(data=self.get_view_c())
        self.fig.canvas.draw_idle()

    def transform_coordinates(self, coordinates, z=None):
        # will consider either all z or a single one
        if z is None:
            ty = self.transforms[:, 0]
            tx = self.transforms[:, 1]
            theta = self.transforms[:, 2]
        else:
            ty = self.transforms[z, 0]
            tx = self.transforms[z, 1]
            theta = self.transforms[z, 2]

        coordinates = apply_transform(ty, tx, theta, coordinates)

        return coordinates

    def apply_scaling(self, data, z=None):
        if z is not None:
            clip_min = self.custom_ranges[z, 0]
            clip_max = self.custom_ranges[z, 1]
        else:
            clip_min = self.custom_ranges[:, 0, None]
            clip_max = self.custom_ranges[:, 1, None]

        data = apply_scaling(data, clip_min, clip_max)

        # at this point the data is between [0, 1]
        return data
    
    def draw_cursor(self, data):
        # keeping in mind that axis=0 is the z axis
        cursor_len = int(0.02 * data.shape[-1])
        data[self.current_z, :cursor_len] = 1.0
        data[self.current_z, -cursor_len:] = 1.0
        return data


    def get_view_a(self):
        view_coords = self.grid_coordinates[:, :, self.current_x, :]
        transformed_coords = self.transform_coordinates(view_coords)
        data = self.apply_scaling(self.image_interpolator(transformed_coords))
        data = self.draw_cursor(data)
        return data

    def get_view_b(self):
        view_coords = self.grid_coordinates[:, self.current_y, :, :]
        transformed_coords = self.transform_coordinates(view_coords)
        data = self.apply_scaling(self.image_interpolator(transformed_coords))
        data = self.draw_cursor(data)
        return data.T

    def get_view_c(self):
        # subsample coordinates for better interactivity
        view_coords = self.grid_coordinates[self.current_z, ::self.downsample, ::self.downsample, :]
        transformed_coords = self.transform_coordinates(view_coords, self.current_z)
        data_view = self.apply_scaling(self.image_interpolator(transformed_coords),
                                       self.current_z)

        data_rgb = np.zeros(data_view.shape + (3,))
        data_rgb[..., :] = data_view[..., None]

        if self.ref_z_mode != NO_REF_LABEL:
            ref_z = self.current_z - 1 if self.ref_z_mode == PREV_REF_LABEL else self.current_z + 1
            if ref_z >= 0 and ref_z <= self.max_z:
                ref_coords = self.grid_coordinates[ref_z, ::self.downsample, ::self.downsample, :]
                transformed_ref_coords = self.transform_coordinates(ref_coords, ref_z)
                data_ref = self.apply_scaling(self.image_interpolator(transformed_ref_coords),
                                              self.current_z)
                data_rgb[..., 0] = data_ref
        return np.clip(data_rgb, 0.0, 1.0)

    def save_results(self, filename):
        """
        Save resulting corrections to npz file.

        Parameters
        ----------
        filename: string or Path
            Output filename.
        """
        np.savez_compressed(filename,
                            custom_ranges=self.custom_ranges,
                            transforms=self.transforms)


def apply_transform(ty, tx, theta, coordinates):
    """
    Apply transformation to coordinates. Coordinates are expected to be
    of shape (nz, ny, nx, 3), with each coordinate given in the order
    (z, y, x).

    Parameters
    ----------
    ty: float or ndarray of shape (nz,)
        Translation along y axis.
    tx: float or ndarray of shape (nz,)
        Translation along x axis.
    theta: float or ndarray of shape (nz,)
        Rotation around z axis in radians. The center of rotation
        is the center of the image.
    Returns
    -------
    coordinates: ndarray (nz, ny, nx, 3)
        Transformed coordinates.
    """
    # Step 1. Rotate coordinates
    center_y = np.max(coordinates[:, :, 1]) / 2.0
    center_x = np.max(coordinates[:, :, 2]) / 2.0
    coordinates = coordinates - np.reshape([0, center_y, center_x], (1, 1, 3))
    rotated_y = np.atleast_2d(np.cos(theta)).T*coordinates[..., 1]\
        - np.atleast_2d(np.sin(theta)).T*coordinates[..., 2]
    rotated_x = np.atleast_2d(np.sin(theta)).T*coordinates[..., 1]\
        + np.atleast_2d(np.cos(theta)).T*coordinates[..., 2]
    coordinates[:, :, 1] = rotated_y + center_y
    coordinates[:, :, 2] = rotated_x + center_x

    # Step 2. Translate coordinates
    coordinates[:, :, 1] += np.atleast_2d(ty).T
    coordinates[:, :, 2] += np.atleast_2d(tx).T

    return coordinates


def apply_scaling(data, vmin, vmax):
    """
    Rescale image intensities from (vmin, vmax) to (0.0, 1.0). Values
    outside the range (vmin, vmax) are clipped.

    Rescaling can be performed with a single range for the whole image
    or with a different range for each ROW. In the case, the first dimension
    of data should correspond to the number of elements in vmin, vmax.

    Parameters
    ----------
    data: ndarray
        The intensities to rescale.
    vmin: float or ndarray of shape (data.shape[0],)
        Minimum value. Will be worth 0 after rescaling.
    vmax: float or ndarray of shape (data.shape[0],)
        Maximum value. Will be worth 1 after rescaling.

    Returns
    -------
    data: ndarray
        Recaled intensities.
    """
    data = np.clip(data, vmin, vmax)
    data -= vmin
    clip_range = vmax - vmin
    if isinstance(clip_range, np.ndarray):
        mask = (clip_range > 0).reshape((-1,))
        data[mask] /= clip_range[mask]
    elif clip_range > 0.0:
        data /= clip_range
    return data


def transform_and_rescale_slice(slice, ty, tx, theta, vmin, vmax):
    """
    Transform and rescale 2D slice. Transform consists of a translation
    (ty, tx) and a rotation theta. Rescaling clips intensities to (vmin, vmax)
    and rescales the resulting values to the range (0, 1).

    Parameters
    ----------
    slice: ndarray of shape (ny, nx)
        Slice to process.
    ty: float
        Translation along y axis (first axis).
    tx: float
        Translation along x axis (second axis).
    theta: float
        Rotation in radians.
    vmin: float
        Minimum value for rescaling.
    vmax: float
        Maximum value for rescaling.

    Returns
    -------
    slice: ndarray of shape (ny, nx)
        Processed slice.
    """
    y = np.arange(slice.shape[0])
    x = np.arange(slice.shape[1])
    image_interpolator = RegularGridInterpolator((y, x), slice, bounds_error=False, fill_value=0)
    grid_coordinates = np.stack(np.meshgrid(0, y, x, indexing='ij'), axis=-1)

    # transform coordinates
    transformed_coordinates = apply_transform(ty, tx, theta, grid_coordinates[0])
    transformed_image = image_interpolator(transformed_coordinates[..., 1:])
    # rescale intensities
    transformed_image = apply_scaling(transformed_image, vmin, vmax)

    return transformed_image