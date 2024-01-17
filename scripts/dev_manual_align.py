import matplotlib.pyplot as plt
import numpy as np
from linumpy.utils_images import normalize, get_overlay, apply_xy_shift
import matplotlib
from matplotlib.widgets import Button, Slider
from pathlib import Path
import nibabel as nib
from skimage.transform import rescale

# TODO: Convert this to a command line tool


matplotlib.use('Qt5Agg')


# Parameters
z = 35

directory = Path("/home/joel/data/2023-07-31-alexia-multiorientation-f1/reconstruction_2d")
file1 = directory / f"mosaic_grid_z{z:02d}_stitched.nii"
file2 = directory / f"mosaic_grid_z{z+1:02d}_stitched.nii"
scaling_factor = 1/16
transform = [0, 0]
shift_limits = [-100, 100]

f1 = directory / f"mosaic_grid_z{z:02d}_stitched.nii"
f2 = directory / f"mosaic_grid_z{z+1:02d}_stitched.nii"

# Load and display the images
im1 = nib.load(f1).get_fdata()
im2 = nib.load(f2).get_fdata()

im1 = normalize(im1)
im2 = normalize(im2)
im1 = rescale(im1, scaling_factor)
im2 = rescale(im2, scaling_factor)

def f(im1, im2, dx, dy):
    # apply the transform
    img2_shifted = apply_xy_shift(im2, im1, dx, dy)

    # Get the overlay
    overlay = get_overlay(im1, img2_shifted)

    return overlay

initial_dx = 0.0
initial_dy = 0.0

# Create the figure and the image that we will manipulate
fig, ax = plt.subplots()
image = ax.imshow(f(im1, im2, initial_dx, initial_dy))
ax.set_title("Overlay")

# Adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.25, bottom=0.25)

# Make the horizontal slider to control the horizontal shift
ax_dx = fig.add_axes([0.25, 0.1, 0.65, 0.03])
dx_slider = Slider(
    ax=ax_dx,
    label="dx",
    valmin=shift_limits[0],
    valmax=shift_limits[1],
    valinit=initial_dx,
)

# Make the vertical slider to control the vertical shift
ax_dy = fig.add_axes([0.1, 0.15, 0.0225, 0.63])
dy_slider = Slider(
    ax=ax_dy,
    label="dy",
    valmin=shift_limits[0],
    valmax=shift_limits[1],
    valinit=initial_dy,
    orientation="vertical"
)

# The function to be called anytime a slider's value changes
def update(val):
    image.set_data(f(im1, im2, dx_slider.val, dy_slider.val))
    fig.canvas.draw_idle()



# Add a button to acceprt or refect the transform
ax_accept = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(ax_accept, 'Accept', color='lightgoldenrodyellow', hovercolor='0.975')
ACCEPTED= False

def accept(event):
    transform[0] = dx_slider.val
    transform[1] = dy_slider.val
    plt.close()

# register the update function with each slider
dx_slider.on_changed(update)
dy_slider.on_changed(update)
button.on_clicked(accept)
plt.show()

print(transform)
transform_hr = [p / scaling_factor for p in transform]
print(f"{transform_hr[0]} {transform_hr[1]}")