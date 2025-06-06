import matplotlib.pyplot as plt
import numpy as np
from linumpy.microscope.oct import OCT
from pathlib import Path
from scipy.interpolate import interp1d
import nibabel as nib
from tqdm.auto import tqdm
from scipy.io import loadmat

#directory = Path(r"D:\joel\2024-11-15-SOCT-UQAM-Calibration-Dispersion-Air-Mirror-woWindow-10x-higherContrast\mirror_z_015")
directory = Path(r"D:\joel\2024-11-15-SOCT-UQAM-Calibration-Dispersion-Air-Mirror-woWindow-10x-higherContrast")
#directory = Path(r"D://joel//2024-11-01-SOCT-UQAM-Finger")
#fringe_file = directory / "fringe"
fringe_file = directory / 'fringes.mat'
output_file = directory / "tomo_widerSpectrum.nii"

# Parameters
suffix = "2024-11-29-air-10x"
calibration_dir = Path(r"C://Users//Public//Documents")
kspace_interpolation_file = calibration_dir / f"kspace_interp_matrix_{suffix}.dat"
kspace_interp_file = calibration_dir / f"kspace_interp_matrix_{suffix}.dat"
window_file = calibration_dir / f"filter_{suffix}.dat"
dispersion_file = calibration_dir / f"phase_{suffix}.dat"

replace_by_default = False

fringe = loadmat(fringe_file)['fringes'][0, 0][0]
# galvo_return = 40
# spectrum_margin = 100
#oct = OCT(fringe_file)

# Load the data and calibration files
#fringe = oct.load_fringe(crop=False).squeeze()
n_samples = fringe.shape[0]
n_alines = fringe.shape[1]
n_bscans = fringe.shape[2]
n_samples = 2048
oversamplingFactor = 8
k_min = 852
k_max = 1620
n_values = k_max - k_min
oversampled_kspace = n_values * oversamplingFactor
dc_margin = 0
output_size = 1024
print(f"The data shape is {fringe.shape}")

# Load the calibration files
window = np.fromfile(window_file, dtype=np.float64)
dispersion = np.fromfile(dispersion_file, dtype=np.float64)
kspace_sparse_interp = np.fromfile(kspace_interpolation_file, dtype=np.float64)
kspace_sparse_interp = np.reshape(kspace_sparse_interp, (len(dispersion), len(kspace_sparse_interp) // len(dispersion)), order="C")

map = np.fromfile(kspace_interp_file, dtype=np.float64)


def reconstruct(fringe, show: bool = False):
    # 1. Input fringes of shape N_Samples x N_Alines
    if show:
        plt.imshow(fringe)
        plt.axhline(k_min, color='red', linestyle='--')
        plt.axhline(k_max, color='red', linestyle='--')
        plt.show()

    # 2. Crop the wavelengths to remove empty spectral regions
    fringe = fringe[k_min:k_max, :]
    if show:
        plt.imshow(fringe)
        plt.title("2. Cropped fringe")
        plt.show()

    # 3. Compute (update) the background from the cropped fringe
    background = fringe.mean(axis=1, keepdims=True)
    if show:
        plt.plot(background)
        plt.title("3. Background")
        plt.show()

    # 4. Remove background
    fringe = fringe - background
    if show:
        plt.imshow(fringe)
        plt.title("4. Fringes w/o background")
        plt.show()

    # 5. Compte a preprocessing tomograph
    preTom = np.fft.fft(fringe, axis=0)
    if show:
        plt.imshow(np.abs(np.log(preTom)))
        plt.title("5. PreTom")
        plt.show()

    # 6. Mask the complex conjugate and (Optional) 7. Mask the DC (optional)
    preTom[0:n_values // 2, :] = 0
    if dc_margin > 0:
        preTom[-dc_margin::, :] = 0
    if show:
        plt.imshow(np.abs(np.log(preTom + 1e-6)))
        plt.title("6/7. Masked PreTom")
        plt.show()

    # 8. Zero-padding to oversample the k-space for the interpolation
    padding_size = oversampled_kspace - preTom.shape[0]
    preTom = np.pad(preTom, ((padding_size, 0), (0, 0)), mode='constant', constant_values=0)
    if show:
        plt.imshow(np.abs(np.log(preTom + 1e-6)))
        plt.title("8. Padded PreTom")
        plt.show()

    # 9. Go back to k-space
    fringe = np.fft.ifft(preTom, axis=0)
    if show:
        plt.imshow(np.abs(fringe))
        plt.title("9. Oversampled Complex Fringes")
        plt.show()

    # 10. K-space linearization via sparse matrix interpolation
    fringe_linear = kspace_sparse_interp @ fringe
    # f = interp1d(np.linspace(0, n_samples, n_samples), fringe, axis=0)
    # fringe_linear = f(map)
    if show:
        plt.imshow(np.abs(fringe_linear))
        plt.title("10. Lineararized K-Space Fringe")
        plt.show()

    # 11. Compensate dispersion and apodization
    complex_window = window * np.exp(-1j * dispersion)
    x = np.linspace(0, 1, complex_window.shape[0])
    y = complex_window.squeeze()
    f = interp1d(x, y)
    complex_window = f(np.linspace(0, 1, fringe_linear.shape[0]))
    fringe_linear = fringe_linear * complex_window.reshape([*complex_window.shape, 1])

    if show:
        plt.imshow(np.abs(fringe_linear))
        plt.title("11. Lineararized K-Space Fringe")
        plt.show()

    # 12. Zero-padding to the output size
    #output_size = n_values // 2
    padding_size = output_size - fringe_linear.shape[0]
    fringe_linear = np.pad(fringe_linear, ((0, padding_size), (0, 0)))
    if show:
        plt.imshow(np.abs(fringe_linear))
        plt.title("12. Lineararized K-Space Fringe, padded")
        plt.show()

    # 13. Circular shift
    shift = kspace_sparse_interp.shape[0] // 2
    fringe_linear = np.roll(fringe_linear, -shift, axis=0)
    if show:
        plt.imshow(np.abs(fringe_linear))
        plt.title("13. Centered processed fringes")
        plt.show()

    # 14. Final FFT
    tomogram = np.fft.fft(fringe_linear, axis=0)
    if show:
        plt.imshow(np.abs(np.log(tomogram + 1e-6)))
        plt.title("14. Tomogram")
        plt.show()

    return tomogram


n_samples = n_values // 2
n_alines = fringe.shape[1]
tomogram = np.zeros((output_size, n_alines, n_bscans), dtype=np.float32)
for i in tqdm(range(n_bscans)):
    tomo = reconstruct(fringe[:, :, i].squeeze(), show=False)
    tomogram[:, :, i] = np.abs(tomo)
tomogram = np.flipud(tomogram)

figure_filename = directory / "mirror_calibrated.png"
plt.imshow(np.log(tomogram.mean(axis=1)), aspect="auto")
plt.xlabel("Mirror depth")
plt.ylabel("B-Scan Depth")
plt.title("Calibrated Tomograms")
plt.savefig(figure_filename, dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()

# Save the volume
# affine = np.eye(4)
# img = nib.Nifti1Image(tomogram, affine)
# nib.save(img, output_file)
