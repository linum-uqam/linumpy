import matplotlib.pyplot as plt
import numpy as np
from linumpy.microscope.oct import OCT
from pathlib import Path
from tqdm.auto import tqdm
from skimage.filters import threshold_otsu, threshold_li
from scipy.io import savemat

directory = Path("/Volumes/USB-JOEL/2024-11-01-SOCT-UQAM-Calibration-Dispersion-45Mirror")
basename = "bouma_z_"
galvo_return = 40
spectrum_margin = 100
fringes_files = list(directory.glob(f"{basename}*"))
print(f"There are {len(fringes_files)} fringes files to process")

# Detect the file number
fringes_files.sort()
fringes_ids = [int(f.stem.split("_")[-1]) for f in fringes_files]

# Load the first fringe and determine the size of the data
oct = OCT(fringes_files[0])
fringe = oct.load_fringe()
n_samples = fringe.shape[0]
n_alines = fringe.shape[1]
n_depths = len(fringes_files)
data = np.zeros((n_samples, n_alines, n_depths), dtype=fringe.dtype)
print(f"The data will have a shape of {data.shape}")

# Load all the data
for i, file in tqdm(zip(fringes_ids, fringes_files), total=n_depths):
    oct = OCT(file)
    fringe = oct.load_fringe()
    data[..., i] = fringe.mean(axis=2)

# Removing galvo return
data = data[:, 0:-galvo_return, ...]

# Detect the min/max wavelentgh with signal, and propose cropping
fringe_avg = data.mean(axis=(1,2))
mask = fringe_avg > threshold_li(fringe_avg)
low_pt_otsu = max(np.where(mask)[0][0] - spectrum_margin, 0)
high_pt_otsu = min(np.where(mask)[0][-1] + spectrum_margin, n_samples)
print(f"Detected spectrum from {low_pt_otsu} to {high_pt_otsu} ({high_pt_otsu-low_pt_otsu} samples)")
plt.plot(data.mean(axis=(1,2)), label="Average fringe")
plt.axvspan(0, low_pt_otsu, color='r', alpha=0.5, label="Cropping region")
plt.axvspan(high_pt_otsu, n_samples, color='r', alpha=0.5)
plt.legend()
plt.show()

# Removing the background for each bscan
data_wo_bg = np.zeros_like(data, dtype=np.float64)
for i in range(data.shape[2]):
    data_wo_bg[..., i] = data[..., i] - data[..., i].mean(axis=1, keepdims=True)

# Perform a quick reconstruction
window = np.hanning(data_wo_bg.shape[0]).reshape((data.shape[0],1,1))
bscans = np.abs(np.log(np.fft.fft(data_wo_bg * window, axis=0)))
plt.imshow(bscans.mean(axis=1), aspect="auto")
plt.show()

# explort matlab file
output = {'fringes': {'ch1': data_wo_bg.astype(np.float32)}, 'omitBscansVec': []}
output_file = directory / 'fringes.mat'
savemat(output_file, output)

