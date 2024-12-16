# Exports the calibration files required by our OCTto be used with PolyMtl/UQAM OCTs
from datetime import datetime
from pathlib import Path

from matplotlib import pyplot as plt
from scipy.io import loadmat
import numpy as np
import json

directory = Path("C://Users//Joël//OneDrive - UQAM//Documents//MATLAB//RobustMapping//Output//Results")
output_directory = Path("C://Users//Public//Documents")
result_file = directory / "UQAM-SOCT-20241115-Simplex-cropped.mat"

name = "2024-11-29-air-10x"
kmin = 852
kmax = 1620
n_samples = kmax - kmin
oversampling_factor = 8
n_samples_kspace = oversampling_factor * n_samples
n_samples_output = 1024

# Load the calibration optimization results
calibration = loadmat(result_file)

# Process the k-space mapping
kspace_mapping = calibration["map"].squeeze()
kspace_mapping = kspace_mapping / max(kspace_mapping) * n_samples_kspace # Adapting to the oversampling factor
kspace_interp_matrix = np.zeros((n_samples//2, n_samples_kspace))
for i in range(n_samples_kspace):
    x = np.arange(n_samples_kspace)
    y = np.zeros((n_samples_kspace,))
    y[i] = 1
    x_new = kspace_mapping

    kspace_interp_matrix[:, i] = np.interp(x_new, x, y)

# Process the dispersion
dispersion = calibration["dispFit"].squeeze()
x = np.linspace(-1, 1, len(dispersion))
x_new = np.linspace(-1, 1, n_samples//2)
dispersion = np.interp(x_new, x, dispersion)

# Prepare the window
window = np.hanning(n_samples//2)

# Prepare the output
figure_filename = output_directory / f"calibration_graphs_{name}.png"
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=[12,4])
axes[0].plot(kspace_mapping)
axes[0].set_title("K-Space Mapping")
axes[1].plot(dispersion)
axes[1].set_title("Dispersion")
axes[2].plot(window)
axes[2].set_title("Window")
plt.savefig(figure_filename, dpi=300, pad_inches=0, bbox_inches='tight')


# Write the calibration files
window_filename = output_directory / f"filter_{name}.dat"
with open(window_filename, "wb") as f:
    window.astype(np.float64).tofile(f)

dispersion_filename = output_directory / f"phase_{name}.dat"
with open(dispersion_filename, "wb") as f:
    dispersion.astype(np.float64).tofile(f)

kspace_interp_matrix_filename = output_directory / f"kspace_interp_matrix_{name}.dat"
with open(kspace_interp_matrix_filename, "wb") as f:
    kspace_interp_matrix.astype(np.float64).tofile(f)

config = {}
config["name"] = name
config["kmin"] = kmin
config["kmax"] = kmax
config["oversampling_factor"] = oversampling_factor
config["n_samples_kspace"] = n_samples_kspace
config["n_samples_output"] = n_samples_output
config["window_filename"] = str(window_filename)
config["dispersion_filename"] = str(dispersion_filename)
config["kspace_interp_matrix_filename"] = str(kspace_interp_matrix_filename)
config["calib_graph_filename"] = str(figure_filename)
config["date"] = datetime.now().strftime("%Y-%m-%d@%Hh%M:%Ss")
config_filename = output_directory / f"config_{name}.json"
with open(config_filename, "w") as f:
    json.dump(config, f)