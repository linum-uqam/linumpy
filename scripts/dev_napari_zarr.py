from pathlib import Path
import napari

# Parameters
zarr_directory = Path("G:/frans/2024-01-12-S9-Coronal/dev_tiles_z05.ome-zarr/")

viewer = napari.Viewer()
viewer.open(zarr_directory, plugin="napari-ome-zarr")
napari.run()