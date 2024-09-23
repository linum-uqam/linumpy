import zipfile
from xml.dom.minidom import parse, parseString
import numpy as np
import matplotlib.pyplot as plt

# Link to Thorlabs documentation.
# https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=OCT

input_file = "/Users/jlefebvre/Library/CloudStorage/Dropbox/Data/concordia-psoct/manual_angle_15_second_trysliceIdx1_0001_ModePolarization3D.oct"

# Load the metadata in the zip file
zip = zipfile.ZipFile(input_file)
print(zip.namelist())

# Read the metadata
metadata_file = "Header.xml"
with zip.open(metadata_file) as f:
    document = parse(f)
print(document)

# Get the data (polarization 1)
raw_data1_file = r"data\Complex.data"
raw_data2_file = r"data\Complex_Cam1.data"

with zip.open(raw_data1_file) as f:
    data1 = f.read()
with zip.open(raw_data2_file) as f:
    data2 = f.read()
print(len(data1), type(data1))

# Convert the data to a numpy array
data1 = np.frombuffer(data1, dtype=np.complex64)
data1 = data1.reshape((300, 300, 1024), order='C')
data2 = np.frombuffer(data2, dtype=np.complex64)
data2 = data2.reshape((300, 300, 1024), order='C')

# Display the data
plt.figure()
plt.imshow(np.abs(data1.mean(axis=2)))
plt.show()
plt.figure()
plt.imshow(np.abs(data2.mean(axis=2)))
plt.show()

# Export as tiff
from imageio import volwrite
tiff_file = input_file.replace(".oct", "_polarization1.tiff")
volwrite(tiff_file, np.abs(data1))


