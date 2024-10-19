import zipfile
from xml.dom.minidom import parse
import numpy as np
from imageio import volwrite
import gc  # For garbage collection

"""This file contains functions for working with Thorlabs files"""

class ThorOCT:
    def __init__(self, path: str = None, compressed_data: zipfile.ZipFile = None):
        self.path = path
        self.compressed_data = compressed_data or (zipfile.ZipFile(path) if path else None)
        self.polarization1 = None
        self.polarization2 = None
        self.size_x = None
        self.size_y= None
        self.size_z = None
        self.header = None
        
    def load(self, erase_raw_data = True, erase_polarization_1 = False, erase_polarization_2 = False):
        if not self.compressed_data:
            raise ValueError("No valid data source provided.")
        self._extract_oct_header()
        self._extract_complex_dimensions()
        self._load_polarized_data(erase_polarization_1, erase_polarization_2)
        
        # If requested, erase the raw data source (compressed_data) itself
        if erase_raw_data:
            print("Erasing the compressed raw data to free up memory.")
            self.compressed_data = None
            gc.collect()  # Force garbage collection
    
    def _extract_oct_header(self) -> None:
        """
        Loads and returns the OCT header metadata.
        """
        try:
            metadata_file = "Header.xml"
            with self.compressed_data.open(metadata_file) as f:
                document = parse(f)
            self.header = document
        except KeyError as e:
            raise FileNotFoundError(f"Error loading header: {e}")
            
    def _extract_complex_dimensions(self) -> None:
        if not self.header:
            raise ValueError("Header must be loaded before extracting dimensions.")
        # Find all DataFile elements
        data_files = self.header.getElementsByTagName('DataFile')

        # Initialize variables to store found data
        complex_data_file = None

        # Loop through each DataFile element and check for the specific values
        for data_file in data_files:
            # Extract text content of the DataFile element
            file_content = data_file.firstChild.data
            
            # Check for specific file paths
            if file_content == "data\\Complex.data":
                complex_data_file = data_file
                print(f"Found Complex Data File: {file_content}")

        # Optionally, access attributes of the found elements
        if complex_data_file:
            self.size_z = int(complex_data_file.getAttribute('SizeZ'))
            self.size_x = int(complex_data_file.getAttribute('SizeX'))
            self.size_y = int(complex_data_file.getAttribute('SizeY'))
            print(f"Complex Data - SizeZ: {self.size_z}, SizeX: {self.size_x}, SizeY: {self.size_y}")

    def _load_polarized_data(self, erase_polarization_1, erase_polarization_2) -> None:
        """
        Loads the polarization data from the zip file.
        """
        try:
            # Files for the polarization data
            raw_data1_file = "data/Complex.data"
            raw_data2_file = "data/Complex_Cam1.data"
            # Load the data for polarization 1 and 2
            if not erase_polarization_1:
                self.polarization1 = self._convert_to_numpy(file = raw_data1_file)
            if not erase_polarization_2:
                self.polarization2 = self._convert_to_numpy(file = raw_data2_file)
        except KeyError as e:
            print(f"Error loading data: {e}")

    def _convert_to_numpy(self, file) -> np.ndarray:
        with self.compressed_data.open(file) as f:
            return np.frombuffer(f.read(), dtype=np.complex64).reshape((self.size_x, self.size_y, self.size_z), order='C')
            
# input_file = "../../Data/2024_07_25_mouse_CB_1slice_4anglesliceIdx1_0001_ModePolarization3D.oct"
# oct = ThorOCT(path = input_file)
# oct.load()
# pol1 = oct.polarization2
# tiff_file = input_file.replace(".oct", "_polarization2.tiff")
# volwrite(tiff_file, np.abs(pol1))