import zipfile
from xml.dom.minidom import parse
import numpy as np
from imageio import volwrite
import gc  # For garbage collection
from pathlib import Path

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
        self.resolution = []
        
    def load(self, erase_raw_data = True, erase_polarization_1 = False, erase_polarization_2 = False, return_complex = False):
        if not self.compressed_data:
            raise ValueError("No valid data source provided.")
        self._extract_oct_header()
        self._extract_complex_dimensions()
        self._load_polarized_data(erase_polarization_1, erase_polarization_2, return_complex)
        
        # If requested, erase the raw data source (compressed_data) itself
        if erase_raw_data:
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

        # Optionally, access attributes of the found elements
        if complex_data_file:
            self.size_z = int(complex_data_file.getAttribute('SizeZ'))
            self.size_x = int(complex_data_file.getAttribute('SizeX'))
            self.size_y = int(complex_data_file.getAttribute('SizeY'))
            range_z = float(complex_data_file.getAttribute('RangeZ'))
            range_y = float(complex_data_file.getAttribute('RangeY'))
            range_x = float(complex_data_file.getAttribute('RangeX'))
            self.resolution = [
                range_x / self.size_x,
                range_y / self.size_y,
                range_z / self.size_z
            ]

    def _load_polarized_data(self, erase_polarization_1, erase_polarization_2, return_complex) -> None:
        """
        Loads the polarization data from the zip file.
        """
        try:
            # Files for the polarization data
            raw_data1_file = "data/Complex.data"
            raw_data2_file = "data/Complex_Cam1.data"
            # Load the data for polarization 1 and 2
            if not erase_polarization_1:
                self.polarization1 = self._convert_to_numpy(file = raw_data1_file, return_complex = return_complex)
            if not erase_polarization_2:
                self.polarization2 = self._convert_to_numpy(file = raw_data2_file, return_complex = return_complex)
        except KeyError as e:
            raise Exception(f"Error loading polarization data: {e}")

    def _convert_to_numpy(self, file, return_complex = False) -> np.ndarray:
        with self.compressed_data.open(file) as f:
            complex_data = np.frombuffer(f.read(), dtype=np.complex64).reshape((self.size_x, self.size_y, self.size_z), order='C')
            if return_complex:
                return complex_data
            # Return the magnitude of complexe by default
            return np.abs(complex_data).astype(np.float32)
        
    @staticmethod
    def extract_positions_from_scan(scan_file_path: str):
        """
        Extracts the x, y positions from the .scan file.

        Parameters:
        - scan_file_path: Path to the .scan file.

        Returns:
        - list containing the x, y index positions.
        - list containing the raw (during acquisition) x, y positions.
        """
        raw_positions = []
        position_indexes = []
        
        with open(scan_file_path, 'r') as file:
            lines = file.readlines()

            # Find the start of the positions section
            positions_section = False
            for line in lines:
                line = line.strip()

                # Mark the start of the positions section
                if line == "------Positions------":
                    positions_section = True
                    continue

                # If in the positions section, extract x, y values
                if positions_section:
                    if line:  # Ignore empty lines
                        # Split by comma and convert to float
                        x, y = map(float, line.split(','))
                        raw_positions.append((x, y, 0))
        
        # hold the index locations for the tile placements in the Zarr file. The PSOCT acquisition (in its current state) is traversing from top right to bottom left.               
        for x in range(3, -1, -1): #range(4): 
            for y in range(4):
                position_indexes.append((x, y, 0))
        return position_indexes, raw_positions
    
    def get_PSOCT_tiles_ids(tiles_directory: str, number_of_angles:int = 2):
        """
        Get the .scan file and all .oct files from the tiles_directory.

        Parameters:
        - tiles_directory: Path to the directory containing the OCT tiles.
        - number_of_angles: Number of acquisition angles.

        Returns:
        - positions: positions of the tiles in 3d
        - grouped_files: list of file paths ordered by angles.
        """
        # Convert the tiles_directory to a Path object
        tiles_path = Path(tiles_directory)
        
        if not tiles_path.is_dir():
            raise ValueError(f"Provided path '{tiles_directory}' is not a valid directory.")
        
        # Initialize variables to store the results
        scan_file = None
        oct_files = []
        grouped_files = [[] for _ in range(number_of_angles)]

        # Iterate through files in the directory
        for file in tiles_path.iterdir():
            # Check for .scan file
            if file.suffix == ".scan":
                scan_file = file
                positions, _ = ThorOCT.extract_positions_from_scan(scan_file)
            # Collect .oct files
            elif file.suffix == ".oct":
                oct_files.append(file)
        
        # If no .scan file is found, raise a warning
        if scan_file is None:
            raise ValueError("Warning: No .scan file found in the directory.")
        
        # If no .oct files are found, raise a warning
        if not oct_files:
            raise ValueError("Warning: No .oct files found in the directory.")
            
        for i, oct_file in enumerate(oct_files):
            angle_index = i % number_of_angles  # Determine the angle based on file index
            grouped_files[angle_index].append(oct_file)
        print(f"File Count for Angle index = {angle_index + 1}: {len(grouped_files[angle_index])}")
        print("Processing the following Files:")
        for file in grouped_files[0]:
            print(f"  - {file}")
        return grouped_files, positions
    
    def preprocess_volume_PSOCT(vol: np.ndarray) -> np.ndarray:
        """Preprocess the volume by rotating and flipping it."""
        vol = vol.transpose(2, 0, 1)
        return vol

            
input_file = "C:\\Users\\Mohamad Hawchar\\Concordia University - Canada\\NeuralABC as-psOCT Samples - data\\2024_07_25_mouse_CB_1slice_2anglesliceIdx1_SUCCESS\\2024_07_25_mouse_CB_1slice_4anglesliceIdx1_0017_ModePolarization3D.oct"
oct = ThorOCT(path = input_file)
oct.load()
pol1 = oct.polarization2
tiff_file = "C:\\Users\\Mohamad Hawchar\\School\\UQAM\\LINUM\\PS-OCT\\m17_polarization2.tiff"
volwrite(tiff_file, np.abs(pol1))