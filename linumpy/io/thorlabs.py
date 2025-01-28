"""
ThorOCT Module

This module provides the ThorOCT class for handling OCT data from ThorLabs PSOCT microscopes.
It includes methods to load, process, and extract metadata and polarization data from compressed
files, as well as utility functions for preprocessing and tile extraction.
"""

import gc
from pathlib import Path
import zipfile
from xml.dom.minidom import parse
import numpy as np


class PreprocessingConfig:
    """
    Configuration class for preprocessing OCT data.
    """
    return_complex: bool
    crop_first_index: int = 320
    crop_second_index: int = 750
    erase_raw_data: bool = (True,)
    erase_polarization_1: bool = (False,)
    erase_polarization_2: bool = (True,)


class ThorOCT:
    """
    A class for handling OCT data from ThorLabs PSOCT microscopes. It provides methods to load,
    process, and extract metadata and data from compressed files.

    Attributes:
        path (str): Path to the compressed data file.
        compressed_data (zipfile.ZipFile): ZipFile object for the compressed data.
        polarization1 (np.ndarray): Data for the first polarization.
        polarization2 (np.ndarray): Data for the second polarization.
        size_x (int): X-dimension size of the data.
        size_y (int): Y-dimension size of the data.
        size_z (int): Z-dimension size of the data.
        header (xml.dom.minidom.Document): Parsed header metadata.
        resolution (list): Resolution values for X, Y, and Z dimensions.
    """

    def __init__(
        self,
        path: str = None,
        compressed_data: zipfile.ZipFile = None,
        config: PreprocessingConfig = None,
    ):
        """
        Initialize the ThorOCT object.

        Parameters:
            path (str): Path to the compressed data file.
            compressed_data (zipfile.ZipFile): ZipFile object containing the data.
        """
        self.path = path
        self.compressed_data = compressed_data or (
            zipfile.ZipFile(path) if path else None
        )
        self.first_polarization = None
        self.second_polarization = None
        self.size_x = None
        self.size_y = None
        self.size_z = None
        self.header = None
        self.resolution = []
        self.ascan_averaging_value = None
        self.config = config

    def load(self) -> None:
        """
        Load the data from the compressed file and extract the header and the complex data.

        Parameters:
        - erase_raw_data: If True, the raw data will be erased after loading the complex data.
        - erase_polarization_1: If True, the polarization 1 data will not be loaded to save space.
        - erase_polarization_2: If True, the polarization 2 data will not be loaded to save space.
        - return_complex: If True, the complex data will be returned.
            Otherwise, the magnitude of the complex data will be returned.

        Raises:
            ValueError: If no valid data source is provided.
        """
        if not self.compressed_data:
            raise ValueError("No valid data source provided.")
        self._extract_oct_header()
        self._extract_complex_dimensions()
        self._load_polarized_data(
            erase_polarization_1=self.config.erase_polarization_1,
            erase_polarization_2=self.config.erase_polarization_2,
        )

        # If requested, erase the raw data source (compressed_data) itself
        if self.config.erase_raw_data:
            self.compressed_data = None
            gc.collect()  # Force garbage collection

    def _extract_oct_header(self) -> None:
        """
        Loads and returns the OCT header metadata.

        Raises:
            FileNotFoundError: If the header file is not found in the compressed data.
        """
        try:
            metadata_file = "Header.xml"
            with self.compressed_data.open(metadata_file) as f:
                document = parse(f)
            self.header = document
        except KeyError as e:
            raise FileNotFoundError(f"Error loading header: {e}") from e

    def _extract_complex_dimensions(self) -> None:
        """
        Extract dimensions and resolution values from the OCT header.

        Raises:
            ValueError: If the header has not been loaded.
        """
        if not self.header:
            raise ValueError("Header must be loaded before extracting dimensions.")
        # Find all DataFile elements
        data_files = self.header.getElementsByTagName("DataFile")
        # Get the <AScans> element
        ascan_element = self.header.getElementsByTagName("AScans")[0]
        # Extract its text content and convert to an integer
        self.ascan_averaging_value = int(ascan_element.firstChild.data.strip())
        # Initialize variables to store found data
        complex_data_file = None
        # Loop through each DataFile element and check for the specific values
        for data_file in data_files:
            # Extract text content of the DataFile element
            file_content = data_file.firstChild.data
            # Check for specific file paths
            if file_content == "data\\Complex.data":
                complex_data_file = data_file
                break

        if complex_data_file:
            self.size_z = int(complex_data_file.getAttribute("SizeZ"))
            self.size_x = int(complex_data_file.getAttribute("SizeX"))
            self.size_y = int(complex_data_file.getAttribute("SizeY"))
            range_z = float(complex_data_file.getAttribute("RangeZ"))
            range_y = float(complex_data_file.getAttribute("RangeY"))
            range_x = float(complex_data_file.getAttribute("RangeX"))
            self.resolution = [
                range_x / self.size_x,
                range_y / self.size_y,
                range_z / self.size_z,
            ]

    def _load_polarized_data(self, erase_polarization_2, erase_polarization_1) -> None:
        """
        Load the polarization data from the compressed file.

        Parameters:
            return_complex (bool): Whether to load complex data.
            erase_polarization_2 (bool): Whether to skip loading polarization 2 data.
            erase_polarization_1 (bool): Whether to skip loading polarization 1 data.

        Raises:
            FileNotFoundError: If required polarization data files are missing.
        """
        try:
            # Files for the polarization data
            raw_data1_file = "data/Complex.data"
            raw_data2_file = "data/Complex_Cam1.data"
            # Load the data for polarization 1 and 2
            if not erase_polarization_1:
                self.first_polarization = self.load_and_process(file=raw_data1_file)
            if not erase_polarization_2:
                self.second_polarization = self.load_and_process(file=raw_data2_file)
        except KeyError as e:
            raise FileNotFoundError(f"Error loading polarization data: {e}") from e

    def _stack_tiles_vertically(self, data: np.ndarray) -> np.ndarray:
        """
        Stacks x tiles on top of each other along the y-axis according to the ascan_averaging_value.

        eg. If ascan_averaging_value = 3, 
        the first 3 tiles will be stacked on top of each other, then the next 3, and so on.
        [0, 1, 2] -> [0] = |2|
                           |1|
                           |0|
        Parameters:
            data (np.ndarray): The input 3D array (SizeZ, SizeX, SizeY).

        Returns:
            np.ndarray: The 3D array with tiles stacked along the y-axis.
        """
        # Ensure the number of tiles is divisible by ascan_averaging_value
        if data.shape[0] % self.ascan_averaging_value != 0:
            raise ValueError(
                f"The number of tiles ({data.shape[0]}) 
                must be divisible by ascan_averaging_value ({self.ascan_averaging_value})."
            )

        stacked_data = []
        for i in range(0, data.shape[0], self.ascan_averaging_value):
            stacked_tile = np.concatenate(
                data[i : i + self.ascan_averaging_value],
                axis=0,  # Stack along the z-axis
            )[
                ::-1
            ]  # Reverse the stacking order so the last tile appears on top
            stacked_data.append(stacked_tile)

        # Combine all stacked tiles into a single array
        stacked_data = np.stack(stacked_data, axis=0)
        # Since we stacked the tiles along the y-axis, we need to adjust the resolution
        self.resolution = [
            self.resolution[0] * self.size_x / stacked_data.shape[0],
            self.resolution[1] * self.size_y / stacked_data.shape[1],
            self.resolution[2],
        ]
        self.size_x = stacked_data.shape[0]
        self.size_y = stacked_data.shape[1]
        return stacked_data

    def _crop_z(
        self, data: np.ndarray, index1: int = 320, index2: int = 750
    ) -> np.ndarray:
        """
        Crops the 3D volume along the Z-axis and keeps the data between the specified indices.

        Parameters:
            data (np.ndarray): The input 3D array (SizeX, SizeY, SizeZ).
            index1 (int): The starting Z index for cropping (inclusive).
            index2 (int): The ending Z index for cropping (exclusive).

        Returns:
            np.ndarray: The cropped 3D array.

        Raises:
            ValueError: If indices are invalid.
        """
        # Ensure valid indices
        if index1 < 0 or index2 > data.shape[2] or index1 >= index2:
            raise ValueError(
                f"Invalid indices: index1={index1}, index2={index2}, data shape={data.shape}"
            )

        # Perform the crop
        cropped_data = data[:, :, index1:index2]
        self.size_z = cropped_data.shape[2]

        return cropped_data

    def _load_raw_data(self, file) -> np.ndarray:
        """
        Load the raw data from the specified file and return it as a NumPy array.

        Parameters:
            file (str): File path in the compressed data.

        Returns:
            np.ndarray: Raw complex data array.
        """
        with self.compressed_data.open(file) as f:
            raw_data = np.frombuffer(f.read(), dtype=np.complex64).reshape(
                (self.size_x, self.size_y, self.size_z), order="C"
            )
        return raw_data

    def _preprocess_data(
        self,
        data: np.ndarray,
    ) -> np.ndarray:
        """
        Preprocess the data, including cropping, stacking, and converting to magnitude.

        Parameters:
            data (np.ndarray): Input complex data array.

        Returns:
            np.ndarray: Preprocessed data array.
        """
        # Perform cropping
        data = self._crop_z(
            data,
            index1=self.config.crop_first_index,
            index2=self.config.crop_second_index,
        )
        # Perform stacking
        data = self._stack_tiles_vertically(data)
        # Return complex or magnitude data
        return data if self.config.return_complex else np.abs(data).astype(np.float64)

    def load_and_process(self, file: str) -> np.ndarray:
        """
        Load raw data from the file and preprocess it.

        Parameters:
            file (str): File path in the compressed data.

        Returns:
            np.ndarray: Fully processed data array.
        """
        raw_data = self._load_raw_data(file)
        processed_data = self._preprocess_data(raw_data)
        return processed_data

    @staticmethod
    def extract_positions_from_scan(
        scan_file_path: str = None, number_of_tiles: int = 0
    ):
        """
        Extracts the raw and index x, y positions from the .scan file.

        Parameters:
        - scan_file_path: Path to the .scan file.
        - number_of_tiles: Number of tiles in the scan.

        Returns:
        - tuple: A tuple containing two lists - index positions and raw positions.
        """
        raw_positions = []
        position_indexes = []

        if scan_file_path:
            with open(file=scan_file_path, mode="r", encoding="utf-8") as file:
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
                            x, y = map(float, line.split(","))
                            raw_positions.append((x, y, 0))

        # hold the index locations for the tile placements in the Zarr file.
        # This follows the current order of acquisition for the PSOCT microscope.
        for x in range(int(np.ceil(np.sqrt(number_of_tiles))) - 1, -1, -1):
            for y in range(int(np.ceil(np.sqrt(number_of_tiles))) - 1, -1, -1):
                position_indexes.append((x, y, 0))
        return position_indexes, raw_positions

    @staticmethod
    def get_psoct_tiles_ids(tiles_directory: str, number_of_angles: int = 2):
        """
        Get the .scan file and all .oct files from the tiles_directory.

        Parameters:
        - tiles_directory: Path to the directory containing the OCT tiles.
        - number_of_angles: Number of acquisition angles.

        Returns:
        - positions: positions of the tiles in 3d
        - grouped_files: list of file paths ordered by angles.

        Raises:
        - ValueError: If the directory or required files are missing.
        """
        # Convert the tiles_directory to a Path object
        tiles_path = Path(tiles_directory)

        if not tiles_path.is_dir():
            raise ValueError(
                f"Provided path '{tiles_directory}' is not a valid directory."
            )

        # Initialize variables to store the results
        scan_file = None
        oct_files = []
        grouped_files = [[] for _ in range(number_of_angles)]
        positions = []
        # Iterate through files in the directory
        for file in tiles_path.iterdir():
            # Check for .scan file
            if file.suffix == ".scan":
                scan_file = file
            # Collect .oct files
            elif file.suffix == ".oct":
                oct_files.append(file)
        positions, _ = ThorOCT.extract_positions_from_scan(scan_file, len(oct_files))

        # If no .oct files are found, raise a warning
        if not oct_files:
            raise ValueError("Warning: No .oct files found in the directory.")

        for i, oct_file in enumerate(oct_files):
            angle_index = (
                i % number_of_angles
            )  # Determine the angle based on file index
            grouped_files[angle_index].append(oct_file)
        print(
            f"File Count for Angle index = {angle_index + 1}: {len(grouped_files[angle_index])}"
        )
        print("Processing the following Files:")
        for file in grouped_files[0]:
            print(f"  - {file}")
        return grouped_files, positions

    @staticmethod
    def orient_volume_psoct(vol: np.ndarray) -> np.ndarray:
        """Transforms the volume to RAS orientation"""
        vol = vol.transpose(2, 0, 1)
        return vol
