import numpy as np
from typing import Any, Optional, Tuple


def write_numpy(npz_path: str, *, data: Optional[Any]=None, metadata: Optional[Any]=None):
    """
    Writes data and metadata to a compressed numpy (.npz) file.
    Data and metadata are wrapped in a numpy array before being written to the file.
    Data and metadata are stored in the 'data' and 'metadata' keys of the .npz file.

    Parameters:
    npz_path (str): The path to the .npz file to write to.
    data (Any, optional): The data to write to the file. Defaults to None.
    metadata (Any, optional): The metadata to write to the file. Defaults to None.
    """
    np.savez_compressed(
        npz_path, 
        data=np.array([data]), 
        metadata=np.array([metadata]),
        types=np.array([{
            'data': type(data),
            'metadata': type(metadata),
        }])
    )


def read_numpy(npz_path: str) -> Tuple[Any, Any]:
    """
    Reads data and metadata from a compressed numpy (.npz) file.

    Parameters:
    npz_path (str): The path to the .npz file to read from.

    Returns:
    Tuple[Any, Any]: A tuple containing the data and metadata read from the file.
    """
    npz = np.load(npz_path, allow_pickle=True)
    return npz['data'][0], npz['metadata'][0]


def read_numpy_data(npz_path: str) -> Tuple[Any, type]:
    """
    Reads only the data from a compressed numpy (.npz) file.

    Parameters:
    npz_path (str): The path to the .npz file to read from.

    Returns:
    Tuple[Any, type]: The data and its type read from the file.
    """
    npz = np.load(npz_path, allow_pickle=True)
    return npz['data'][0], npz['types'][0]['data']


def read_numpy_metadata(npz_path: str) -> Tuple[Any, type]:
    """
    Reads only the metadata from a compressed numpy (.npz) file.

    Parameters:
    npz_path (str): The path to the .npz file to read from.

    Returns:
    Tuple[Any, type]: The metadata and its type read from the file.
    """
    npz = np.load(npz_path, allow_pickle=True)
    return npz['metadata'][0], npz['types'][0]['metadata']


def _example_one():
    # Usage example
    from skimage import data as skdata

    array = skdata.coins()
    write_numpy(
        './coins.npz', 
        data=array, 
        metadata={
            'example': 'this is an example of metadata'
        }
    )

    data, metadata = read_numpy('./coins.npz')
    print(f'data : {data.shape}')
    print(f'metadata : {metadata}')


def _example_two():
    class Person:
        def __init__(self, name, age):
            self.name = name
            self.age = age

        def __repr__(self):
            return f'Person(name={self.name} age={self.age})'
        
    person = Person('John', 30)
    write_numpy(
        './person.npz', 
        data=person, 
        metadata={
            'example': 'this file contains a person object'
        }
    )
    
    data, data_type = read_numpy_data('./person.npz')
    print(data)
    print(data_type)

    metadata, metadata_type = read_numpy_metadata('./person.npz')
    print(metadata)
    print(metadata_type)

