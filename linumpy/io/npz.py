import numpy as np


def write_numpy(npz_path, *, data=None, metadata=None):
    np.savez_compressed(npz_path, data=data, metadata=metadata)


def read_numpy(npz_path) -> tuple:
    data = np.load(npz_path, allow_pickle=True)
    return data['data'], data['metadata']


def read_numpy_data(npz_path):
    npz = np.load(npz_path, allow_pickle=True)
    return npz['data']


def read_numpy_metadata(npz_path):
    npz = np.load(npz_path, allow_pickle=True)
    return npz['metadata']


def _example():
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