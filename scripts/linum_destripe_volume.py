"""
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from linumpy.destripe.general_stripe_remover import GeneralStripeRemover
from linumpy.io.zarr import read_omezarr, save_omezarr

import time
import dask.array as da

import argparse


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_image',
                   help='Input image in .ome.zarr format.')
    p.add_argument('out_image',
                   help='Output image in .ome.zarr format.')
    p.add_argument('--out_screenshot',
                   help='Optional screenshot filename.')
    p.add_argument('--use_gpu', action='store_true',
                   help='Use GPU. Else runs on CPU.')
    p.add_argument('--mu', nargs=2, type=float, default=[0.17, 0.003],
                   help='Optimizer parameters. Recommended parameters:\n'
                        '- [0.17, 0.003] or [0.23, 0.003]:\n'
                        '    If stripes are thin and impairment is low.\n'
                        '- [0.33, 0.003] or [0.4,0.007]:\n'
                        '    If stripes are wider and corruptions severely\n'
                        '    influence the visual impression.\n'
                        '- [0.5, 0.017]:\n'
                        '    If corruptions are severe and stripes are of\n'
                        '    short length (on the scale of structures).')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Open 3D image
    vol, res = read_omezarr(args.in_image)
    if not np.allclose(res[:-1], res[1:]):
        parser.error(f'Isotropic input expected. Got: {res}.')

    img = torch.from_numpy(vol[:]).float()

    # assuming images are stacked along first axis
    # stripes will be along second and last axis
    direction = (0, 0, 1)

    # Normalize image
    img = (img - img.min())/(img.max() - img.min())

    # Process image
    t1 = time.time()
    result = GeneralStripeRemover(img,iterations=1000, proj=True,
                                  mu=args.mu, resz=1,
                                  direction=direction,
                                  GPU=args.use_gpu, verbose=True)
    t2 = time.time()
    print("\nProcessing time: ",t2-t1)

    if args.out_screenshot is not None:
        residual = (result - img).abs()

        # Visualize result
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        fig.set_dpi(100)
        axes[0,0].imshow(img[:,:,img.shape[2]//2],cmap='gray')
        axes[0,1].imshow(result[:,:,img.shape[2]//2],cmap='gray')
        axes[0,2].imshow(residual[:,:,img.shape[2]//2], cmap='gray')
        axes[1,0].imshow(img[:,img.shape[1]//2,:],cmap='gray')
        axes[1,1].imshow(result[:,img.shape[1]//2,:],cmap='gray')
        axes[1,2].imshow(residual[:,img.shape[1]//2,:],cmap='gray')
        axes[2,0].imshow(img[img.shape[0]//2,:,:],cmap='gray')
        axes[2,1].imshow(result[img.shape[0]//2,:,:],cmap='gray')
        axes[2,2].imshow(residual[img.shape[0]//2,:,:],cmap='gray')
        plt.tight_layout()
        fig.savefig(args.out_screenshot)

    out = result.numpy()
    save_omezarr(da.from_array(out), args.out_image, res)


if __name__ == '__main__':
    main()
