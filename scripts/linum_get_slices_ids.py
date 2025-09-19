#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""List the slice ids from a tiles directory"""

import argparse
import csv
from pathlib import Path
import numpy as np

from linumpy.reconstruction import get_tiles_ids


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("directory",
                   help="Tiles directory")
    p.add_argument("output_file",
                   help="Output CSV file")
    return p


def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Extract the parameters
    tiles_directory = Path(args.directory)
    output_file = Path(args.output_file)
    assert output_file.name.endswith(".csv"), "The output file must be as csv file"

    # Detect the tiles
    tiles, tiles_ids = get_tiles_ids(tiles_directory)

    # Extract the list of slice ids
    z_list = []
    for id in tiles_ids:
        z_list.append(id[2])
    z_list = list(set(z_list))

    # Save the ids to a csv file
    data = np.array([z_list, ]).T
    with open(output_file, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        writer.writerow(["slice_id"])
        writer.writerows(data)


if __name__ == "__main__":
    main()
