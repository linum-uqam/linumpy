#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import shutil
from pathlib import Path

from tqdm.auto import tqdm


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description="Move slices from a flat directory into subdirectories based on their names.")
    p.add_argument("--dir", help="Directory containing the slices.")
    return p


def get_file_list(tiles_directory: Path) -> list[Path]:
    """
    List all the tiles in a given directory.

    :param tiles_directory: Path to the directory containing the tiles.
    :type tiles_directory: Path
    :return: List of tile file paths.
    :rtype: list[Path]
    """
    # List all folders in the path
    folders = [f for f in tiles_directory.iterdir() if f.is_dir()]
    return folders


def copy_files(tiles_directory: Path, folders: list) -> None:
    """
    Create the new folders per z slice and copy the files from the old folders to the new ones.

    :param tiles_directory: Path to the directory containing the tiles.
    :type tiles_directory: Path
    :param folders: List of folders containing the tiles.
    :type folders: list[Path]

    :return: None
    """
    # Create new folders per z slice
    for folder in tqdm(folders, desc="Copying data from old folders to new z slice folders", unit="folder"):
        z_slice = folder.name.split("_")[-1]
        new_folder = tiles_directory / f"{z_slice}"
        new_folder.mkdir(exist_ok=True)
        # Copy all contents from the old folder to the new folder, keeping the structure but removing the z slice from the folder name
        for item in folder.iterdir():
            if item.is_file():
                new_sub_folder = new_folder / folder.name
                new_sub_folder.mkdir(exist_ok=True)
                shutil.copy(item, new_sub_folder / item.name)


def check_files(tiles_directory: Path, old_folders: list) -> None:
    """
    Check if all files were moved correctly by comparing the new folders with the old ones.

    :param tiles_directory: Path to the directory containing the tiles.
    :type tiles_directory: Path
    :param old_folders: List of old folders containing the tiles.
    :type old_folders: list[Path]
    """
    print("Checking if all files were moved correctly...")
    equal = True
    for folder in tqdm(old_folders, desc="Checking folders", unit="folder"):
        z_slice = folder.name.split("_")[-1]
        new_folder = tiles_directory / f"{z_slice}"
        new_sub_folder = new_folder / folder.name
        # Check if the new sub folder exists
        if not new_sub_folder.exists():
            print(f"New sub folder {new_sub_folder} does not exist")
            equal = False
            continue
        # Check if the files in the old folder are in the new sub folder
        for item in folder.iterdir():
            if item.is_file():
                if not (new_sub_folder / item.name).exists():
                    print(f"File {item.name} from {folder} is not in {new_sub_folder}")
                    equal = False

    if not equal:
        print("Not all files were moved correctly, check the output above.")
    else:
        print("All files were moved correctly.")


def remove_old_folders(old_folders: list) -> None:
    """
    Remove the old folders after confirming with the user.

    :param old_folders: List of old folders to remove.
    :type old_folders: list[Path]
    :return: None
    """
    confirm = input(f"Are you sure you want to remove {len(old_folders)} old folders? (y/n): ")
    if confirm.lower() != 'y':
        print("Aborting removal of old folders.")
        exit()
    else:
        for folder in tqdm(old_folders, desc="Removing old folders", unit="folder"):
            shutil.rmtree(folder)


def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()
    tiles_directory = Path(args.dir)

    if not tiles_directory.exists():
        print(f"Source directory {tiles_directory} does not exist.")
        exit(1)

    # Get the list of folders in the source directory
    old_folders = get_file_list(tiles_directory)
    # Copy files from the old folders to the new z slice folders
    copy_files(tiles_directory, old_folders)
    # Check if all files were moved correctly
    check_files(tiles_directory, old_folders)
    # Remove the old folders
    remove_old_folders(old_folders)


if __name__ == "__main__":
    main()
