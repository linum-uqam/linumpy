#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Move slices from a flat directory into subdirectories based on their names.
"""
import argparse
import filecmp
import re
import shutil
from pathlib import Path

from tqdm.auto import tqdm


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__)
    p.add_argument("--dir", help="Directory containing the slices.", required=True)
    return p


def get_folder_list(tiles_directory: Path) -> list[Path]:
    """
    List all subdirectories in a given directory, excluding those that match the "zxx" pattern.

    :param tiles_directory: Path to the directory containing the tiles.
    :type tiles_directory: Path
    :return: List of subdirectory paths (excluding "zxx" directories).
    :rtype: list[Path]
    """
    # List all folders in the path
    folders = [f for f in tiles_directory.iterdir() if f.is_dir()]
    # Exclude folders that match the pattern "^z\d\d$" (e.g., "z12", "z34")
    pattern = re.compile(r"^z\d\d$")
    folders = [f for f in folders if not pattern.match(f.name)]
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
        # Create the sub-folder for the current folder
        new_sub_folder = new_folder / folder.name
        new_sub_folder.mkdir(exist_ok=True)
        # Copy all contents from the old folder to the new folder
        for item in folder.iterdir():
            if item.is_file():
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
    for folder in (pbar := tqdm(old_folders, desc="Checking folders", unit="folder")):
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
            pbar.set_postfix({"Checking": f"{folder.name}/{item.name}"})
            if item.is_file():
                if not (new_sub_folder / item.name).exists():
                    print(f"File {item.name} from {folder} is not in {new_sub_folder}")
                    equal = False
                else:
                    # Compare the files
                    if not filecmp.cmp(item, new_sub_folder / item.name, shallow=False):
                        print(f"File {item.name} from {folder} is different in {new_sub_folder}")
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
        p.error(f"Directory {tiles_directory} does not exist.")

    # Get the list of folders in the source directory
    old_folders = get_folder_list(tiles_directory)

    if not old_folders:
        p.error("No old tile folders found in the directory.")

    # Copy files from the old folders to the new z slice folders
    copy_files(tiles_directory, old_folders)
    # Check if all files were moved correctly
    check_files(tiles_directory, old_folders)
    # Remove the old folders
    remove_old_folders(old_folders)


if __name__ == "__main__":
    main()
