#!/usr/bin/env python3
"""
Clean up raw data acquisitions by removing binary data files while preserving metadata.

This script:
- Removes all .bin files (raw data that has been processed)
- Removes processing files (ROI files, tile cleaning images)
- Removes OS cache files (.DS_Store, Thumbs.db, etc.)
- Keeps metadata.json and info.txt files
- Moves quick stitch images to the quick_stitches directory
- Moves all slice directories to a metadata subdirectory
- Maintains the directory structure

Usage:
    soct_clean_raw_data.py <data_directory> [--dry-run]

Arguments:
    data_directory: Path to the subject data directory (e.g., /path/to/sub-24)
    --dry-run: Show what would be done without actually doing it
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ensure_directory(directory: Path, dry_run: bool = False) -> None:
    """
    Create a directory if it doesn't exist.

    Args:
        directory: Path to the directory to create
        dry_run: If True, only log what would be done
    """
    if not dry_run and not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def move_item(source: Path, destination: Path, destination_label: str, dry_run: bool = False) -> bool:
    """
    Move a file or directory from source to destination.

    Args:
        source: Source path to move
        destination: Destination path
        destination_label: Label for logging (e.g., "quick_stitches/", "metadata/")
        dry_run: If True, only log what would be done

    Returns:
        True if moved (or would be moved), False if skipped
    """
    # Check if destination already exists
    if destination.exists():
        logger.warning(f"{source.name} already exists in destination, skipping: {source.name}")
        return False

    if dry_run:
        logger.info(f"[DRY RUN] Would move: {source} -> {destination}")
    else:
        shutil.move(str(source), str(destination))
        logger.info(f"Moved: {source.name} -> {destination_label}")

    return True


def find_bin_files(data_dir: Path) -> list[Path]:
    """Find all .bin files in the data directory."""
    return list(data_dir.rglob("*.bin"))


def find_quick_stitches(data_dir: Path) -> list[Path]:
    """Find quick stitch images in tile directories that need to be moved."""
    quick_stitches = []

    # Look for quick_stitch files in tiles directories
    for slice_dir in data_dir.glob("slice_z*"):
        tiles_dir = slice_dir / "tiles"
        if tiles_dir.exists():
            # Find quick stitch images in the tiles directory
            for qs_file in tiles_dir.glob("quick_stitch_*.jpg"):
                quick_stitches.append(qs_file)
            for qs_file in tiles_dir.glob("quick_stitch_*.png"):
                quick_stitches.append(qs_file)

    return quick_stitches


def move_quick_stitches(data_dir: Path, dry_run: bool = False) -> int:
    """
    Move quick stitch images to the quick_stitches directory.
    Note: The original files in the tiles directories will be deleted after moving.

    Returns:
        Number of files moved
    """
    quick_stitch_dir = data_dir / "quick_stitches"
    quick_stitches = find_quick_stitches(data_dir)

    if not quick_stitches:
        logger.info("No quick stitch images found to move")
        return 0

    # Create quick_stitches directory if it doesn't exist
    ensure_directory(quick_stitch_dir, dry_run)

    moved_count = 0
    for qs_file in quick_stitches:
        dest_file = quick_stitch_dir / qs_file.name
        if move_item(qs_file, dest_file, "quick_stitches/", dry_run):
            moved_count += 1

    return moved_count


def find_cache_files(data_dir: Path) -> list[Path]:
    """Find common OS cache files (macOS, Windows, Linux)."""
    cache_files = []

    # macOS cache files
    cache_files.extend(data_dir.rglob(".DS_Store"))
    cache_files.extend(data_dir.rglob("._*"))  # macOS resource forks

    # Windows cache files
    cache_files.extend(data_dir.rglob("Thumbs.db"))
    cache_files.extend(data_dir.rglob("Desktop.ini"))

    # Linux/general cache
    cache_files.extend(data_dir.rglob(".directory"))  # KDE
    cache_files.extend(data_dir.rglob("*~"))  # Backup files

    return list(cache_files)


def find_processing_files(data_dir: Path) -> list[Path]:
    """Find ROI and tile cleaning files that can be deleted after processing."""
    processing_files = []

    # Look for ROI files (roi_z*.png)
    processing_files.extend(data_dir.rglob("roi_z*.png"))

    # Look for tile cleaning files (both png and tif)
    processing_files.extend(data_dir.rglob("tile_cleaning.png"))
    processing_files.extend(data_dir.rglob("tile_cleaning.tif"))
    processing_files.extend(data_dir.rglob("tile_cleaning.tiff"))

    return list(processing_files)


def delete_processing_files(data_dir: Path, dry_run: bool = False) -> int:
    """
    Delete processing files (ROI and tile cleaning images).

    Returns:
        Number of files deleted
    """
    processing_files = find_processing_files(data_dir)

    if not processing_files:
        logger.info("No processing files found to delete")
        return 0

    deleted_count = 0
    for proc_file in processing_files:
        if dry_run:
            logger.info(f"[DRY RUN] Would delete processing file: {proc_file}")
        else:
            proc_file.unlink()
            logger.info(f"Deleted processing file: {proc_file}")

        deleted_count += 1

    return deleted_count


def delete_cache_files(data_dir: Path, dry_run: bool = False) -> int:
    """
    Delete OS cache files.

    Returns:
        Number of files deleted
    """
    cache_files = find_cache_files(data_dir)

    if not cache_files:
        logger.info("No cache files found to delete")
        return 0

    deleted_count = 0
    for cache_file in cache_files:
        if dry_run:
            logger.info(f"[DRY RUN] Would delete cache file: {cache_file}")
        else:
            cache_file.unlink()
            logger.info(f"Deleted cache file: {cache_file}")

        deleted_count += 1

    return deleted_count


def delete_bin_files(data_dir: Path, dry_run: bool = False) -> int:
    """
    Delete all .bin files in the data directory.

    Returns:
        Number of files deleted
    """
    bin_files = find_bin_files(data_dir)

    if not bin_files:
        logger.info("No .bin files found to delete")
        return 0

    deleted_count = 0
    total_size = 0

    for bin_file in bin_files:
        file_size = bin_file.stat().st_size
        total_size += file_size

        if dry_run:
            logger.info(f"[DRY RUN] Would delete: {bin_file} ({file_size / (1024 ** 2):.2f} MB)")
        else:
            bin_file.unlink()
            logger.info(f"Deleted: {bin_file}")

        deleted_count += 1

    logger.info(f"Total size of .bin files: {total_size / (1024 ** 3):.2f} GB")

    return deleted_count


def move_slices_to_metadata(data_dir: Path, dry_run: bool = False) -> int:
    """
    Move all slice directories to a metadata subdirectory.

    Returns:
        Number of slice directories moved
    """
    metadata_dir = data_dir / "metadata"
    slice_dirs = sorted(data_dir.glob("slice_z*"))

    if not slice_dirs:
        logger.info("No slice directories found to move")
        return 0

    # Create metadata directory if it doesn't exist
    ensure_directory(metadata_dir, dry_run)

    moved_count = 0
    for slice_dir in slice_dirs:
        dest_dir = metadata_dir / slice_dir.name
        if move_item(slice_dir, dest_dir, "metadata/", dry_run):
            moved_count += 1

    return moved_count


def verify_structure(data_dir: Path) -> bool:
    """
    Verify that the data directory has the expected structure.

    Returns:
        True if structure is valid, False otherwise
    """
    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        return False

    if not data_dir.is_dir():
        logger.error(f"Path is not a directory: {data_dir}")
        return False

    # Check for at least one slice directory
    slice_dirs = list(data_dir.glob("slice_z*"))
    if not slice_dirs:
        logger.error("No slice directories found (expected slice_z*)")
        return False

    logger.info(f"Found {len(slice_dirs)} slice directories")

    return True


def clean_raw_data(data_dir: Path, dry_run: bool = False) -> dict:
    """
    Main function to clean raw data.

    Returns:
        Dictionary with statistics about the cleanup
    """
    logger.info(f"Cleaning raw data in: {data_dir}")

    if dry_run:
        logger.info("DRY RUN MODE - No files will be modified")

    # Verify structure
    if not verify_structure(data_dir):
        logger.error("Data directory structure verification failed")
        return {"success": False}

    # Move quick stitches
    logger.info("\n=== Moving quick stitch images ===")
    moved_count = move_quick_stitches(data_dir, dry_run)

    # Delete .bin files
    logger.info("\n=== Deleting .bin files ===")
    deleted_count = delete_bin_files(data_dir, dry_run)

    # Delete processing files (ROI and tile cleaning)
    logger.info("\n=== Deleting processing files ===")
    processing_deleted = delete_processing_files(data_dir, dry_run)

    # Delete cache files
    logger.info("\n=== Deleting cache files ===")
    cache_deleted = delete_cache_files(data_dir, dry_run)

    # Move slice directories to metadata folder
    logger.info("\n=== Moving slice directories to metadata folder ===")
    slices_moved = move_slices_to_metadata(data_dir, dry_run)

    # Summary
    logger.info("\n=== Cleanup Summary ===")
    logger.info(f"Quick stitch images moved: {moved_count}")
    logger.info(f"Binary files deleted: {deleted_count}")
    logger.info(f"Processing files deleted: {processing_deleted}")
    logger.info(f"Cache files deleted: {cache_deleted}")
    logger.info(f"Slice directories moved to metadata: {slices_moved}")

    return {
        "success": True,
        "moved_count": moved_count,
        "deleted_count": deleted_count,
        "processing_deleted": processing_deleted,
        "cache_deleted": cache_deleted,
        "slices_moved": slices_moved,
    }


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Clean up raw data acquisitions by removing binary files and organizing quick stitches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would be deleted
  %(prog)s /path/to/sub-24 --dry-run
  
  # Actually clean the data
  %(prog)s /path/to/sub-24
        """
    )

    parser.add_argument(
        "data_directory",
        type=Path,
        help="Path to the subject data directory (e.g., /path/to/sub-24)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually doing it"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Confirm if not dry run
    if not args.dry_run:
        print(f"\nWARNING: This will DELETE all .bin files in {args.data_directory}")
        response = input("Are you sure you want to continue? [y/N]: ")
        if response.lower() != 'y':
            print("Operation cancelled")
            return 0

    # Run the cleanup
    result = clean_raw_data(args.data_directory, args.dry_run)

    if result["success"]:
        logger.info("\nCleanup completed successfully")
        return 0
    else:
        logger.error("\nCleanup failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
