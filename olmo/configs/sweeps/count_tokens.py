import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import os # Needed for file size

def count_tokens_in_directory(directory_path_str, force_dtype_str=None):
    """
    Counts the total number of tokens (elements) in all .bin files
    within a directory. Handles np.load compatible files and attempts
    to handle raw binary files by guessing the dtype based on file size
    or using a forced dtype.

    Args:
        directory_path_str (str): The path to the directory to scan.
        force_dtype_str (str, optional): Force interpretation of raw binary files
                                         as this dtype ('int16' or 'int32').
                                         Defaults to None (auto-guess).

    Returns:
        int: The total number of tokens found, or None if the directory is invalid.
    """
    directory_path = Path(directory_path_str)
    if not directory_path.is_dir():
        print(f"Error: Provided path '{directory_path_str}' is not a valid directory.")
        return None

    print(f"Scanning directory: {directory_path.resolve()}")
    if force_dtype_str:
        print(f"Attempting to force raw binary dtype to: {force_dtype_str}")

    bin_files = list(directory_path.rglob('*.bin'))

    if not bin_files:
        print("Warning: No '.bin' files found in the specified directory.")
        return 0

    print(f"Found {len(bin_files)} '.bin' files. Counting tokens...")

    total_tokens = 0
    files_processed = 0
    files_skipped = 0
    # NumPy .npy magic string - b'\x93NUMPY'

    forced_dtype = None
    if force_dtype_str:
        if force_dtype_str.lower() == 'int16':
            forced_dtype = np.int16
        elif force_dtype_str.lower() == 'int32':
            forced_dtype = np.int32
        else:
            print(f"Warning: Invalid --dtype '{force_dtype_str}'. Ignoring.")


    for file_path in tqdm(bin_files, unit="file"):
        file_str = str(file_path)
        processed_file = False
        try:
            # --- Attempt 1: Standard np.load (no pickle) ---
            try:
                array = np.load(file_str, mmap_mode='r', allow_pickle=False)
                total_tokens += array.size
                files_processed += 1
                del array
                processed_file = True
            except ValueError:
                pass # Failed, likely needs pickle or is raw binary
            except Exception as e:
                 tqdm.write(f"Warning: Skipping file {file_path.name} - Initial np.load(allow_pickle=False) failed unexpectedly: {e}")
                 files_skipped += 1
                 processed_file = True # Mark as handled (skipped)

            if processed_file: continue

            # --- Attempt 2: Standard np.load (with pickle) ---
            try:
                array = np.load(file_str, mmap_mode='r', allow_pickle=True)
                if isinstance(array, np.ndarray):
                    total_tokens += array.size
                    files_processed += 1
                    del array
                    processed_file = True
                    # tqdm.write(f"Note: Loaded {file_path.name} using allow_pickle=True.")
                else:
                    # Loaded something, but not a numpy array
                    tqdm.write(f"Warning: Skipping file {file_path.name} - Loaded with pickle, but is not a NumPy array (type: {type(array)}).")
                    files_skipped += 1
                    if hasattr(array, 'close'): array.close()
                    del array
                    processed_file = True

            except Exception: # Includes "Failed to interpret as pickle"
                 pass # Failed, likely raw binary or corrupt

            if processed_file: continue

            # --- Attempt 3: Assume raw binary and use np.fromfile ---
            try:
                file_size = os.path.getsize(file_str)
                if file_size == 0:
                    # tqdm.write(f"Note: Skipping empty file {file_path.name}.")
                    files_skipped += 1
                    continue # Skip empty files

                dtype_to_try = None
                itemsize = 0

                if forced_dtype:
                    dtype_to_try = forced_dtype
                    itemsize = np.dtype(dtype_to_try).itemsize
                    if file_size % itemsize != 0:
                         tqdm.write(f"Warning: Skipping file {file_path.name} - File size {file_size} not divisible by item size {itemsize} for forced dtype {force_dtype_str}.")
                         files_skipped += 1
                         continue
                else:
                    # Auto-guess dtype based on divisibility
                    if file_size % 2 == 0: # Check int16 first
                        dtype_to_try = np.int16
                        itemsize = 2
                        # If also divisible by 4, could be int32, but int16 is common for tokens. Stick with int16 guess.
                    elif file_size % 4 == 0: # Check int32 if not divisible by 2
                        dtype_to_try = np.int32
                        itemsize = 4
                    # Add elif file_size % 8 == 0: for int64 if needed

                if dtype_to_try:
                    num_tokens = file_size // itemsize
                    total_tokens += num_tokens
                    files_processed += 1
                    # tqdm.write(f"Note: Interpreted {file_path.name} as raw binary with guessed dtype {dtype_to_try.__name__}.")
                else:
                    tqdm.write(f"Warning: Skipping file {file_path.name} - Could not guess raw binary dtype based on file size {file_size}.")
                    files_skipped += 1

            except OSError as e:
                 tqdm.write(f"Warning: Skipping file {file_path.name} - OS error during raw file check: {e}")
                 files_skipped += 1
            except Exception as e:
                 tqdm.write(f"Warning: Skipping file {file_path.name} - Unexpected error during raw file processing: {e}")
                 files_skipped += 1


        except Exception as e:
            # Catch-all for unexpected errors for this file during the outer try
            tqdm.write(f"ERROR: Unexpected error processing file {file_path.name}: {e}")
            files_skipped += 1


    print(f"\nProcessed {files_processed} files.")
    if files_skipped > 0:
         print(f"Skipped {files_skipped} files due to errors or incompatible formats.")

    return total_tokens

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count total tokens in NumPy '.bin' files within a directory. Handles np.load and raw binary formats."
    )
    parser.add_argument(
        "directory",
        help="Path to the directory containing the .bin files."
    )
    parser.add_argument(
        "--dtype",
        choices=['int16', 'int32'], # Limit choices or remove for flexibility
        default=None,
        help="Force interpretation of raw binary files as this dtype (e.g., int16, int32). Default: auto-guess."
    )
    args = parser.parse_args()

    final_count = count_tokens_in_directory(args.directory, force_dtype_str=args.dtype)

    if final_count is not None:
        print("\n--------------------------------------------------")
        print(f"Total number of tokens found: {final_count:,}")
        print("--------------------------------------------------")