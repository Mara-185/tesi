#!/usr/bin/env python3
"""Produces the _interpreted.h5 file from _scan.h5 files."""
import argparse
import glob
import os
import traceback
from tqdm import tqdm
from analysis import Analysis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_file", nargs="*",
        help="The _scan.h5 file(s). If not given, looks in output_data/module_0/chip_0.")
    parser.add_argument("-f", "--overwrite", action="store_true",
                        help="Overwrite the _interpreted.h5 when already present.")
    args = parser.parse_args()

    files = []
    if args.input_file:  # If anything was given on the command line
        for pattern in args.input_file:
            files.extend(glob.glob(pattern, recursive=True))
    else:
        files.extend(glob.glob("output_data/module_0/chip_0/*_scan.h5"))
    files.sort()

    if not args.overwrite:
        files = [fp for fp in files if not os.path.isfile(os.path.splitext(fp)[0] + "_interpreted.h5")]

    for fp in tqdm(files, unit="File"):
        try:
            print("Processing", fp)
            with Analysis(raw_data_file=fp) as a:
                a.analyze_data()
        except Exception:
            print(traceback.format_exc())
