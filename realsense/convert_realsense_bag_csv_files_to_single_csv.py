from argparse import ArgumentParser
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np


def main():
    print("Starting...")

    parser = ArgumentParser()
    parser.add_argument("import_path", help="Path to csv files to convert.")
    parser.add_argument("export_path", help="Path to csv file to export to.")
    parser.add_argument("--cpu", '-c', help="Number of CPUs used to convert", required=False, default=12, type=int)
    args = parser.parse_args()

    print(f"  Convert from: \"{args.import_path}\"")
    print(f"  Export to:    \"{args.export_path}\"")

    files = sorted(Path(args.import_path).glob("*.npy"))

    print(f"  Converting using {args.cpu} CPU Cores...")
    pool = Pool(args.cpu)
    realsense_data = np.asarray(pool.map(np.load, files))

    print("  Saving...")
    np.savez_compressed(args.export_path, realsense_data=realsense_data)

    print("Done")


if __name__ == '__main__':
    main()
