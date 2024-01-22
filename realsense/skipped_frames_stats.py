import argparse

import numpy as np
import os


def main():
    skipped_frames = []
    with open(args.input) as file:
        skipped_frames = [int(line.strip()) for line in file]

    consecutive_skipped_frames = []
    current_run = [skipped_frames[0]]
    for current, next in zip(skipped_frames, skipped_frames[1:]):

        # if not current_run:
        #     current_run.append(current)
        #     continue

        if current + 1 == next:
            current_run.append(next)
        else:
            consecutive_skipped_frames.append(current_run)
            current_run = [next]


        # if next == current + 1:
        #     current_run.append(current)
        # elif len(current_run) > 0:
        #     current_run.append(current)
        #     consecutive_skipped_frames.append(current_run)
        #     current_run = []
        #     consecutive_skipped_frames.append([current])
        # else:
        #     consecutive_skipped_frames.append([current])
    consecutive_skipped_frames.append(current_run)

    print(skipped_frames)
    print(consecutive_skipped_frames)
    flat_list = [item for sublist in consecutive_skipped_frames for item in sublist]

    print(flat_list)

    run_lengths = np.array([len(x) for x in consecutive_skipped_frames])
    print(f"Mean run length: {run_lengths.mean()}")
    print(f"Max run length: {run_lengths.max()}")
    print(f"Min run length: {run_lengths.min()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="path to file to read skipped frames one for each line")
    args = parser.parse_args()

    main()
