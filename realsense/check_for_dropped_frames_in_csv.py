from argparse import ArgumentParser
from pathlib import Path
import pandas as pd


def main():
    print("Starting...")

    parser = ArgumentParser()
    parser.add_argument("input_path", help="Path to metadata csv files.")
    args = parser.parse_args()

    files = sorted(Path(args.input_path).glob("*metadata*.txt"))
    print(f"Analyzing {len(files)} files...")

    df = pd.read_csv(files[0], delimiter=': ', header=None).T
    df = df.drop(df.index[1])

    for file in files:
        cur_df = pd.read_csv(file, delimiter=': ', header=None).T.iloc[1:, :]
        df = pd.concat([df, cur_df], ignore_index=True)

    df.columns = df.iloc[0]
    df = df.drop(df.index[0])

    frames = pd.to_numeric(df["Frame Counter"])
    recorded_frames_set = set(frames)
    all_frames_in_range_set = set(range(frames.min(), frames.max()))
    missing_frames = sorted(list(all_frames_in_range_set - recorded_frames_set))

    print(f"Following frames are missing: {missing_frames}")


if __name__ == '__main__':
    main()