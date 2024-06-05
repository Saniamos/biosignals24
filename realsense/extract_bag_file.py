import argparse

import pyrealsense2 as rs
import numpy as np
import os

def main():
    if not os.path.exists(args.directory):
        os.mkdir(args.directory)

    skipped_frames = []
    try:
        config = rs.config()
        rs.config.enable_device_from_file(config, args.input, False)
        pipeline = rs.pipeline()
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 60)
        profile = pipeline.start(config)
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)

        last_frame_number = -1
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            current_frame_number = depth_frame.get_frame_number() - 1

            if current_frame_number % 100 == 0:
                print("Saving frame:", current_frame_number)

            depth_image = np.asanyarray(depth_frame.get_data())
            np.save(f"{args.directory}/frame_{current_frame_number:05d}", depth_image)

            if last_frame_number + 1 != current_frame_number:
                frame_skip = list(range(last_frame_number + 1, current_frame_number))
                skipped_frames += frame_skip
                print(f"Frame(s) skipped: {frame_skip} (overall: {len(skipped_frames)})")

            last_frame_number = current_frame_number

    except RuntimeError:
        print("No more frames arrived, reached end of BAG file!")
        print(f"{len(skipped_frames)} frames were skipped during export!")
        with open(args.skipped_frames, 'w+') as file:
            for skipped_frame in skipped_frames:
                file.write(f"{skipped_frame}\n")

        print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", type=str, help="Path to save the images")
    parser.add_argument("-i", "--input", type=str, help="Bag file to read")
    parser.add_argument("-s", "--skipped-frames", type=str, help="File to log skipped frames.")
    args = parser.parse_args()

    main()
