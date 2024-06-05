import pyrealsense2 as rs
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime

@DeprecationWarning
def main():
    parser = ArgumentParser()
    parser.add_argument("--fps", "-f",
                        help="Frames per Second setting of camera. Needs to be at least double of clock signal.",
                        type=float, default=60)
    parser.add_argument("--resolution", "-r", help="Length of 'on' portion of signal in micro seconds.", type=tuple,
                        default=(640, 480))
    parser.add_argument("--timeout", "-t", help="Max time in seconds to wait for new frame.", type=int, default=120)
    parser.add_argument("--output_dir", "-o", help="Path of output dir to save csv to.", required=True)
    args = parser.parse_args()

    # Create output dir
    cur_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.output_dir = Path(args.output_dir, cur_date_time)
    # args.output_file = f"{args.output_dir}/{cur_date_time}.csv"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Configure depth stream
    pipeline = rs.pipeline()
    config = rs.config()
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    # set Inter Cam Sync Mode to 4 to wait for incoming Clock signal
    depth_sensor = device.first_depth_sensor()
    depth_sensor.set_option(rs.option.inter_cam_sync_mode, 4)

    config.enable_stream(rs.stream.depth, args.resolution[0], args.resolution[1], rs.format.z16, args.fps)

    # Start streaming
    pipeline.start(config)

    timeout_ms = args.timeout * 1000

    frame_count = 0
    last_timestamp = -1.0

    # video_writer = cv2.VideoWriter(args.output_file, cv2.VideoWriter_fourcc(*'MPEG'), args.fps, args.resolution)
    try:
        while True:
            # with open(args.output_file, "w+") as out_file:
                frames = pipeline.wait_for_frames(timeout_ms=timeout_ms)
                timeout_ms = 1000
                depth_frame = frames.get_depth_frame()

                if not depth_frame:
                    continue
                if frames.frame_number > frame_count + 1:
                    print(f"Old Frame: {frame_count} ", end='')
                    print(f"New Frame: {frames.frame_number}")

                frame_count = frames.frame_number
                current_timestamp = depth_frame.timestamp

                # Convert images to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())

                print_stats(frame_count, current_timestamp, last_timestamp)
                last_timestamp = current_timestamp

                # np.savetxt(out_file, depth_image.flatten())
                # video_writer.write(depth_image)

                np.save(f"{args.output_dir}/frame_{frame_count}", depth_image)

    except (RuntimeError, KeyboardInterrupt) as e:
        print("No frame received, clock signal seems to be turned off!")

    pipeline.stop()
    video_writer.release()
    print(f"Total Frames: {frame_count}")
    pass


def print_stats(frame_count: int, current_timestamp: float, last_timestamp: float):
    fps = 1000/(current_timestamp - last_timestamp)
    print(f"Detected Frames: {frame_count:05d}, fps: {fps:03.2f} \r", end='')


if __name__ == '__main__':
    main()
