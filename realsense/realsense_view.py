# License: Apache 2.0. See LICENSE file in root directory.
# Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

import pyrealsense2 as rs
import numpy as np
import cv2
from argparse import ArgumentParser

# Video settings
# Font settings
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (0, 20)
fontScale = 0.6
fontColor = (255, 255, 255)


def main():
    parser = ArgumentParser()
    parser.add_argument("--fps", "-f",
                        help="Frames per Second setting of camera. Needs to be at least double of clock signal.",
                        type=int, default=60)
    parser.add_argument("--resolution", "-r", help="Length of 'on' portion of signal in micro seconds.", type=tuple,
                        default=(640, 480))
    parser.add_argument("--no-sync", '--s', help="Do not wait for sync signal for image capture.", action='store_true')
    args = parser.parse_args()
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    # set Inter Cam Sync Mode to 4 to wait for incoming PWM signal
    depth_sensor = device.first_depth_sensor()
    if args.no_sync:
        sync_mode = 0
    else:
        sync_mode = 4
    depth_sensor.set_option(rs.option.inter_cam_sync_mode, sync_mode)

    config.enable_stream(rs.stream.depth, args.resolution[0], args.resolution[1], rs.format.z16, args.fps)
    config.enable_stream(rs.stream.infrared, 1, args.resolution[0], args.resolution[1], rs.format.y8, args.fps)

    # Start streaming
    pipeline.start(config)

    # Frame count
    frame = 0

    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

    while True:
        frames = pipeline.wait_for_frames(timeout_ms=120 * 1000)
        depth_frame = frames.get_depth_frame()
        infrared_frame = frames.get_infrared_frame(1)
        if not depth_frame or not infrared_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        infrared_image = np.asanyarray(infrared_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        infrared_colormap = cv2.cvtColor(infrared_image, cv2.COLOR_GRAY2BGR)

        images = np.hstack((infrared_colormap, depth_colormap))

        # write frame count to image
        cv2.putText(images, f"Frame: {frame}", bottomLeftCornerOfText, font, fontScale, fontColor)
        frame += 1

        # Show images
        cv2.imshow('RealSense', images)
        if cv2.waitKey(1) == 27:
            break

    # Stop streaming
    cv2.destroyAllWindows()
    pipeline.stop()
    print(f"Total Frames: {frame}")


if __name__ == '__main__':
    main()
