import pandas as pd
from datasets.RealSenseOptitrackDataset import RealSenseOptitrackDataset_Joints
import loss.CompositionalLoss


class RealSenseOptitrackDataset_Relative_Bones(RealSenseOptitrackDataset_Joints):
    """Realsense Optitrack dataset with relative Bone data"""

    def _prepare_optitrack_data(self, csv_file):
        optitrack_data = super()._prepare_optitrack_data(csv_file)

        # convert from global joints to relative bones
        for joint in parents.keys():
            optitrack_data[joint] = optitrack_data[parents[joint]] - optitrack_data[joint]
        return optitrack_data
