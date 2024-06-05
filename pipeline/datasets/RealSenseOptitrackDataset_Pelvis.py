import numpy as np
import pandas as pd
from datasets.RealSenseOptitrackDataset import RealSenseOptitrackDataset

import os

class RealSenseOptitrackDataset_Pelvis(RealSenseOptitrackDataset):
    """Realsense Optitrack dataset only for pelvis detection"""
    
#     def __init__(self, csv_file, root_dir, limit_to_percentage=None, transform=None, drop_rotation=False):
#         super(self, csv_file, root_dir, limit_to_percentage, transform, drop_rotation)
            
    def _prepare_optitrack_data(self, csv_file):
        optitrack_data = super()._prepare_optitrack_data(csv_file)
        
        # only keep hip joint data
        optitrack_data = optitrack_data[['Hip']]
        
        return optitrack_data
