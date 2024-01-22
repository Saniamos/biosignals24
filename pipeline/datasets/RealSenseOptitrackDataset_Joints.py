import pandas as pd
from datasets.RealSenseOptitrackDataset import RealSenseOptitrackDataset
import numpy as np


class RealSenseOptitrackDataset_Joints(RealSenseOptitrackDataset):
    """Realsense Optitrack dataset with Joint data"""
    
#     def __init__(self, csv_file, root_dir, limit_to_percentage=None, transform=None, drop_rotation=False):
#         super(self, csv_file, root_dir, limit_to_percentage, transform, drop_rotation)
            
    def _prepare_optitrack_data(self, csv_file):
        optitrack_data = pd.read_csv(csv_file, skiprows=0, header=[0, 1], index_col=0)
        optitrack_data = optitrack_data.sort_index(axis=1)
#         optitrack_data = optitrack_data * 100  # convert m to cm
#         self.joint_names = [x.replace("Jonah Full Body_", "") for x in optitrack_data.columns.levels[0].to_list()]
        self.joint_names = optitrack_data.columns.levels[0].to_list()
        
        levels = pd.MultiIndex.from_arrays(
            [
                np.array([[x]*3 for x in self.joint_names]).flatten(), 
                np.full(len(self.joint_names) * 3, 'Position'),
                np.array([['X', 'Y', 'Z'] for _ in self.joint_names]).flatten()
            ])
        optitrack_data.columns = levels
        
        # Drop buggy hips
        optitrack_data = optitrack_data.drop(columns="Hips")
        optitrack_data.columns = optitrack_data.columns.remove_unused_levels()
        
        # delete duplicate entries
#         optitrack_data = optitrack_data.loc[0:21361]
        
        return optitrack_data
