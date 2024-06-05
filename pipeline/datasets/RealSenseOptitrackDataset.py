import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data.dataset import Dataset

import os

marker_set = {'Ab',
              'Chest',
              'Head',
              'Hip',
              'LFArm',
              'LFoot',
              'LHand',
              'LShin',
              'LShoulder',
              'LThigh',
              'LToe',
              'LUArm',
              'Neck',
              'RFArm',
              'RFoot',
              'RHand',
              'RShin',
              'RShoulder',
              'RThigh',
              'RToe',
              'RUArm'}

class RealSenseOptitrackDataset(Dataset):
    """Realsense Optitrack dataset."""
    
    def __init__(self, csv_file, root_dir, limit_to_percentage=None,
                 transform=None, drop_rotation=False, normalize=False):
        """
        Args:
            csv_file (string): Path to the csv file with optitrack data as annotations.
            root_dir (string): Directory with all the npy arrays containing realsense data.
            limit_to_percentage (float): Only load given percentage of recorded data (ex. for test set)
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.normalize = normalize
        self.nan_percentage = -1

        self.optitrack_data = self._load_and_init_optitrack_data(csv_file, f"{root_dir}/../skipped_frames.txt")
        if limit_to_percentage:
            self.optitrack_data = self.optitrack_data[:int(self.__len__()*(limit_to_percentage / 100))]
            
        if drop_rotation:
            column_list = self.optitrack_data.columns.to_list()
            column_list = [x for x in column_list if 'Rotation' in x]
            self.optitrack_data = self.optitrack_data.drop(columns=column_list)

    def _prepare_optitrack_data(self, csv_file):
        optitrack_data = pd.read_csv(csv_file, skiprows=2, header=[0, 1, 3, 4], index_col=0)
        optitrack_data = optitrack_data.sort_index(axis=1)
        optitrack_data = optitrack_data.rename(
            columns={'Unnamed: 1_level_0': "Time", 'Unnamed: 1_level_1': "Time", 'Unnamed: 1_level_2': "Time"})
        optitrack_data = optitrack_data.drop('Marker', axis=1)
        optitrack_data = optitrack_data.droplevel(level=0, axis=1)
        optitrack_data = optitrack_data.drop('Time', axis=1)

        # rename Bones
        self.bone_names = [x.replace("Jonah Full Body:", "") for x in optitrack_data.columns.levels[0].to_list()]
        optitrack_data.columns = optitrack_data.columns.set_levels(self.bone_names, level=0)

        # calc percentage of NaN values
        isna = optitrack_data.isna().sum().sum()
        number_of_values = optitrack_data.shape[0] * optitrack_data.shape[1]
        self.nan_percentage = isna / number_of_values * 100
        
        # drop and fill empty values
        optitrack_data = optitrack_data.dropna(axis=1, how='all')
        optitrack_data = optitrack_data.interpolate(axis=0, limit_direction='both')

        return optitrack_data

    def _load_and_init_optitrack_data(self, csv_file, skipped_frames_file):
        
        # Load optitrack data from csv and prepare it
        optitrack_data = self._prepare_optitrack_data(csv_file)

        # remove frames dropped by realsense recording
        skipped_frames = np.loadtxt(skipped_frames_file)
        optitrack_data = optitrack_data[~optitrack_data.index.isin(skipped_frames)]

        # normalize data
        if self.normalize:
            self.optitrack_mean = optitrack_data.mean()
            self.optitrack_std = optitrack_data.std()
            optitrack_data = (optitrack_data - self.optitrack_mean) / self.optitrack_std
            self.optitrack_mean_cuda = torch.from_numpy(self.optitrack_mean.to_numpy()).cuda()
            self.optitrack_std_cuda = torch.from_numpy(self.optitrack_std.to_numpy()).cuda()

        # Adjust number of samples so there are the same amount of realsense and optitrack entries.
        realsense_length = len([name for name in os.listdir(self.root_dir) if os.path.isfile(os.path.join(self.root_dir, name))])
        if len(optitrack_data) > realsense_length:
            optitrack_data = optitrack_data[:realsense_length]
        return optitrack_data

    
    def _load_image(self, frame_number):
        img_name = os.path.join(self.root_dir, f"frame_{frame_number:05}.npy")
        return np.load(img_name)

    
    def __len__(self):
        return len(self.optitrack_data)

    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        optitrack_entry = self.optitrack_data.iloc[idx]
        optitrack_point = torch.from_numpy(optitrack_entry.to_numpy())

        realsense_image = self._load_image(optitrack_entry.name)
        realsense_image = torch.from_numpy(realsense_image.astype("float"))
            
        # add channel dimension
        realsense_image = realsense_image[None, :]
        
        sample = {'realsense': realsense_image, 'optitrack': optitrack_point}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def denormalize(self, df: pd.DataFrame):
        return df * self.optitrack_std + self.optitrack_mean
    
    def denormalize_cuda(self, outputs: Tensor):
        for i, output in enumerate(outputs):
            outputs[i] = output * self.optitrack_std_cuda + self.optitrack_mean_cuda
        return outputs

               