"""
This script calculates the stride length from a motive export
"""
import argparse

import numpy as np
import pandas as pd
from scipy.signal import lfilter, find_peaks
from scipy.spatial.distance import cdist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csvPath", help="Path to csv file to use as source")
    args = parser.parse_args()

    df = pd.read_csv(args.csvPath, skiprows=2, header=[0, 1, 3, 4], index_col=0)
    df = df.rename(columns={'Unnamed: 1_level_0': "Time", 'Unnamed: 1_level_1': "Time", 'Unnamed: 1_level_2': "Time"})

    df = df.dropna(axis=1, how='all')

    time = df['Time', 'Time']
    l_foot = df['Bone', 'Jonah_Masterarbeit_Lower_Body:LFoot']
    r_foot = df['Bone', 'Jonah_Masterarbeit_Lower_Body:RFoot']

    l_foot = pd.concat([time, l_foot], axis=1)
    r_foot = pd.concat([time, r_foot], axis=1)

    matrix_l_foot = calculate_stride_length_matrix_per_foot(l_foot)
    matrix_r_foot = calculate_stride_length_matrix_per_foot(r_foot)
    print("Left foot:")
    print(matrix_l_foot)

    print("right foot:")
    print(matrix_r_foot)

    print("Done")


def calculate_stride_length_matrix_per_foot(foot: pd.DataFrame, start_index=2):
    end_index = len(foot)

    add_velocities(foot)
    foot["Velocity"] = foot["Velocity"].interpolate()

    foot_filtered = iir_filter(foot[("Velocity", "all")][start_index:end_index])
    valley_environments = calculate_valley_environment(foot_filtered)

    centroids = []
    for valley in valley_environments:
        centroids.append(list(foot.iloc[valley]["Position"].mean()))

    return cdist(centroids, centroids)


def add_velocities(df: pd.DataFrame):
    for index, _ in df.iterrows():
        if index == 0: continue
        px_i = df.loc[index, ("Position", "X")]
        py_i = df.loc[index, ("Position", "Y")]
        pz_i = df.loc[index, ("Position", "Z")]
        t_i = df.loc[index, ("Time", "Time (Seconds)")]

        px_i1 = df.loc[index - 1, ("Position", "X")]
        py_i1 = df.loc[index - 1, ("Position", "Y")]
        pz_i1 = df.loc[index - 1, ("Position", "Z")]
        t_i1 = df.loc[index - 1, ("Time", "Time (Seconds)")]

        delta_t = t_i - t_i1
        delta_x = (px_i - px_i1)
        delta_y = (py_i - py_i1)
        delta_z = (pz_i - pz_i1)
        df.at[index, ("Velocity", "X")] = delta_x / delta_t
        df.at[index, ("Velocity", "Y")] = delta_y / delta_t
        df.at[index, ("Velocity", "Z")] = delta_z / delta_t
        df.at[index, ("Velocity", "all")] = np.sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2) / delta_t


# IIR Filter
def iir_filter(data, n=20, a=1):
    b = [1.0 / n] * n
    return lfilter(b, a, data)


def calculate_valley_environment(data: np.ndarray, height=100, threshold=None, distance=20, prominence=100):
    valleys = find_valleys(data, height, threshold, distance, prominence)
    lowest_sum_environments = calculate_lowest_sum_environment(valleys, data)
    return lowest_sum_environments


def find_valleys(data: np.ndarray, height, threshold, distance, prominence):
    amax = np.amax(data)
    flipped_data = np.apply_along_axis(lambda x: x * -1 + amax, 0, data)
    valleys = find_peaks(flipped_data, height, threshold, distance, prominence)
    return valleys[0]


def calculate_avg_indices_distance(data: np.ndarray):
    avg_distance = 0
    length = len(data)
    for i in range(length):
        if i == length - 1: break
        avg_distance += data[i + 1] - data[i]
    avg_distance /= length
    return avg_distance


def calculate_lowest_sum_environment(valleys: np.ndarray, data: np.ndarray, width_factor=0.4):
    """

    :param valleys: list with indices of all valleys
    :param data: data to compute lowest sum environment for
    :param width_factor: Between 0 and 1. Gibt an wie breit jede Umgebung ist. 0 entspricht nur einem Punkt, 1 entspricht der Breite des durchschnittlichen Abstands zwischen 2 Punkten.
    :return: 2d array containing all indices for each valley enviromnent
    """
    average_distance = calculate_avg_indices_distance(valleys)

    width = int(width_factor * average_distance)

    lowest_sum_indices = []
    for valley in np.nditer(valleys):
        valley_lowest_sum = np.Infinity
        valley_lowest_sum_indices = []
        for starting_index in range(valley - width, valley):
            current_sum = np.sum(data[starting_index:starting_index + width])
            if current_sum < valley_lowest_sum:
                valley_lowest_sum = current_sum
                valley_lowest_sum_indices = list(range(starting_index, starting_index + width))
        lowest_sum_indices.append(valley_lowest_sum_indices)

    return np.asarray(lowest_sum_indices)


if __name__ == '__main__':
    main()
