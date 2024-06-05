import torch
from torch import Tensor
from torch.linalg import norm
from scipy.spatial import procrustes
import numpy as np
from scipy.spatial.transform import Rotation as R

from loss.CompositionalLoss import parents_index

def compare_rotation(a: R, b: R) -> float:
    """
    Du Q. Huynh. 2019. Metrics for 3D Rotations: Comparison and Analysis.
    Eq 18.
    The metric gives values in the range [0, √2].
    """
    return min(np.linalg.norm(a.as_quat() - b.as_quat(), ord=2), 
               np.linalg.norm(a.as_quat() + b.as_quat(), ord=2))


def mpbrpe(outputs: Tensor, targets: Tensor, reduction='mean') -> Tensor:
    """
    Mean per keypoint and rotation positional error.
    Wie mpkpe nur für die Gelenkposition und Rotation Repräsentation
    """
    batch_size = outputs.shape[0]
    result = torch.zeros((batch_size, 2), requires_grad=False).cuda()
    number_of_bones = outputs.shape[1] / 6
    
    for i, (output, target) in enumerate(zip(outputs, targets)):
        output = output.split(6)
        target = target.split(6)
        for prediction_bone_rot, ground_truth_bone_rot in zip(output, target):
            bone    = prediction_bone_rot[  0:3]
            bone_gt = ground_truth_bone_rot[0:3]
            rot     = R.from_euler('xyz', prediction_bone_rot[  3:6].cpu().numpy(), degrees=True)
            rot_gt  = R.from_euler('xyz', ground_truth_bone_rot[3:6].cpu().numpy(), degrees=True)
            
            result[i][0] += norm(bone_gt - bone, ord=2)
            result[i][1] += compare_rotation(rot_gt, rot)
        result[i][0] /= number_of_bones  
        result[i][1] /= number_of_bones 
    
    if   reduction == 'mean': return result.mean(0)
    elif reduction == 'sum':  return result.sum(0)
    else: raise RuntimeError('invalid reduction!')
        

def mpkpe(outputs: Tensor, targets: Tensor, reduction='mean') -> Tensor:
    """
    Mean per keypoint positional error.
    """
    batch_size = outputs.shape[0]
    result = torch.zeros((batch_size), requires_grad=False).cuda()
    number_of_keypoints = outputs.shape[1] / 3
    
    for i, (output, target) in enumerate(zip(outputs, targets)):
        output = output.split(3)
        target = target.split(3)
        for keypoint, keypoint_gt in zip(output, target):
            result[i] += norm(keypoint_gt - keypoint, ord=2)
            
        result[i] /= number_of_keypoints
    
    if   reduction == 'mean': return result.mean(0)
    elif reduction == 'sum':  return result.sum(0)
    else: raise RuntimeError('invalid reduction!')

    
def pck(outputs: Tensor, targets: Tensor, threshold_cm=15) -> Tensor:
    batch_size = outputs.shape[0]
    number_of_joints = outputs.shape[1] / 3
    result = torch.zeros(batch_size, requires_grad=False).cuda()
    
    for i, (output, target) in enumerate(zip(outputs, targets)):
        output = output.split(3)
        target = target.split(3)
        for prediction_joint, ground_truth_joint in zip(output, target):
            result[i] += 1 if norm(ground_truth_joint - prediction_joint, ord=2) <= threshold_cm else 0
        result[i] /= number_of_joints
    
    return result.mean() * 100


def pmpkpe(outputs: Tensor, targets: Tensor, reduction='mean', has_rot_data=False) -> Tensor:
    batch_size = outputs.shape[0]
    result = torch.zeros(batch_size, requires_grad=False).cuda()
    
    for i, (output, target) in enumerate(zip(outputs, targets)):
        if has_rot_data:
            output = output.split(6)
            output = np.asarray([x.cpu().detach().numpy()[0:3] for x in output])
            target = target.split(6)
            target = np.asarray([x.cpu().detach().numpy()[0:3] for x in target])
            number_of_joints = outputs.shape[1] / 6
        else:
            output = output.split(3)
            output = np.asarray([x.cpu().detach().numpy() for x in output])
            target = target.split(3)
            target = np.asarray([x.cpu().detach().numpy() for x in target])
            number_of_joints = outputs.shape[1] / 3
        
        # save mean and norm to destandardize after procrustes
        target_mean = np.mean(target, 0)
        output_mean = np.mean(output, 0)
        target_norm = np.linalg.norm(target - target_mean)
        output_norm = np.linalg.norm(output - output_mean)
        
        standard_target, aligned_output, disparity = procrustes(target, output)
        
        # destandardize
        aligned_output *= output_norm
        aligned_output += output_mean
        standard_target *= target_norm
        standard_target += output_mean
                
        standard_target = torch.from_numpy(standard_target).cuda()
        aligned_output = torch.from_numpy(aligned_output).cuda()
                
        for prediction_joint, ground_truth_joint in zip(aligned_output, standard_target):
            
            result[i] += norm(ground_truth_joint - prediction_joint, ord=2)
        result[i] /= number_of_joints
    
    if   reduction == 'mean': return result.mean()
    elif reduction == 'sum':  return result.sum()
    else: raise RuntimeError('invalid reduction!')


def bone_lengths(inputs: Tensor, has_rot_data: bool, total_bone_lengths=None, net_output='joints'):
    if has_rot_data:
        number_bones = int(inputs.shape[1] / 6)
    else:
        number_bones = int(inputs.shape[1] / 3)
    
    if not total_bone_lengths:
        total_bone_lengths = tuple([[] for _ in range(number_bones)])
        
    for input in inputs:
        if has_rot_data:
            input = input.split(6)
            input = np.asarray([x.cpu().detach().numpy()[0:3] for x in input])
        else:
            input = input.split(3)
            input = np.asarray([x.cpu().detach().numpy() for x in input])
        
        if net_output != 'bones': 
            bones = [input[parents_index[i]] - x for i, x in enumerate(input)]
        else:
            bones = input

        # cut Hip, because its length is always 0
#         del bones[3]
        
        for bone, bone_length in zip(bones, total_bone_lengths):
            bone_length.append(np.linalg.norm(bone))
    return total_bone_lengths

"""
Distanz zwischen Histogrammen nach: Cha, S.-H. & Srihari, S. N. On measuring the distance between histograms. Pattern Recognition 35, 1355-1370 (2002).
"""
def distance_ordinal_histogram(hist1, hist2, bins):
    prefixsum = 0
    h_dist = 0
    for i in range(bins):
        prefixsum += hist1[i] - hist2[i]
        h_dist += abs(prefixsum)
    return h_dist
        
        
                         
        
    
    
    
    

