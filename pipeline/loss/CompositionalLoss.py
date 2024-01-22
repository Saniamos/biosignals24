import torch
from torch import nn
from torch import Tensor
import networkx as nx
from itertools import combinations

parents = {
    'Ab': 'Hip',
    'Chest': 'Ab',
    'Head': 'Neck',
    'Hip': 'Hip',
    'LFArm': 'LUArm',
    'LFoot': 'LShin',
    'LHand': 'LFArm',
    'LShin': 'LThigh',
    'LShoulder': 'Chest',
    'LThigh': 'Hip',
    'LToe': 'LFoot',
    'LUArm': 'LShoulder',
    'Neck': 'Chest',
    'RFArm': 'RUArm',
    'RFoot': 'RShin',
    'RHand': 'RFArm',
    'RShin': 'RThigh',
    'RShoulder': 'Chest',
    'RThigh': 'Hip',
    'RToe': 'RFoot',
    'RUArm': 'RShoulder'
}


def get_joint_index(joint_name: str):
    return list(parents.keys()).index(joint_name)


parents_index = {get_joint_index(x): get_joint_index(parents[x]) for x in list(parents.keys())}

graph = nx.Graph()
graph_index = nx.Graph()
graph.add_edges_from([(x, parents[x]) for x in parents.keys()])
graph_index.add_edges_from([(x, parents_index[x]) for x in parents_index.keys()])
all_shortest_paths = dict(nx.all_pairs_shortest_path(graph))
all_shortest_paths_index = dict(nx.all_pairs_shortest_path(graph_index))


def sgn(path, m: int):
    return 1 if parents_index[path[m]] == path[m + 1] else -1


def get_ith_bone(input: Tensor, i: int) -> Tensor:
    batch_size = input.shape[0]
    i *= 3
    bone_index = torch.tensor([[i, i + 1, i + 2]], requires_grad=False).expand(batch_size, -1).cuda()
    return input.gather(1, bone_index)


def get_long_range_relative_position(input: Tensor, joint_u: int, joint_v: int) -> Tensor:
    path = all_shortest_paths_index[joint_u][joint_v]
    result = torch.zeros([input.shape[0], 3], requires_grad=False).cuda()
    for m in range(len(path) - 1):
        bone_m = get_ith_bone(input, path[m])
        result += sgn(path, m) * bone_m
    return result


def compose_output(outputs: Tensor) -> Tensor:
    results = torch.zeros(outputs.shape).cuda()
    for output, result in zip(outputs, results):
        joints = output.split(3)
        for i, joint in enumerate(joints):
            for bone_path_index in all_shortest_paths_index[3][i]:
                result[i*3:(i+1)*3] += joints[bone_path_index]
    return results
  
            

class CompositionalLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.P = list(combinations(parents_index.keys(), r=2))
        self.avg_reduction = reduction == 'mean'

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = torch.zeros(1, requires_grad=True).cuda()

        for u, v in self.P:
            delta_j = get_long_range_relative_position(input, int(u), int(v))
            delta_j_gt = get_ith_bone(target, u) - get_ith_bone(target, v)
                        
            delta_batch = delta_j - delta_j_gt
            
            delta_norms = torch.zeros(delta_batch.shape[0], requires_grad=True).cuda()
            for i, delta in enumerate(delta_batch):
                delta_norms[i] = torch.linalg.norm(delta, ord=1)
            
            if self.avg_reduction:
                loss += delta_norms.mean()
            else:
                loss += delta_norms.sum()
        return loss





