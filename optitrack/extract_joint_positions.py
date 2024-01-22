"""
Script to convert skeleton from .bvh file to .csv file.
Usage: Import .bvh into Blender session_* > Open Scripting Window in Blender > Open this Script in Blender > Run this script
"""
import bpy
import os

# export to blend file location
basedir = os.path.dirname(bpy.data.filepath)

#Name of file
session = 6
NAME = f"joints_session_{session}_from_bvh.csv"

export_path = f"{basedir}/{NAME}"

if not basedir:
    raise Exception("Blend file is not saved")


def get_line_string(frame_number, joint_names, joint_pos):
    result = f"{frame_number}"
    for joint in joint_names:
        pos = joint_pos[joint]
        result += f",{pos.x},{pos.y},{pos.z}"
    return result + "\n"

def get_header_string(joint_names):
    first_header = ""
    second_header = "Frame"
    for joint in joint_names:
        first_header += f",{joint},{joint},{joint}"
        second_header += ",X,Y,Z"
    return f"{first_header}\n{second_header}\n"

print(f"Exporting to file \"{NAME}\"... ", end='')

obj = bpy.data.objects[f"session_{session}"]
joint_names = sorted([x.name for x in obj.pose.bones])

sce = bpy.context.scene


with open(export_path, 'w+') as file:
    
    file.write(get_header_string(joint_names))
    
    for f in range(sce.frame_start, sce.frame_end+1):
        sce.frame_set(f)

        obj = bpy.data.objects[f"session_{session}"]
        joint_names = sorted([x.name for x in obj.pose.bones])

        joint_pos = {}

        for bone in obj.pose.bones:
            position = obj.matrix_world @ bone.matrix @ bone.location
            joint_pos[bone.name] = position
        
        file.write(get_line_string(f, joint_names, joint_pos))

print("Done!")
