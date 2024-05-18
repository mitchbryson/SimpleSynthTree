"""
generate_example_tree.py: generates an example synthetic tree point cloud and exports to
a PLY file
Author: Mitch Bryson
"""

from treesim import gen_simtree
from ply import export_points_ply
from las import export_points_las

# Generate a single synthetic tree example and export as a PLY file
points = gen_simtree(Np=4096)
export_points_ply('example001.ply', points)

# Generate a single synthetic tree example and export as a LAS file
points = gen_simtree(Np=4096)
export_points_las('example001.las', points)