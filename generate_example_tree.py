"""
generate_example_tree.py: generates an example synthetic tree point cloud and exports to
a PLY file
Author: Mitch Bryson
"""

from treesim import gen_simtree
from ply import export_points_ply

# Generate a single synthetic tree example and export as a PLY file
points = gen_simtree(Np=4096)
export_points_ply('example001.ply', points)
