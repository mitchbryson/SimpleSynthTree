# Simple Synth Tree
A python-based simulation model for generating synthetic tree point clouds.

Source code associated with "Using synthetic tree data in deep learning-based individual tree segmentation using LiDAR point clouds" by Mitch Bryson, Feiyu Wang and James Allworth.

## Usage
see "generate_example_tree.py":

    points = gen_simtree(Np=4096)
    export_points_ply('example001.ply', points)
