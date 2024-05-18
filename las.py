import numpy as np
import laspy

def export_points_las(filename, points):
    """
    Export points to a LAS file
    :param filename: output file name
    :param points: point cloud data
    """
    # Create a new LAS data
    las = laspy.create(point_format=2, file_version='1.2')

    # Set the point records
    las.X = points[:, 0]
    las.Y = points[:, 1]
    las.Z = points[:, 2]
    las.classification = points[:, 3].astype(np.uint8)  # Classification must be an unsigned 8-bit integer

    # Write the LAS file
    las.write(filename)