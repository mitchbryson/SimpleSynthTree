"""
ply.py: Functions for export points in PLY format
Author: Mitch Bryson
"""

# export_points_ply: exports a point cloud in ASCII PLY format
def export_points_ply(filepath, xyzl):
    
    f = open(filepath, "w");
    f.write('ply\n')
    f.write('format ascii 1.0\n')
    f.write('element vertex %d\n'%(xyzl.shape[0]))
    f.write('property float x\n')
    f.write('property float y\n')
    f.write('property float z\n')
    f.write('property uchar diffuse_red\n')
    f.write('property uchar diffuse_green\n')
    f.write('property uchar diffuse_blue\n')
    f.write('end_header\n')
    c = 0
    for i in range(xyzl.shape[0]):
        if xyzl[i,3] == 0: # foliage point (green)
            (R,G,B) = (0,255,0)
        else: # stem point (red)
            (R,G,B) = (255,0,0)
        f.write('%.4f %.4f %.4f %d %d %d\n'%(xyzl[i,0],xyzl[i,1],xyzl[i,2],int(R),int(G),int(B)))
    f.close()

