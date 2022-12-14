import open3d as o3d
import numpy as np
p = o3d.io.read_point_cloud('out.xyz')
o3d.visualization.draw_geometries([p])