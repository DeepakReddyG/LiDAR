import laspy
import numpy as np
import open3d as o3d

print("Opening LAS file...")

with laspy.open("UPark_Merged_PS_NAD83_G18_USFT_las.las") as f:

    print("Total points:", f.header.point_count)

    # Read ONLY first chunk
    points = []

    for chunk in f.chunk_iterator(7500000):

        x = chunk.x
        y = chunk.y
        z = chunk.z

        pts = np.vstack((x, y, z)).T

        points.append(pts)

        break  # only first chunk

points = np.concatenate(points, axis=0)

print("Loaded points:", points.shape)

pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(points)

o3d.visualization.draw_geometries([pcd])
