import laspy
import numpy as np
import open3d as o3d
import argparse

def main(las_file, output_type):
    print("Opening LAS file...")

    with laspy.open(las_file) as f:
        print(f"Total points: {f.header.point_count}")

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

    if output_type == "depth":
        project_depth(pcd)
    elif output_type == "intensity":
        project_intensity(pcd)
    else:
        print("Invalid output type. Please choose 'depth' or 'intensity'.")
        exit(1)

def project_depth(pcd):
    # Code to generate depth projection
    pass

def project_intensity(pcd):
    # Code to generate intensity projection
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process LiDAR data and generate projections.")
    parser.add_argument("las_file", type=str, help="Path to the LAS file")
    parser.add_argument("--output-type", type=str, choices=["depth", "intensity"], default="depth",
                        help="Type of projection to generate (default: depth)")
    
    args = parser.parse_args()
    main(args.las_file, args.output_type)
