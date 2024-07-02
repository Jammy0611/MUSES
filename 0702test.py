import argparse
import json
import open3d as o3d
import numpy as np
import os

class LidarPointCloudViewer:
    def __init__(self, data_root):
        self.data_root = data_root
        self.load_lidar_json()

    def load_lidar_json(self):
        json_file_path = os.path.join(self.data_root, 'annotations_REC0006_frame_040219.json')
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"File '{json_file_path}' does not exist.")
        
        with open(json_file_path, 'r') as json_file:
            self.data = json.load(json_file)
        print("Loaded JSON structure:", self.data)  # 打印JSON文件的結構

    def display_point_cloud(self):
        # Extract points and annotations from JSON
        lidar_points = self.data.get("lidar_points", [])
        annotations = self.data.get("annotations", [])
        
        if not lidar_points:
            raise ValueError("No lidar points found in the JSON file.")
        if not annotations:
            raise ValueError("No annotations found in the JSON file.")

        # Convert points to numpy array
        points = np.array(lidar_points)
        print("Points loaded from JSON:", points)  # Debug output

        # Ensure the points are 3D
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Lidar points should be an Nx3 array representing 3D coordinates.")

        # Convert annotations to numpy array and map them to colors
        colors = np.zeros((len(annotations), 3))  # Default color is black
        label_to_color = {
            0: [0, 0, 0],       # Background
            1: [255, 0, 0],     # Red for label 1
            2: [0, 255, 0],     # Green for label 2
            3: [0, 0, 255],     # Blue for label 3
            # Add more mappings if needed
        }

        for i, annotation in enumerate(annotations):
            if annotation is not None:
                colors[i] = label_to_color.get(annotation, [255, 255, 255])  # Default color is white if label not found
        
        # Create Open3D point cloud object
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalize colors to [0, 1] range
        
        # Visualize the point cloud
        o3d.visualization.draw_geometries([point_cloud])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lidar Point Cloud Viewer with Open3D')
    parser.add_argument('--data_root', default='data/muses/lidarped', help='Root path for data containing lidar.json')
    args = parser.parse_args()

    viewer = LidarPointCloudViewer(args.data_root)
    viewer.display_point_cloud()
