import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
from PIL import Image

def radar_to_bev_image(radar_point_cloud, bev_width=800, bev_height=800, resolution=0.1):
    bev_image = np.zeros((bev_height, bev_width), dtype=np.uint8)
    x = radar_point_cloud[:, 0]
    y = radar_point_cloud[:, 1]
    intensity = radar_point_cloud[:, 2]
    bev_x = np.int32((x / resolution) + (bev_width / 2))
    bev_y = np.int32((y / resolution) + (bev_height / 2))
    bev_x = np.clip(bev_x, 0, bev_width - 1)
    bev_y = np.clip(bev_y, 0, bev_height - 1)
    bev_image[bev_y, bev_x] = np.clip(intensity * 255, 0, 255).astype(np.uint8)
    return bev_image

def visualize_bev(bev_image, title='BEV Image'):
    bev_image_pil = Image.fromarray(bev_image)
    bev_image_rotated = bev_image_pil.rotate(-90, expand=True)
    bev_image_rotated_np = np.array(bev_image_rotated)
    plt.figure(figsize=(10, 10))
    plt.imshow(bev_image_rotated_np, cmap='gray')
    plt.title(title)
    plt.show()

def load_raw_radar_data(radar_path):
    radar_range_azimuth = cv2.imread(radar_path, cv2.IMREAD_GRAYSCALE)
    return radar_range_azimuth

def radar_to_point_cloud(ra_image, max_range=100, fov=2*np.pi):
    height, width = ra_image.shape
    ranges = np.linspace(0, max_range, height)
    angles = np.linspace(-fov / 2, fov / 2, width)
    x = np.outer(ranges, np.cos(angles)).flatten()
    y = np.outer(ranges, np.sin(angles)).flatten()
    intensity = ra_image.flatten() / 255.0
    point_cloud = np.vstack((x, y, intensity)).T
    return point_cloud

def load_ground_truth_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def project_ground_truth_to_bev(ground_truth_data, bev_width=800, bev_height=800, resolution=0.1):
    bev_image = np.zeros((bev_height, bev_width), dtype=np.uint8)
    for point in ground_truth_data['lidar_points']:
        x, y, z = point
        bev_x = int((x / resolution) + (bev_width / 2))
        bev_y = int((y / resolution) + (bev_height / 2))
        if 0 <= bev_x < bev_width and 0 <= bev_y < bev_height:
            bev_image[bev_y, bev_x] = 255 
    if 'annotations' in ground_truth_data:
        for idx, annotation in enumerate(ground_truth_data['annotations']):
            if annotation is not None:
                x, y, z = ground_truth_data['lidar_points'][idx]
                bev_x = int((x / resolution) + (bev_width / 2))
                bev_y = int((y / resolution) + (bev_height / 2))
                if 0 <= bev_x < bev_width and 0 <= bev_y < bev_height:
                    bev_image[bev_y, bev_x] = 128 
    return bev_image

def compare_bev_images(bev_image_radar, bev_image_lidar):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    axs[0].imshow(bev_image_radar, cmap='gray')
    axs[0].set_title('Radar BEV Image')
    axs[1].imshow(bev_image_lidar, cmap='gray')
    axs[1].set_title('LiDAR BEV Image')
    plt.show()

radar_path = 'data/muses/radar/train/clear/day/REC0006_frame_040219_radar.png'  
ra_image = load_raw_radar_data(radar_path)
radar_point_cloud = radar_to_point_cloud(ra_image, max_range=100, fov=2*np.pi)
radar_bev_image = radar_to_bev_image(radar_point_cloud, bev_width=800, bev_height=800, resolution=0.1)

json_path = 'data/muses/lidarped/annotations_REC0006_frame_040219.json' 
ground_truth_data = load_ground_truth_json(json_path)
lidar_bev_image = project_ground_truth_to_bev(ground_truth_data)

compare_bev_images(radar_bev_image, lidar_bev_image)


