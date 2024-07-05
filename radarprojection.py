import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
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

def visualize_bev(bev_image):
    bev_image_pil = Image.fromarray(bev_image)
    bev_image_rotated = bev_image_pil.rotate(-90, expand=True)
    bev_image_rotated_np = np.array(bev_image_rotated)
    plt.figure(figsize=(10, 10))
    plt.imshow(bev_image_rotated_np, cmap='gray')
    plt.title('BEV Image')
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

def process_all_images(input_dir, output_dir, bev_width=800, bev_height=800, resolution=0.1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".png"):
                radar_path = os.path.join(root, file)
                ra_image = load_raw_radar_data(radar_path)
                radar_point_cloud = radar_to_point_cloud(ra_image, max_range=100, fov=2*np.pi)
                bev_image = radar_to_bev_image(radar_point_cloud, bev_width, bev_height, resolution)
                output_path = os.path.join(output_dir, file)
                cv2.imwrite(output_path, bev_image)
                print(f"Processed and saved BEV image: {output_path}")

if __name__ == "__main__":
    input_directory = "data/muses/radar/train/"
    output_directory = "data/muses/radarped/"
    process_all_images(input_directory, output_directory)
