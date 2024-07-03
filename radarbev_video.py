import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from matplotlib.animation import FuncAnimation

def radar_to_bev_image(radar_point_cloud, bev_width=800, bev_height=800, resolution=0.1):
    """
    Convert radar point cloud to a BEV (Bird's Eye View) image.

    Args:
        radar_point_cloud (np.array): The radar point cloud data.
        bev_width (int): The width of the BEV image (default is 800).
        bev_height (int): The height of the BEV image (default is 800).
        resolution (float): The resolution of the BEV image in meters per pixel (default is 0.1).

    Returns:
        np.array: The BEV image.
    """
    # Initialize BEV image
    bev_image = np.zeros((bev_height, bev_width), dtype=np.uint8)

    # Extract x, y, and intensity from the point cloud
    x = radar_point_cloud[:, 0]
    y = radar_point_cloud[:, 1]
    intensity = radar_point_cloud[:, 2]

    # Calculate BEV coordinates
    bev_x = np.int32((x / resolution) + (bev_width / 2))
    bev_y = np.int32((y / resolution) + (bev_height / 2))

    # Clip the coordinates to the image size
    bev_x = np.clip(bev_x, 0, bev_width - 1)
    bev_y = np.clip(bev_y, 0, bev_height - 1)

    # Fill the BEV image with intensity values
    bev_image[bev_y, bev_x] = np.clip(intensity * 255, 0, 255).astype(np.uint8)

    return bev_image

def load_raw_radar_data(radar_path):
    """
    Load radar data from the specified path.

    Args:
        radar_path (str): The path to the radar data.

    Returns:
        np.array: The raw azimuth range radar data.
    """
    radar_range_azimuth = cv2.imread(radar_path, cv2.IMREAD_GRAYSCALE)
    return radar_range_azimuth

def radar_to_point_cloud(ra_image, max_range=100, fov=2*np.pi):
    """
    Convert radar RA image to point cloud.

    Args:
        ra_image (np.array): The radar RA image.
        max_range (float): The maximum range for the radar (default is 100).
        fov (float): The field of view of the radar in radians (default is 2π).

    Returns:
        np.array: The radar point cloud.
    """
    height, width = ra_image.shape
    ranges = np.linspace(0, max_range, height)
    angles = np.linspace(-fov / 2, fov / 2, width)

    x = np.outer(ranges, np.cos(angles)).flatten()
    y = np.outer(ranges, np.sin(angles)).flatten()
    intensity = ra_image.flatten() / 255.0  # Normalize the intensity

    point_cloud = np.vstack((x, y, intensity)).T

    return point_cloud

def load_and_process_images(radar_directory):
    """
    Load and process radar images from a specified directory.

    Args:
        radar_directory (str): The path to the directory containing radar images.

    Returns:
        list: A list of processed BEV images.
    """
    bev_images = []
    for filename in os.listdir(radar_directory):
        if filename.endswith(".png"):  # Check if the file is a .png image
            radar_path = os.path.join(radar_directory, filename)
            ra_image = load_raw_radar_data(radar_path)
            radar_point_cloud = radar_to_point_cloud(ra_image, max_range=100, fov=2*np.pi)
            bev_image = radar_to_bev_image(radar_point_cloud, bev_width=800, bev_height=800, resolution=0.1)
            bev_images.append(bev_image)
    return bev_images

def animate_bev_images(bev_images):
    """
    Animate a list of BEV images using Matplotlib.

    Args:
        bev_images (list): A list of BEV images.

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    def update(frame):
        ax.clear()
        ax.imshow(frame, cmap='gray')
        ax.set_title('BEV Image')

    ani = FuncAnimation(fig, update, frames=bev_images, interval=200)
    plt.show()

# 示例使用方法
if __name__ == "__main__":
    radar_directory = "data/muses/radar/train/clear/day/"
    bev_images = load_and_process_images(radar_directory)
    animate_bev_images(bev_images)
