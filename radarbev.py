import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

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

def visualize_bev(bev_image):
    """
    Visualize the BEV image using Matplotlib.

    Args:
        bev_image (np.array): The BEV image.

    Returns:
        None
    """
    # Convert numpy array to PIL image for rotation
    bev_image_pil = Image.fromarray(bev_image)
    # Rotate the image 90 degrees clockwise
    bev_image_rotated = bev_image_pil.rotate(-90, expand=True)
    # Convert back to numpy array
    bev_image_rotated_np = np.array(bev_image_rotated)

    plt.figure(figsize=(10, 10))
    plt.imshow(bev_image_rotated_np, cmap='gray')
    plt.title('BEV Image')
    plt.show()

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

# 示例使用方法
if __name__ == "__main__":
    radar_path = "data/muses/radar/train/clear/day/REC0006_frame_040219_radar.png"
    ra_image = load_raw_radar_data(radar_path)
    radar_point_cloud = radar_to_point_cloud(ra_image, max_range=100, fov=2*np.pi)
    bev_image = radar_to_bev_image(radar_point_cloud, bev_width=800, bev_height=800, resolution=0.1)
    visualize_bev(bev_image)
