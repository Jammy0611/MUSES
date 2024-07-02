import os
import json
import numpy as np
import cv2
from processing.utils import load_muses_calibration_data, filter_and_project_pcd_to_image, motion_compensate_pcd, create_image_from_point_cloud, enlarge_points_in_image

def load_lidar_data(lidar_path):
    loaded_pcd = np.fromfile(lidar_path, dtype=np.float64)
    return loaded_pcd.reshape((-1, 6))

def load_points_in_image_lidar(lidar_path, calib_data, scene_meta_data=None, motion_compensation=False, muses_root=None, target_shape=(1920, 1080)):
    assert os.path.exists(lidar_path), f"Lidar data {lidar_path} does not exist"
    pcd_points = load_lidar_data(lidar_path)
    if motion_compensation:
        assert scene_meta_data is not None, "Scene meta data is required for motion compensation"
        assert muses_root is not None, "MUSES root directory is required for motion compensation"
        lidar2gnss = calib_data["extrinsics"]["lidar2gnss"]
        pcd_points = motion_compensate_pcd(muses_root, scene_meta_data, pcd_points, lidar2gnss, ts_channel_num=5)
    K_rgb = calib_data["intrinsics"]["rgb"]["K"]
    lidar2rgb = calib_data["extrinsics"]["lidar2rgb"]
    uv_img_cords_filtered, pcd_points_filtered = filter_and_project_pcd_to_image(pcd_points, lidar2rgb, K_rgb, target_shape)
    return uv_img_cords_filtered, pcd_points_filtered, pcd_points

def load_lidar_projection(lidar_path, calib_data, scene_meta_dict=None, motion_compensation=False, muses_root=None, target_shape=(1920, 1080), enlarge_lidar_points=False):
    uv_img_cords_filtered, pcd_points_filtered, pcd_points = load_points_in_image_lidar(lidar_path, calib_data, scene_meta_data=scene_meta_dict, motion_compensation=motion_compensation, muses_root=muses_root, target_shape=target_shape)
    image = create_image_from_point_cloud(uv_img_cords_filtered, pcd_points_filtered, target_shape)
    if enlarge_lidar_points:
        image = enlarge_points_in_image(image, kernel_shape=(2, 2))
    return image, uv_img_cords_filtered, pcd_points

def main(args):
    calib_data_path = os.path.join(args.muses_root, 'calib.json')
    if not os.path.exists(calib_data_path):
        raise FileNotFoundError(f"File '{calib_data_path}' does not exist.")
    
    calib_data = load_muses_calibration_data(args.muses_root)
    
    meta_file = os.path.join(args.muses_root, 'meta.json')
    output_folder = os.path.join(args.muses_root, 'lidarped')
    
    if not os.path.exists(meta_file):
        raise FileNotFoundError(f"File '{meta_file}' does not exist.")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    with open(meta_file, 'r') as f:
        try:
            meta_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from '{meta_file}': {e}")
    
    for entry_name, metadata in meta_data.items():
        if isinstance(metadata, dict) and 'path_to_lidar' in metadata:
            lidar_filename = metadata['path_to_lidar']
            split = metadata.get('split', 'train')
        else:
            print(f"Expected lidar_filename to be a string, got {type(metadata)} instead.")
            continue
        
        lidar_path = os.path.join(args.muses_root, lidar_filename)
        if not os.path.exists(lidar_path):
            print(f"LiDAR file '{lidar_path}' not found, skipping.")
            continue
        
        if split in ['train', 'val']:
            gt_semantic_folder = os.path.join(args.muses_root, 'gt_semantic', split, metadata['weather'], metadata['time_of_day'])
            gt_image_name = entry_name + '_gt_labelIds.png'
            gt_image_path = os.path.join(gt_semantic_folder, gt_image_name)
            
            if not os.path.exists(gt_image_path):
                print(f"GT image '{gt_image_path}' not found, skipping.")
                continue
        else:
            print(f"Skipping GT semantic images for {entry_name} as 'split' is '{split}'.")
            continue 
        
        projected_image, lidar_points_2d, lidar_points_3d = load_lidar_projection(
            lidar_path, 
            calib_data, 
            motion_compensation=args.motion_compensation, 
            target_shape=(1920, 1080), 
            enlarge_lidar_points=True
        )
        
        gt_image = cv2.imread(gt_image_path, cv2.IMREAD_GRAYSCALE)
        if gt_image is None:
            print(f"Failed to load GT image '{gt_image_path}', skipping.")
            continue
        
        if projected_image.shape[:2] != gt_image.shape[:2]:
            print(f"GT image '{gt_image_path}' and projected image shape do not match, skipping.")
            continue
        
        lidar_annotations = []
        for point in lidar_points_2d:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < gt_image.shape[1] and 0 <= y < gt_image.shape[0]:
                label = int(gt_image[y, x])
                lidar_annotations.append(label)
            else:
                lidar_annotations.append(None)
        
        output_data = {
            'lidar_points': lidar_points_3d[:, :3].tolist(),  # Extract only the XYZ coordinates
            'annotations': lidar_annotations
        }
        output_filename = os.path.join(output_folder, f'annotations_{entry_name}.json')
        with open(output_filename, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"Saved annotations for {entry_name} as {output_filename}.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='LiDAR projection tool')
    parser.add_argument('--muses_root', default='data/muses/', help='Root path for data')
    parser.add_argument('--motion_compensation', action='store_true', help='Enable motion compensation (default: False)')
    args = parser.parse_args()
    main(args)
