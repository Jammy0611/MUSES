import numpy as np
import json
import os
import matplotlib.pyplot as plt

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

def visualize_bev(bev_image):
    plt.figure(figsize=(10, 10))
    plt.imshow(bev_image, cmap='gray')
    plt.title('BEV Image')
    plt.show()

def save_bev_image(bev_image, output_path):
    plt.imsave(output_path, bev_image, cmap='gray')

def process_all_ground_truth_jsons(json_directory, output_directory, bev_width=800, bev_height=800, resolution=0.1):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    json_files = [f for f in os.listdir(json_directory) if f.endswith('.json')]
    
    for json_file in json_files:
        json_path = os.path.join(json_directory, json_file)
        ground_truth_data = load_ground_truth_json(json_path)
        bev_image = project_ground_truth_to_bev(ground_truth_data, bev_width, bev_height, resolution)
        
        output_path = os.path.join(output_directory, json_file.replace('.json', '.png'))
        save_bev_image(bev_image, output_path)
        print(f"Saved BEV image for {json_file} as {output_path}")


json_directory = 'data/muses/lidarped'
output_directory = 'data/muses/gt_lidarbev_images'
process_all_ground_truth_jsons(json_directory, output_directory)
