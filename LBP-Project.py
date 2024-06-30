# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 12:27:24 2024

@author: aa
"""

import os
import cv2
import numpy as np
import pandas as pd

def calculate_lbp_features(image, radius=3, num_points=16):
    def get_pixel(img, center, x, y):
        try:
            if img[x][y] >= center:
                return 1 
            else: 
                return 0
        except IndexError:
            return 0

    def generate_neighbors(radius, num_points):
        neighbors = []
        for i in range(num_points):
            theta = 2 * np.pi * i / num_points
            x = int(radius * np.cos(theta))
            y = int(radius * np.sin(theta))
            neighbors.append((x, y))
        return neighbors

    def lbp_calculated_pixel(img, x, y, radius, num_points):
        center = img[x][y]
        neighbors = generate_neighbors(radius, num_points)

        lbp_value = 0

        for i, (dx, dy) in enumerate(neighbors):
            nx, ny = x + dx, y + dy
            lbp_value += get_pixel(img, center, nx, ny) * (2 ** i)

        return lbp_value

    height, width = image.shape
    img_lbp = np.zeros((height, width), np.uint8)

    for i in range(height):
        for j in range(width):
            img_lbp[i, j] = lbp_calculated_pixel(image, i, j, radius, num_points)

    feature_vector = img_lbp.flatten()
    return feature_vector

def process_images(directory, is_stroke, radius=1, num_points=8):
    image_stats = []
    image_paths = [os.path.join(directory, filename) for filename in os.listdir(directory)]

    for path in image_paths:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Read the image in grayscale
        kernel = np.ones((7, 7), np.uint8)

        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
        cleaned_image = cv2.morphologyEx(blurred_image, cv2.MORPH_OPEN, kernel)

        # Calculate LBP features
        lbp_features = calculate_lbp_features(cleaned_image, radius=radius, num_points=num_points)

        # Store statistics
        stats = {'Image': os.path.basename(path), 'Is_Stroke': is_stroke, 'LBP_Features': lbp_features}
        image_stats.append(stats)
        stroke_df = pd.DataFrame(image_stats)
        stroke_df.to_csv('lbp_features_Normalm.csv', index=False)

        print(stats)

    return image_stats

# Directories for images
stroke_directory = 'G:\\MphilCS\\2ndSemester\\MIA\\Project\\DataSet\\Data\\Brain_Data_Organised\\Stroke'
normal_directory = 'E:\\2ndSemester\\MIA\Project\\DataSet\Data\\Brain_Data_Organised\\Normal'

# Process images and collect statistics
# stroke_stats = process_images(stroke_directory, is_stroke=1)
normal_stats = process_images(normal_directory, is_stroke=0)


print("LBP feature extraction and saving complete.")
