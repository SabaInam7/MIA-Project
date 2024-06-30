# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 23:38:13 2024

@author: aa
"""

import cv2
import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

def calculate_statistics(image, is_stroke):
    flattened = image.flatten()
    stats = {
        "Is_Stroke": is_stroke,
        'Mean': np.mean(flattened),
        'Median': np.median(flattened),
        'Variance': np.var(flattened),
        'Standard Deviation': np.std(flattened),
        'Skewness': skew(flattened, bias=False),
        'Kurtosis': kurtosis(flattened, bias=False),
        'Mean Absolute Deviation': np.mean(np.abs(flattened - np.mean(flattened))),
        'Median Absolute Deviation': np.median(np.abs(flattened - np.median(flattened))),
        'Local Contrast': np.mean([np.std(image[i:i+3, j:j+3]) for i in range(0, image.shape[0]-2) for j in range(0, image.shape[1]-2)]),
        'Local Probability': np.mean(image)
    }
    return stats

def process_images(directory, is_stroke):
    image_stats = []
    image_paths = [os.path.join(directory, filename) for filename in os.listdir(directory)]

    for path in image_paths:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Read the image in grayscale
        kernel = np.ones((7, 7), np.uint8)

        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
        cleaned_image = cv2.morphologyEx(blurred_image, cv2.MORPH_OPEN, kernel)

        stats = calculate_statistics(cleaned_image, is_stroke)
        print(stats)
        stats['Image'] = os.path.basename(path)
        image_stats.append(stats)
        
        normal_df = pd.DataFrame(image_stats)
        normal_df.to_csv('image_FOFS_Normal10.csv', index=False)
    return image_stats

# Directories for images
stroke_directory = 'G:\\MphilCS\\2ndSemester\\MIA\\Project\\DataSet\\Data\\Brain_Data_Organised\\stroke_1'
normal_directory = 'G:\\MphilCS\\2ndSemester\\MIA\\Project\\DataSet\\Data\\Brain_Data_Organised\\Normal'
normal_partial_directory='G:\\MphilCS\\2ndSemester\\MIA\\normal'
# stroke_stats = process_images(stroke_directory, is_stroke=1)
normal_stats = process_images(normal_partial_directory, is_stroke=0)


