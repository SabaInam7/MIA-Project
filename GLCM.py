
import numpy as np
import cv2
import os
import pandas as pd

def calculate_texture_features(matrix, is_stroke, d=1, angle=0):
    def create_glcm(matrix, d, angle):
        height, width = matrix.shape
        max_value = np.max(matrix)

        glcm = np.zeros((max_value + 1, max_value + 1))

        if angle == 0:
            dx, dy = 1, 0
        elif angle == 45:
            dx, dy = 1, -1
        elif angle == 90:
            dx, dy = 0, -1
        elif angle == 135:
            dx, dy = -1, -1

        for i in range(height):
            for j in range(width):
                value = matrix[i, j]
                ni, nj = i + dy * d, j + dx * d  # Shifted coordinates
                # Check if shifted coordinates are within the matrix bounds
                if 0 <= ni < height and 0 <= nj < width:
                    neighbor_value = matrix[ni, nj]
                    glcm[value, neighbor_value] += 1

        # Transpose the GLCM matrix
        glcm_T = np.transpose(glcm)

        # Add the GLCM and its transpose to construct the symmetric co-occurrence matrix
        symmetric_glcm = glcm + glcm_T

        # Normalize the symmetric GLCM
        symmetric_glcm /= np.sum(symmetric_glcm)

        return symmetric_glcm

    def calculate_energy(glcm):
        energy = np.sum(glcm ** 2)
        return energy

    def calculate_contrast(glcm):
        contrast = 0
        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                contrast += (i - j) ** 2 * glcm[i, j]
        return contrast

    def calculate_correlation(glcm):
        N = glcm.shape[0]
        i = np.arange(N)
        j = np.arange(N)

        # Calculate Px(i) and Py(j)
        Px = np.sum(glcm, axis=1)
        Py = np.sum(glcm, axis=0)

        # Calculate means (mu_x and mu_y)
        mu_x = np.sum(i * Px)
        mu_y = np.sum(j * Py)

        # Calculate standard deviations (sigma_x and sigma_y)
        sigma_x = np.sqrt(np.sum(Px * (i - mu_x) ** 2))
        sigma_y = np.sqrt(np.sum(Py * (j - mu_y) ** 2))

        # Correlation calculation
        correlation = 0
        for i in range(N):
            for j in range(N):
                correlation += (i - mu_x) * (j - mu_y) * glcm[i, j]

        correlation /= (sigma_x * sigma_y)
        return correlation

    def calculate_entropy(glcm, epsilon=1e-8):
        entropy = -np.sum(glcm * np.log(glcm + epsilon))
        return entropy

    def calculate_homogeneity(glcm):
        homogeneity = 0
        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                homogeneity += glcm[i, j] / (1 + abs(i - j))
        return homogeneity

    GLCM = create_glcm(matrix, d=d, angle=angle)

    energy = calculate_energy(GLCM)
    contrast = calculate_contrast(GLCM)
    correlation = calculate_correlation(GLCM)
    entropy = calculate_entropy(GLCM)
    homogeneity = calculate_homogeneity(GLCM)

    features = {
        "Is_Stroke": is_stroke,
        "Energy": energy,
        "Contrast": contrast,
        "Correlation": correlation,
        "Entropy": entropy,
        "Homogeneity": homogeneity,
    }
    return features

def process_images(directory, is_stroke):
    image_stats = []
    image_paths = [os.path.join(directory, filename) for filename in os.listdir(directory)]

    for path in image_paths:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Read the image in grayscale
        kernel = np.ones((7, 7), np.uint8)

        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
        cleaned_image = cv2.morphologyEx(blurred_image, cv2.MORPH_OPEN, kernel)

        stats = calculate_texture_features(cleaned_image, is_stroke, d=1, angle=0)
        stats['Image'] = os.path.basename(path)
        image_stats.append(stats)
        print(stats)
    return image_stats

# Directories for images
stroke_directory = 'G:\\MphilCS\\2ndSemester\\MIA\\Project\\DataSet\\Data\\Brain_Data_Organised\\Stroke'
normal_directory = 'G:\\MphilCS\\2ndSemester\\MIA\\Project\\DataSet\\Data\\Brain_Data_Organised\\Normal'

# Process images and collect statistics
stroke_stats = process_images(stroke_directory, is_stroke=1)
normal_stats = process_images(normal_directory, is_stroke=0)

# Combine statistics into a DataFrame
stroke_df = pd.DataFrame(stroke_stats)
stroke_df.to_csv('image_GLCM_Stroke.csv', index=False)

normal_df = pd.DataFrame(normal_stats)
normal_df.to_csv('image_GLCM_Normal.csv', index=False)

combined_df = pd.concat([stroke_df, normal_df], ignore_index=True)

# Save statistics to a CSV file
combined_df.to_csv('image_GLCM_combined.csv', index=False)
