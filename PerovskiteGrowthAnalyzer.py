# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 13:38:48 2023

@author: tinajero
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def process_image(image_name):
    stored_area = 0
    
    # Load the image
    img = cv2.imread(image_name)

    if img is None:
        print(f"Unable to load the image: {image_name}")
        return

    # Define the points of the area of interest
    area_pts = np.array([[985, 145], [1200, 140], [1200, 225], [985, 230]])
    cv2.drawContours(img, [area_pts], -1, (255, 255, 255), 3)
    
    # Calculate the area of the polygon
    polygon_area = cv2.contourArea(area_pts)

    print(f"Polygon area: {polygon_area} square pixels")
    
    # Calculate the length of each side of the polygon
    side_lengths = []
    num_points = len(area_pts)
    
    for i in range(num_points):
        current_point = area_pts[i]
        next_point = area_pts[(i + 1) % num_points]  # Use the next point, wrapping back to the first point at the end
    
        distance = np.linalg.norm(next_point - current_point)  # Calculate the distance between the two points
        side_lengths.append(distance)
    
    # Show the area of the polygon
    print(f"Polygon area: {polygon_area} square pixels")
    
    # Show the lengths of each side of the polygon
    for i, length in enumerate(side_lengths):
        print(f"Length of side {i + 1}: {length} pixels")

    # Create a mask for the area of interest
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [area_pts], (255, 255, 255))

    # Apply the mask to the original image and HSV images
    masked_img = cv2.bitwise_and(img, mask)
    imghsv = cv2.cvtColor(masked_img, cv2.COLOR_BGR2HSV)

    # Define color ranges
    yellow_lower = np.array([0, 50, 50], np.uint8)
    yellow_upper = np.array([23.5, 255, 255], np.uint8)
    green_lower = np.array([36, 50, 50], np.uint8)
    green_upper = np.array([75, 255, 255], np.uint8)

    # Create masks using the color ranges and apply the area of interest mask
    mask1 = cv2.inRange(imghsv, yellow_lower, yellow_upper)
    mask2 = cv2.inRange(imghsv, green_lower, green_upper)

    # Find contours in the masks
    yellow_contours, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    green_contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cont = 0
    
    # Draw yellow contours
    for c in yellow_contours:
        area = cv2.contourArea(c)
        if area > 20:
            stored_area += area
            cont += 1
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(img, str(cont), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    print(stored_area) 
    # Folder where the processed images will be saved
    output_folder = "processed_images"

    # Ensure the folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create the output file name (same as the input name)
    output_file_name = os.path.join(output_folder, os.path.basename(image_name))

    # Save the processed image in the output folder
    cv2.imwrite(output_file_name, img)
    
    # cv2.imshow('Input Image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return stored_area  # Return the value of stored_area
    

# Get the current path of the Python file
current_path = os.path.dirname(os.path.abspath(__file__))

# Use glob to find all jpg files in the current folder
jpg_files = [file for file in os.listdir(current_path) if file.endswith('.jpg')]

# Store the number of files in another variable
num_files = len(jpg_files)

# Print the list of jpg file names and the number of files
print("JPG files in the folder:", jpg_files)
print("Number of files:", num_files)

stored_areas = []
study_times = []

for i in range(num_files):
    resulting_area = process_image(jpg_files[i])
    stored_areas.append(resulting_area)    

for e in range(num_files):
    time = e * 2
    study_times.append(time)
    
print(stored_areas)
print(study_times)

# Normalization of areas
# Multiply each element of the list by the constant using list comprehension
for i in range(len(stored_areas)):
    stored_areas[i] *= 42.51
    stored_areas[i] /= 16250

# Smoothing
# Convert lists to a pandas series to use the rolling function
series = pd.Series(stored_areas, index=study_times)

# Choose a smoothing window, for example, 3.
# This means the smoothing will be calculated with the current point and the previous and next points.
window = 3

# Apply the moving average. The rolling().mean() function does this for us.
smoothed_data = series.rolling(window=window).mean()

# Now, create the plot
plt.figure(figsize=(10, 5))
plt.plot(study_times, stored_areas, label='Original', color='blue', marker='o')
plt.plot(smoothed_data, label='Smoothed', color='red', linestyle='--')
plt.title('Detected Area vs. Time (Smoothed)')
plt.xlabel('Time (minutes)')
plt.ylabel('Detected Area (mm²)')
plt.legend()
plt.grid(True)

# Save the plot to a PDF file
plt.savefig('area_vs_time_plot.pdf')

# Save the list data to a text file
with open('area_vs_time_data.txt', 'w') as text_file:
    for time, area in zip(study_times, stored_areas):
        text_file.write(f'{time}, {area}\n')

# Display the plot
plt.show()
