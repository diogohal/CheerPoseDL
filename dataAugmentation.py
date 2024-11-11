import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
import numpy as np
import cv2

def flipImage(input_dir, output_dir): 
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Loop over each image in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Check for image file extensions
            # Load the image
            image_path = os.path.join(input_dir, filename)
            image = load_img(image_path)
            image_array = img_to_array(image)  # Convert image to array format

            # Flip the image horizontally
            flipped_image_array = np.fliplr(image_array)  # Flip the image array left-to-right

            # Save the flipped image
            save_path = os.path.join(output_dir, f'flipped_{filename}')
            save_img(save_path, flipped_image_array)
            print(f'Saved flipped image: {save_path}')

def random_perspective_points(cols, rows, max_offset=0.2):
    """Generate randomized destination points for perspective transformation."""
    # Define destination points with small random variations
    dst_points = np.float32([
        [cols * np.random.uniform(0, max_offset), rows * np.random.uniform(0, max_offset)],             # Top-left
        [cols * (1 - np.random.uniform(0, max_offset)), rows * np.random.uniform(0, max_offset)],       # Top-right
        [cols * np.random.uniform(0, max_offset), rows * (1 - np.random.uniform(0, max_offset))],       # Bottom-left
        [cols * (1 - np.random.uniform(0, max_offset)), rows * (1 - np.random.uniform(0, max_offset))]  # Bottom-right
    ])
    return dst_points

def changePerspective(input_dir, output_dir, num_variations=4, max_offset=0.2):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Loop over each image in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Check for image file extensions
            # Load the image
            image_path = os.path.join(input_dir, filename)
            image = load_img(image_path)
            image_array = img_to_array(image)  # Convert to array format for OpenCV
            rows, cols, _ = image_array.shape

            # Define source points for the original image
            src_points = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])

            for i in range(num_variations):
                # Generate random destination points
                dst_points = random_perspective_points(cols, rows, max_offset)

                # Calculate the perspective transformation matrix and apply it
                matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                transformed_image_array = cv2.warpPerspective(image_array, matrix, (cols, rows))

                # Save the perspective-transformed image with a unique filename
                save_path = os.path.join(output_dir, f'perspective_{i+1}_{filename}')
                save_img(save_path, transformed_image_array)
                print(f'Saved perspective-transformed image: {save_path}')

# Paths for your input and output directories

figures = ['lib', 'arabesque', 'bow', 'heel', 'scorpion', 'scale']
figures = ['bow']
for figure in figures:
    input_dir = f'Dataset/{figure}'
    output_dir = f'Dataset/{figure}'

    flipImage(input_dir, output_dir)
    if(figure != 'lib' and figure != 'scale'):
        changePerspective(input_dir, output_dir, num_variations=4)
    else:
        changePerspective(input_dir, output_dir, num_variations=1)