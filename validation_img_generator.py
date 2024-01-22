import os
import shutil
import random

# Path to the root directory containing training and validation folders
# Use an absolute path to the root directory
root_directory = r'D:\plant species identifier'

# Percentage of images to move to validation (20% in this case)
validation_percentage = 20

# Loop through each class folder in the training directory
for class_folder in os.listdir(os.path.join(root_directory, 'training')):
    class_path = os.path.join(root_directory, 'training', class_folder)
    
    # Create a corresponding folder in the validation directory if it doesn't exist
    validation_class_path = os.path.join(root_directory, 'validation', class_folder)
    os.makedirs(validation_class_path, exist_ok=True)
    
    # List all the images in the class folder
    images = os.listdir(class_path)
    
    # Calculate the number of images to move to validation
    num_images = len(images)
    num_validation_images = int(num_images * (validation_percentage / 100))
    
    # Randomly select images to move to validation
    validation_images = random.sample(images, num_validation_images)
    
    # Move the selected images to the validation directory
    for image in validation_images:
        src_path = os.path.join(class_path, image)
        dest_path = os.path.join(validation_class_path, image)
        shutil.move(src_path, dest_path)

print("Images copied to validation successfully.")