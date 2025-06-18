import os
import shutil
import random

# Define paths
dataset_dir = 'yolo_helmet_dataset'
images_dir = os.path.join(dataset_dir, 'images')
labels_dir = os.path.join(dataset_dir, 'labels')
train_images_dir = os.path.join(dataset_dir, 'train/images')
train_labels_dir = os.path.join(dataset_dir, 'train/labels')
valid_images_dir = os.path.join(dataset_dir, 'valid/images')
valid_labels_dir = os.path.join(dataset_dir, 'valid/labels')
test_images_dir = os.path.join(dataset_dir, 'test/images')
test_labels_dir = os.path.join(dataset_dir, 'test/labels')

# Create directories
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(valid_images_dir, exist_ok=True)
os.makedirs(valid_labels_dir, exist_ok=True)
os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(test_labels_dir, exist_ok=True)

# Get list of images
image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
random.shuffle(image_files)

# Split ratios: 70% train, 20% validation, 10% test
total = len(image_files)
train_count = int(0.7 * total)
valid_count = int(0.2 * total)
test_count = total - train_count - valid_count

train_files = image_files[:train_count]
valid_files = image_files[train_count:train_count + valid_count]
test_files = image_files[train_count + valid_count:]

# Copy files to respective folders
for f in train_files:
    shutil.copy(os.path.join(images_dir, f), train_images_dir)
    shutil.copy(os.path.join(labels_dir, f.replace('.png', '.txt')), train_labels_dir)
for f in valid_files:
    shutil.copy(os.path.join(images_dir, f), valid_images_dir)
    shutil.copy(os.path.join(labels_dir, f.replace('.png', '.txt')), valid_labels_dir)
for f in test_files:
    shutil.copy(os.path.join(images_dir, f), test_images_dir)
    shutil.copy(os.path.join(labels_dir, f.replace('.png', '.txt')), test_labels_dir)

print("Dataset split complete. Check 'yolo_helmet_dataset/train', 'yolo_helmet_dataset/valid', and 'yolo_helmet_dataset/test' folders.")