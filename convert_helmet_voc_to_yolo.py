import os
import xml.etree.ElementTree as ET
import shutil

# Define paths
dataset_dir = 'hard-hat-detection'  # Update this to the exact Kaggle dataset folder name
output_dir = 'yolo_helmet_dataset'
images_dir = os.path.join(dataset_dir, 'images')
annotations_dir = os.path.join(dataset_dir, 'annotations')
output_images_dir = os.path.join(output_dir, 'images')
output_labels_dir = os.path.join(output_dir, 'labels')

# Create output directories
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

# Class names (update based on your dataset's annotations)
classes = ['helmet', 'no_helmet']  # Adjust if your dataset uses different names

for xml_file in os.listdir(annotations_dir):
    if not xml_file.endswith('.xml'):
        continue
    # Parse XML
    xml_path = os.path.join(annotations_dir, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    image_name = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    # Output YOLO label file
    label_file = os.path.join(output_labels_dir, xml_file.replace('.xml', '.txt'))
    with open(label_file, 'w') as f:
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in classes:
                continue
            class_id = classes.index(class_name)
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            # Convert to YOLO format
            x_center = (xmin + xmax) / 2.0 / width
            y_center = (ymin + ymax) / 2.0 / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    # Copy image
    shutil.copy(os.path.join(images_dir, image_name), output_images_dir)

print("Conversion complete. Check 'yolo_helmet_dataset' folder.")