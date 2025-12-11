import os
import json
import cv2
import numpy as np
from pathlib import Path

def create_yolo_dataset():
    """Convert IDD Lite dataset to YOLO format"""
    
    # Create directories for YOLO format
    os.makedirs("dataset/images/train", exist_ok=True)
    os.makedirs("dataset/images/val", exist_ok=True)
    os.makedirs("dataset/labels/train", exist_ok=True)
    os.makedirs("dataset/labels/val", exist_ok=True)
    
    # IDD Lite class mapping (simplified for vehicle detection)
    class_mapping = {
        0: 6,   # person -> person
        1: 0,   # car -> car
        2: 1,   # bus -> bus  
        3: 2,   # truck -> truck
        4: 3,   # autorickshaw -> autorickshaw
        5: 4,   # motorcycle -> motorcycle
        6: 5,   # bicycle -> bicycle
        # Add other mappings as needed
    }
    
    def process_split(split):
        img_dir = f"idd-lite/idd20k_lite/leftImg8bit/{split}"
        gt_dir = f"idd-lite/idd20k_lite/gtFine/{split}"
        
        if not os.path.exists(img_dir):
            print(f"Warning: {img_dir} not found")
            return
            
        image_paths = []
        
        # Process each sequence folder
        for seq_folder in os.listdir(img_dir):
            seq_img_path = os.path.join(img_dir, seq_folder)
            seq_gt_path = os.path.join(gt_dir, seq_folder)
            
            if not os.path.isdir(seq_img_path):
                continue
                
            # Process images in sequence
            for img_file in os.listdir(seq_img_path):
                if not (img_file.endswith('.png') or img_file.endswith('.jpg')):
                    continue
                    
                img_path = os.path.join(seq_img_path, img_file)
                
                # Copy image to YOLO dataset structure
                dst_img_path = f"dataset/images/{split}/{seq_folder}_{img_file}"
                
                # For now, just create symbolic links or copy files
                # This is a simplified version - you may need to implement
                # proper mask-to-bbox conversion based on your specific needs
                
                try:
                    # Copy image
                    import shutil
                    shutil.copy2(img_path, dst_img_path)
                    
                    # Create empty label file for now (you'll need to implement proper conversion)
                    label_file = dst_img_path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt')
                    with open(label_file, 'w') as f:
                        pass  # Empty file for now
                        
                    image_paths.append(dst_img_path)
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        # Create train/val.txt files
        with open(f"dataset/{split}.txt", 'w') as f:
            for img_path in image_paths:
                f.write(f"{img_path}\n")
        
        print(f"Processed {len(image_paths)} images for {split}")

    # Process train and val splits
    process_split('train')
    process_split('val')
    
    print("Dataset conversion completed!")
    print("Note: This creates empty label files. You'll need to implement")
    print("proper segmentation mask to bounding box conversion for your specific use case.")

if __name__ == "__main__":
    create_yolo_dataset()