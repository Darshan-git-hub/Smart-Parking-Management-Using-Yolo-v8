from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import numpy as np

def test_parking_model():
    """Test the trained parking space classification model"""
    
    # Load the trained model
    model = YOLO("models/parking_classifier.pt")
    
    print("🚗 Parking Space Detection Model Test")
    print("=" * 50)
    
    # Test on sample images from test set
    test_empty_dir = Path("parking dataset/test/empty")
    test_occupied_dir = Path("parking dataset/test/occupied")
    
    # Test empty spaces
    print("\n📍 Testing Empty Parking Spaces:")
    empty_correct = 0
    empty_total = 0
    
    for img_path in list(test_empty_dir.glob("*"))[:5]:  # Test first 5 images
        results = model(str(img_path))
        prediction = results[0].names[results[0].probs.top1]
        confidence = results[0].probs.top1conf.item()
        
        is_correct = prediction == "empty"
        empty_correct += is_correct
        empty_total += 1
        
        status = "✅" if is_correct else "❌"
        print(f"{status} {img_path.name}: {prediction} ({confidence:.3f})")
    
    # Test occupied spaces
    print("\n🚙 Testing Occupied Parking Spaces:")
    occupied_correct = 0
    occupied_total = 0
    
    for img_path in list(test_occupied_dir.glob("*"))[:5]:  # Test first 5 images
        results = model(str(img_path))
        prediction = results[0].names[results[0].probs.top1]
        confidence = results[0].probs.top1conf.item()
        
        is_correct = prediction == "occupied"
        occupied_correct += is_correct
        occupied_total += 1
        
        status = "✅" if is_correct else "❌"
        print(f"{status} {img_path.name}: {prediction} ({confidence:.3f})")
    
    # Calculate accuracy
    total_correct = empty_correct + occupied_correct
    total_samples = empty_total + occupied_total
    accuracy = total_correct / total_samples * 100
    
    print(f"\n📊 Test Results:")
    print(f"Empty spaces accuracy: {empty_correct}/{empty_total} ({empty_correct/empty_total*100:.1f}%)")
    print(f"Occupied spaces accuracy: {occupied_correct}/{occupied_total} ({occupied_correct/occupied_total*100:.1f}%)")
    print(f"Overall accuracy: {total_correct}/{total_samples} ({accuracy:.1f}%)")
    
    return model

def detect_parking_spaces_in_image(model, image_path, output_path=None):
    """
    Detect parking spaces in a full parking lot image
    This is a demo function - in practice you'd need to define parking space regions
    """
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return
    
    # For demo purposes, let's divide the image into grid sections
    # In a real application, you'd have predefined parking space coordinates
    height, width = img.shape[:2]
    
    # Create a 4x3 grid of parking spaces (demo)
    rows, cols = 3, 4
    space_height = height // rows
    space_width = width // cols
    
    results_img = img.copy()
    empty_count = 0
    total_spaces = rows * cols
    
    print(f"\n🔍 Analyzing parking lot image: {image_path}")
    print(f"Grid size: {rows}x{cols} = {total_spaces} spaces")
    
    for row in range(rows):
        for col in range(cols):
            # Extract parking space region
            y1 = row * space_height
            y2 = (row + 1) * space_height
            x1 = col * space_width
            x2 = (col + 1) * space_width
            
            space_img = img[y1:y2, x1:x2]
            
            # Resize to model input size
            space_resized = cv2.resize(space_img, (224, 224))
            
            # Predict
            results = model(space_resized, verbose=False)
            prediction = results[0].names[results[0].probs.top1]
            confidence = results[0].probs.top1conf.item()
            
            # Draw bounding box and label
            color = (0, 255, 0) if prediction == "empty" else (0, 0, 255)  # Green for empty, Red for occupied
            cv2.rectangle(results_img, (x1, y1), (x2, y2), color, 2)
            
            # Add text label
            label = f"{prediction} ({confidence:.2f})"
            cv2.putText(results_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            if prediction == "empty":
                empty_count += 1
    
    # Add summary text
    summary = f"Empty: {empty_count}/{total_spaces} spaces"
    cv2.putText(results_img, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Save result
    if output_path:
        cv2.imwrite(output_path, results_img)
        print(f"✅ Result saved to: {output_path}")
    
    print(f"📊 Parking Analysis: {empty_count} empty spaces out of {total_spaces} total")
    
    return results_img, empty_count, total_spaces

if __name__ == "__main__":
    # Test the model
    model = test_parking_model()
    
    # Demo: Test on a sample image (if available)
    sample_images = list(Path("parking dataset/test/occupied").glob("*.jpg"))
    if sample_images:
        sample_img = str(sample_images[0])
        print(f"\n🎯 Demo: Analyzing sample parking lot image...")
        detect_parking_spaces_in_image(model, sample_img, "parking_analysis_demo.jpg")
    
    print(f"\n🎉 Parking space detection model is ready!")
    print(f"📁 Model saved at: models/parking_classifier.pt")
    print(f"🔧 Usage: model = YOLO('models/parking_classifier.pt')")
    print(f"🔧 Predict: results = model('path/to/parking_space_image.jpg')")