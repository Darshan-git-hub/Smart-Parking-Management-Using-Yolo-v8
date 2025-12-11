from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import argparse

class ParkingSpaceDetector:
    """
    A comprehensive parking space detection system using the trained classifier
    """
    
    def __init__(self, model_path="models/parking_classifier.pt"):
        """Initialize the parking space detector"""
        self.model = YOLO(model_path)
        print(f"✅ Loaded parking space detection model: {model_path}")
    
    def classify_single_space(self, image_path_or_array, verbose=True):
        """
        Classify a single parking space as empty or occupied
        
        Args:
            image_path_or_array: Path to image file or numpy array
            verbose: Whether to print results
            
        Returns:
            tuple: (prediction, confidence)
        """
        results = self.model(image_path_or_array, verbose=False)
        prediction = results[0].names[results[0].probs.top1]
        confidence = results[0].probs.top1conf.item()
        
        if verbose:
            status = "🟢 EMPTY" if prediction == "empty" else "🔴 OCCUPIED"
            print(f"{status} - Confidence: {confidence:.3f}")
        
        return prediction, confidence
    
    def analyze_parking_lot_grid(self, image_path, grid_size=(4, 6), output_path=None):
        """
        Analyze a parking lot by dividing it into a grid of parking spaces
        
        Args:
            image_path: Path to parking lot image
            grid_size: (cols, rows) for grid division
            output_path: Path to save annotated result
            
        Returns:
            dict: Analysis results
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        height, width = img.shape[:2]
        cols, rows = grid_size
        
        # Calculate space dimensions
        space_width = width // cols
        space_height = height // rows
        
        results_img = img.copy()
        empty_spaces = []
        occupied_spaces = []
        
        print(f"\n🔍 Analyzing parking lot: {Path(image_path).name}")
        print(f"📐 Grid: {cols}x{rows} = {cols*rows} spaces")
        print(f"📏 Space size: {space_width}x{space_height} pixels")
        
        # Analyze each grid cell
        for row in range(rows):
            for col in range(cols):
                # Extract parking space region
                x1 = col * space_width
                y1 = row * space_height
                x2 = x1 + space_width
                y2 = y1 + space_height
                
                # Extract and resize space image
                space_img = img[y1:y2, x1:x2]
                space_resized = cv2.resize(space_img, (224, 224))
                
                # Classify the space
                prediction, confidence = self.classify_single_space(space_resized, verbose=False)
                
                # Store results
                space_info = {
                    'row': row, 'col': col,
                    'coords': (x1, y1, x2, y2),
                    'prediction': prediction,
                    'confidence': confidence
                }
                
                if prediction == "empty":
                    empty_spaces.append(space_info)
                    color = (0, 255, 0)  # Green for empty
                    text_color = (0, 200, 0)
                else:
                    occupied_spaces.append(space_info)
                    color = (0, 0, 255)  # Red for occupied
                    text_color = (0, 0, 200)
                
                # Draw bounding box
                cv2.rectangle(results_img, (x1, y1), (x2, y2), color, 2)
                
                # Add space number and status
                space_num = row * cols + col + 1
                label = f"#{space_num}"
                status = "E" if prediction == "empty" else "O"
                
                # Draw labels
                cv2.putText(results_img, label, (x1+5, y1+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                cv2.putText(results_img, status, (x1+5, y2-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        # Add summary information
        total_spaces = len(empty_spaces) + len(occupied_spaces)
        occupancy_rate = len(occupied_spaces) / total_spaces * 100
        
        # Draw summary box
        summary_text = [
            f"Total Spaces: {total_spaces}",
            f"Empty: {len(empty_spaces)}",
            f"Occupied: {len(occupied_spaces)}",
            f"Occupancy: {occupancy_rate:.1f}%"
        ]
        
        # Background for text
        cv2.rectangle(results_img, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.rectangle(results_img, (10, 10), (300, 120), (255, 255, 255), 2)
        
        for i, text in enumerate(summary_text):
            cv2.putText(results_img, text, (20, 35 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Save result if requested
        if output_path:
            cv2.imwrite(output_path, results_img)
            print(f"💾 Annotated image saved: {output_path}")
        
        # Print results
        print(f"\n📊 Analysis Results:")
        print(f"🟢 Empty spaces: {len(empty_spaces)}")
        print(f"🔴 Occupied spaces: {len(occupied_spaces)}")
        print(f"📈 Occupancy rate: {occupancy_rate:.1f}%")
        
        return {
            'total_spaces': total_spaces,
            'empty_spaces': empty_spaces,
            'occupied_spaces': occupied_spaces,
            'occupancy_rate': occupancy_rate,
            'annotated_image': results_img
        }
    
    def analyze_predefined_spaces(self, image_path, space_coordinates, output_path=None):
        """
        Analyze parking spaces with predefined coordinates
        
        Args:
            image_path: Path to parking lot image
            space_coordinates: List of (x1, y1, x2, y2) tuples for each space
            output_path: Path to save annotated result
            
        Returns:
            dict: Analysis results
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        results_img = img.copy()
        empty_count = 0
        occupied_count = 0
        
        print(f"\n🔍 Analyzing {len(space_coordinates)} predefined parking spaces")
        
        for i, (x1, y1, x2, y2) in enumerate(space_coordinates):
            # Extract parking space
            space_img = img[y1:y2, x1:x2]
            space_resized = cv2.resize(space_img, (224, 224))
            
            # Classify
            prediction, confidence = self.classify_single_space(space_resized, verbose=False)
            
            # Count and visualize
            if prediction == "empty":
                empty_count += 1
                color = (0, 255, 0)
            else:
                occupied_count += 1
                color = (0, 0, 255)
            
            # Draw rectangle and label
            cv2.rectangle(results_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(results_img, f"{i+1}: {prediction[:1].upper()}", 
                       (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        if output_path:
            cv2.imwrite(output_path, results_img)
        
        total = len(space_coordinates)
        occupancy_rate = occupied_count / total * 100
        
        print(f"📊 Results: {empty_count} empty, {occupied_count} occupied ({occupancy_rate:.1f}% full)")
        
        return {
            'total_spaces': total,
            'empty_count': empty_count,
            'occupied_count': occupied_count,
            'occupancy_rate': occupancy_rate,
            'annotated_image': results_img
        }

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="Parking Space Detection System")
    parser.add_argument("--image", type=str, help="Path to parking lot image")
    parser.add_argument("--model", type=str, default="models/parking_classifier.pt", 
                       help="Path to trained model")
    parser.add_argument("--grid", type=str, default="4,6", 
                       help="Grid size as 'cols,rows' (e.g., '4,6')")
    parser.add_argument("--output", type=str, help="Output path for annotated image")
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = ParkingSpaceDetector(args.model)
    
    if args.image:
        # Parse grid size
        cols, rows = map(int, args.grid.split(','))
        
        # Analyze parking lot
        results = detector.analyze_parking_lot_grid(
            args.image, 
            grid_size=(cols, rows),
            output_path=args.output
        )
        
        print(f"\n✅ Analysis complete!")
        
    else:
        # Demo mode - test on sample images
        print("🎯 Demo Mode: Testing on sample images from dataset")
        
        # Test single space classification
        test_empty = "parking dataset/test/empty/spot1.jpg"
        test_occupied = "parking dataset/test/occupied/spot100.jpg"
        
        if Path(test_empty).exists():
            print(f"\n📸 Testing empty space: {test_empty}")
            detector.classify_single_space(test_empty)
        
        if Path(test_occupied).exists():
            print(f"\n📸 Testing occupied space: {test_occupied}")
            detector.classify_single_space(test_occupied)

if __name__ == "__main__":
    main()