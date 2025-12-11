from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

class ParkingVideoAnalyzer:
    """Analyze parking spaces in video using the trained classifier"""
    
    def __init__(self, model_path="models/parking_classifier.pt"):
        self.model = YOLO(model_path)
        print(f"✅ Loaded parking classifier: {model_path}")
    
    def analyze_parking_video(self, video_path, output_path=None, grid_size=(6, 4)):
        """
        Analyze parking video frame by frame
        
        Args:
            video_path: Path to parking video
            output_path: Path to save annotated video
            grid_size: (cols, rows) for parking space grid
        """
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"🎥 Video Info: {width}x{height} @ {fps}FPS, {total_frames} frames")
        
        # Setup video writer if output path provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"💾 Will save to: {output_path}")
        
        # Calculate grid dimensions
        cols, rows = grid_size
        space_width = width // cols
        space_height = height // rows
        total_spaces = cols * rows
        
        print(f"🅿️ Parking Grid: {cols}x{rows} = {total_spaces} spaces")
        print(f"📏 Space size: {space_width}x{space_height} pixels")
        
        frame_count = 0
        
        # Process video frame by frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Analyze parking spaces in current frame
            annotated_frame, empty_count, occupied_count = self.analyze_frame(
                frame, grid_size, space_width, space_height
            )
            
            # Add frame info and statistics
            occupancy_rate = (occupied_count / total_spaces) * 100
            
            # Create info panel
            info_text = [
                f"Frame: {frame_count}/{total_frames}",
                f"Empty: {empty_count} spaces",
                f"Occupied: {occupied_count} spaces", 
                f"Occupancy: {occupancy_rate:.1f}%"
            ]
            
            # Draw info panel background
            panel_height = 100
            cv2.rectangle(annotated_frame, (10, 10), (300, panel_height), (0, 0, 0), -1)
            cv2.rectangle(annotated_frame, (10, 10), (300, panel_height), (255, 255, 255), 2)
            
            # Draw info text
            for i, text in enumerate(info_text):
                y_pos = 30 + i * 20
                cv2.putText(annotated_frame, text, (20, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show progress
            if frame_count % 30 == 0:  # Print every 30 frames
                progress = (frame_count / total_frames) * 100
                print(f"🔄 Processing: {progress:.1f}% - Empty: {empty_count}, Occupied: {occupied_count}")
            
            # Display frame (optional - comment out for faster processing)
            cv2.imshow("Parking Analysis", annotated_frame)
            
            # Save frame if output video specified
            if output_path:
                out.write(annotated_frame)
            
            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("⏹️ Stopped by user")
                break
        
        # Cleanup
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"✅ Processed {frame_count} frames")
        if output_path:
            print(f"💾 Output saved to: {output_path}")
    
    def analyze_frame(self, frame, grid_size, space_width, space_height):
        """
        Analyze parking spaces in a single frame
        
        Returns:
            tuple: (annotated_frame, empty_count, occupied_count)
        """
        cols, rows = grid_size
        annotated_frame = frame.copy()
        empty_count = 0
        occupied_count = 0
        
        # Analyze each parking space
        for row in range(rows):
            for col in range(cols):
                # Calculate space coordinates
                x1 = col * space_width
                y1 = row * space_height
                x2 = x1 + space_width
                y2 = y1 + space_height
                
                # Extract parking space region
                space_img = frame[y1:y2, x1:x2]
                
                # Skip if space is too small
                if space_img.shape[0] < 10 or space_img.shape[1] < 10:
                    continue
                
                # Resize to model input size
                space_resized = cv2.resize(space_img, (224, 224))
                
                # Classify the space
                results = self.model(space_resized, verbose=False)
                prediction = results[0].names[results[0].probs.top1]
                confidence = results[0].probs.top1conf.item()
                
                # Count spaces and set colors
                if prediction == "empty":
                    empty_count += 1
                    color = (0, 255, 0)  # Green for empty
                    text_color = (0, 200, 0)
                    status = "E"
                else:
                    occupied_count += 1
                    color = (0, 0, 255)  # Red for occupied
                    text_color = (0, 0, 200)
                    status = "O"
                
                # Draw bounding box
                thickness = 2 if confidence > 0.8 else 1
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                
                # Add space number and status
                space_num = row * cols + col + 1
                
                # Draw space number
                cv2.putText(annotated_frame, f"{space_num}", (x1+5, y1+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                
                # Draw status
                cv2.putText(annotated_frame, status, (x1+5, y2-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                
                # Draw confidence if low
                if confidence < 0.9:
                    cv2.putText(annotated_frame, f"{confidence:.2f}", 
                               (x2-40, y1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return annotated_frame, empty_count, occupied_count

def main():
    """Main function to run parking video analysis"""
    
    # Initialize analyzer
    analyzer = ParkingVideoAnalyzer()
    
    # Video paths
    input_video = "videos/parking.mp4"
    output_video = "videos/parking_analysis.mp4"
    
    # Check if input video exists
    if not Path(input_video).exists():
        print(f"❌ Error: Video not found at {input_video}")
        return
    
    print("🚗 Starting Parking Video Analysis")
    print("=" * 50)
    
    # Analyze the parking video
    # You can adjust grid_size based on your parking lot layout
    analyzer.analyze_parking_video(
        video_path=input_video,
        output_path=output_video,
        grid_size=(6, 4)  # 6 columns, 4 rows = 24 parking spaces
    )
    
    print("\n🎉 Analysis Complete!")
    print(f"📁 Input: {input_video}")
    print(f"📁 Output: {output_video}")
    print("\n💡 Tips:")
    print("- Adjust grid_size parameter to match your parking lot layout")
    print("- Press 'q' during playback to stop early")
    print("- Comment out cv2.imshow() line for faster processing")

if __name__ == "__main__":
    main()