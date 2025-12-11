from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_parking_video_summary(video_path="videos/parking.mp4", model_path="models/parking_classifier.pt"):
    """
    Analyze parking video and provide summary statistics
    """
    
    # Load model
    model = YOLO(model_path)
    print(f"✅ Loaded model: {model_path}")
    
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
    duration = total_frames / fps
    
    print(f"\n🎥 Video Analysis Summary")
    print("=" * 40)
    print(f"📹 File: {Path(video_path).name}")
    print(f"📐 Resolution: {width}x{height}")
    print(f"⏱️ Duration: {duration:.1f} seconds ({total_frames} frames @ {fps}FPS)")
    
    # Different grid configurations to try
    grid_configs = [
        (4, 3, "Small lot (12 spaces)"),
        (6, 4, "Medium lot (24 spaces)"), 
        (8, 5, "Large lot (40 spaces)")
    ]
    
    print(f"\n🔍 Testing different parking lot configurations:")
    
    for cols, rows, description in grid_configs:
        print(f"\n📊 {description} - Grid: {cols}x{rows}")
        
        # Reset video to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Sample frames for analysis (every 30 frames)
        sample_frames = []
        empty_counts = []
        occupied_counts = []
        
        frame_count = 0
        while frame_count < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()
            if not ret:
                break
            
            # Analyze this frame
            empty_count, occupied_count = analyze_frame_grid(frame, model, (cols, rows))
            empty_counts.append(empty_count)
            occupied_counts.append(occupied_count)
            sample_frames.append(frame_count)
            
            frame_count += 30  # Sample every 30 frames
        
        # Calculate statistics
        total_spaces = cols * rows
        avg_empty = np.mean(empty_counts)
        avg_occupied = np.mean(occupied_counts)
        avg_occupancy = (avg_occupied / total_spaces) * 100
        
        min_empty = min(empty_counts)
        max_empty = max(empty_counts)
        
        print(f"   🅿️ Total spaces: {total_spaces}")
        print(f"   📈 Average occupancy: {avg_occupancy:.1f}%")
        print(f"   🟢 Average empty: {avg_empty:.1f} spaces")
        print(f"   🔴 Average occupied: {avg_occupied:.1f} spaces")
        print(f"   📊 Empty range: {min_empty}-{max_empty} spaces")
    
    cap.release()
    
    # Recommend best grid size
    print(f"\n💡 Recommendations:")
    print(f"   • Use grid 6x4 (24 spaces) for typical parking lot analysis")
    print(f"   • Adjust grid size based on actual parking lot layout")
    print(f"   • For better accuracy, define actual parking space coordinates")

def analyze_frame_grid(frame, model, grid_size):
    """Analyze a single frame with given grid size"""
    
    cols, rows = grid_size
    height, width = frame.shape[:2]
    
    space_width = width // cols
    space_height = height // rows
    
    empty_count = 0
    occupied_count = 0
    
    for row in range(rows):
        for col in range(cols):
            # Extract space
            x1 = col * space_width
            y1 = row * space_height
            x2 = x1 + space_width
            y2 = y1 + space_height
            
            space_img = frame[y1:y2, x1:x2]
            
            if space_img.shape[0] < 10 or space_img.shape[1] < 10:
                continue
            
            # Resize and classify
            space_resized = cv2.resize(space_img, (224, 224))
            results = model(space_resized, verbose=False)
            prediction = results[0].names[results[0].probs.top1]
            
            if prediction == "empty":
                empty_count += 1
            else:
                occupied_count += 1
    
    return empty_count, occupied_count

def create_parking_timeline(video_path="videos/parking.mp4", model_path="models/parking_classifier.pt"):
    """Create a timeline chart of parking occupancy"""
    
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample every 15 frames for timeline
    timestamps = []
    occupancy_rates = []
    
    print(f"\n📈 Creating occupancy timeline...")
    
    frame_count = 0
    while frame_count < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        if not ret:
            break
        
        # Analyze with 6x4 grid
        empty_count, occupied_count = analyze_frame_grid(frame, model, (6, 4))
        total_spaces = 24
        occupancy_rate = (occupied_count / total_spaces) * 100
        
        timestamp = frame_count / fps
        timestamps.append(timestamp)
        occupancy_rates.append(occupancy_rate)
        
        frame_count += 15
    
    cap.release()
    
    # Create and save plot
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, occupancy_rates, 'b-', linewidth=2)
    plt.fill_between(timestamps, occupancy_rates, alpha=0.3)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Occupancy Rate (%)')
    plt.title('Parking Lot Occupancy Over Time')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    
    # Add statistics
    avg_occupancy = np.mean(occupancy_rates)
    max_occupancy = max(occupancy_rates)
    min_occupancy = min(occupancy_rates)
    
    plt.axhline(y=avg_occupancy, color='r', linestyle='--', alpha=0.7, 
                label=f'Average: {avg_occupancy:.1f}%')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('parking_occupancy_timeline.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Occupancy Statistics:")
    print(f"   Average: {avg_occupancy:.1f}%")
    print(f"   Maximum: {max_occupancy:.1f}%")
    print(f"   Minimum: {min_occupancy:.1f}%")
    print(f"💾 Timeline chart saved as: parking_occupancy_timeline.png")

if __name__ == "__main__":
    # Run summary analysis
    analyze_parking_video_summary()
    
    # Create timeline (optional - requires matplotlib)
    try:
        create_parking_timeline()
    except ImportError:
        print("\n📊 Install matplotlib to generate occupancy timeline: pip install matplotlib")
    except Exception as e:
        print(f"\n⚠️ Could not create timeline: {e}")