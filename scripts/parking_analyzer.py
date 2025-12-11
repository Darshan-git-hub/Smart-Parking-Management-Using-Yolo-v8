#!/usr/bin/env python3
"""
Parking Space Analyzer - Complete solution for parking space detection
"""

from ultralytics import YOLO
import cv2
import argparse
from pathlib import Path

def quick_analysis(video_path="videos/parking.mp4"):
    """Quick analysis of parking video"""
    
    print("🚗 Quick Parking Analysis")
    print("=" * 30)
    
    # Load model
    model = YOLO("models/parking_classifier.pt")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
    
    # Get middle frame for analysis
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame = total_frames // 2
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
    ret, frame = cap.read()
    
    if not ret:
        print("❌ Cannot read frame")
        cap.release()
        return
    
    # Analyze with 6x4 grid (24 spaces)
    height, width = frame.shape[:2]
    cols, rows = 6, 4
    space_width = width // cols
    space_height = height // rows
    
    empty_count = 0
    occupied_count = 0
    
    print(f"📐 Analyzing {cols}x{rows} grid ({cols*rows} spaces)")
    
    for row in range(rows):
        for col in range(cols):
            x1 = col * space_width
            y1 = row * space_height
            x2 = x1 + space_width
            y2 = y1 + space_height
            
            space_img = frame[y1:y2, x1:x2]
            space_resized = cv2.resize(space_img, (224, 224))
            
            results = model(space_resized, verbose=False)
            prediction = results[0].names[results[0].probs.top1]
            
            if prediction == "empty":
                empty_count += 1
            else:
                occupied_count += 1
    
    total_spaces = cols * rows
    occupancy_rate = (occupied_count / total_spaces) * 100
    
    print(f"\n📊 Results (Frame {middle_frame}):")
    print(f"🟢 Empty spaces: {empty_count}")
    print(f"🔴 Occupied spaces: {occupied_count}")
    print(f"📈 Occupancy rate: {occupancy_rate:.1f}%")
    
    cap.release()

def main():
    """Main function with command line interface"""
    
    parser = argparse.ArgumentParser(description="Parking Space Analyzer")
    parser.add_argument("--video", "-v", default="videos/parking.mp4", 
                       help="Path to parking video")
    parser.add_argument("--mode", "-m", choices=["quick", "full", "summary"], 
                       default="quick", help="Analysis mode")
    parser.add_argument("--output", "-o", help="Output video path")
    parser.add_argument("--grid", "-g", default="6,4", 
                       help="Grid size as 'cols,rows'")
    
    args = parser.parse_args()
    
    # Check if video exists
    if not Path(args.video).exists():
        print(f"❌ Video not found: {args.video}")
        return
    
    # Check if model exists
    if not Path("models/parking_classifier.pt").exists():
        print("❌ Parking classifier model not found!")
        print("💡 Run: python scripts/train_parking.py")
        return
    
    print(f"🎯 Mode: {args.mode}")
    print(f"📹 Video: {args.video}")
    
    if args.mode == "quick":
        quick_analysis(args.video)
        
    elif args.mode == "full":
        from scripts.detect_parking_video import ParkingVideoAnalyzer
        
        cols, rows = map(int, args.grid.split(','))
        output_path = args.output or "videos/parking_analysis_full.mp4"
        
        analyzer = ParkingVideoAnalyzer()
        analyzer.analyze_parking_video(
            video_path=args.video,
            output_path=output_path,
            grid_size=(cols, rows)
        )
        
    elif args.mode == "summary":
        from scripts.parking_summary import analyze_parking_video_summary
        analyze_parking_video_summary(args.video)
    
    print(f"\n✅ Analysis complete!")

if __name__ == "__main__":
    main()