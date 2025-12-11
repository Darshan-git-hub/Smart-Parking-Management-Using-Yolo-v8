from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

class VehicleAllocationAnalyzer:
    """
    Analyze incoming vehicles and available parking spaces
    """
    
    def __init__(self):
        # Load models
        self.vehicle_model = YOLO("yolov8s.pt")
        self.parking_model = YOLO("models/parking_classifier.pt")
        
        print("✅ Vehicle Allocation Analyzer initialized")
    
    def count_incoming_vehicles(self, video_path):
        """Count vehicles (cars and trucks) in the input video"""
        
        print(f"\n🚗 Analyzing incoming vehicles from: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return 0
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"📹 Video info: {total_frames} frames @ {fps} FPS")
        
        # Track unique vehicles
        tracked_vehicles = set()
        vehicle_count = 0
        frame_count = 0
        
        # Sample every 10 frames for efficiency
        sample_interval = 10
        
        while frame_count < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Detect vehicles (cars=2, trucks=7)
            results = self.vehicle_model.track(
                frame,
                classes=[2, 7],  # Only cars and trucks
                conf=0.4,
                persist=True,
                verbose=False
            )
            
            # Count unique tracked vehicles
            if results[0].boxes is not None and results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
                
                for track_id in track_ids:
                    if track_id not in tracked_vehicles:
                        tracked_vehicles.add(track_id)
                        vehicle_count += 1
                        print(f"🆕 Vehicle #{vehicle_count} detected (Track ID: {track_id})")
            
            frame_count += sample_interval
            
            # Show progress
            if frame_count % (sample_interval * 20) == 0:
                progress = (frame_count / total_frames) * 100
                print(f"🔄 Progress: {progress:.1f}% - Vehicles found: {vehicle_count}")
        
        cap.release()
        
        print(f"✅ Total incoming vehicles detected: {vehicle_count}")
        return vehicle_count
    
    def count_available_parking_spaces(self, video_path, grid_size=(6, 4)):
        """Count available parking spaces from parking video"""
        
        print(f"\n🅿️ Analyzing parking spaces from: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return 0, 0
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video info: {width}x{height}, {total_frames} frames")
        
        # Setup parking grid
        cols, rows = grid_size
        total_slots = cols * rows
        slot_width = width // cols
        slot_height = height // rows
        
        print(f"🏗️ Parking grid: {cols}x{rows} = {total_slots} slots")
        
        # Analyze middle frame for parking status
        middle_frame = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Cannot read frame")
            cap.release()
            return 0, 0
        
        empty_slots = 0
        occupied_slots = 0
        
        print(f"🔍 Analyzing parking slots...")
        
        for row in range(rows):
            for col in range(cols):
                slot_num = row * cols + col + 1
                
                # Extract slot region
                x1 = col * slot_width
                y1 = row * slot_height
                x2 = x1 + slot_width
                y2 = y1 + slot_height
                
                slot_region = frame[y1:y2, x1:x2]
                
                if slot_region.shape[0] < 10 or slot_region.shape[1] < 10:
                    continue
                
                # Resize and classify
                slot_resized = cv2.resize(slot_region, (224, 224))
                results = self.parking_model(slot_resized, verbose=False)
                prediction = results[0].names[results[0].probs.top1]
                confidence = results[0].probs.top1conf.item()
                
                if prediction == "empty":
                    empty_slots += 1
                    status = "🟢 EMPTY"
                else:
                    occupied_slots += 1
                    status = "🔴 OCCUPIED"
                
                print(f"   Slot_{slot_num:02d}: {status} (confidence: {confidence:.2f})")
        
        cap.release()
        
        print(f"\n📊 Parking Analysis Results:")
        print(f"🟢 Empty slots: {empty_slots}")
        print(f"🔴 Occupied slots: {occupied_slots}")
        print(f"📈 Occupancy rate: {(occupied_slots/total_slots)*100:.1f}%")
        
        return empty_slots, total_slots
    
    def analyze_allocation_capacity(self, input_video, parking_video):
        """Main analysis function"""
        
        print("🎯 Vehicle Allocation Capacity Analysis")
        print("=" * 50)
        
        # Step 1: Count incoming vehicles
        incoming_vehicles = self.count_incoming_vehicles(input_video)
        
        # Step 2: Count available parking spaces
        available_spaces, total_spaces = self.count_available_parking_spaces(parking_video)
        
        # Step 3: Calculate allocation
        print(f"\n📋 ALLOCATION ANALYSIS")
        print("=" * 30)
        
        can_allocate = min(incoming_vehicles, available_spaces)
        cannot_allocate = max(0, incoming_vehicles - available_spaces)
        
        print(f"🚗 Incoming vehicles: {incoming_vehicles}")
        print(f"🅿️ Available spaces: {available_spaces}")
        print(f"🏢 Total parking spaces: {total_spaces}")
        
        print(f"\n✅ Vehicles that CAN be allocated: {can_allocate}")
        print(f"❌ Vehicles that CANNOT be allocated: {cannot_allocate}")
        
        if incoming_vehicles <= available_spaces:
            print(f"🎉 SUCCESS: All {incoming_vehicles} vehicles can be parked!")
            print(f"🆓 Remaining free spaces: {available_spaces - incoming_vehicles}")
        else:
            print(f"⚠️ CAPACITY EXCEEDED: {cannot_allocate} vehicles will not find parking")
            print(f"📊 Utilization: {(can_allocate/total_spaces)*100:.1f}% of total capacity")
        
        # Summary statistics
        print(f"\n📈 SUMMARY STATISTICS")
        print("=" * 20)
        print(f"Allocation success rate: {(can_allocate/incoming_vehicles)*100:.1f}%")
        print(f"Parking utilization: {((total_spaces-available_spaces+can_allocate)/total_spaces)*100:.1f}%")
        
        return {
            'incoming_vehicles': incoming_vehicles,
            'available_spaces': available_spaces,
            'total_spaces': total_spaces,
            'can_allocate': can_allocate,
            'cannot_allocate': cannot_allocate
        }

def main():
    """Main function"""
    
    # Check if parking model exists
    if not Path("models/parking_classifier.pt").exists():
        print("❌ Parking classifier model not found!")
        print("💡 Run: python scripts/train_parking.py")
        return
    
    # Video paths
    input_video = "videos/input.mp4"      # Video with incoming vehicles
    parking_video = "videos/parking.mp4"  # Video showing parking lot
    
    # Check if videos exist
    if not Path(input_video).exists():
        print(f"❌ Input video not found: {input_video}")
        return
    
    if not Path(parking_video).exists():
        print(f"❌ Parking video not found: {parking_video}")
        return
    
    # Run analysis
    analyzer = VehicleAllocationAnalyzer()
    results = analyzer.analyze_allocation_capacity(input_video, parking_video)
    
    # Save results to file
    with open("allocation_analysis_results.txt", "w") as f:
        f.write("VEHICLE ALLOCATION ANALYSIS RESULTS\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Incoming vehicles: {results['incoming_vehicles']}\n")
        f.write(f"Available parking spaces: {results['available_spaces']}\n")
        f.write(f"Total parking spaces: {results['total_spaces']}\n")
        f.write(f"Vehicles that can be allocated: {results['can_allocate']}\n")
        f.write(f"Vehicles that cannot be allocated: {results['cannot_allocate']}\n")
        f.write(f"Allocation success rate: {(results['can_allocate']/results['incoming_vehicles'])*100:.1f}%\n")
    
    print(f"\n💾 Results saved to: allocation_analysis_results.txt")

if __name__ == "__main__":
    main()