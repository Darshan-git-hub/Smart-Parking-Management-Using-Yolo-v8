from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict, deque
import time
from pathlib import Path

class SmartParkingSystem:
    """
    Smart Parking System that combines vehicle detection, tracking, and parking allocation
    """
    
    def __init__(self, vehicle_model_path="yolov8s.pt", parking_model_path="models/parking_classifier.pt"):
        # Load models
        self.vehicle_model = YOLO(vehicle_model_path)
        self.parking_model = YOLO(parking_model_path)
        
        # Vehicle tracking
        self.track_history = defaultdict(lambda: deque(maxlen=30))
        self.vehicle_count = 0
        self.tracked_vehicles = {}
        self.vehicle_classes = [2, 7]  # car=2, truck=7 in COCO dataset
        
        # Parking management
        self.parking_slots = {}
        self.allocated_vehicles = {}
        self.free_slots = set()
        
        # Statistics
        self.total_vehicles_detected = 0
        self.vehicles_parked = 0
        self.vehicles_left = 0
        
        print("✅ Smart Parking System initialized")
        print(f"🚗 Vehicle detection: {vehicle_model_path}")
        print(f"🅿️ Parking classification: {parking_model_path}")
    
    def initialize_parking_slots(self, grid_size=(6, 4), frame_width=640, frame_height=360):
        """Initialize parking slot grid"""
        cols, rows = grid_size
        slot_width = frame_width // cols
        slot_height = frame_height // rows
        
        slot_id = 1
        for row in range(rows):
            for col in range(cols):
                x1 = col * slot_width
                y1 = row * slot_height
                x2 = x1 + slot_width
                y2 = y1 + slot_height
                
                slot_name = f"Slot_{slot_id:02d}"
                self.parking_slots[slot_name] = {
                    'coords': (x1, y1, x2, y2),
                    'status': 'unknown',
                    'occupied_by': None
                }
                self.free_slots.add(slot_name)
                slot_id += 1
        
        print(f"🅿️ Initialized {len(self.parking_slots)} parking slots")
    
    def detect_vehicles(self, frame):
        """Detect and track vehicles in frame"""
        # Run vehicle detection with tracking
        results = self.vehicle_model.track(
            frame, 
            classes=self.vehicle_classes,  # Only cars and trucks
            conf=0.3,
            persist=True,
            verbose=False
        )
        
        detected_vehicles = []
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confidences = results[0].boxes.conf.float().cpu().tolist()
            classes = results[0].boxes.cls.int().cpu().tolist()
            
            for box, track_id, conf, cls in zip(boxes, track_ids, confidences, classes):
                x, y, w, h = box
                
                # Generate unique vehicle ID if new
                if track_id not in self.tracked_vehicles:
                    self.vehicle_count += 1
                    vehicle_id = f"V{self.vehicle_count:03d}"
                    self.tracked_vehicles[track_id] = {
                        'vehicle_id': vehicle_id,
                        'class': 'car' if cls == 2 else 'truck',
                        'first_seen': time.time(),
                        'last_seen': time.time(),
                        'assigned_slot': None,
                        'status': 'detected'
                    }
                    self.total_vehicles_detected += 1
                    print(f"🆕 New vehicle detected: {vehicle_id} ({self.tracked_vehicles[track_id]['class']})")
                else:
                    self.tracked_vehicles[track_id]['last_seen'] = time.time()
                
                # Store detection info
                vehicle_info = self.tracked_vehicles[track_id].copy()
                vehicle_info.update({
                    'track_id': track_id,
                    'bbox': (int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)),
                    'confidence': conf,
                    'center': (int(x), int(y))
                })
                
                detected_vehicles.append(vehicle_info)
                
                # Update track history
                self.track_history[track_id].append((int(x), int(y)))
        
        return detected_vehicles
    
    def analyze_parking_slots(self, frame):
        """Analyze parking slot occupancy"""
        for slot_name, slot_info in self.parking_slots.items():
            x1, y1, x2, y2 = slot_info['coords']
            
            # Extract slot region
            slot_region = frame[y1:y2, x1:x2]
            if slot_region.shape[0] < 10 or slot_region.shape[1] < 10:
                continue
            
            # Resize and classify
            slot_resized = cv2.resize(slot_region, (224, 224))
            results = self.parking_model(slot_resized, verbose=False)
            prediction = results[0].names[results[0].probs.top1]
            confidence = results[0].probs.top1conf.item()
            
            # Update slot status
            old_status = slot_info['status']
            slot_info['status'] = prediction
            
            # Manage free slots
            if prediction == 'empty':
                self.free_slots.add(slot_name)
                if slot_info['occupied_by']:
                    # Vehicle left
                    vehicle_id = slot_info['occupied_by']
                    print(f"🚗💨 Vehicle {vehicle_id} left {slot_name}")
                    slot_info['occupied_by'] = None
                    if vehicle_id in self.allocated_vehicles:
                        del self.allocated_vehicles[vehicle_id]
                    self.vehicles_left += 1
            else:
                self.free_slots.discard(slot_name)
    
    def allocate_parking(self, detected_vehicles):
        """Allocate parking slots to detected vehicles"""
        for vehicle in detected_vehicles:
            vehicle_id = vehicle['vehicle_id']
            track_id = vehicle['track_id']
            
            # Skip if already allocated
            if vehicle_id in self.allocated_vehicles:
                continue
            
            # Check if vehicle is near any parking slot
            vehicle_center = vehicle['center']
            
            for slot_name, slot_info in self.parking_slots.items():
                if slot_info['status'] == 'occupied' and not slot_info['occupied_by']:
                    x1, y1, x2, y2 = slot_info['coords']
                    slot_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    
                    # Calculate distance
                    distance = np.sqrt((vehicle_center[0] - slot_center[0])**2 + 
                                     (vehicle_center[1] - slot_center[1])**2)
                    
                    # If vehicle is close to occupied slot, assign it
                    if distance < 80:  # Threshold for assignment
                        slot_info['occupied_by'] = vehicle_id
                        self.allocated_vehicles[vehicle_id] = slot_name
                        self.tracked_vehicles[track_id]['assigned_slot'] = slot_name
                        self.tracked_vehicles[track_id]['status'] = 'parked'
                        self.vehicles_parked += 1
                        print(f"🅿️ Vehicle {vehicle_id} → {slot_name}")
                        break
    
    def draw_annotations(self, frame, detected_vehicles):
        """Draw all annotations on frame"""
        annotated_frame = frame.copy()
        
        # Draw parking slots
        for slot_name, slot_info in self.parking_slots.items():
            x1, y1, x2, y2 = slot_info['coords']
            
            # Color based on status
            if slot_info['status'] == 'empty':
                color = (0, 255, 0)  # Green
                text_color = (0, 200, 0)
            else:
                color = (0, 0, 255)  # Red
                text_color = (0, 0, 200)
            
            # Draw slot rectangle
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw slot name
            cv2.putText(annotated_frame, slot_name, (x1+5, y1+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            
            # Draw occupying vehicle ID if any
            if slot_info['occupied_by']:
                cv2.putText(annotated_frame, slot_info['occupied_by'], 
                           (x1+5, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        
        # Draw detected vehicles
        for vehicle in detected_vehicles:
            x1, y1, x2, y2 = vehicle['bbox']
            vehicle_id = vehicle['vehicle_id']
            vehicle_class = vehicle['class']
            track_id = vehicle['track_id']
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            
            # Draw vehicle info
            label = f"{vehicle_id} ({vehicle_class})"
            cv2.putText(annotated_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Draw tracking trail
            points = list(self.track_history[track_id])
            if len(points) > 1:
                for i in range(1, len(points)):
                    cv2.line(annotated_frame, points[i-1], points[i], (0, 255, 255), 2)
        
        # Draw statistics panel
        self.draw_statistics_panel(annotated_frame)
        
        # Draw allocation list
        self.draw_allocation_list(annotated_frame)
        
        return annotated_frame
    
    def draw_statistics_panel(self, frame):
        """Draw statistics panel"""
        # Background
        cv2.rectangle(frame, (10, 10), (350, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (350, 120), (255, 255, 255), 2)
        
        # Statistics
        stats = [
            f"Total Vehicles: {self.total_vehicles_detected}",
            f"Currently Tracked: {len([v for v in self.tracked_vehicles.values() if time.time() - v['last_seen'] < 5])}",
            f"Vehicles Parked: {self.vehicles_parked}",
            f"Free Slots: {len(self.free_slots)}/{len(self.parking_slots)}"
        ]
        
        for i, stat in enumerate(stats):
            cv2.putText(frame, stat, (20, 35 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def draw_allocation_list(self, frame):
        """Draw vehicle allocation list"""
        height, width = frame.shape[:2]
        
        # Background for allocation list
        list_width = 250
        list_height = min(300, len(self.allocated_vehicles) * 25 + 40)
        
        x_start = width - list_width - 10
        y_start = 10
        
        cv2.rectangle(frame, (x_start, y_start), 
                     (x_start + list_width, y_start + list_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (x_start, y_start), 
                     (x_start + list_width, y_start + list_height), (255, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "Vehicle Allocation", (x_start + 10, y_start + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Allocation list
        y_offset = 50
        for vehicle_id, slot_name in list(self.allocated_vehicles.items())[:10]:  # Show max 10
            allocation_text = f"{vehicle_id} -> {slot_name}"
            cv2.putText(frame, allocation_text, (x_start + 10, y_start + y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += 25
    
    def process_video(self, video_path, output_path=None, grid_size=(6, 4)):
        """Process video with smart parking system"""
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"🎥 Processing video: {width}x{height} @ {fps}FPS, {total_frames} frames")
        
        # Initialize parking slots
        self.initialize_parking_slots(grid_size, width, height)
        
        # Setup video writer
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"💾 Will save to: {output_path}")
        
        frame_count = 0
        
        print("\n🚀 Starting Smart Parking Analysis...")
        print("=" * 50)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 1. Detect and track vehicles
            detected_vehicles = self.detect_vehicles(frame)
            
            # 2. Analyze parking slots
            self.analyze_parking_slots(frame)
            
            # 3. Allocate parking
            self.allocate_parking(detected_vehicles)
            
            # 4. Draw annotations
            annotated_frame = self.draw_annotations(frame, detected_vehicles)
            
            # 5. Display frame
            cv2.imshow("Smart Parking System", annotated_frame)
            
            # 6. Save frame
            if output_path:
                out.write(annotated_frame)
            
            # Progress update
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"🔄 Progress: {progress:.1f}% - Active vehicles: {len(detected_vehicles)}, Free slots: {len(self.free_slots)}")
            
            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("⏹️ Stopped by user")
                break
        
        # Cleanup
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        print(f"\n📊 Final Statistics:")
        print(f"🚗 Total vehicles detected: {self.total_vehicles_detected}")
        print(f"🅿️ Vehicles parked: {self.vehicles_parked}")
        print(f"🚗💨 Vehicles left: {self.vehicles_left}")
        print(f"🆓 Free slots: {len(self.free_slots)}/{len(self.parking_slots)}")
        
        if output_path:
            print(f"💾 Output saved: {output_path}")

def main():
    """Main function"""
    
    # Check if models exist
    if not Path("models/parking_classifier.pt").exists():
        print("❌ Parking classifier not found! Run: python scripts/train_parking.py")
        return
    
    # Initialize system
    system = SmartParkingSystem()
    
    # Process video
    input_video = "videos/input.mp4"  # Change this to your video
    output_video = "videos/smart_parking_analysis.mp4"
    
    if not Path(input_video).exists():
        print(f"❌ Video not found: {input_video}")
        print("Available videos:")
        for video in Path("videos").glob("*.mp4"):
            print(f"  - {video}")
        return
    
    # Run analysis
    system.process_video(
        video_path=input_video,
        output_path=output_video,
        grid_size=(6, 4)  # Adjust based on your parking lot
    )

if __name__ == "__main__":
    main()