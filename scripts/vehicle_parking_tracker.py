from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict, deque
import time

class VehicleParkingTracker:
    """
    Vehicle tracking and parking allocation system
    Tracks cars and trucks, assigns them to parking slots
    """
    
    def __init__(self):
        # Load models
        self.vehicle_model = YOLO("yolov8s.pt")
        self.parking_model = YOLO("models/parking_classifier.pt")
        
        # Vehicle tracking
        self.vehicle_count = 0
        self.tracked_vehicles = {}
        self.track_history = defaultdict(lambda: deque(maxlen=30))
        
        # Parking slots
        self.parking_slots = {}
        self.allocated_vehicles = {}
        self.free_slots = set()
        
        # Statistics
        self.total_vehicles = 0
        self.vehicles_parked = 0
        
        print("✅ Vehicle Parking Tracker initialized")
    
    def setup_parking_grid(self, frame_width, frame_height, grid_size=(6, 4)):
        """Setup parking slot grid"""
        cols, rows = grid_size
        slot_width = frame_width // cols
        slot_height = frame_height // rows
        
        for row in range(rows):
            for col in range(cols):
                slot_id = row * cols + col + 1
                slot_name = f"Slot_{slot_id}"
                
                x1 = col * slot_width
                y1 = row * slot_height
                x2 = x1 + slot_width
                y2 = y1 + slot_height
                
                self.parking_slots[slot_name] = {
                    'coords': (x1, y1, x2, y2),
                    'status': 'unknown',
                    'occupied_by': None
                }
                self.free_slots.add(slot_name)
        
        print(f"🅿️ Setup {len(self.parking_slots)} parking slots")
    
    def detect_and_track_vehicles(self, frame):
        """Detect cars and trucks with tracking"""
        # Detect only cars (2) and trucks (7)
        results = self.vehicle_model.track(
            frame,
            classes=[2, 7],  # car=2, truck=7
            conf=0.4,
            persist=True,
            verbose=False
        )
        
        vehicles = []
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            classes = results[0].boxes.cls.int().cpu().tolist()
            confidences = results[0].boxes.conf.float().cpu().tolist()
            
            for box, track_id, cls, conf in zip(boxes, track_ids, classes, confidences):
                x, y, w, h = box
                
                # Create new vehicle ID if not seen before
                if track_id not in self.tracked_vehicles:
                    self.vehicle_count += 1
                    vehicle_id = f"V{self.vehicle_count:02d}"
                    
                    self.tracked_vehicles[track_id] = {
                        'vehicle_id': vehicle_id,
                        'type': 'car' if cls == 2 else 'truck',
                        'assigned_slot': None,
                        'first_seen': time.time()
                    }
                    self.total_vehicles += 1
                    print(f"🆕 New vehicle: {vehicle_id} ({self.tracked_vehicles[track_id]['type']})")
                
                # Update tracking history
                center = (int(x), int(y))
                self.track_history[track_id].append(center)
                
                # Store vehicle info
                vehicle_info = {
                    'track_id': track_id,
                    'vehicle_id': self.tracked_vehicles[track_id]['vehicle_id'],
                    'type': self.tracked_vehicles[track_id]['type'],
                    'bbox': (int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)),
                    'center': center,
                    'confidence': conf,
                    'assigned_slot': self.tracked_vehicles[track_id]['assigned_slot']
                }
                
                vehicles.append(vehicle_info)
        
        return vehicles
    
    def analyze_parking_spaces(self, frame):
        """Analyze each parking slot"""
        for slot_name, slot_info in self.parking_slots.items():
            x1, y1, x2, y2 = slot_info['coords']
            
            # Extract slot region
            slot_region = frame[y1:y2, x1:x2]
            if slot_region.shape[0] < 10 or slot_region.shape[1] < 10:
                continue
            
            # Classify slot
            slot_resized = cv2.resize(slot_region, (224, 224))
            results = self.parking_model(slot_resized, verbose=False)
            prediction = results[0].names[results[0].probs.top1]
            
            # Update slot status
            slot_info['status'] = prediction
            
            if prediction == 'empty':
                self.free_slots.add(slot_name)
                # If slot became empty, remove vehicle assignment
                if slot_info['occupied_by']:
                    vehicle_id = slot_info['occupied_by']
                    slot_info['occupied_by'] = None
                    if vehicle_id in self.allocated_vehicles:
                        del self.allocated_vehicles[vehicle_id]
            else:
                self.free_slots.discard(slot_name)
    
    def allocate_vehicles_to_slots(self, vehicles):
        """Allocate vehicles to parking slots"""
        for vehicle in vehicles:
            vehicle_id = vehicle['vehicle_id']
            track_id = vehicle['track_id']
            
            # Skip if already allocated
            if vehicle_id in self.allocated_vehicles:
                continue
            
            # Find closest occupied slot without assignment
            vehicle_center = vehicle['center']
            min_distance = float('inf')
            best_slot = None
            
            for slot_name, slot_info in self.parking_slots.items():
                if slot_info['status'] == 'occupied' and not slot_info['occupied_by']:
                    x1, y1, x2, y2 = slot_info['coords']
                    slot_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    
                    distance = np.sqrt((vehicle_center[0] - slot_center[0])**2 + 
                                     (vehicle_center[1] - slot_center[1])**2)
                    
                    if distance < min_distance and distance < 100:  # Within reasonable range
                        min_distance = distance
                        best_slot = slot_name
            
            # Assign vehicle to slot
            if best_slot:
                self.parking_slots[best_slot]['occupied_by'] = vehicle_id
                self.allocated_vehicles[vehicle_id] = best_slot
                self.tracked_vehicles[track_id]['assigned_slot'] = best_slot
                self.vehicles_parked += 1
                print(f"🅿️ Vehicle {vehicle_id} → {best_slot}")
    
    def draw_frame(self, frame, vehicles):
        """Draw all annotations on frame"""
        annotated = frame.copy()
        
        # Draw parking slots
        for slot_name, slot_info in self.parking_slots.items():
            x1, y1, x2, y2 = slot_info['coords']
            
            # Color based on status
            if slot_info['status'] == 'empty':
                color = (0, 255, 0)  # Green
            else:
                color = (0, 0, 255)  # Red
            
            # Draw slot
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, slot_name, (x1+5, y1+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Show assigned vehicle
            if slot_info['occupied_by']:
                cv2.putText(annotated, slot_info['occupied_by'], 
                           (x1+5, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw vehicles
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle['bbox']
            vehicle_id = vehicle['vehicle_id']
            vehicle_type = vehicle['type']
            track_id = vehicle['track_id']
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 2)
            
            # Draw vehicle label
            label = f"{vehicle_id} ({vehicle_type})"
            cv2.putText(annotated, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Draw tracking trail
            points = list(self.track_history[track_id])
            for i in range(1, len(points)):
                cv2.line(annotated, points[i-1], points[i], (0, 255, 255), 2)
        
        # Draw info panels
        self.draw_info_panel(annotated)
        self.draw_allocation_list(annotated)
        
        return annotated
    
    def draw_info_panel(self, frame):
        """Draw information panel"""
        # Background
        cv2.rectangle(frame, (10, 10), (300, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, 100), (255, 255, 255), 2)
        
        # Info text
        info = [
            f"Total Vehicles: {self.total_vehicles}",
            f"Vehicles Parked: {self.vehicles_parked}",
            f"Free Slots: {len(self.free_slots)}/{len(self.parking_slots)}"
        ]
        
        for i, text in enumerate(info):
            cv2.putText(frame, text, (20, 35 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def draw_allocation_list(self, frame):
        """Draw vehicle allocation list"""
        height, width = frame.shape[:2]
        
        # Position on right side
        x_start = width - 220
        y_start = 10
        list_height = min(200, len(self.allocated_vehicles) * 20 + 40)
        
        # Background
        cv2.rectangle(frame, (x_start, y_start), 
                     (x_start + 210, y_start + list_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (x_start, y_start), 
                     (x_start + 210, y_start + list_height), (255, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "Vehicle Allocation", (x_start + 5, y_start + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # List allocations
        y_offset = 40
        for vehicle_id, slot_name in list(self.allocated_vehicles.items())[:8]:
            text = f"{vehicle_id} -> {slot_name}"
            cv2.putText(frame, text, (x_start + 5, y_start + y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            y_offset += 20
    
    def process_video(self, input_path, output_path):
        """Process video with vehicle tracking and parking allocation"""
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {input_path}")
            return
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"🎥 Video: {width}x{height} @ {fps}FPS, {total_frames} frames")
        
        # Setup parking grid
        self.setup_parking_grid(width, height)
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        print("\n🚀 Starting Vehicle Tracking & Parking Analysis...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 1. Detect and track vehicles
            vehicles = self.detect_and_track_vehicles(frame)
            
            # 2. Analyze parking spaces
            self.analyze_parking_spaces(frame)
            
            # 3. Allocate vehicles to slots
            self.allocate_vehicles_to_slots(vehicles)
            
            # 4. Draw annotations
            annotated_frame = self.draw_frame(frame, vehicles)
            
            # 5. Save and display
            out.write(annotated_frame)
            cv2.imshow("Vehicle Parking Tracker", annotated_frame)
            
            # Progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"🔄 {progress:.1f}% - Vehicles: {len(vehicles)}, Free slots: {len(self.free_slots)}")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"\n✅ Processing complete!")
        print(f"📊 Total vehicles detected: {self.total_vehicles}")
        print(f"🅿️ Vehicles parked: {self.vehicles_parked}")
        print(f"💾 Output saved: {output_path}")

def main():
    """Main function"""
    tracker = VehicleParkingTracker()
    
    # Process the input video
    input_video = "videos/input.mp4"
    output_video = "videos/vehicle_parking_analysis.mp4"
    
    tracker.process_video(input_video, output_video)

if __name__ == "__main__":
    main()