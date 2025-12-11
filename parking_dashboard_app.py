import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ultralytics import YOLO
import tempfile
import os
from pathlib import Path
import time
from collections import defaultdict, deque
import base64

# Set page config
st.set_page_config(
    page_title="Smart Parking Analysis Dashboard",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ParkingAnalyzer:
    """Main analyzer class for parking and vehicle detection"""
    
    def __init__(self):
        self.vehicle_model = None
        self.parking_model = None
        self.load_models()
    
    def create_video_writer(self, output_path, fps, width, height):
        """Create a video writer with the best available codec"""
        codecs_to_try = [
            ('XVID', '.avi'),  # Most reliable
            ('MJPG', '.avi'),  # Motion JPEG
            ('mp4v', '.mp4'),  # MP4 fallback
            ('X264', '.mp4'),  # Alternative H264
        ]
        
        for codec, ext in codecs_to_try:
            try:
                # Adjust output path extension if needed
                test_output = output_path
                if not output_path.endswith(ext):
                    test_output = output_path.rsplit('.', 1)[0] + ext
                
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(test_output, fourcc, fps, (width, height))
                
                if out.isOpened():
                    st.info(f"✅ Using codec: {codec} for {test_output}")
                    return out, test_output
                else:
                    out.release()
            except Exception as e:
                st.warning(f"⚠️ Codec {codec} failed: {e}")
                continue
        
        st.error(f"❌ Could not create video writer for {output_path}")
        return None, output_path
    
    def load_models(self):
        """Load YOLO models"""
        try:
            self.vehicle_model = YOLO("yolov8s.pt")
            if Path("models/parking_classifier.pt").exists():
                self.parking_model = YOLO("models/parking_classifier.pt")
            else:
                st.error("❌ Parking classifier model not found! Please train the model first.")
        except Exception as e:
            st.error(f"Error loading models: {e}")
    
    def analyze_incoming_vehicles(self, video_path, progress_bar=None):
        """Analyze incoming vehicles from video"""
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0, []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        tracked_vehicles = set()
        vehicle_timeline = []
        frame_count = 0
        sample_interval = 10
        
        while frame_count < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Detect vehicles
            results = self.vehicle_model.track(
                frame,
                classes=[2, 7],  # cars and trucks
                conf=0.4,
                persist=True,
                verbose=False
            )
            
            # Count unique vehicles
            current_vehicles = 0
            if results[0].boxes is not None and results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
                
                for track_id in track_ids:
                    if track_id not in tracked_vehicles:
                        tracked_vehicles.add(track_id)
                
                current_vehicles = len(track_ids)
            
            # Record timeline
            timestamp = frame_count / fps
            vehicle_timeline.append({
                'time': timestamp,
                'vehicles': current_vehicles,
                'cumulative_vehicles': len(tracked_vehicles)
            })
            
            frame_count += sample_interval
            
            # Update progress
            if progress_bar:
                progress = min(1.0, frame_count / total_frames)
                progress_bar.progress(progress)
        
        cap.release()
        return len(tracked_vehicles), vehicle_timeline
    
    def analyze_parking_spaces(self, video_path, grid_size=(6, 4), progress_bar=None):
        """Analyze parking spaces over time"""
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {}, []
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        cols, rows = grid_size
        total_slots = cols * rows
        slot_width = width // cols
        slot_height = height // rows
        
        occupancy_timeline = []
        slot_analysis = {}
        
        # Sample frames for timeline
        sample_frames = np.linspace(0, total_frames-1, min(20, total_frames), dtype=int)
        
        for i, frame_num in enumerate(sample_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            empty_count = 0
            occupied_count = 0
            
            # Analyze each slot
            for row in range(rows):
                for col in range(cols):
                    slot_id = row * cols + col + 1
                    
                    x1 = col * slot_width
                    y1 = row * slot_height
                    x2 = x1 + slot_width
                    y2 = y1 + slot_height
                    
                    slot_region = frame[y1:y2, x1:x2]
                    
                    if slot_region.shape[0] < 10 or slot_region.shape[1] < 10:
                        continue
                    
                    # Classify slot
                    slot_resized = cv2.resize(slot_region, (224, 224))
                    results = self.parking_model(slot_resized, verbose=False)
                    prediction = results[0].names[results[0].probs.top1]
                    confidence = results[0].probs.top1conf.item()
                    
                    if prediction == "empty":
                        empty_count += 1
                    else:
                        occupied_count += 1
                    
                    # Store slot info (from middle frame)
                    if i == len(sample_frames) // 2:
                        slot_analysis[f"Slot_{slot_id:02d}"] = {
                            'status': prediction,
                            'confidence': confidence,
                            'coords': (x1, y1, x2, y2)
                        }
            
            # Record timeline
            timestamp = frame_num / fps
            occupancy_rate = (occupied_count / total_slots) * 100
            
            occupancy_timeline.append({
                'time': timestamp,
                'empty_slots': empty_count,
                'occupied_slots': occupied_count,
                'occupancy_rate': occupancy_rate
            })
            
            # Update progress
            if progress_bar:
                progress = min(1.0, (i + 1) / len(sample_frames))
                progress_bar.progress(progress)
        
        cap.release()
        return slot_analysis, occupancy_timeline
    
    def process_videos_with_annotations(self, vehicle_video_path, parking_video_path):
        """Process videos and create annotated versions"""
        
        # Create output directory
        output_dir = Path("processed_videos")
        output_dir.mkdir(exist_ok=True)
        
        # Process vehicle video - get the actual output path returned
        vehicle_output_base = output_dir / f"annotated_vehicle_{int(time.time())}"
        vehicle_output_path = self.create_annotated_vehicle_video(vehicle_video_path, str(vehicle_output_base))
        
        # Process parking video - get the actual output path returned
        parking_output_base = output_dir / f"annotated_parking_{int(time.time())}"
        parking_output_path = self.create_annotated_parking_video(parking_video_path, str(parking_output_base))
        
        return vehicle_output_path, parking_output_path
    
    def create_annotated_vehicle_video(self, input_path, output_path):
        """Create annotated vehicle detection video with robust error handling"""
        
        cap = None
        out = None
        final_output_path = output_path
        
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                st.error(f"Could not open input video: {input_path}")
                return output_path
            
            # Get video properties with safe defaults
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
            
            # Ensure minimum dimensions
            if width <= 0 or height <= 0:
                width, height = 640, 480
            
            # Use WebM VP8 format for optimal web playback
            final_output_path = output_path.rsplit('.', 1)[0] + '.webm'
            fourcc = cv2.VideoWriter_fourcc(*'VP80')
            out = cv2.VideoWriter(final_output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                st.error(f"Could not create video writer for {final_output_path}")
                return output_path
            
            vehicle_count = 0
            tracked_vehicles = set()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                try:
                    # Protect YOLO inference with try/except
                    results = self.vehicle_model.track(
                        frame,
                        classes=[2, 7],
                        conf=0.4,
                        persist=True,
                        verbose=False
                    )
                    
                    # Draw annotations
                    if results[0].boxes is not None:
                        boxes = results[0].boxes.xywh.cpu()
                        if results[0].boxes.id is not None:
                            track_ids = results[0].boxes.id.int().cpu().tolist()
                            classes = results[0].boxes.cls.int().cpu().tolist()
                            
                            for box, track_id, cls in zip(boxes, track_ids, classes):
                                x, y, w, h = box
                                x1, y1, x2, y2 = int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)
                                
                                # Clip coordinates to frame size
                                x1 = max(0, min(x1, width-1))
                                y1 = max(0, min(y1, height-1))
                                x2 = max(0, min(x2, width-1))
                                y2 = max(0, min(y2, height-1))
                                
                                if track_id not in tracked_vehicles:
                                    tracked_vehicles.add(track_id)
                                    vehicle_count += 1
                                
                                # Draw bounding box
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                
                                # Draw label
                                vehicle_type = "Car" if cls == 2 else "Truck"
                                label = f"V{track_id:03d} ({vehicle_type})"
                                cv2.putText(frame, label, (x1, y1-10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                except Exception as e:
                    st.warning(f"Frame processing error: {e}")
                    # Continue with unprocessed frame
                
                # Draw statistics
                cv2.rectangle(frame, (10, 10), (300, 80), (0, 0, 0), -1)
                cv2.putText(frame, f"Total Vehicles: {vehicle_count}", (20, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Currently Visible: {len(results[0].boxes) if 'results' in locals() and results[0].boxes is not None else 0}", 
                           (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                out.write(frame)
            
        except Exception as e:
            st.error(f"Video processing error: {e}")
        
        finally:
            # Always release video resources
            if cap is not None:
                cap.release()
            if out is not None:
                out.release()
        
        return final_output_path
    
    def create_annotated_parking_video(self, input_path, output_path, grid_size=(6, 4)):
        """Create annotated parking space video with robust error handling"""
        
        cap = None
        out = None
        final_output_path = output_path
        
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                st.error(f"Could not open input video: {input_path}")
                return output_path
            
            # Get video properties with safe defaults
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
            
            # Ensure minimum dimensions
            if width <= 0 or height <= 0:
                width, height = 640, 480
            
            # Use WebM VP8 format for optimal web playback
            final_output_path = output_path.rsplit('.', 1)[0] + '.webm'
            fourcc = cv2.VideoWriter_fourcc(*'VP80')
            out = cv2.VideoWriter(final_output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                st.error(f"Could not create video writer for {final_output_path}")
                return output_path
            
            cols, rows = grid_size
            slot_width = width // cols
            slot_height = height // rows
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                empty_count = 0
                occupied_count = 0
                
                # Analyze and draw parking slots
                for row in range(rows):
                    for col in range(cols):
                        slot_id = row * cols + col + 1
                        
                        x1 = col * slot_width
                        y1 = row * slot_height
                        x2 = x1 + slot_width
                        y2 = y1 + slot_height
                        
                        # Clip slot coordinates to frame size
                        x1 = max(0, min(x1, width-1))
                        y1 = max(0, min(y1, height-1))
                        x2 = max(0, min(x2, width))
                        y2 = max(0, min(y2, height))
                        
                        if x2 <= x1 or y2 <= y1:
                            continue
                        
                        slot_region = frame[y1:y2, x1:x2]
                        
                        if slot_region.shape[0] < 10 or slot_region.shape[1] < 10:
                            continue
                        
                        try:
                            # Protect YOLO inference with try/except
                            slot_resized = cv2.resize(slot_region, (224, 224))
                            results = self.parking_model(slot_resized, verbose=False)
                            prediction = results[0].names[results[0].probs.top1]
                        except Exception as e:
                            # Default classification becomes "empty" when error occurs
                            prediction = "empty"
                            st.warning(f"Slot classification error: {e}")
                        
                        # Draw slot
                        if prediction == "empty":
                            color = (0, 255, 0)  # Green
                            empty_count += 1
                        else:
                            color = (0, 0, 255)  # Red
                            occupied_count += 1
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"S{slot_id}", (x1+5, y1+20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Draw statistics
                total_slots = cols * rows
                occupancy_rate = (occupied_count / total_slots) * 100 if total_slots > 0 else 0
                
                cv2.rectangle(frame, (10, 10), (300, 100), (0, 0, 0), -1)
                cv2.putText(frame, f"Empty: {empty_count}", (20, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Occupied: {occupied_count}", (20, 55), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, f"Occupancy: {occupancy_rate:.1f}%", (20, 75), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                out.write(frame)
            
        except Exception as e:
            st.error(f"Video processing error: {e}")
        
        finally:
            # Always release video resources
            if cap is not None:
                cap.release()
            if out is not None:
                out.release()
        
        return final_output_path

def main():
    """Main Streamlit app"""
    
    # Title and header
    st.title("🅿️ Smart Parking Analysis Dashboard")
    st.markdown("---")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = ParkingAnalyzer()
    
    # Sidebar for inputs
    st.sidebar.header("📹 Video Inputs")
    
    # File uploaders
    vehicle_video = st.sidebar.file_uploader(
        "Upload Incoming Vehicles Video",
        type=['mp4', 'avi', 'mov'],
        help="Video showing incoming vehicles (cars and trucks)"
    )
    
    parking_video = st.sidebar.file_uploader(
        "Upload Parking Lot Video", 
        type=['mp4', 'avi', 'mov'],
        help="Video showing the parking lot"
    )
    
    # Grid size configuration
    st.sidebar.header("⚙️ Configuration")
    cols = st.sidebar.slider("Parking Grid Columns", 3, 10, 6)
    rows = st.sidebar.slider("Parking Grid Rows", 2, 8, 4)
    grid_size = (cols, rows)
    
    # Show existing local videos
    st.sidebar.header("📁 Local Videos")
    videos_dir = Path("uploaded_videos")
    processed_dir = Path("processed_videos")
    
    if videos_dir.exists():
        uploaded_videos = list(videos_dir.glob("*.mp4"))
        if uploaded_videos:
            st.sidebar.write(f"📹 Uploaded: {len(uploaded_videos)} videos")
        else:
            st.sidebar.write("📹 No uploaded videos")
    
    if processed_dir.exists():
        processed_videos = list(processed_dir.glob("*.mp4"))
        if processed_videos:
            st.sidebar.write(f"🎬 Processed: {len(processed_videos)} videos")
        else:
            st.sidebar.write("🎬 No processed videos")
    
    # Process button
    process_button = st.sidebar.button("🚀 Analyze Videos", type="primary")
    
    # Main content area
    if vehicle_video and parking_video and process_button:
        
        # Create local videos directory if it doesn't exist
        videos_dir = Path("uploaded_videos")
        videos_dir.mkdir(exist_ok=True)
        
        # Save uploaded files locally with proper names
        vehicle_path = videos_dir / f"vehicle_video_{int(time.time())}.mp4"
        parking_path = videos_dir / f"parking_video_{int(time.time())}.mp4"
        
        # Write video files
        with open(vehicle_path, "wb") as f:
            f.write(vehicle_video.read())
        
        with open(parking_path, "wb") as f:
            f.write(parking_video.read())
        
        st.success(f"✅ Videos saved locally: {vehicle_path.name} and {parking_path.name}")
        
        # Convert to string paths for processing
        vehicle_path = str(vehicle_path)
        parking_path = str(parking_path)
        
        # Analysis section
        st.header("📊 Analysis Results")
        
        # Create columns for progress bars
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🚗 Analyzing Incoming Vehicles")
            vehicle_progress = st.progress(0)
        
        with col2:
            st.subheader("🅿️ Analyzing Parking Spaces")
            parking_progress = st.progress(0)
        
        # Perform analysis
        try:
            # Analyze vehicles
            total_vehicles, vehicle_timeline = st.session_state.analyzer.analyze_incoming_vehicles(
                vehicle_path, vehicle_progress
            )
            
            # Analyze parking
            slot_analysis, occupancy_timeline = st.session_state.analyzer.analyze_parking_spaces(
                parking_path, grid_size, parking_progress
            )
            
            # Calculate allocation results
            empty_slots = sum(1 for slot in slot_analysis.values() if slot['status'] == 'empty')
            total_slots = len(slot_analysis)
            can_allocate = min(total_vehicles, empty_slots)
            cannot_allocate = max(0, total_vehicles - empty_slots)
            
            # Display results
            st.markdown("---")
            st.header("📋 Vehicle Allocation Analysis")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🚗 Incoming Vehicles", total_vehicles)
            
            with col2:
                st.metric("🅿️ Available Spaces", empty_slots)
            
            with col3:
                st.metric("✅ Can Allocate", can_allocate, 
                         delta=f"{(can_allocate/total_vehicles)*100:.1f}% success rate")
            
            with col4:
                st.metric("❌ Cannot Allocate", cannot_allocate,
                         delta=f"{(cannot_allocate/total_vehicles)*100:.1f}% overflow")
            
            # Detailed results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Allocation Summary")
                
                if total_vehicles <= empty_slots:
                    st.success(f"✅ SUCCESS: All {total_vehicles} vehicles can be parked!")
                    st.info(f"🆓 Remaining spaces: {empty_slots - total_vehicles}")
                else:
                    st.error(f"⚠️ CAPACITY EXCEEDED: {cannot_allocate} vehicles need alternative parking")
                    st.warning(f"📊 Parking will be at 100% capacity")
                
                # Statistics table
                stats_data = {
                    'Metric': ['Total Vehicles', 'Available Spaces', 'Can Allocate', 'Cannot Allocate', 'Success Rate'],
                    'Value': [total_vehicles, empty_slots, can_allocate, cannot_allocate, f"{(can_allocate/total_vehicles)*100:.1f}%"]
                }
                st.table(pd.DataFrame(stats_data))
            
            with col2:
                st.subheader("🅿️ Parking Slot Status")
                
                # Parking slot visualization
                slot_data = []
                for slot_name, slot_info in slot_analysis.items():
                    slot_data.append({
                        'Slot': slot_name,
                        'Status': slot_info['status'].title(),
                        'Confidence': f"{slot_info['confidence']:.2f}"
                    })
                
                df_slots = pd.DataFrame(slot_data)
                
                # Color code the dataframe
                def color_status(val):
                    if val == 'Empty':
                        return 'background-color: lightgreen'
                    else:
                        return 'background-color: lightcoral'
                
                styled_df = df_slots.style.applymap(color_status, subset=['Status'])
                st.dataframe(styled_df, use_container_width=True)
            
            # Occupancy timeline chart
            if occupancy_timeline:
                st.subheader("📈 Parking Occupancy Over Time")
                
                df_occupancy = pd.DataFrame(occupancy_timeline)
                
                fig = px.line(df_occupancy, x='time', y='occupancy_rate',
                             title='Parking Lot Occupancy Rate Over Time',
                             labels={'time': 'Time (seconds)', 'occupancy_rate': 'Occupancy Rate (%)'})
                
                fig.add_hline(y=50, line_dash="dash", line_color="orange", 
                             annotation_text="50% Capacity")
                fig.add_hline(y=80, line_dash="dash", line_color="red", 
                             annotation_text="80% Capacity (High)")
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional occupancy metrics
                avg_occupancy = df_occupancy['occupancy_rate'].mean()
                max_occupancy = df_occupancy['occupancy_rate'].max()
                min_occupancy = df_occupancy['occupancy_rate'].min()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Average Occupancy", f"{avg_occupancy:.1f}%")
                with col2:
                    st.metric("📈 Peak Occupancy", f"{max_occupancy:.1f}%")
                with col3:
                    st.metric("📉 Minimum Occupancy", f"{min_occupancy:.1f}%")
            
            # Vehicle timeline chart
            if vehicle_timeline:
                st.subheader("🚗 Vehicle Detection Over Time")
                
                df_vehicles = pd.DataFrame(vehicle_timeline)
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=df_vehicles['time'], 
                    y=df_vehicles['vehicles'],
                    mode='lines+markers',
                    name='Currently Visible',
                    line=dict(color='blue')
                ))
                
                fig.add_trace(go.Scatter(
                    x=df_vehicles['time'], 
                    y=df_vehicles['cumulative_vehicles'],
                    mode='lines+markers',
                    name='Total Detected',
                    line=dict(color='green')
                ))
                
                fig.update_layout(
                    title='Vehicle Detection Timeline',
                    xaxis_title='Time (seconds)',
                    yaxis_title='Number of Vehicles',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Process and display annotated videos
            st.markdown("---")
            st.header("🎥 Processed Videos")
            
            with st.spinner("Creating annotated videos..."):
                vehicle_output, parking_output = st.session_state.analyzer.process_videos_with_annotations(
                    vehicle_path, parking_path
                )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🚗 Vehicle Detection Video")
                if os.path.exists(vehicle_output):
                    try:
                        with open(vehicle_output, 'rb') as video_file:
                            video_bytes = video_file.read()
                        st.video(video_bytes)
                    except Exception as e:
                        st.error(f"Could not load vehicle detection video: {e}")
                else:
                    st.error("Could not generate vehicle detection video")
            
            with col2:
                st.subheader("🅿️ Parking Space Detection Video")
                if os.path.exists(parking_output):
                    try:
                        with open(parking_output, 'rb') as video_file:
                            video_bytes = video_file.read()
                        st.video(video_bytes)
                    except Exception as e:
                        st.error(f"Could not load parking space video: {e}")
                else:
                    st.error("Could not generate parking space video")
            
            # Keep files for future reference (optional cleanup)
            st.info(f"📁 Files saved in: uploaded_videos/ and processed_videos/")
            
            # Optional: Add cleanup button
            if st.button("🗑️ Clean up temporary files"):
                try:
                    if os.path.exists(vehicle_path):
                        os.unlink(vehicle_path)
                    if os.path.exists(parking_path):
                        os.unlink(parking_path)
                    if os.path.exists(vehicle_output):
                        os.unlink(vehicle_output)
                    if os.path.exists(parking_output):
                        os.unlink(parking_output)
                    st.success("✅ Temporary files cleaned up!")
                except Exception as e:
                    st.warning(f"Could not clean up some files: {e}")
                
        except Exception as e:
            st.error(f"Error during analysis: {e}")
            st.exception(e)
    
    elif not vehicle_video or not parking_video:
        # Instructions
        st.info("👆 Please upload both videos in the sidebar to begin analysis")
        
        st.markdown("""
        ## 📋 How to Use This Dashboard
        
        1. **Upload Videos**: Use the sidebar to upload:
           - 🚗 **Incoming Vehicles Video**: Shows vehicles entering the area
           - 🅿️ **Parking Lot Video**: Shows the parking lot with spaces
        
        2. **Configure Settings**: Adjust the parking grid size to match your lot layout
        
        3. **Run Analysis**: Click "Analyze Videos" to process both videos
        
        4. **View Results**: The dashboard will show:
           - 📊 Vehicle allocation capacity analysis
           - 🅿️ Parking space availability
           - 📈 Occupancy trends over time
           - 🎥 Annotated videos with detections
        
        ## 🎯 Features
        
        - **Real-time Analysis**: Process videos to detect vehicles and parking spaces
        - **Allocation Logic**: Determine how many vehicles can be accommodated
        - **Visual Dashboard**: Interactive charts and metrics
        - **Annotated Videos**: See detection results overlaid on original videos
        - **Detailed Reports**: Comprehensive analysis with success rates and utilization
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit, YOLO, and OpenCV")

if __name__ == "__main__":
    main()