from ultralytics import YOLO
import cv2

if __name__ == '__main__':
    # Load pretrained YOLOv8 model (since our trained model has no labels)
    model = YOLO("yolov8s.pt")
    
    # Open video file
    cap = cv2.VideoCapture("videos/input.mp4")
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        exit()
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties: {width}x{height} at {fps} FPS")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("videos/output.mp4", fourcc, fps, (width, height))
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run detection (filter for vehicle classes: car, truck, bus, motorcycle)
        results = model(frame, conf=0.3, classes=[2, 3, 5, 7])  # COCO classes for vehicles
        annotated = results[0].plot()
        
        # Count detections
        detections = len(results[0].boxes) if results[0].boxes is not None else 0
        
        # Add frame info
        cv2.putText(annotated, f"Frame: {frame_count} | Detections: {detections}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display and save
        cv2.imshow("Vehicle Detection", annotated)
        out.write(annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Detection stopped by user")
            break
    
    print(f"Processed {frame_count} frames")
    print("Output saved to videos/output.mp4")
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
