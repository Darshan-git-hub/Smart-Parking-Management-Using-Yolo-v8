from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO("models/best.pt")

# Load a single frame from the video
cap = cv2.VideoCapture("videos/input.mp4")
ret, frame = cap.read()

if ret:
    print(f"Frame shape: {frame.shape}")
    
    # Try detection with very low confidence
    results = model(frame, conf=0.01, verbose=True)
    
    print(f"Number of detections: {len(results[0].boxes) if results[0].boxes is not None else 0}")
    
    if results[0].boxes is not None:
        for i, box in enumerate(results[0].boxes):
            print(f"Detection {i}: confidence={box.conf.item():.3f}, class={box.cls.item()}")
    
    # Save annotated image
    annotated = results[0].plot()
    cv2.imwrite("test_detection.jpg", annotated)
    print("Test detection saved as test_detection.jpg")
    
else:
    print("Could not read frame from video")

cap.release()