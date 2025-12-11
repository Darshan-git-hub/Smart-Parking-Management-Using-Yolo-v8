from ultralytics import YOLO
import torch

if __name__ == '__main__':
    # Limit GPU memory usage to 2GB
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.5)  # Adjust based on your GPU memory
        torch.cuda.empty_cache()

    model = YOLO("yolov8s.pt")  # pretrained model

    # Train the model
    results = model.train(
        data="idd_lite.yaml",
        epochs=10,
        imgsz=416,  # Smaller image size to use less memory
        batch=4,   # Further reduced batch size
        device=0 if torch.cuda.is_available() else 'cpu',  # Use GPU if available
        workers=0  # Disable multiprocessing on Windows
    )

    # Save the trained model
    model.save("models/best.pt")
    print("Model saved to models/best.pt")