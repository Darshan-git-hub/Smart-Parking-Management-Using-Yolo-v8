from ultralytics import YOLO
import torch
import os
from pathlib import Path

if __name__ == '__main__':
    # Limit GPU memory usage
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.5)
        torch.cuda.empty_cache()
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Load YOLOv8 classification model
    model = YOLO("yolov8s-cls.pt")  # YOLOv8 small classification model
    
    print("Starting parking space classification training...")
    print("Dataset structure:")
    print("- Empty spaces: parking dataset/train/empty/")
    print("- Occupied spaces: parking dataset/train/occupied/")
    
    # Count dataset samples
    train_empty = len(list(Path("parking dataset/train/empty").glob("*")))
    train_occupied = len(list(Path("parking dataset/train/occupied").glob("*")))
    test_empty = len(list(Path("parking dataset/test/empty").glob("*")))
    test_occupied = len(list(Path("parking dataset/test/occupied").glob("*")))
    
    print(f"\nDataset Statistics:")
    print(f"Training - Empty: {train_empty}, Occupied: {train_occupied}")
    print(f"Testing - Empty: {test_empty}, Occupied: {test_occupied}")
    print(f"Total training samples: {train_empty + train_occupied}")
    
    # Train the model
    results = model.train(
        data="parking dataset",  # Path to dataset root
        epochs=20,               # Number of training epochs
        imgsz=224,              # Image size for classification
        batch=16,               # Batch size
        device=0 if torch.cuda.is_available() else 'cpu',
        workers=0,              # Disable multiprocessing on Windows
        patience=5,             # Early stopping patience
        save=True,              # Save checkpoints
        plots=True,             # Generate training plots
        val=True,               # Validate during training
        project="runs/classify", # Project directory
        name="parking_spaces"   # Experiment name
    )
    
    # Save the trained model to models directory
    model_save_path = "models/parking_classifier.pt"
    model.save(model_save_path)
    print(f"\nModel saved to: {model_save_path}")
    
    # Print training results
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {results.results_dict.get('metrics/accuracy_top1', 'N/A')}")
    
    # Test the model on a sample image
    print("\nTesting model on sample images...")
    
    # Test on empty space
    empty_sample = list(Path("parking dataset/test/empty").glob("*"))[0]
    empty_result = model(str(empty_sample))
    print(f"Empty space prediction: {empty_result[0].names[empty_result[0].probs.top1]} "
          f"(confidence: {empty_result[0].probs.top1conf:.3f})")
    
    # Test on occupied space  
    occupied_sample = list(Path("parking dataset/test/occupied").glob("*"))[0]
    occupied_result = model(str(occupied_sample))
    print(f"Occupied space prediction: {occupied_result[0].names[occupied_result[0].probs.top1]} "
          f"(confidence: {occupied_result[0].probs.top1conf:.3f})")
    
    print(f"\nModel ready for use! Load it with: YOLO('{model_save_path}')")