from ultralytics import YOLO

model = YOLO('yolo11s.pt')  # Use nano model for faster training

# Training configuration
results = model.train(
    data='/home/nocal/ted/RVSS_Sunday_Drivers/scripts/stopsigndata3/data.yaml',  # Your dataset config file from Roboflow
    epochs=50,                 # Start with 50 epochs
    imgsz=120,                # Image size
    batch=64,                 # Batch size
    patience=10,              # Early stopping patience
    device='cuda',            # Use GPU if available
    workers=8,                # Number of worker threads
)
