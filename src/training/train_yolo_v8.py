from ultralytics import YOLO

def train():
    # Load the base model
    model = YOLO("yolov8n.pt")

    # Train the model
    results = model.train(
        data="dataset.yaml",
        epochs=10,
        imgsz=[750, 458], # Tip: Reduce this (e.g., imgsz=320) for significant speedups
        batch=16,
        name="fruit_ninja_v1",
        device='cpu',  # Tip: Change to 0 (or [0, 1] for multiple GPUs) if you have an NVIDIA GPU
        workers=8,     # Parallelize data loading across CPU cores
        cache=True     # Cache images in RAM to speed up training (requires sufficient RAM)
    )

if __name__ == "__main__":
    train()
