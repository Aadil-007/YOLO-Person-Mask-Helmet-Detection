from ultralytics import YOLO

def train_model():
    model = YOLO('yolov8n.pt')
    model.train(
        data='data.yaml',  # Your mask dataset config
        epochs=50,  # Increased epochs
        imgsz=640,
        batch=8,
        name='mask_detection_improved',
        device=0  # GPU
    )
    print("Training complete. Check 'runs/detect/mask_detection_improved' for results.")

if __name__ == '__main__':
    train_model()