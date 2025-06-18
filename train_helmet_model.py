from ultralytics import YOLO

def train_model():
    # Load the YOLOv8 nano model (pre-trained on COCO)
    model = YOLO('yolov8n.pt')

    model.train(
    data='helmet_data.yaml',
    epochs=50,  # Increased epochs
    imgsz=640,
    batch=8,
    name='helmet_detection_improved',
    device=0
)
    print("Training complete. Check 'runs/detect/helmet_detection' for results.")

if __name__ == '__main__':
    train_model()