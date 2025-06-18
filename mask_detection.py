import cv2
from ultralytics import YOLO

# Load models with improved weights
person_model = YOLO('yolov8n.pt')  # Pre-trained person detection
mask_model = YOLO('runs/detect/mask_detection_improved/weights/best.pt')  # Improved mask model
helmet_model = YOLO('runs/detect/helmet_detection_improved/weights/best.pt')  # Improved helmet model

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Person detection
    person_results = person_model(frame)
    for result in person_results:
        boxes = result.boxes
        for box in boxes:
            if int(box.cls) == 0:  # Person class
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf.item()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for person
                cv2.putText(frame, f'Person: {confidence:.2f}', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    # Mask detection
    mask_results = mask_model(frame)
    for result in mask_results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls)
            label = mask_model.names[class_id]  # e.g., 'with_mask', 'without_mask'
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf.item()
            color = (255, 0, 0) if label == 'with_mask' else (0, 0, 255)  # Blue for mask, red for no-mask
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{label}: {confidence:.2f}', (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

    # Helmet detection
    helmet_results = helmet_model(frame)
    for result in helmet_results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls)
            label = helmet_model.names[class_id]  # e.g., 'helmet'
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf.item()
            color = (0, 255, 255)  # Yellow for helmet
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{label}: {confidence:.2f}', (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

    # Display frame
    cv2.imshow('Person, Mask, and Helmet Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()