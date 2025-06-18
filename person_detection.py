import cv2
from ultralytics import YOLO

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # Using the nano model for speed

# Open the webcam (default is 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform inference on the frame
    results = model(frame)

    # Filter for person detections (class ID 0 is 'person' in YOLO)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            if int(box.cls) == 0:  # Check if the detected object is a person
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf.item()
                # Draw rectangle and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'Person: {confidence:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with detections
    cv2.imshow('Person Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()