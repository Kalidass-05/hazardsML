import cv2
import os
from ultralytics import YOLO
from datetime import datetime

# Load the trained model
model_path = r"C:/Users/lathi/OneDrive/Desktop/hazard detection/runs/detect/train5/weights/best.pt" # Update with your actual model path
model = YOLO(model_path)

# Create a folder to save pothole images
save_folder = "detected_potholes"
os.makedirs(save_folder, exist_ok=True)

# Open the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Flag to check if pothole is detected
    pothole_detected = False  

    # Draw bounding boxes on the frame
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            conf = box.conf[0]  # Confidence score
            cls = int(box.cls[0])  # Class index

            if conf > 0.5:  # Only show high-confidence detections
                label = f"Pothole ({conf:.2f})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red bounding box
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                pothole_detected = True  # Set flag to True

    # If a pothole is detected, save the image
    if pothole_detected:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Generate a unique timestamp
        image_path = os.path.join(save_folder, f"pothole_{timestamp}.jpg")  
        cv2.imwrite(image_path, frame)  # Save the image
        print(f"Saved pothole image: {image_path}")

    # Show the webcam feed with detections
    cv2.imshow("Pothole Detection", frame)

    # Press 'q' to exit the webcam feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
