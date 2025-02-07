import cv2
import os
import base64
import requests
import json
from ultralytics import YOLO
from datetime import datetime

# Load the trained YOLO model
model_path = r"./best_model.pt"  # Update with your actual model path
model = YOLO(model_path)

# API Endpoint (Replace with your actual MongoDB backend URL)
url = "http://localhost:3000/api/emergency"

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)
    pothole_detected = False  

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0]  # Confidence score
            cls = int(box.cls[0])  # Class index

            if conf > 0.5:  # High-confidence detections only
                label = f"Pothole ({conf:.2f})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                pothole_detected = True  

    if pothole_detected:
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = f"pothole_{timestamp}.jpg"
        cv2.imwrite(image_path, frame)

        # Convert image to Base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        # Prepare API payload
        payload = json.dumps({
            "latitude": 34.052235,  # Replace with actual GPS data if available
            "longitude": -118.243683,
            "cameraId": "CAM002",
            "impact": "Medium",
            "image": f"data:image/jpeg;base64,{base64_image}"
        })

        # API headers
        headers = {'Content-Type': 'application/json'}

        # Send request to API
        response = requests.post(url, headers=headers, data=payload)

        print(f"Saved pothole image: {image_path}")
        print(f"Server Response: {response.text}")

    # Show webcam feed
    cv2.imshow("Pothole Detection", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
