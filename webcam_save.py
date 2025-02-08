import cv2
import os
import base64
import requests
import json
from ultralytics import YOLO
from datetime import datetime

# Use a smaller model for better performance
model_path = "/home/pi/ML_models/yolov8n.pt"
model = YOLO(model_path)

# API Endpoint (Replace with your actual database API)
url = "http://localhost:3000/api/hazards"

# Create a folder to save pothole images
save_folder = "detected_potholes"
os.makedirs(save_folder, exist_ok=True)

# Set up OpenCV
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  

# ✅ Check if camera is detected
if not cap.isOpened():
    print("❌ ERROR: Camera not detected! Check if it's in use by another app.")
    exit()

print("✅ Camera opened successfully!")

frame_skip = 2  # Process every 2nd frame
frame_count = 0
captured_images = 0  # Counter to track the number of images captured

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ ERROR: Frame capture failed.")
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # Skip frames to improve speed

    results = model(frame)  # Run detection

    # Flag to check if pothole is detected
    pothole_detected = False  

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            conf = box.conf[0]  # Confidence score
            cls = int(box.cls[0])  # Class index

            if conf > 0.6:  # Only store high-confidence detections
                label = f"Pothole ({conf:.2f})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red bounding box
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                pothole_detected = True

    # If a pothole is detected and we haven't captured 2 images yet
    if pothole_detected and captured_images < 2:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Generate a unique timestamp
        image_path = os.path.join(save_folder, f"pothole_{timestamp}.jpg")  
        cv2.imwrite(image_path, frame)  # Save the image
        captured_images += 1  # Increment counter
        print(f"✅ Saved pothole image ({captured_images}/2): {image_path}")

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
        try:
            response = requests.post(url, headers=headers, data=payload)
            print(f"📡 Server Response: {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"❌ ERROR: Failed to send data to backend: {e}")
    
    # Show the webcam feed with detections
    cv2.imshow("Pothole Detection", frame)

    # Break the loop after capturing 2 images
    if captured_images >= 2:
        print("✅ Captured 2 images. Exiting...")
        break

    # Press 'q' to exit manually
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()