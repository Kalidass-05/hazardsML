import cv2
import os
import base64
import json
import pymongo  # MongoDB client
from ultralytics import YOLO
from datetime import datetime
from pymongo import MongoClient

# MongoDB Atlas Connection
MONGO_URI = "mongodb+srv://jv8110909191:ASas12.,@cluster0.qsdf4.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DB_NAME = "test"
COLLECTION_NAME = "hazards"

# Connect to MongoDB Atlas
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Load the trained model
model_path = r"./best_model.pt"  # Update with your actual model path
model = YOLO(model_path)

# Open the webcam
cap = cv2.VideoCapture(0)

# ✅ Check if camera is detected
if not cap.isOpened():
    print("❌ ERROR: Camera not detected! Check if it's in use by another app.")
    exit()

print("✅ Camera opened successfully!")

captured_images = 0  # Counter to track the number of images captured

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ ERROR: Frame capture failed. Restarting camera...")
        cap.release()
        cap = cv2.VideoCapture(0)
        continue

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

    # If a pothole is detected and we haven't captured 2 images yet
    if pothole_detected and captured_images < 2:
        timestamp = datetime.utcnow()  # Store timestamp in UTC
        image_filename = f"pothole_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"

        # Convert image to Base64
        _, buffer = cv2.imencode(".jpg", frame)
        base64_image = base64.b64encode(buffer).decode('utf-8')

        # Prepare MongoDB document
        pothole_data = {
            "latitude": 10.9390,  # Default to 0
            "longitude": 76.9522,  # Default to 0
            "cameraId": "CAM002",
            "image": f"data:image/jpeg;base64,{base64_image}",
            "status": "not-responded",  # Default status
            "time": timestamp  # MongoDB Date format
        }

        # Insert into MongoDB Atlas
        try:
            collection.insert_one(pothole_data)
            print(f"✅ Pothole data saved in MongoDB Atlas: {pothole_data}")
        except Exception as e:
            print(f"❌ ERROR: Failed to insert into MongoDB: {e}")

        captured_images += 1  # Increment counter

    # Show the webcam feed with detections
    cv2.imshow("Pothole Detection", frame)

    # Break the loop after capturing 2 images
    if captured_images >= 2:
        print("✅ Captured 2 images. Exiting...")
        break

    # Press 'q' to exit manually
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()