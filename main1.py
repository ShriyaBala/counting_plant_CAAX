import cv2
import numpy as np

# Load YOLOv3 model
net = cv2.dnn.darkNet('yolov3.weights', 'yolov3.cfg')

# Load the custom class labels for crop plants
with open('crop_plant.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Load the image
image = cv2.imread('crop_image.jpg')
height, width = image.shape[:2]

# Prepare the input blob for YOLO model
blob = cv2.dnn.blobFromImage(
    image, 1/255.0, (416, 416), swapRB=True, crop=False)

# Set the input blob to the network
net.setInput(blob)

# Get the output layer names
layer_names = net.getUnconnectedOutLayersNames()

# Run forward pass to get detections
detections = net.forward(layer_names)

# Initialize variables for counting crop plants
crop_plants_count = 0

# Loop through the detections
for detection in detections:
    for obj in detection:
        scores = obj[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5:  # You can adjust this confidence threshold as needed
            # Use the correct class name for crop plants
            if classes[class_id] == 'crop_plant':
                crop_plants_count += 1

# Print the count of crop plants
print("Number of crop plants in the image:", crop_plants_count)

# Display the image with bounding boxes (optional)
for detection in detections:
    for obj in detection:
        scores = obj[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5:  # You can adjust this confidence threshold as needed
            # Use the correct class name for crop plants
            if classes[class_id] == 'crop_plant':
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)

                # Calculate bounding box coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Draw bounding box
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image with bounding boxes (optional)
cv2.imshow('Crop Plants Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
