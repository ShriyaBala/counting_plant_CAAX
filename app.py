import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

import supervision as sv
import main  # Assuming 'main.py' is your main script containing the 'main()' function
def main_app():
    st.title("Object Detection App")

    # Create a sidebar to select the model
    model_choice = st.sidebar.radio("Select Model", ("YOLOv8", "YOLOv3"))

    if model_choice == "YOLOv8":
        st.title("YOLOv8 Object Detection")
        # Add code for YOLOv8 model here
        ZONE_POLYGON = np.array([
            [0, 0],
            [0.5, 0],
            [0.5, 1],
            [0, 1]
        ])

        uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "mp4"])

        if uploaded_file is not None:
            frame = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
            model = YOLO("yolov8n.pt")

            # Perform object detection
          
            result = model(frame, agnostic_nms=True)[0]
            detections = sv.Detections.from_yolov8(result)

            # Annotate the frame with bounding boxes
            box_annotator = sv.BoxAnnotator(
                thickness=2,
                text_thickness=2,
                text_scale=1
            )

            labels = [
        f"{model.model.names[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, _
        in detections
    ]

            annotated_frame = box_annotator.annotate(
                scene=frame,
                detections=detections,
                labels=labels
            )

            # Display the annotated frame
            st.image(annotated_frame, channels="BGR")

    elif model_choice == "YOLOv3":
        st.title("YOLOv3 Object Detection")
        # Add code for YOLOv3 model here
        net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

        # Load the custom class labels for crop plants
        with open('crop_plant.names', 'r') as f:
            classes = f.read().strip().split('\n')

        uploaded_file = st.file_uploader("Upload an image (jpg)", type=["jpg"])

        if uploaded_file is not None:
            image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)

            # Prepare the input blob for YOLO model
            blob = cv2.dnn.blobFromImage(
                image, 1/255.0, (416, 416), swapRB=True, crop=False
            )

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

            # Annotate the image with bounding boxes (optional)
            for detection in detections:
                for obj in detection:
                    scores = obj[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > 0.5:  # You can adjust this confidence threshold as needed
                        # Use the correct class name for crop plants
                        if classes[class_id] == 'crop_plant':
                            center_x = int(obj[0] * image.shape[1])
                            center_y = int(obj[1] * image.shape[0])
                            w = int(obj[2] * image.shape[1])
                            h = int(obj[3] * image.shape[0])

                            # Calculate bounding box coordinates
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)

                            # Draw bounding box
                            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the image with bounding boxes (optional)
            st.image(image, channels="BGR")
            st.write("Number of crop plants in the image:", crop_plants_count)

if __name__ == "__main__":
    main_app()
