import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

# Define the path to the image you want to analyze
image_path = "images\caax4.jpeg"  # Replace with your image path


def main():
    # Load the image
    frame = cv2.imread(image_path)

    model = YOLO("yolov8n.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    result = model(frame, agnostic_nms=True)[0]
    detections = sv.Detections.from_yolov8(result)
    labels = [
        f"{model.model.names[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, _
        in detections
    ]
    frame = box_annotator.annotate(
        scene=frame,
        detections=detections,
        labels=labels
    )

    cv2.imshow("yolov8", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
