import cv2
import argparse
from ultralytics import YOLO
import numpy as np


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 on video")
    parser.add_argument(
        "--video-path",
        required=True,
        help="Path to the video file"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    cap = cv2.VideoCapture(args.video_path)  # Load video file

    model = YOLO("yolov8l.pt")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        result = model(frame, agnostic_nms=True)[0]

        # Extract bounding boxes and labels
        boxes = result.xyxy[0].cpu().numpy()
        confidences = result.pred[0][:, 4].cpu().numpy()
        class_ids = result.pred[0][:, 5].cpu().numpy()

        for i in range(len(boxes)):
            box = boxes[i]
            x, y, w, h = map(int, box[:4])
            class_id = int(class_ids[i])
            confidence = confidences[i]

            label = f"{model.names[class_id]} {confidence:.2f}"
            color = (0, 255, 0)  # Green color for bounding boxes

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x, y), (w, h), color, 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the frame with bounding boxes and labels
        cv2.imshow("YOLOv8 Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
