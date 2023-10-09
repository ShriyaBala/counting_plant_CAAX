import cv2
import argparse
from ultralytics import YOLO


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
        rendered_frame = result.render()

        # Display the rendered frame in the Windows terminal
        print(rendered_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
