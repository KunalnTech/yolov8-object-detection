"""
Real-time Object Detection with YOLOv8
Detects and classifies objects in images, video files, or live webcam feed.
"""

from ultralytics import YOLO
import cv2
import argparse

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME  = "yolov8n.pt"   # nano model — fast, good for laptops
CONF_THRESH = 0.4            # confidence threshold (0-1)
IMG_SIZE    = 640            # inference image size

def load_model():
    print(f"Loading YOLOv8 model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)
    print("Model loaded successfully!")
    return model

def detect_image(model, image_path, save=True):
    """Run detection on a single image."""
    results = model.predict(
        source=image_path,
        conf=CONF_THRESH,
        imgsz=IMG_SIZE,
        save=save,
    )
    for result in results:
        boxes = result.boxes
        print(f"\nDetected {len(boxes)} objects:")
        for box in boxes:
            cls_id     = int(box.cls[0])
            cls_name   = model.names[cls_id]
            confidence = float(box.conf[0])
            print(f"  {cls_name}: {confidence:.1%}")
    return results

def detect_webcam(model):
    """Run real-time detection on webcam feed."""
    print("Starting webcam... Press 'q' to quit.")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            source=frame,
            conf=CONF_THRESH,
            imgsz=IMG_SIZE,
            verbose=False,
        )

        # Draw bounding boxes
        annotated = results[0].plot()

        # Show FPS
        cv2.putText(
            annotated, "YOLOv8 | Press Q to quit",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 0), 2
        )

        cv2.imshow("YOLOv8 Object Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Object Detection")
    parser.add_argument("--source", type=str, default="webcam",
                        help="'webcam' or path to image/video file")
    args = parser.parse_args()

    model = load_model()

    if args.source == "webcam":
        detect_webcam(model)
    else:
        detect_image(model, args.source)
