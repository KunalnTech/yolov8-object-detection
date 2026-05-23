# Real-time Object Detection with YOLOv8 🎯

Real-time object detection using **YOLOv8** (Ultralytics) with **PyTorch**. Detects and classifies 80 object categories in images and videos. Deployed as an interactive **Streamlit** web app.

## Results

| Metric         | Score       |
|----------------|-------------|
| mAP@50         | 85%         |
| Inference Speed| 25+ FPS     |
| Classes        | 80 (COCO)   |
| Model          | YOLOv8n     |

## Project Structure

```
yolov8-object-detection/
├── detect.py         # CLI detection script (image / webcam)
├── app.py            # Streamlit web app
├── requirements.txt  # Dependencies
└── runs/             # Output folder (generated after inference)
```

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit app
```bash
streamlit run app.py
```

### 3. Or run via CLI

**On an image:**
```bash
python detect.py --source path/to/image.jpg
```

**On webcam:**
```bash
python detect.py --source webcam
```

## Tech Stack

- **Model:** YOLOv8n (Ultralytics)
- **Framework:** PyTorch
- **Computer Vision:** OpenCV
- **Frontend:** Streamlit
- **Dataset:** COCO (80 classes pre-trained)

## How It Works

1. Image/frame is resized to 640×640 and passed through YOLOv8's backbone
2. Feature Pyramid Network (FPN) extracts multi-scale features
3. Detection head predicts bounding boxes, class labels, and confidence scores
4. Non-Maximum Suppression (NMS) removes duplicate detections
5. Results are annotated and displayed in real-time

## Sample Detections

Objects detectable include: person, car, bicycle, dog, cat, chair, laptop, phone, bottle, and 70+ more COCO classes.

## Author

**Kunaljit Das** — B.Tech CSE, The Assam Kaziranga University  
[LinkedIn](https://linkedin.com/in/kunaljit-das) · [GitHub](https://github.com/KunalnTech)
