import gradio as gr
import cv2, os
from modules.detector import YOLOv8FaceDetector
from modules.recognizer import ArcFaceRecognizer
from modules.logger import FaceLogger
from modules.utils import annotate_frame

def launch_app(config):
    detector = YOLOv8FaceDetector(
        model_path=config["paths"]["yolov8_model"],
        conf_thres=config["system"]["confidence_threshold"],
        use_gpu=config["system"]["use_gpu"]
    )
    recognizer = ArcFaceRecognizer(
        model_path=config["paths"]["arcface_model"],
        db_path=config["paths"]["face_db"],
        thres=config["system"]["recognition_threshold"],
        use_gpu=config["system"]["use_gpu"]
    )
    logger = FaceLogger()

    feeds = {}

    def add_feed(name, path):
        feeds[name] = path
        return list(feeds.keys())

    def upload_video(video_file):
        name = os.path.basename(video_file.name)
        feeds[name] = video_file.name
        return list(feeds.keys())

    def process_feed(feed_name):
        if feed_name not in feeds:
            return None
        cap = cv2.VideoCapture(feeds[feed_name])
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            detections = detector.detect(frame)
            for bbox in detections:
                emb, name = recognizer.recognize(frame, bbox)
                if emb is not None:
                    logger.log(name)
                frame = annotate_frame(frame, bbox, name)
            yield frame[:, :, ::-1]

    with gr.Blocks() as demo:
        gr.Markdown("## 🎥 Face Recognition System")

        with gr.Row():
            feed_dropdown = gr.Dropdown(choices=[], label="Available Feeds")
            add_name = gr.Textbox(label="Feed Name")
            add_path = gr.Textbox(label="Feed Path / RTSP")
            add_btn = gr.Button("➕ Add Feed")
            upload_btn = gr.File(label="Upload Video", type="file")

        video_out = gr.Image(label="Output", type="numpy")

        add_btn.click(add_feed, inputs=[add_name, add_path], outputs=feed_dropdown)
        upload_btn.upload(upload_video, inputs=upload_btn, outputs=feed_dropdown)
        feed_dropdown.change(process_feed, inputs=feed_dropdown, outputs=video_out)

    demo.launch(share=True)
