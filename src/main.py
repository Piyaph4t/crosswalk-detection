import os
from ultralytics import YOLO

# Define the path to your video file. Assuming it's in the /content/ directory.
video_path = 'test.mp4'

# Perform inference on the video.
# 'save=True' ensures the annotated video is saved to disk.
# 'stream=True' processes the video frame by frame for better handling.
model_ncnn = YOLO("../assets/model/yolo26n_ncnn_model")
results = model_ncnn.predict(source=video_path, save=True, stream=True)

