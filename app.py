 
from flask import Flask, Response, render_template, request,jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import cv2 as cv
import numpy as np
import random
from ultralytics import YOLO
import os
import io
import time
import threading
from main import FindLaneLines  # Import for lane and curve detection
from gtts import gTTS  # Google Text-to-Speech
import pygame  # Library to play the sound

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Initialize Pygame for playing audio
pygame.mixer.init()

# Load class names for object detection
base_dir = r"C:\Users\de_jayson\Desktop\mini project\Ad_LaneCurve detection\models\utils"
file_path = os.path.join(base_dir, "coco.txt")

if os.path.exists(file_path):
    with open(file_path, "r") as f:
        class_list = f.read().splitlines()
else:
    raise FileNotFoundError(f"coco.txt not found at {file_path}")


# Generate random colors for each class
detection_colors = [tuple(random.randint(0, 255) for _ in range(3)) for _ in range(len(class_list))]

# Load a pretrained YOLOv8 model
model = YOLO("weights/yolov8n.pt", "v8")

# Initialize the FindLaneLines class for lane detection
findLaneLines = FindLaneLines()

# Global variables to store the selected video source and mode
video_source = 0  # Default to live camera
detection_mode = 'lane'  # Default to lane detection

def play_audio_async(text):
    """
    Generate and play audio feedback in a separate thread.
    Prevents blocking the main video loop.
    """
    try:
        tts = gTTS(text=text, lang='en')
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)

        pygame.mixer.music.load(audio_fp, 'mp3')
        pygame.mixer.music.play()
    except Exception as e:
        print(f"Audio playback error: {e}")


def generate_frames():
    cap = cv.VideoCapture(video_source)
    if not cap.isOpened():
        print("Cannot open video source")
        return

    last_play_time = time.time()
    frame_count = 0  # for throttling object detection

    feedback_messages = [
        "Good lane Keeping",
        "You're doing great, stay focused",
        "Bad lane keeping",
        "Excellent lane control, well done",
        "You almost drift off, good you're back on track",
        "Nice driving! Stay alert and keep up the good work",
        "Your lane keeping is on point, keep going!",
        "Watch out, you are getting too close to the lane boundary",
        "You're swerving, try to maintain a straighter line",
        "Caution! you're not staying centered in the lane"
    ]

    while True:
        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("End of video reached, stopping...")
                cap.set(cv.CAP_PROP_POS_FRAMES, 0)  # restart video from beginning
                continue
                #exit loop instead of spamming errors

            processed_frame = frame

            if detection_mode == 'lane':
                processed_frame = findLaneLines.forward(frame)

                if time.time() - last_play_time >= 5:  # every 5 sec
                    lane_feedback = random.choice(feedback_messages)
                    threading.Thread(
                        target=play_audio_async,
                        args=(lane_feedback,),
                        daemon=True
                    ).start()
                    last_play_time = time.time()

            elif detection_mode == 'object':
                # Run detection every 3rd frame for speed
                if frame_count % 3 == 0:
                    detect_param = model.predict(source=[frame], conf=0.45, save=False)
                    DP = detect_param[0].numpy()

                    if len(DP) != 0:
                        for i in range(len(detect_param[0])):
                            boxes = detect_param[0].boxes
                            box = boxes[i]
                            clsID = box.cls.numpy()[0]
                            conf = box.conf.numpy()[0]
                            bb = box.xyxy.numpy()[0]

                            cv.rectangle(
                                frame,
                                (int(bb[0]), int(bb[1])),
                                (int(bb[2]), int(bb[3])),
                                detection_colors[int(clsID)],
                                3,
                            )

                            cv.putText(
                                frame,
                                class_list[int(clsID)],
                                (int(bb[0]), int(bb[1]) - 10),
                                cv.FONT_HERSHEY_COMPLEX,
                                1,
                                (255, 255, 255),
                                2,
                            )

                        processed_frame = frame

            frame_count += 1

            # Encode and yield
            ret, buffer = cv.imencode('.jpg', processed_frame)
            if not ret:
                continue

            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except Exception as e:
            print(f"Error in generate_frames loop: {e}")
            continue  # keep streaming alive even after errors

    cap.release()


@app.route('/')
def index():
    return render_template('home.html')

@app.route('/detect')
def detect():
    return render_template('detect.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/object_detection')
def object_detection():
    return render_template('object.html')

@app.route('/lane')
def lane_detection():
    return render_template('lane.html')

@app.route('/performance')
def performance():
    return {"status": "ok", "fps": 24}  # replace with real values later


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_source', methods=['POST'])
def set_source():
    try:
        source = request.form.get('source')
        mode = request.form.get('mode')

        if 'video' in request.files:
            file = request.files['video']
            filename = secure_filename(file.filename)
            filepath = os.path.join("uploads", filename)
            file.save(filepath)

            # Set video source
            global video_source, detection_mode
            video_source = filepath
            detection_mode = mode or 'lane'

            return jsonify(success=True, message="Video uploaded and processing started!")
        else:
            return jsonify(success=False, error="No video uploaded")
    except Exception as e:
        return jsonify(success=False, error=str(e))


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)


# Trial on  both lane and performance
# import random  # For randomly selecting feedback

# def generate_frames():
#     cap = cv.VideoCapture(video_source)
#     if not cap.isOpened():
#         print("Cannot open video source")
#         return

#     # Variable to track the last time audio was played
#     last_play_time = time.time()  # Initialize with the current time

#     # List of dynamic feedback messages
#     feedback_messages = [
#         "Good lane keeping, keep it up!",
#         "You're doing great, stay focused!",
#         "Excellent lane control, well done!",
#         "Keep driving safe and stay in your lane!",
#         "Your lane keeping is on point, keep going!",
#         "Nice driving! Stay alert and keep up the good work!"
#     ]

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         processed_frame = frame

#         if detection_mode == 'lane':
#             # Process the frame for lane detection
#             processed_frame = findLaneLines.forward(frame)

#             # Determine if the lane is straight or curved
#             if findLaneLines.is_straight_lane:
#                 lane_feedback = "Straight lane detected, keep driving steady!"
#             else:
#                 lane_feedback = "Curve ahead, adjust your steering carefully!"

#             # Randomly select a feedback message for performance
#             performance_feedback = random.choice(feedback_messages)

#             # Combine the lane detection feedback and performance feedback
#             full_feedback = f"{lane_feedback} {performance_feedback}"

#             # Check if 5 seconds have passed since the last audio play
#             if time.time() - last_play_time >= 5:  # 5 seconds delay
#                 try:
#                     # Generate speech audio in-memory
#                     tts = gTTS(text=full_feedback, lang='en')
#                     audio_fp = io.BytesIO()
#                     tts.write_to_fp(audio_fp)
#                     audio_fp.seek(0)

#                     # Play the audio directly from memory
#                     pygame.mixer.music.load(audio_fp, 'mp3')
#                     pygame.mixer.music.play()

#                     # Update the last play time
#                     last_play_time = time.time()

#                 except Exception as e:
#                     print(f"Error playing audio: {e}")

#         elif detection_mode == 'object':
#             detect_param = model.predict(source=[frame], conf=0.45, save=False)
#             DP = detect_param[0].numpy()

#             if len(DP) != 0:
#                 for i in range(len(detect_param[0])):
#                     boxes = detect_param[0].boxes
#                     box = boxes[i]
#                     clsID = box.cls.numpy()[0]
#                     conf = box.conf.numpy()[0]
#                     bb = box.xyxy.numpy()[0]

#                     cv.rectangle(
#                         frame,
#                         (int(bb[0]), int(bb[1])),
#                         (int(bb[2]), int(bb[3])),
#                         detection_colors[int(clsID)],
#                         3,
#                     )

#                     font = cv.FONT_HERSHEY_COMPLEX
#                     cv.putText(
#                         frame,
#                         class_list[int(clsID)],
#                         (int(bb[0]), int(bb[1]) - 10),
#                         font,
#                         1,
#                         (255, 255, 255),
#                         2,
#                     )

#                 processed_frame = frame

#         ret, buffer = cv.imencode('.jpg', processed_frame)
#         frame = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#     cap.release()

