#  Original Working Code

# from flask import Flask, Response, render_template, request, redirect, url_for
# import cv2 as cv
# import numpy as np
# import random
# from ultralytics import YOLO
# import os
# from main import FindLaneLines  # Import for lane and curve detection
# # from gtts import gTTS
# # import pygame

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads/'

# # Initialize pygame for playing audio
# # pygame.mixer.init()
# # Load class names for object detection
# with open("C:\\Users\\Ben Alpha\\Python Works\\My Project\\Model\\utils\\coco.txt", "r") as f:
#     class_list = f.read().split("\n")

# # Generate random colors for each class
# detection_colors = [tuple(random.randint(0, 255) for _ in range(3)) for _ in range(len(class_list))]

# # Load a pretrained YOLOv8 model
# model = YOLO("weights/yolov8n.pt", "v8")

# # Initialize the FindLaneLines class for lane detection
# findLaneLines = FindLaneLines()

# # Global variables to store the selected video source and mode
# video_source = 0  # Default to live camera
# detection_mode = 'lane'  # Default to lane detection

# # Object Detection Code
# def generate_frames():
#     cap = cv.VideoCapture(video_source)
#     if not cap.isOpened():
#         print("Cannot open video source")
#         return

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Initialize processed_frame with the original frame
#         processed_frame = frame

#         # Process frame based on the selected mode
#         if detection_mode == 'lane':
#             processed_frame = findLaneLines.forward(frame)
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


# @app.route('/')
# def index():
#     return render_template('home.html')

# @app.route('/detect')
# def detect():
#     return render_template('detect.html')

# @app.route('/about')
# def about():
#     return render_template('about.html')

# @app.route('/object_detection')
# def object_detection():
#     return render_template('object.html')

# @app.route('/lane')
# def lane_detection():
#     return render_template('lane.html')

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/set_source', methods=['POST'])
# def set_source():
#     global video_source, detection_mode
#     source = request.form.get('source')
#     detection_mode = request.form.get('mode')  # Get the mode (lane or object)

#     if source == 'live':
#         video_source = 0  # Live camera, but only for lane detection
#         detection_mode = 'lane'  # Live feed is only used for lane detection
#     elif source == 'upload':
#         file = request.files['video']
#         if file:
#             file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#             file.save(file_path)
#             video_source = file_path  # Uploaded video
#             # Object detection mode is selected by the user via the mode field

#     return redirect(url_for('detect'))

# if __name__ == '__main__':
#     if not os.path.exists(app.config['UPLOAD_FOLDER']):
#         os.makedirs(app.config['UPLOAD_FOLDER'])
#     app.run(debug=True, use_reloader=False)  # Disable auto-reloader


#  Speecch  Sound

# Speech Trial 2
# from flask import Flask, Response, render_template, request, redirect, url_for
# import cv2 as cv
# import numpy as np
# import random
# from ultralytics import YOLO
# import os
# from main import FindLaneLines  # Import for lane and curve detection
# from gtts import gTTS
# import pygame

# # Initialize pygame mixer for playing audio
# pygame.mixer.init()

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads/'

# # Load class names for object detection
# with open("C:\\Users\\Ben Alpha\\Python Works\\My Project\\Model\\utils\\coco.txt", "r") as f:
#     class_list = f.read().split("\n")

# # Generate random colors for each class
# detection_colors = [tuple(random.randint(0, 255) for _ in range(3)) for _ in range(len(class_list))]

# # Load a pretrained YOLOv8 model
# model = YOLO("weights/yolov8n.pt", "v8")

# # Initialize the FindLaneLines class for lane detection
# findLaneLines = FindLaneLines()

# # Global variables to store the selected video source and mode
# video_source = 0  # Default to live camera
# detection_mode = 'lane'  # Default to lane detection

# # To control feedback audio playback
# lane_feedback_played = False
# curve_feedback_played = False

# def generate_frames():
#     global lane_feedback_played, curve_feedback_played
#     cap = cv.VideoCapture(video_source)
#     if not cap.isOpened():
#         print("Cannot open video source")
#         return

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Initialize processed_frame with the original frame
#         processed_frame = frame

#         # Process frame based on the selected mode
#         if detection_mode == 'lane':
#             # Get lane detection result and feedback (e.g., whether it's a straight or curved lane)
#             processed_frame, lane_status = findLaneLines.forward(frame)  # Assuming forward() returns feedback on lane status
            
#             # Check if lane is straight or curved
#             if lane_status == 'straight' and not lane_feedback_played:
#                 straight_feedback = "You are driving on a straight lane."  # Feedback message
#                 tts = gTTS(text=straight_feedback, lang='en')
#                 temp_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'straight_lane.mp3')
#                 tts.save(temp_audio_path)
#                 pygame.mixer.music.load(temp_audio_path)
#                 pygame.mixer.music.play()
#                 lane_feedback_played = True  # Prevent repeated playback for straight lane
#                 curve_feedback_played = False  # Reset curve feedback

#             elif lane_status == 'curve' and not curve_feedback_played:
#                 curve_feedback = "You are approaching a curve."  # Feedback message
#                 tts = gTTS(text=curve_feedback, lang='en')
#                 temp_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'curve_ahead.mp3')
#                 tts.save(temp_audio_path)
#                 pygame.mixer.music.load(temp_audio_path)
#                 pygame.mixer.music.play()
#                 curve_feedback_played = True  # Prevent repeated playback for curves
#                 lane_feedback_played = False  # Reset straight lane feedback

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


# @app.route('/')
# def index():
#     return render_template('home.html')


# @app.route('/detect')
# def detect():
#     return render_template('detect.html')


# @app.route('/about')
# def about():
#     return render_template('about.html')


# @app.route('/object_detection')
# def object_detection():
#     return render_template('object.html')


# @app.route('/lane')
# def lane_detection():
#     return render_template('lane.html')


# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# @app.route('/set_source', methods=['POST'])
# def set_source():
#     global video_source, detection_mode
#     source = request.form.get('source')
#     detection_mode = request.form.get('mode')  # Get the mode (lane or object)

#     if source == 'live':
#         video_source = 0  # Live camera, but only for lane detection
#         detection_mode = 'lane'  # Live feed is only used for lane detection
#     elif source == 'upload':
#         file = request.files['video']
#         if file:
#             file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#             file.save(file_path)
#             video_source = file_path  # Uploaded video
#             # Object detection mode is selected by the user via the mode field

#     return redirect(url_for('detect'))


# if __name__ == '__main__':
#     if not os.path.exists(app.config['UPLOAD_FOLDER']):
#         os.makedirs(app.config['UPLOAD_FOLDER'])
#     app.run(debug=True)

# TRial workinng with sound but has errors
# from flask import Flask, Response, render_template, request, redirect, url_for
# import cv2 as cv
# import numpy as np
# import random
# from ultralytics import YOLO
# import os
# from main import FindLaneLines  # Import for lane and curve detection
# from gtts import gTTS  # Google Text-to-Speech
# import pygame  # Library to play the sound

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads/'

# # Initialize Pygame for playing audio
# pygame.mixer.init()

# # Load class names for object detection
# with open("C:\\Users\\Ben Alpha\\Python Works\\My Project\\Model\\utils\\coco.txt", "r") as f:
#     class_list = f.read().split("\n")

# # Generate random colors for each class
# detection_colors = [tuple(random.randint(0, 255) for _ in range(3)) for _ in range(len(class_list))]

# # Load a pretrained YOLOv8 model
# model = YOLO("weights/yolov8n.pt", "v8")

# # Initialize the FindLaneLines class for lane detection
# findLaneLines = FindLaneLines()

# # Global variables to store the selected video source and mode
# video_source = 0  # Default to live camera
# detection_mode = 'lane'  # Default to lane detection


# def generate_frames():
#     cap = cv.VideoCapture(video_source)
#     if not cap.isOpened():
#         print("Cannot open video source")
#         return

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Initialize processed_frame with the original frame
#         processed_frame = frame

#         # Process frame based on the selected mode
#         if detection_mode == 'lane':
#             processed_frame = findLaneLines.forward(frame)
            
#             # Generate feedback using Google Text-to-Speech (gTTS)
#             lane_feedback = "You are keeping the lane correctly"  # Message to the user
            
#             # Convert text to speech using gTTS
#             tts = gTTS(text=lane_feedback, lang='en')
#             tts.save("lane_detected.mp3")  # Save the speech to a file
            
#             # Play the saved audio file
#             pygame.mixer.music.load("lane_detected.mp3")
#             pygame.mixer.music.play()

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


# @app.route('/')
# def index():
#     return render_template('home.html')

# @app.route('/detect')
# def detect():
#     return render_template('detect.html')

# @app.route('/about')
# def about():
#     return render_template('about.html')

# @app.route('/object_detection')
# def object_detection():
#     return render_template('object.html')

# @app.route('/lane')
# def lane_detection():
#     return render_template('lane.html')

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/set_source', methods=['POST'])
# def set_source():
#     global video_source, detection_mode
#     source = request.form.get('source')
#     detection_mode = request.form.get('mode')  # Get the mode (lane or object)

#     if source == 'live':
#         video_source = 0  # Live camera, but only for lane detection
#         detection_mode = 'lane'  # Live feed is only used for lane detection
#     elif source == 'upload':
#         file = request.files['video']
#         if file:
#             file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#             file.save(file_path)
#             video_source = file_path  # Uploaded video
#             # Object detection mode is selected by the user via the mode field
            

#     return redirect(url_for('detect'))


# if __name__ == '__main__':
#     if not os.path.exists(app.config['UPLOAD_FOLDER']):
#         os.makedirs(app.config['UPLOAD_FOLDER'])
#     app.run(debug=True)
