"""
Lane Lines Detection pipeline

Usage:
    main.py [--video] INPUT_PATH OUTPUT_PATH 

Options:

-h --help                               show this screen
--video                                 process video file instead of image
"""

import numpy as np
import matplotlib.image as mpimg
import cv2
import os
from docopt import docopt
from IPython.display import HTML, Video
from moviepy.editor import VideoFileClip
from param import output
from sklearn.linear_model import enet_path
from CameraCalibration import CameraCalibration
from Thresholding import *
from PerspectiveTransformation import *
from LaneLines import *

class FindLaneLines:
    """ This class is for parameter tunning.

    Attributes:
        ...
    """
    def __init__(self):
        """ Init Application"""
        self.calibration = CameraCalibration('camera_cal', 9, 6)
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation()
        self.lanelines = LaneLines()

    def forward(self, img):
        out_img = np.copy(img)
        img = self.calibration.undistort(img)
        img = self.transform.forward(img)
        img = self.thresholding.forward(img)
        img = self.lanelines.forward(img)
        img = self.transform.backward(img)
      # Ensure the images have the same dimensions and number of channels
        if out_img.shape != img.shape:
            img = cv2.resize(img, (out_img.shape[1], out_img.shape[0]))
            if len(out_img.shape) == 3 and len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0)
        out_img = self.lanelines.plot(out_img)
        return out_img

    def process_image(self, input_path, output_path):
        img = mpimg.imread(input_path)
        out_img = self.forward(img)
        mpimg.imsave(output_path, out_img)

        def process_video(self, input_path, output_path):
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"❌ Video not found: {input_path}")

        # Open video
        clip = VideoFileClip(input_path).resize(height=480)
        target_fps = clip.fps if clip.fps else 24

        # Video writer (OpenCV backend avoids freezes)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, target_fps,
                              (int(clip.w * clip.resize(height=480).w / clip.w * clip.size[0]), int(480)))

        # Process frame by frame
        for frame in clip.iter_frames(fps=target_fps, dtype="uint8"):
            try:
                processed = self.forward(frame)
            except Exception as e:
                print(f"⚠️ Frame skipped: {e}")
                processed = frame

            # Ensure color format (MoviePy uses RGB, OpenCV needs BGR)
            processed_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
            out.write(processed_bgr)

        out.release()
        print(f"✅ Video saved to {output_path}")


def main():
    args = docopt(__doc__)
    input_path = args['INPUT_PATH']
    output_path = args['OUTPUT_PATH']
    
    findLaneLines = FindLaneLines()
    if args['--video']:
        findLaneLines.process_video(input_path, output_path)
    else:
        findLaneLines.process_image(input_path, output_path)

    
if __name__ == "__main__":
    main()

# Use below to run the file
# C:\Anaconda\python.exe main.py --video challenge_video.mp4 output_video.mp4



# # Trial code
# """
# Lane Lines Detection pipeline

# Usage:
#     main.py [--video] INPUT_PATH OUTPUT_PATH 

# Options:

# -h --help                               show this screen
# --video                                 process video file instead of image
# """

# import numpy as np
# import matplotlib.image as mpimg
# import cv2
# from docopt import docopt
# from IPython.display import HTML, Video
# from moviepy.editor import VideoFileClip
# from param import output
# import time
# from sklearn.linear_model import enet_path
# from CameraCalibration import CameraCalibration
# from Thresholding import *
# from PerspectiveTransformation import *
# from LaneLines import *
# from gtts import gTTS  # For audio feedback
# import pygame  # For playing audio
# import io

# # Initialize Pygame
# pygame.mixer.init()

# class FindLaneLines:
#     """ This class is for detecting lane lines and giving feedback.
    
#     Attributes:
#         ...
#     """
#     def __init__(self):
#         """ Init Application"""
#         self.calibration = CameraCalibration('camera_cal', 9, 6)
#         self.thresholding = Thresholding()
#         self.transform = PerspectiveTransformation()
#         self.lanelines = LaneLines()
        
#         # Track straight and curve lane states
#         self.last_lane_state = None
        
#         # Audio feedback variables
#         self.last_play_time = 0  # Track the last time audio was played

#     def forward(self, img):
#         out_img = np.copy(img)
#         img = self.calibration.undistort(img)
#         img = self.transform.forward(img)
#         img = self.thresholding.forward(img)
#         img, lane_state = self.lanelines.forward(img)  # Update to return lane state (straight/curve)
#         img = self.transform.backward(img)
        
#         # Ensure the images have the same dimensions and number of channels
#         if out_img.shape != img.shape:
#             img = cv2.resize(img, (out_img.shape[1], out_img.shape[0]))
#             if len(out_img.shape) == 3 and len(img.shape) == 2:
#                 img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
#         out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0)
#         out_img = self.lanelines.plot(out_img)

#         # Determine lane feedback based on the detected state
#         if lane_state == "straight":
#             feedback = "You're driving on a straight lane, good job!"
#         elif lane_state == "curve":
#             feedback = "A curve is ahead, adjust your steering!"
#         else:
#             feedback = None

#         # Provide feedback if lane state changes or 5 seconds have passed
#         if feedback and (lane_state != self.last_lane_state or time.time() - self.last_play_time > 5):
#             try:
#                 tts = gTTS(text=feedback, lang='en')
#                 audio_fp = io.BytesIO()
#                 tts.write_to_fp(audio_fp)
#                 audio_fp.seek(0)

#                 pygame.mixer.music.load(audio_fp, 'mp3')
#                 pygame.mixer.music.play()

#                 self.last_lane_state = lane_state
#                 self.last_play_time = time.time()

#             except Exception as e:
#                 print(f"Error playing feedback audio: {e}")

#         return out_img

#     def process_image(self, input_path, output_path):
#         img = mpimg.imread(input_path)
#         out_img = self.forward(img)
#         mpimg.imsave(output_path, out_img)

#     def process_video(self, input_path, output_path):
#         clip = VideoFileClip(input_path)
#         out_clip = clip.fl_image(self.forward)
#         out_clip.write_videofile(output_path, audio=False)

# def main():
#     args = docopt(__doc__)
#     input_path = args['INPUT_PATH']
#     output_path = args['OUTPUT_PATH']
    
#     findLaneLines = FindLaneLines()
#     if args['--video']:
#         findLaneLines.process_video(input_path, output_path)
#     else:
#         findLaneLines.process_image(input_path, output_path)

    
# if __name__ == "__main__":
#     main()

# # Use below to run the file
# # C:\Anaconda\python.exe main.py --video challenge_video.mp4 output_video.mp4
