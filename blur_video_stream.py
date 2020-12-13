import cv2
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
from utils import *

# Set this variable to False if you want to view a background image
# and also set the path to your background image
# Background Image Source: https://pixabay.com/photos/monoliths-clouds-storm-ruins-sky-5793364/
BLUR = False
BG_PTH = "bg1.jpg"

# Load the DeepLabv3 model to memory
model = utils.load_model()

# Read the background image to memory
bg_image = cv2.imread(BG_PTH)
bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)

# Start a video cam session
video_session = cv2.VideoCapture(0)

# Define a blurring value kernel size for cv2's Gaussian Blur
blur_value = (51, 51)

# Define two axes for showing the mask and the true video in realtime
# And set the ticks to none for both the axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 8))
ax2.set_title("Mask")

ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])

# Create two image objects to picture on top of the axes defined above
im1 = ax1.imshow(utils.grab_frame(video_session))
im2 = ax2.imshow(utils.grab_frame(video_session))

# Switch on the interactive mode in matplotlib
plt.ion()
plt.show()

# Read frames from the video, make realtime predictions and display the same
while True:
    frame = utils.grab_frame(video_session)

    # Ensure there's something in the image (not completely blacnk)
    if np.any(frame):

        # Read the frame's width, height, channels and get the labels' predictions from utilities
        width, height, channels = frame.shape
        labels = utils.get_pred(frame, model)
        
        if BLUR:
            # Wherever there's empty space/no person, the label is zero 
            # Hence identify such areas and create a mask (replicate it across RGB channels)
            mask = labels == 0
            mask = np.repeat(mask[:, :, np.newaxis], channels, axis = 2)

            # Apply the Gaussian blur for background with the kernel size specified in constants above
            blur = cv2.GaussianBlur(frame, blur_value, 0)
            frame[mask] = blur[mask]
            ax1.set_title("Blurred Video")
        else:
            # The PASCAL VOC dataset has 20 categories of which Person is the 16th category
            # Hence wherever person is predicted, the label returned will be 15
            # Subsequently repeat the mask across RGB channels 
            mask = labels == 15
            mask = np.repeat(mask[:, :, np.newaxis], 3, axis = 2)
            
            # Resize the image as per the frame capture size
            bg = cv2.resize(bg_image, (height, width))
            bg[mask] = frame[mask]
            frame = bg
            ax1.set_title("Background Changed Video")

        # Set the data of the two images to frame and mask values respectively
        im1.set_data(frame)
        im2.set_data(mask * 255)
        plt.pause(0.01)
        
    else:
        break

# Empty the cache and switch off the interactive mode
torch.cuda.empty_cache()
plt.ioff()
