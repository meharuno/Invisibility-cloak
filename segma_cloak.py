import cv2
import numpy as np
import mediapipe as mp
import time

# Initialize camera
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segment = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)  # Try 0 (indoor)

# Capture the background frame (wait a bit before capturing)
print("Get ready — capturing background in 5 seconds...")
time.sleep(5)

bg_frames = []
for i in range(30):
    ret, bg_frame = cap.read()
    if ret:
        bg_frames.append(cv2.flip(bg_frame, 1))
background = np.median(bg_frames, axis=0).astype(np.uint8)
bg_matched = cv2.convertScaleAbs(background, alpha=1.02, beta=2)
print("Background captured! starting cloak...")
kernel = np.ones((10,10), np.uint8)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get segmentation mask
    result = segment.process(rgb_frame)
    mask = result.segmentation_mask

    mask_eroded = cv2.erode((mask * 255).astype(np.uint8), kernel, iterations=1)
    mask_eroded = mask_eroded / 255.0  # bring back to 0-1 range
    mask_blur = cv2.GaussianBlur(mask_eroded, (55, 55), 0)


    #clean blending
    mask_blur_3ch = np.stack((mask_blur,) * 3, axis=-1)
    output_frame = (mask_blur_3ch * bg_matched + (1 - mask_blur_3ch) * frame).astype(np.uint8)
    # Show the result
    cv2.imshow('Segma Cloak 2.0', output_frame)

    #to show the raw segmentation mask (press 'm' key)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('m'):
        # Show mask window
        cv2.imshow('Segmentation Mask', mask)

cap.release()
cv2.destroyAllWindows()
