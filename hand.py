# Python Air Painter
#
# Description:
# This program uses your computer's camera to create a virtual drawing canvas.
# It leverages OpenCV for camera handling and display, and Google's MediaPipe
# library for advanced hand tracking.
#
# Features:
# - Pinch to Draw: Drawing only occurs when your thumb and index finger are close together.
# - Selection Mode: Point your index finger to interact with the UI.
# - Color Selection: Change colors by pointing at the color swatches in the toolbar.
# - Brush & Eraser Size Control: Point at the +/- icons to change the brush or eraser size.
# - Clear Canvas: A dedicated button to clear your artwork.
# - Attractive UI: A semi-transparent toolbar and enhanced visual feedback for interactions.
# - Full Camera View: The application displays the complete, mirrored camera feed.
#
# How to Run:
# 1. Make sure you have Python installed on your computer.
# 2. Install the required libraries by running the following commands in your terminal:
#    pip install opencv-python
#    pip install mediapipe
# 3. Save this code as a Python file (e.g., `air_painter.py`).
# 4. Run the file from your terminal:
#    python air_painter.py
# 5. Press the 'q' key to quit the application.

import cv2
import numpy as np
import mediapipe as mp
import math

# --- Setup ---
# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize OpenCV Video Capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Get camera frame dimensions and set a consistent size
cap.set(3, 1280)
cap.set(4, 720)
ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame from camera.")
    exit()
h, w, c = frame.shape

# --- Drawing Canvas & Variables ---
canvas = np.zeros((h, w, 3), np.uint8)
xp, yp = 0, 0
brush_thickness = 15
eraser_thickness = 100
draw_color = (255, 100, 50)  # Default: Blue (BGR)

# --- UI Elements & Layout ---
toolbar_height = 80
# Define UI elements as dictionaries for easier management
ui_elements = {
    "colors": [
        {"pos": (20, 15), "size": 50, "color": (50, 50, 255)},   # Red
        {"pos": (90, 15), "size": 50, "color": (50, 255, 50)},  # Green
        {"pos": (160, 15), "size": 50, "color": (255, 100, 50)},# Blue
        {"pos": (230, 15), "size": 50, "color": (0, 255, 255)},  # Yellow
    ],
    "eraser": {"pos": (320, 15), "size": 50, "color": (255, 255, 255)}, # White
    "brush_size": {
        "minus": {"pos": (w - 240, 40), "radius": 20},
        "plus": {"pos": (w - 160, 40), "radius": 20},
        "text_pos": (w - 200, 52)
    },
    "clear": {"pos": (w - 75, 15), "size": (60, 50)}
}

def draw_ui(img):
    """Draws all UI elements onto the image."""
    # Create a semi-transparent toolbar
    toolbar = img[0:toolbar_height, 0:w]
    white_rect = np.ones(toolbar.shape, dtype=np.uint8) * 255
    res = cv2.addWeighted(toolbar, 0.5, white_rect, 0.5, 1.0)
    img[0:toolbar_height, 0:w] = res

    # Draw color swatches
    for swatch in ui_elements["colors"]:
        x, y = swatch["pos"]
        size = swatch["size"]
        cv2.rectangle(img, (x, y), (x + size, y + size), swatch["color"], cv2.FILLED)
        if swatch["color"] == draw_color:
            cv2.rectangle(img, (x, y), (x + size, y + size), (0, 0, 0), 4)

    # Draw eraser
    eraser = ui_elements["eraser"]
    ex, ey = eraser["pos"]
    esize = eraser["size"]
    cv2.rectangle(img, (ex, ey), (ex + esize, ey + esize), eraser["color"], cv2.FILLED)
    cv2.putText(img, "Eraser", (ex, ey + esize + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    if eraser["color"] == draw_color:
        cv2.rectangle(img, (ex, ey), (ex + esize, ey + esize), (0, 0, 0), 4)

    # Draw Brush Size controls
    bs = ui_elements["brush_size"]
    cv2.circle(img, bs["minus"]["pos"], bs["minus"]["radius"], (200, 200, 200), cv2.FILLED)
    cv2.putText(img, "-", (bs["minus"]["pos"][0] - 8, bs["minus"]["pos"][1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.circle(img, bs["plus"]["pos"], bs["plus"]["radius"], (200, 200, 200), cv2.FILLED)
    cv2.putText(img, "+", (bs["plus"]["pos"][0] - 8, bs["plus"]["pos"][1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(img, f"Size: {brush_thickness}", bs["text_pos"], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

    # Draw Clear button
    clear = ui_elements["clear"]
    cx, cy = clear["pos"]
    csize_x, csize_y = clear["size"]
    cv2.rectangle(img, (cx, cy), (cx + csize_x, cy + csize_y), (128, 128, 128), cv2.FILLED)
    cv2.putText(img, "CLEAR", (cx + 5, cy + 35), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
    
    # Instructions
    cv2.putText(img, "Pinch to Draw | Point to Select | 'q' to Quit", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# --- Main Loop ---
while True:
    success, img = cap.read()
    if not success: break
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        x1, y1 = int(index_tip.x * w), int(index_tip.y * h)
        x2, y2 = int(thumb_tip.x * w), int(thumb_tip.y * h)
        
        distance = math.hypot(x2 - x1, y2 - y1)

        # A. Pinch Gesture (Drawing Mode)
        if distance < 40:
            cv2.circle(img, (x1, y1), int(brush_thickness/2), draw_color, cv2.FILLED)
            if xp == 0 and yp == 0: xp, yp = x1, y1
            
            current_thickness = eraser_thickness if draw_color == (255, 255, 255) else brush_thickness
            cv2.line(canvas, (xp, yp), (x1, y1), draw_color, current_thickness)
            xp, yp = x1, y1
        # B. Pointing Gesture (Selection Mode)
        else:
            xp, yp = 0, 0
            cv2.circle(img, (x1, y1), 10, draw_color, 2) # Selection cursor

            if y1 < toolbar_height: # Finger is in toolbar
                # Check color swatches
                for swatch in ui_elements["colors"]:
                    sx, sy = swatch["pos"]; ssize = swatch["size"]
                    if sx < x1 < sx + ssize and sy < y1 < sy + ssize:
                        draw_color = swatch["color"]
                
                # Check eraser
                ex, ey = ui_elements["eraser"]["pos"]; esize = ui_elements["eraser"]["size"]
                if ex < x1 < ex + esize and ey < y1 < ey + esize:
                    draw_color = ui_elements["eraser"]["color"]

                # Check brush size controls
                bs_minus = ui_elements["brush_size"]["minus"]
                bs_plus = ui_elements["brush_size"]["plus"]
                if math.hypot(x1 - bs_minus["pos"][0], y1 - bs_minus["pos"][1]) < bs_minus["radius"]:
                    brush_thickness = max(2, brush_thickness - 1)
                if math.hypot(x1 - bs_plus["pos"][0], y1 - bs_plus["pos"][1]) < bs_plus["radius"]:
                    brush_thickness = min(50, brush_thickness + 1)

                # Check clear button
                cx, cy = ui_elements["clear"]["pos"]; csize_x, csize_y = ui_elements["clear"]["size"]
                if cx < x1 < cx + csize_x and cy < y1 < cy + csize_y:
                    canvas = np.zeros((h, w, 3), np.uint8)
    else:
        xp, yp = 0, 0

    # Combine canvas and camera feed
    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, canvas)

    # Draw UI on top
    draw_ui(img)

    cv2.imshow("Air Painter", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
