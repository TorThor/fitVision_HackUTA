import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
import cv2

# Define body part edges
EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def cal_elbow_angle(point1, point2, point3):
    a = np.array(point1) # Ankle
    b = np.array(point2) # Shoulder
    c = np.array(point3) # Wrist

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)


# Function to calculate the angle between three points (ex. shoulder, elbow, wrist for pushup)
def cal_shoulder_angle(point1, point2, point3):
    a = np.array(point1) # Shoulder
    b = np.array(point2) # Elbow
    c = np.array(point3) # Wrist

    # Calculate vectors
    ba = a - b
    bc = c - b

    # Calculate cosine of the angle
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle) # Gives angle in radians

    return np.degrees(angle) # Gives angle in degrees

# Function to detect push-up state based on keypoints
def pushup_rep_counter(keypoints_with_scores, rep_count, last_position):
    # Extract the shoulder, elbow, and wrist keypoints (adjust indices as necessary)
    keypoints = keypoints_with_scores[0][0]

    # For left arm: shoulder = 5, elbow = 7, wrist = 9, ankle = 0
    shoulder = [keypoints[5][1], keypoints[5][0]]
    elbow = [keypoints[7][1], keypoints[7][0]]
    wrist = [keypoints[9][1], keypoints[9][0]]
    ankle = [keypoints[15][1], keypoints[15][0]]

    # Calculate the angle at the elbow
    elbow_angle = cal_shoulder_angle(shoulder, elbow, wrist)
    shoulder_angle = cal_elbow_angle(ankle, shoulder, wrist)

    # Detect push-up based on elbow angle
    if elbow_angle <= 90 and last_position == "up":
        last_position = "down"
        print("DOWN Elbow angle:", f'{elbow_angle:.2f}')
        print("DOWN Shoulder angle:", f'{shoulder_angle:.2f}')
    elif elbow_angle > 165 and last_position == "down" and shoulder_angle > 64 and keypoints[15][2] >= 0.3: 
        last_position = "up"
        rep_count += 1
        print("UP Elbow angle:", f'{elbow_angle:.2f}')
        print("UP Shoulder angle:", f'{shoulder_angle:.2f}')

    return rep_count, last_position, elbow_angle, shoulder_angle

# Define keypoint and connection drawing functions
def draw_keypoints(frame, keypoints, confidence_threshold=0.3):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 6, (255, 255, 255), -1)

def draw_edges(frame, keypoints, edges, confidence_threshold=0.3):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (160, 160, 160), 2)

def draw_angles(frame, elbow_angle, center=(100, 100), thresholds=(0,0), radius=50, max_angle=180):
    """
    Draws a dynamic arc representing the elbow angle on the frame.
    
    Args:
        frame: The video frame where the arc will be drawn.
        elbow_angle: The current elbow angle to visualize.
        center: The center of the arc (x, y coordinates).
        radius: The radius of the arc.
        max_angle: The maximum angle of the arc (180 degrees by default).
    """
    # Normalize elbow_angle between 0 and max_angle (e.g., 0 to 180 degrees)
    normalized_angle = min(max(elbow_angle, 0), max_angle)
    
    # Set color based on the angle (you can change this logic as needed)
    if normalized_angle >= thresholds[1]: # ex. pushup: 165
        color = (0, 255, 0)  # Green for a larger angle (closer to 180 degrees)
    elif thresholds[0] <= normalized_angle < thresholds[1]: # ex. pushup: 90, 165
        color = (255, 255, 0)  # Yellow for intermediate angles
    else:
        color = (0, 255, 0)  # Red for lower angles (close to 0)

    # Calculate the start and end angle for the arc
    start_angle = 0  # Starting at 0 degrees
    end_angle = int(normalized_angle)  # End at the current elbow angle
    
    # Draw the arc using OpenCV's ellipse function (draws partial circles)
    cv2.ellipse(
        frame,               # The frame to draw on
        center,              # The center of the arc
        (radius, radius),    # Size of the arc (equal radius for circular arc)
        0,                   # Angle of rotation of the ellipse (0 for upright)
        start_angle,         # Starting angle of the arc
        end_angle,           # Ending angle (determined by elbow_angle)
        color,               # Color of the arc
        3                    # Thickness of the arc line
    )

    return color

# Load the TensorFlow Lite model
movenet_path = os.path.join(os.getcwd(), 'backend', 'MoveNet_model', '3.tflite')
interpreter = tf.lite.Interpreter(model_path=movenet_path)
interpreter.allocate_tensors()

# Capture video from the default camera (adjust the index if needed)
cap = cv2.VideoCapture(1)  # Use 0 for the default camera
rep_count = 0 
last_position = "up"

# Adjust the input image size to match MoveNet's expected size (192x192)
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image. Try one more time (happens occasionally).")
        break

    # Reshape the image to fit the model input requirements (192x192)
    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 256, 256)
    input_image = tf.cast(img, dtype=tf.float32)

    # Setup input and output for the interpreter
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Make predictions using the model
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

    # Track push-ups based on the keypoints
    rep_count, last_position, elbow_angle, shoulder_angle = pushup_rep_counter(keypoints_with_scores, rep_count, last_position)

    # Render keypoints and connections on the frame
    draw_edges(frame, keypoints_with_scores, EDGES, 0.3)
    draw_keypoints(frame, keypoints_with_scores, 0.3)

    # Draw the dynamic arc representing the elbow angle
    color1 = draw_angles(frame, elbow_angle, center=(100, 100), thresholds=(90, 165), radius=50, max_angle=180)
    color2 = draw_angles(frame, shoulder_angle, center=(100, 240), thresholds=(49, 60), radius=50, max_angle=180)

    # Display the rep count on the frame and elbow_angle
    cv2.putText(frame, f'Reps: {rep_count}', (45, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'{elbow_angle:.0f}', (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color1, 2, cv2.LINE_AA)
    cv2.putText(frame, 'Elbow angle', (45, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color1, 2, cv2.LINE_AA)
    cv2.putText(frame, f'{shoulder_angle:.0f}', (80, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color2, 2, cv2.LINE_AA)
    cv2.putText(frame, 'Shoulder angle', (45, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color2, 2, cv2.LINE_AA)


    # Display the frame with keypoints and edges
    cv2.imshow('MoveNet Push-Up Counter', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()