import cv2
from poses.pushup import PushUp
from draw_methods.draw_methods import DrawMethods
from keypoints import EDGES
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np

# Load TensorFlow Lite model
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
movenet_path = os.path.join(os.getcwd(), 'backend', 'MoveNet_model', '3.tflite')
interpreter = tf.lite.Interpreter(model_path=movenet_path)
interpreter.allocate_tensors()

# Initialize exercise and drawing classes
exercise = PushUp()
drawer = DrawMethods()

# Capture video from the default camera
cap = cv2.VideoCapture(1)
rep_count = 0
last_position = "up"

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image. Try again. If persists, change VideoCapture(1) to VideoCapture(0).")
        break

    # Prepare the image for MoveNet
    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 256, 256)
    input_image = tf.cast(img, dtype=tf.float32)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()

    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

    # Track push-ups and calculate angles
    rep_count, last_position, elbow_angle, shoulder_angle = exercise.pushup_rep_counter(keypoints_with_scores, rep_count, last_position)

    # Draw keypoints, edges, and angles
    drawer.draw_edges(frame, keypoints_with_scores, EDGES)
    drawer.draw_keypoints(frame, keypoints_with_scores)
    
    color1 = drawer.draw_angles(frame, elbow_angle, center=(100, 100), thresholds=(90, 165))
    color2 = drawer.draw_angles(frame, shoulder_angle, center=(100, 240), thresholds=(56, 66))
    
    # Display counts and angles
    cv2.putText(frame, f'Reps: {rep_count}', (45, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'{elbow_angle:.0f}', (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color1, 2, cv2.LINE_AA)
    cv2.putText(frame, 'Elbow angle', (45, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color1, 2, cv2.LINE_AA)
    cv2.putText(frame, f'{shoulder_angle:.0f}', (80, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color2, 2, cv2.LINE_AA)
    cv2.putText(frame, 'Shoulder angle', (45, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color2, 2, cv2.LINE_AA)

    cv2.imshow('MoveNet Push-Up Counter', frame)

    # Exit on 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()