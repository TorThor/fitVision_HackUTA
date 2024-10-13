import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2

movenet_path = os.path.join(os.getcwd(), 'backend', 'MoveNet_model', '3.tflite')
# print("Movenet path:", movenet_path)
interpreter = tf.lite.Interpreter(model_path=movenet_path)
interpreter.allocate_tensors()

cap = cv2.VideoCapture(1)
while cap.isOpened():
    ret, frame = cap.read()

    cv2.imshow('MoveNet Thunder', frame)

    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()