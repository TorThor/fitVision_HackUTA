import cv2
import numpy as np

class DrawMethods:
    def __init__(self):
        pass

    def draw_keypoints(self, frame, keypoints, confidence_threshold=0.3):
        y, x, _ = frame.shape
        shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
        for kp in shaped:
            ky, kx, kp_conf = kp
            if kp_conf > confidence_threshold:
                cv2.circle(frame, (int(kx), int(ky)), 6, (255, 255, 255), -1)

    def draw_edges(self, frame, keypoints, edges, confidence_threshold=0.3):
        y, x, _ = frame.shape
        shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
        for edge, color in edges.items():
            p1, p2 = edge
            y1, x1, c1 = shaped[p1]
            y2, x2, c2 = shaped[p2]
            if (c1 > confidence_threshold) & (c2 > confidence_threshold):
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (160, 160, 160), 2)

    def draw_angles(self, frame, angle, center=(100, 100), thresholds=(90, 165), radius=50, max_angle=180):
        normalized_angle = min(max(angle, 0), max_angle)
        color = (0, 255, 0) if normalized_angle >= thresholds[1] else (255, 255, 0) if thresholds[0] <= normalized_angle < thresholds[1] else (0, 255, 0)
        start_angle = 0
        end_angle = int(normalized_angle)
        cv2.ellipse(frame, center, (radius, radius), 0, start_angle, end_angle, color, 3)
        return color