import numpy as np

class PushUp:
    def __init__(self):
        pass

    def cal_shoulder_angle(self, point1, point2, point3):
        a = np.array(point1)
        b = np.array(point2)
        c = np.array(point3)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cosine_angle))

    def cal_elbow_angle(self, point1, point2, point3):
        a = np.array(point1)
        b = np.array(point2)
        c = np.array(point3)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cosine_angle))

    def pushup_rep_counter(self, keypoints_with_scores, rep_count, last_position):
        keypoints = keypoints_with_scores[0][0]
        shoulder = [keypoints[5][1], keypoints[5][0]]
        elbow = [keypoints[7][1], keypoints[7][0]]
        wrist = [keypoints[9][1], keypoints[9][0]]
        ankle = [keypoints[15][1], keypoints[15][0]]

        elbow_angle = self.cal_shoulder_angle(shoulder, elbow, wrist)
        shoulder_angle = self.cal_elbow_angle(ankle, shoulder, wrist)

        if elbow_angle <= 90 and last_position == "up":
            last_position = "down"
        elif elbow_angle > 165 and last_position == "down" and shoulder_angle > 64 and keypoints[15][2] >= 0.3:
            last_position = "up"
            rep_count += 1

        return rep_count, last_position, elbow_angle, shoulder_angle
