import cv2
import mediapipe as mp
import time

# https://google.github.io/mediapipe/solutions/pose.html

class poseDetector():
    def __init__(self,
                 static_image_mode = False,
                 model_complexity = 1,
                 smooth_landmarks = True,
                 enable_segmentation = False,
                 smooth_segmentation = True,
                 min_detection_confidence = 0.5,
                 min_tracking_confidence = 0.5):

        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence


        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.static_image_mode,
                                     self.model_complexity,
                                     self.smooth_landmarks,
                                     self.enable_segmentation,
                                     self.smooth_segmentation,
                                     self.min_detection_confidence,
                                     self.min_tracking_confidence)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
                for id, lm in enumerate(self.results.pose_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # print('ID =', id, '\n', cx, cy)

                    # mark nose (id==0)
                    if id == 0:
                        cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)
        return img

    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks.landmark:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)

        return lmList


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) !=0:
            print(lmList[14])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, 'fps=' + str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
