import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



    # cv2.imshow("Image", imgGrey)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
