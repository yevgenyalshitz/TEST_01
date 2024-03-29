import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print('ID =', id, '\nhand landmarks:\n', lm)
                # check img height, width, channels:
                h,w,c = img.shape

                # 20 hand fingers position convert to pixels position:
                cx,cy = int(lm.x*w), int(lm.y*h)
                print('ID =', id, '\n', cx,cy)

                # sign hand center position
                if id == 0:
                    cv2.circle(img, (cx,cy), 10, (255,0,255), cv2.FILLED)
                # # sign thumb position
                # if id == 4:
                #     cv2.circle(img, (cx,cy), 10, (0,255,0), cv2.FILLED)


            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, 'fps=' + str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    # cv2.imshow("Image", imgGrey)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
