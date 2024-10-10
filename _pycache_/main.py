import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

brushThickness = 15
eraserThickness = 50

folderPath = "HeaderImages"
myList = os.listdir(folderPath)

overlayList = []
for imgPath in myList:
    image = cv2.imread(f"{folderPath}/{imgPath}")
    overlayList.append(image)

header = overlayList[0]
drawColor = (255, 0, 255)

imageCanvas = np.zeros((720, 1280, 3), np.uint8)

vidCap = cv2.VideoCapture(1)
vidCap.set(3, 1280)
vidCap.set(4, 720)

detector = htm.HandTrackingModule(detectionConf=0.85)

xprev, yprev = 0, 0

while True:
    # Import Image
    success, img = vidCap.read()
    img = cv2.flip(img, 1)

    # Find Hand Landmarks

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        topOfIndexFingerX, topOfIndexFingerY = lmList[8][1], lmList[8][2]
        topOfMiddleFingerX, topOfMiddleFingerY = lmList[12][1], lmList[12][2]

        # Check which fingers are up

        fingers = detector.fingersUp()

        # If Selection Mode - Two fingers up
        if fingers[1] and fingers[2]:
            xprev, yprev = 0, 0
            # Checking for click
            if topOfIndexFingerY < 100:
                if 145 < topOfIndexFingerX < 215:
                    header = overlayList[1]
                    drawColor = (0, 0, 255)
                elif 460 < topOfIndexFingerX < 510:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)

                elif 780 < topOfIndexFingerX < 825:
                    header = overlayList[3]
                    drawColor = (255, 0, 0)
                elif 1000 < topOfIndexFingerX < 1085:
                    header = overlayList[4]
                    drawColor = (0, 0, 0)

            cv2.rectangle(img, (topOfIndexFingerX, topOfIndexFingerY - 25),
                          (topOfMiddleFingerX, topOfMiddleFingerY + 25),
                          drawColor, cv2.FILLED)

        # Drawing Mode
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (topOfIndexFingerX, topOfIndexFingerY),
                       15, drawColor, cv2.FILLED)
            if xprev == 0 and yprev == 0:
                xprev, yprev = topOfIndexFingerX, topOfIndexFingerY

            if drawColor == (0, 0, 0):
                cv2.line(img, (xprev, yprev), (topOfIndexFingerX, topOfIndexFingerY),
                         drawColor, eraserThickness)
                cv2.line(imageCanvas, (xprev, yprev), (topOfIndexFingerX, topOfIndexFingerY),
                         drawColor, eraserThickness)
            else:
                cv2.line(img, (xprev, yprev), (topOfIndexFingerX, topOfIndexFingerY),
                         drawColor, brushThickness)
                cv2.line(imageCanvas, (xprev, yprev), (topOfIndexFingerX, topOfIndexFingerY),
                         drawColor, brushThickness)

            xprev, yprev = topOfIndexFingerX, topOfIndexFingerY

    imgGray = cv2.cvtColor(imageCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imageCanvas)

    # Setting the header image
    img[0:100, 0:1280] = header
    #img = cv2.addWeighted(img, 0.5, imageCanvas, 0.5, 0)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
