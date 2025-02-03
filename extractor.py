import cv2
import numpy as np
import os

def extract(imagePath, outputFolder, padding=8, width=15, height=10):

    image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return
    
    _, binary = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
    edge, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not edge:
        return
    
    edge = sorted(edge, key=cv2.contourArea, reverse=True)
    cellContours = edge[1:((width*height)+1)] 
    os.makedirs(outputFolder, exist_ok=True)

    count = 0
    for contour in cellContours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 10:
            cell = image[y + padding:y + h - padding, x + padding:x + w - padding]
            cell = cv2.GaussianBlur(cell, (3, 3), 0)

            srcPts = np.array([[x + padding, y + padding], [x + w - padding, y + padding],
                                [x + w - padding, y + h - padding], [x + padding, y + h - padding]], dtype="float32")
            dstPts = np.array([[0, 0], [63, 0], [63, 63], [0, 63]], dtype="float32")

            imagePrt = cv2.warpPerspective(image, cv2.getPerspectiveTransform(srcPts, dstPts), (64, 64))
            imagePrt = cv2.bitwise_not(imagePrt)
            _, binary = cv2.threshold(imagePrt, 128, 255, cv2.THRESH_BINARY)
            imagePrt = cv2.GaussianBlur(binary, (5, 5), 0)
            cv2.imwrite(os.path.join(outputFolder, f"{count}.png"), imagePrt)

            count += 1

inputFolder = "입력"
outputRootFolder = "출력"

for filename in os.listdir(inputFolder):
    imagePath = os.path.join(inputFolder, filename)
    outputFolder = os.path.join(outputRootFolder, os.path.splitext(filename)[0])
    extract(imagePath, outputFolder)
