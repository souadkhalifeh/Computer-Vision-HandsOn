import cv2 as cv 
import numpy as np 

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_eye.xml")

img = cv.imread("../data/Nadia_Murad.jpg")
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#detect faces in the image and return their coordinates as a list of rectangles (x,y,w,h) 
# where x and y are the coordinates of the top-left corner of the rectangle 
# and w and h are the width and height of the rectangle respectively
faces = face_cascade.detectMultiScale(gray, 1.3, 5) # the first parameter is the image, the second parameter is the scale factor that specifies how much the image size is reduced at each image scale, and the third parameter is the minimum number of neighbors that a rectangle should have to be retained

for (x,y,w,h) in faces:
    cv.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2) # draw a rectangle around the detected face in the original image, the first parameter is the image, the second parameter is the top-left corner of the rectangle, the third parameter is the bottom-right corner of the rectangle, the fourth parameter is the color of the rectangle in BGR format, and the fifth parameter is the thickness of the rectangle


    roi_gray = gray[y:y+h, x:x+w] # extract the region of interest (the detected face) from the grayscale image
    roi_color = img[y:y+h, x:x+w] # extract the region of interest (the detected face) from the original image

    eyes = eye_cascade.detectMultiScale(roi_gray,1.8) # detect eyes in the region of interest (the detected face) and return their coordinates as a list of rectangles (ex,ey,ew,eh) where ex and ey are the coordinates of the top-left corner of the rectangle and ew and eh are the width and height of the rectangle respectively

    for (ex,ey,ew,eh) in eyes:
        cv.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2) # draw a rectangle around the detected eye in the region of interest

cv.imshow("Detected Faces", img) 
cv.waitKey(0) 