import cv2
import time
#define a function to capture video from the webcam and display it in a window
cam = cv2.VideoCapture(0)


#define frames 
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

time.sleep(2) #to give the camera some time to warm up

while True:
    ret, frame = cam.read() #read a frame from the webcam
    if not ret:
        print("Failed to retrieve frame")
        break\
        
    if frame.shape[0] and frame.shape[1]: #check if the frame is not empty
        cv2.imshow("Webcam", frame) #display the frame in a window called "Webcam"
    else:
        print("Error: Invalid dimensions for the frame")    

    if cv2.waitKey(1) & 0xFF == ord('q'): #wait for the user to press the 'q' key to exit
        break


cam.release() #release the webcam
cv2.destroyAllWindows() #close all windows