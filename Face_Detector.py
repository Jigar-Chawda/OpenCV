import cv2
from random import randrange

# Load pre-trained data from opencv (harr cascade algorithm) 
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose an image to detect faces in
# img = cv2.imread('Cast.jpg')

# To capture video from webcam
webcam = cv2.VideoCapture('video.mp4')


# Iterate forever over frames
while True:

    # Read the current frame
    successful_frame_read, frame = webcam.read()
    
    # Must convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (randrange(256) , randrange(256), randrange(256)), 5)

    cv2.imshow('Python Face Detector', frame)
    key = cv2.waitKey(1)

    # Stop if Q key is pressed
    if key==81 or key==113:
        break


# Release the VideoCapture object
webcam.release()

print("Code Complete")