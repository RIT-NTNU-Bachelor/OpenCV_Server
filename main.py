
import cv2
import socket

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# detector = FaceDetector(minDetectionCon=0.8)

haar_cascade =cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ('127.0.0.1', 5052)

while True:
    success, img = cap.read()
    
    # Converting image to grayscale 
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    
    # Applying the face detection method on the grayscale image 
    faces_rect = haar_cascade.detectMultiScale(gray_img, 1.1, 9) 
    
    # Iterating through rectangles of detected faces 
    for (x, y, w, h) in faces_rect: 
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2) 
    
    cv2.imshow('Detected faces', img)
    cv2.waitKey(1)