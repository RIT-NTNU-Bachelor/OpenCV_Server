import cv2
import socket
import time
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from udp_server import send_udp_data

frame_rate = 45
prev = 0


# Setup for the information for the UDP server. 
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ('127.0.0.1', 5052)

cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)

while True:
    time_elapsed = time.time() - prev
    res, image = cap.read()

    if time_elapsed > 1. / frame_rate:
        prev = time.time()

        success, img = cap.read()
        img, faces = detector.findFaceMesh(img, draw=False)

        if faces:
            face = faces[0]
            faceCenter = face[1]
            leftEye = face[145]
            rightEye = face[374]
            w, _ = detector.findDistance(leftEye, rightEye)
            W = 6.3

            # Finding distance
            f = 655
            d = int((W * f) / w)

            # Append z-coordinates.
            faceCenter[0] = max(faceCenter[0] // 5, 0)

            faceCoordinatesXYZ = faceCenter
            faceCoordinatesXYZ.append(d)

            cvzone.putTextRect(img, f'Coords: {faceCoordinatesXYZ}',
                               (face[10][0] - 100, face[10][1] - 50),
                               scale=2)

            # Send data using UDP
            send_udp_data(sock, serverAddressPort, faceCoordinatesXYZ, log=True)

        cv2.imshow("Image", img)
        cv2.waitKey(1)