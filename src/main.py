import cv2
import socket
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector

# Importing the UDP Function for transmitting data
from udp_server import send_udp_data

# Importing the function for face detection in the models module. 
from models.code.cvzone import detect_face_cvzone

# Importing the instance of detector
from constants import CVZONE_DETECTOR_MAX_ONE



# Setup for the information for the UDP server. 
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ('127.0.0.1', 5052)


def main():
    # Start the video capture
    cap = cv2.VideoCapture(0)

    while True:
        _, img = cap.read()

        faces = detect_face_cvzone(img, CVZONE_DETECTOR_MAX_ONE)

        if faces:
            face = faces[0]
            faceCenter = face[1]
            leftEye = face[145]
            rightEye = face[374]
            w, _ = CVZONE_DETECTOR_MAX_ONE.findDistance(leftEye, rightEye)
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


# Run the main function when the file is run
if __name__ == "__main__":
    main()