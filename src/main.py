# Import packages for viewing the image, sending the data and tracking the face
import cv2
import socket
import cvzone

# Importing the UDP Function for transmitting data
from udp_server import send_udp_data

# Import the function for estimating the depth (Z)
from depth import estimate_depth

# Importing the function for face detection in the models module. 
from models.code.cvzone import detect_face_cvzone

# Importing the instance of detector
from constants import CVZONE_DETECTOR_MAX_ONE

# Setup for the information for the UDP server. 
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address_port = ('127.0.0.1', 5052)


def main():
    # Start the video capture
    cap = cv2.VideoCapture(0)

    # Never ending loop
    while True:
        # Capture the image 
        _, img = cap.read()

        # Detect faces with the desired model 
        faces = detect_face_cvzone(img, CVZONE_DETECTOR_MAX_ONE)

        # If there is a list of faces to detect 
        if faces and len(faces) > 0:
            # Retrieve the first face and only evaluate that 
            face = faces[0]

            # Get the X and Y cord for the face in the center 
            face_center = face[1]

            # Try to estimate the distance, ignore if not found.
            depth = estimate_depth(face)
            if depth == None:
                continue

            
            # Create an instance for containing the coordinates in a list 
            # Format will be [X, Y, Z]
            face_coordinates = face_center
            face_coordinates.append(depth)

            cvzone.putTextRect(img, f'Coords: {face_coordinates}',
                            (face[10][0] - 100, face[10][1] - 50),
                            scale=2)

            # Send data using UDP
            send_udp_data(sock, server_address_port, face_coordinates, log=True)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


# Run the main function when the file is run
if __name__ == "__main__":
    main()