import os
from os import walk
import sys
import dlib
import cv2
import numpy as np

# Constants for the save_test function 
COLOR = (0, 255, 0) # Color of landmark or boarder
OUTPUT_PARENT_DIR = "data/results/unit_test_output/" # output folder relative to the root folder of the project


# Adds the parent 'src' directory to the system path.
# This allows the test files to import modules from the src directory.
def set_project_path_for_tests():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.join(current_dir, '../src')  
    sys.path.insert(0, parent_dir)


# Function to draw a single rectangle 
# Given the face coordinates and the image that the rectangle should be drawn on
def draw_rectangle(face_coords, image):
    x, y, width, height = face_coords
    cv2.rectangle(image, (x, y), (x + width, y + height), COLOR, 2)

# Draw each landmark on a given image
# Takes the list of landmarks and the image that the rectangle should be drawn on
def draw_landmark(landmarks, image):
    for point in landmarks:
        cv2.circle(image, (point[0], point[1]), 1, COLOR, -1)

# Draw the rectangle from Dlib output
# Takes the dlib.rectangle object for drawing the rectangle 
def draw_rectangle_from_dlib(face_rectangle, img_with_box):
    cv2.rectangle(img_with_box, (face_rectangle.left(), face_rectangle.top()), (face_rectangle.right(), face_rectangle.bottom()), COLOR, 2)

# Function that saves the image to the 
def save_image_to_file(img, file_name):
    full_path = os.path.join(OUTPUT_PARENT_DIR, file_name)
    cv2.imwrite(full_path, img)

# Function for saving test output as a single image
# Will detect how many faces was detected and then draw each face on the image
# Works with all models and with both rectangles and landmarks (draws both to the image)
# Saves the image in a directory with the given filename
def save_test(img, file_name, face):
    # Create a new image copy for drawing the rectangle or landmarks on
    image_with_box = img.copy()

    # Check what model it is based on the datatype and then draw the landmark/rectangle accordantly 
    if isinstance(face, tuple) or (isinstance(face, np.ndarray) and face.ndim == 1): # Single face as a tuple or 1D numpy array 
        draw_rectangle(face, image_with_box)
    elif isinstance(face, np.ndarray) and face.ndim == 2: # Multiple faces in a 2D numpy array
        for single_face in face:
            draw_rectangle(single_face, image_with_box)
    elif isinstance(face, dlib.rectangle):  # Single dlib rectangle => 1 face
        draw_rectangle_from_dlib(face, image_with_box)
    elif isinstance(face, dlib.rectangles):  # Multiple dlib rectangles => 2 faces
        for face_rectangle in face:
           draw_rectangle_from_dlib(face_rectangle, image_with_box)
    elif isinstance(face, list): # DNN or CVZone
        if all(isinstance(f, tuple) and len(f) == 4 for f in face):  # List of tuples for DNN => 2 faces
            for f in face:
                draw_rectangle(f, image_with_box)
        elif len(face) == 468:  # CVZone => 1 face
            draw_landmark(face,image_with_box)
        elif all(isinstance(f, list) for f in face) and all(len(f) == 468 for f in face):  #CVZone => 2 faces
            for face_landmarks in face:
                draw_landmark(face_landmarks, image_with_box)

    # Save the image to the file 
    save_image_to_file(image_with_box, file_name)


# Function for retrieving all images in a given parent folder
def get_images_from_dataset(path_to_parent_dir):
    # List of images found and saved
    images = []

    # Iterate over each file 
    for (_, _, filenames) in walk(path_to_parent_dir):
        for file in filenames:
            full_path = path_to_parent_dir + file.strip()
            current_image = cv2.imread(full_path)
            images.append(current_image)
        break

    return images

