import os
import sys
import dlib
import cv2
import numpy as np

# Constants for the save_test function 
COLOR = (0, 255, 0) # Color of landmark or boarder
PARENT_DIR = "data/unit_test_output/" # output folder relative to the root folder of the project

def set_project_path_for_tests():
    """Adds the parent 'src' directory to the system path.
    This allows the test files to import modules from the src directory.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.join(current_dir, '../src')  
    sys.path.insert(0, parent_dir)


def draw_rectangle(face_coords, image):
    """Function to draw a single rectangle

    Args:
    - face_coords (tuple): Tuple with x, y, width and height of the bounding box
    - image (Matlike): Image object obtained from the OpenCV imread function 
    """
    x, y, width, height = face_coords
    cv2.rectangle(image, (x, y), (x + width, y + height), COLOR, 2)


def draw_landmark(landmarks, image):
    """Function to draw a landmark.
    Draws a circle in the image for each landmark in the given list of landmarks

    Args:
        landmarks (list): List of landmarks where each landmark has a x and y position
        image (Matlike): Image object obtained from the OpenCV imread function 
    """
    for point in landmarks:
        cv2.circle(image, (point[0], point[1]), 1, COLOR, -1)

def draw_rectangle_from_dlib(face_rectangle, img_with_box):
    """Function that draws the given dlib rectangle to the given image

    Args:
        face_rectangle (dlib.rectangle): A dlib rectangle object
        img_with_box (Matlike): Image object obtained from the OpenCV imread function
    """
    cv2.rectangle(img_with_box, (face_rectangle.left(), face_rectangle.top()), (face_rectangle.right(), face_rectangle.bottom()), COLOR, 2)
    

def save_test(img, file_name, face):
    """ Function for saving test output as a single image
    Will detect how many faces was detected and then draw each face on the image
    Works with all models and with both rectangles and landmarks (draws both to the image)
    Saves the image in a directory with the given filename

    Args:
        img (Matlike): OpenCV image that the bounding boxes will be drawn on
        file_name (string): name of the file of the test 
        face(list| tuple | np.array | dlib.rectangle | dlib.rectangles): Output from detecting face with 
    """
    # Create a new image copy for drawing the rectangle or landmarks on
    image_with_box = img.copy()

    # Check that none if the input was none 
    if img is None or len(file_name) == 0 or face is None:
        print("ERROR: could not save unit test to image due to invalid given args")
        return 

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
    full_path = os.path.join(PARENT_DIR, file_name)
    cv2.imwrite(full_path, image_with_box)
