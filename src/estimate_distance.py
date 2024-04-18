from constants.model_constants import CVZONE_DETECTOR_MAX_ONE, EYE_DISTANCE_INDEX
from constants.config_constants import FOCAL_LENGTH, INTEROCULAR_DISTANCE


def estimate_depth(face, img_width, focal_length=FOCAL_LENGTH):
    """
    Estimate the Z-coordinate (depth) for a detected face using the distance between the eyes.
    It uses the CVZone 

    Parameters:
    - face: A list representing a detected face with specific indices for eyes.
    - img_width: Width of the image in pixels.
    - focal_length: Focal length of the camera in pixels.

    Returns:
    - d: The estimated depth (Z-coordinate).
    """
    try:
        # Accessing eye positions
        if EYE_DISTANCE_INDEX['left_eye'] < len(face) and EYE_DISTANCE_INDEX['right_eye'] < len(face):
            leftEye = face[EYE_DISTANCE_INDEX['left_eye']]
            rightEye = face[EYE_DISTANCE_INDEX['right_eye']]
        else:
            raise IndexError("Eye indices are out of the range of the face array.")

        # Calculate pixel distance between eyes
        w, _ = CVZONE_DETECTOR_MAX_ONE.findDistance(leftEye, rightEye)

        # Convert interocular distance from cm to pixels
        sensor_width_mm = 48  # Placeholder, replace with actual sensor width if known
        inter_in_pixels = (INTEROCULAR_DISTANCE / sensor_width_mm) * img_width

        # Estimate depth using interocular distance in pixels and focal length in pixels
        d = int((inter_in_pixels * focal_length) / w)

        print(f"Interocular distance (pixels): {inter_in_pixels}, Pixel distance (w): {w}, Depth (d): {d}")
        return d
    except Exception as e:
        print(f"ERROR: An error occurred while trying to estimate Z: {e}")
        return None