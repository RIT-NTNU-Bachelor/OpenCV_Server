# Additional Scripts

## Camera Initialization and Configuration Value Calculation

This script initializes a camera, captures a frame, and calculates central coordinates for configuration in the [Unreal Engine Client](https://github.com/RIT-NTNU-Bachelor/Unreal-facetracking-client)

### Script Overview

- **Camera Initialization**: Opens the camera to capture images.
- **Image Capture**: Attempts to capture a single frame.
- **Error Handling**: Verifies camera availability and successful image capture.
- **Configuration Values**: Calculates and prints the center coordinates of the image for Unreal Engine configuration.


The configured values that you retrive and need to add to the Client is:

- **CX**: Center of the frame along the x axis
- **CY**: Center of the frame along the y axis