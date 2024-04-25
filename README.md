<div align="center">
    <h1>OpenCV Face Tracking Server</h1>
    <i>A repository by: Kjetil Indrehus, Sander Hauge & Martin Johannessen</i>
</div>

<div align="center">
    <br />
    <a href="https://www.python.org/downloads/release/python-3100/">
        <img alt="click" src="https://img.shields.io/badge/Python%20Version-3.10-blue" />
    </a>
</div> <br />

![Screenshot from 2024-04-01 20-56-30](https://github.com/RIT-NTNU-Bachelor/OpenCV_Server/assets/66110094/c6a54bb3-ea92-4ba2-9d1c-49edf7c3dc8d)

(**Note:** The image is for illustrating the server tracking a users face, and then sending the coordinates over UDP)

This project implements a server application using OpenCV for real-time face tracking. It detects the face of a user via a webcam and sends the coordinates to a client application using UDP protocol. This allows for the development of interactive applications that respond to user's face position in real-time.

> This repository represents a component of the software for the thesis "User-face tracking for Unreal Interface". The thesis is authored by Martin Hegnum Johannessen, Kjetil Karstensen Indrehus, and Sander Tøkje Hauge. Feel free to see to checkout the [Github Organization created for the thesis](https://github.com/RIT-NTNU-Bachelor)


### Table of Contents

**[Requirements](#Requirements)**<br>
**[Installation](#Installation)**<br>
**[Usage](#Usage)**<br>
**[Unit testing](#Unit-testing)**<br>
**[Scripts](#Scripts)**<br>
**[Example Code](#Example-Code)**<br>
**[Project Structure](#Project-Structure)**<br>
**[System Description](#System-Description)**<br>
**[Case Studies](#Case-Studies)**<br>
**[License](#License)**<br>


## Requirements 

To successfully run this project, the following requirements must be met: 
- Python Version 3.10.X
- Installed all packages listed in `requirements.txt`
- Access to webcam. Please ensure that your PC has a built-in webcam or is connected to an external webcam.


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the  OpenCV Server. All packages that are required are found within the `requirements.txt` file. The server is developed and tested with python 3.10.X, so make sure that the current version that of python is the same or a higher version. 


### Installation without Virtual Environment

Install the packages with pip: 

```terminal
pip install -r requirements.txt
```

`setup.sh` does this for you, and also checks if you have the correct Python version. 

### Installation with Virtual Environment (VS Code Setup)

1. Open the command pallet by pressing <kbd>Ctrl</kbd>+<kbd>Alt</kbd> + <kbd>p</kbd>
2. Write the command `Python:Create Environment...`, and select it:

![Screenshot from 2024-04-19 13-57-58](https://github.com/RIT-NTNU-Bachelor/OpenCV_Server/assets/66110094/878971c9-5888-4292-b5cb-b428b5dd5306)

4. Select `Venv`

![Screenshot from 2024-04-19 13-58-29](https://github.com/RIT-NTNU-Bachelor/OpenCV_Server/assets/66110094/733c5b6a-2d4f-43c0-83db-618bd086b71b)


6. Select your Python 3.10.x interpreter:

![Screenshot from 2024-04-19 13-59-12](https://github.com/RIT-NTNU-Bachelor/OpenCV_Server/assets/66110094/94050719-77b0-4464-8f69-e3f1e2f16a3a)

7. Check of the box for installing the `requirements.txt`:

![Screenshot from 2024-04-19 13-59-50](https://github.com/RIT-NTNU-Bachelor/OpenCV_Server/assets/66110094/c2f30447-e2a0-4f38-9d4e-f5144906fbaf)

8. Wait until the environment is set up! In the terminal, you will get the `venv` text before your command prompt like this. That means that it successfully created, and the environment is active:

![Screenshot from 2024-04-19 14-05-08](https://github.com/RIT-NTNU-Bachelor/OpenCV_Server/assets/66110094/82be9c71-5922-4f41-9ab9-b308f6954948)


10. Use the commands in the [Usage](#usage) section, or the scripts (`run.sh` / `test.sh`)


## Usage

The repository includes code for all of the following methods:
- Haar
- Hog
- DNN
- MMOD
- CVZone

The source code for each method is found within the `src/models/code/` folder, and the complementary pre-trained models are in the `src/models/trained_models/` folder. 

Every face detection algorithm has the same function signature. They differ only in extra parameters. Most of these already have a default value:
```python
def detect_face_xxx(img, ... , detectMultipleFaces=False):
```

The main file is structured in a way to make it easy to change what face detection algorithm is being used. Run the main file with the following command: 

```terminal
python src/main.py
```


## Unit testing

This repository includes unit test for testing the different algorithms. The following are tested: 
- All models are loaded correctly from the dependencies.
- All models do not find a face in an image without a face
- All models find the single face in an image
- All models find both faces in an image

To run all unit tests, simply run:

```terminal
python -m unittest -v
```


## Scripts

The repository also contains bash scripts. These scripts simplify the process of setting up, unit test and run the project. 
Note that this is additional material, and are not required for neither setup, testing and running the project. Each script can be run by first adding executable permissions to them and run them in the terminal. The following scripts are included:

- Run: File `run.sh`
- Unit test: File `test.sh` 
- Setup: File `setup.sh` 

Run a script by: 

```bash
chmod +x ./run.sh
./run.sh
```


### Example Code
This repository uses a modular design for all the face detection modules. This makes it easy to change the code and switch between models. For instance: 

```python
# Import OpenCV for reading image
import cv2

# Importing the functions for the face detection models 
from models.code.dnn import detect_face_dnn
from models.code.haar import detect_face_haar
from models.code.hog import detect_face_hog
from models.code.cvzone import detect_face_cvzone
from models.code.mmod import detect_face_mmod

# Import the detectors from src/constants.py
from constants import DNN_NET, CVZONE_DETECTOR_MAX_ONE, HAAR_CLASSIFIER, HOG_DETECTOR, MMOD_DETECTOR

# Opening a sample image
image_path = "path/to/your/image.jpg"
img = cv2.imread(image_path)

# Example of using a method to detect face models 
faces_dnn = detect_face_dnn(img, DNN_NET)
faces_haar = detect_face_haar(img, HAAR_CLASSIFIER)
faces_hog = detect_face_hog(img, HOG_DETECTOR)
faces_cvzone = detect_face_cvzone(img, CVZONE_DETECTOR_MAX_ONE)
faces_mmod = detect_face_mmod(img,MMOD_DETECTOR)

# Print the amount of faces found within the image
print(f"DNN Detected Faces: {faces_dnn}")
print(f"Haar Detected Faces: {faces_haar}")
print(f"HOG Detected Faces: {faces_hog}")
print(f"CVZone Detected Faces: {faces_cvzone}")
print(f"MMOD Detected Faces: {faces_mmod}")


```

## Project Structure

```
OpenCV_Server

│    
├─ .gitattributes
├─ .gitignore
├─ LICENSE
├─ README.md
├─ data
│  ├─  unit_test_output
│  │   ├─ cvzone_one_face_output.png
│  │   ├─ cvzone_two_faces_output.png
│  │   ├─ dnn_one_face_output.png
│  │   ├─ dnn_two_faces_output.png
│  │   ├─ haar_one_face_output.png
│  │   ├─ haar_two_faces_output.png
│  │   ├─ hog_one_face_output.png
│  │   ├─ hog_two_faces_output.png
│  │   ├─ mmod_one_face_output.png
│  │   └─ mmod_two_faces_output.png
│  └─ unit_test
│     ├─ Lenna.png
│     ├─ Peppers.png
│     └─ TwoFaces.jpg

├─ docs
│  ├─ cvzone.md
│  ├─ depth.md
│  ├─ dnn.md
│  ├─ haar.md
│  ├─ hog.md
│  ├─ index.md
│  ├─ mmod.md
│  ├─ scripts.md
│  └─ udp_server.md
├─ mkdocs.yml
├─ requirements.txt
├─ run.sh
├─ scripts
│  ├─ camera-center.py
│  └─ run-all.py
├─ setup.sh
├─ src
│  ├─ constants.py
│  ├─ depth.py
│  ├─ main.py
│  ├─ models
│  │  ├─ __init__.py
│  │  ├─ code
│  │  │  ├─ __init__.py
│  │  │  ├─ cvzone.py
│  │  │  ├─ dnn.py
│  │  │  ├─ haar.py
│  │  │  ├─ hog.py
│  │  │  └─ mmod.py
│  │  └─ trained_models
│  │     ├─ deploy.prototxt
│  │     ├─ haarcascade_frontalface_default.xml
│  │     ├─ mmod_human_face_detector.dat
│  │     ├─ opencv_face_detector.pbtxt
│  │     ├─ opencv_face_detector_uint8.pb
│  │     └─ res10_300x300_ssd_iter_140000_fp16.caffemodel
│  └─ udp_server.py
├─ test.sh
└─ tests
   ├─ __init__.py
   ├─ test_load_model.py
   ├─ test_models_negative.py
   ├─ test_models_one_face.py
   ├─ test_models_two_faces.py
   └─ test_utils.py

```

## System Description

The system consists of two primary components: the Face Tracking server and the Rendering Engine Client. The Face Tracking Server uses OpenCV to process frames in real-time. For each frame, it detects the position of the face. Once the position is ascertained, these coordinates are sent using the UDP (User Datagram Protocol) to ensure fast and efficient transmission. The Rendering Engine Client then takes over by parsing the received coordinates. Using this data, the engine re-renders the scene to align with the new position of the face. This cycle of detection, transmission, and rendering continues seamlessly with each frame, allowing for a responsive and dynamic integration of face tracking data with the rendered content: 

![image](https://github.com/RIT-NTNU-Bachelor/Unreal-facetracking-client/assets/66110094/5c48a2a6-4d80-40b1-8c07-f7020125e143)

## Case Studies

**NOTE:** As of PR [#54](https://github.com/RIT-NTNU-Bachelor/OpenCV_Server/issues/54), this repo does not contain the case studies mentioned in the thesis. It is moved to its own repository. The arguments for the separation are: 

- **Reduced Repository Size:** Separating case studies into their own repository keeps the main server lightweight, enhancing performance and ease of cloning and setup for new users.
- **Improved Code Cohesion:** By isolating the case studies, we can maintain a high level of cohesion within each repository, ensuring that each component focuses on a specific set of responsibilities.
- **Simplified Version Control:** With separate repositories, tracking and reverting changes becomes more straightforward, as the history of commits in each repo is more relevant to its specific content.
- **Consistent Codebase Across Repositories:** Maintaining the same `models` module in both repositories ensures consistency and interoperability of features, critical for functionality and integration testing.
- **Focused Issue Tracking and Documentation:** Issues, pull requests, and documentation can be more precisely tailored to the repository's content, improving clarity and effectiveness in project management.
- **Client Convenience:** Separating the case studies into their own repository simplifies the user experience. The user can utilize each repo without the added complexity of the other repository.  

**Important:** The case study repository utilizes the same code as found in the `src/models` module of the main server. As such, any significant changes made to the `models` module must be reflected in both repositories to ensure consistency with the case studies

## License

[MIT](https://github.com/RIT-NTNU-Bachelor/OpenCV_Server/blob/main/LICENSE)
