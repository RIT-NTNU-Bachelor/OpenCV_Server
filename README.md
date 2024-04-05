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

(**Note:** The image shows the server tracking a users face, and then sending the coordinates over UDP)

This project implements a server application using OpenCV for real-time face tracking. It detects the face of a user via a webcam and sends the coordinates to a client application using UDP protocol. This allows for the development of interactive applications that respond to user's face position in real-time.

> This repository represents a component of the software for the thesis "User-face tracking for Unreal Interface". The thesis is authored by Martin Hegnum Johannessen, Kjetil Karstensen Indrehus, and Sander Tøkje Hauge. Feel free to see to checkout the [Github Organization created for the thesis](https://github.com/RIT-NTNU-Bachelor)


### Table of Contents

**[Requirements](#Requirements)**<br>
**[Installation](#Installation)**<br>
**[Usage](#Usage)**<br>
**[System Description](#System-Description)**<br>
**[Case Studies](#Case-Studies)**<br>
**[License](#License)**<br>


## Requirements 

To successfully run this project, the following requirements must be met: 
- Python Version 3.10.X
- Installed all packages in `requirements.txt`
- Access to webcam. Please confirm that your PC has a built-in webcam or connect an external webcam.


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the  OpenCV Server. All packages that are required are found within the `requirements.txt` file. The server is developed and tested with python 3.10.X, so make sure that the current version that of python is the same or a higher version. Install the packages with pip: 

```bash
pip install -r requirements.txt
```

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

The main file is structured in a way to make it easy to change what face detection algorithm is being used

```python
# TODO: Add clean code of main.py when finished
```


## System Description

The system consists of two primary components: the Face Tracking server and the Rendering Engine Client. The Face Tracking Server uses OpenCV to process frames in real-time. For each frame, it detects the position of the face. Once the position is ascertained, these coordinates are sent using the UDP (User Datagram Protocol) to ensure fast and efficient transmission. The Rendering Engine Client then takes over by parsing the received coordinates. Using this data, the engine re-renders the scene to align with the new position of the face. This cycle of detection, transmission, and rendering continues seamlessly with each frame, allowing for a responsive and dynamic integration of face tracking data with the rendered content: 

![image](https://github.com/RIT-NTNU-Bachelor/Unreal-facetracking-client/assets/66110094/5c48a2a6-4d80-40b1-8c07-f7020125e143)

## Case Studies

For deciding the best face detection algorithm for the thesis, see the case studies created in the `analysis/` folder (created in JupyterHub files). The two main factors for a real-time face detection system is efficacy and accuracy. These factors was also discussed, and agreed upon with the bachelor thesis client. There are two case studies

1. Case Study - Comparing Accuracy for Real-Time Face Detection Models
This case study measures the FPS over a dataset with ~2800 images. It presents the result in plotted in multiple graphs. See the [case study for more information](https://github.com/RIT-NTNU-Bachelor/OpenCV_Server/blob/main/analysis/compare_face_detection_accuracy.ipynb)

2. Case Study - Comparing Efficacy for a Real-Time Face Detection System
This case study measures the memory usage of each algorithm. It plots the result as peak mega byte usage during a stress test of the algorithms. For this case, the python package `memory_profiler` is used. The result is plotted in multiple graphs. See the [case study for more information](https://github.com/RIT-NTNU-Bachelor/OpenCV_Server/blob/main/analysis/compare_face_detection_efficacy.ipynb)


**Note:** The case studies does not represent an absolute truth. There are most likely something that makes it not objectively true. Use them with caution.

Code Owners are allowed to use the Github Workflow that triggers the compiling of the case studies. The workflow also downloads the dataset needed. The code for the workflow can be found [here](https://github.com/RIT-NTNU-Bachelor/OpenCV_Server/blob/main/.github/workflows/jupiterhub_workflow.yml). 

Estimated time of compiling both case studies (Note: the repository includes compiled notebooks): 

```txt
~ 2h, 46 min
```

## License

[MIT](https://github.com/RIT-NTNU-Bachelor/OpenCV_Server/blob/main/LICENSE)
