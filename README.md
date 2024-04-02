# OpenCV Face Tracking Server

This project implements a server application using OpenCV for real-time face tracking. It detects the face of a user via a webcam and sends the coordinates to a client application using UDP protocol. This allows for the development of interactive applications that respond to user's face position in real-time.

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

## License

[MIT](https://github.com/RIT-NTNU-Bachelor/OpenCV_Server/blob/main/LICENSE)