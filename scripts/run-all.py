import argparse
import os
import time
import cv2
import dlib
import numpy as np

# This script is inspired by learnopencv's script located at:
# https://github.com/spmallick/learnopencv/blob/master/FaceDetectionComparison/run-all.cpp

# Load models
face_cascade = cv2.CascadeClassifier("../src/models/trained_models/haarcascade_frontalface_default.xml")
hog_detector = dlib.get_frontal_face_detector()
mmod_detector = dlib.cnn_face_detection_model_v1("../src/models/trained_models/mmod_human_face_detector.dat")

def detect_faces_haar(cascade, frame, in_height=300):
    frame_copy = frame.copy()
    frame_height, frame_width = frame_copy.shape[:2]
    in_width = int((frame_width / frame_height) * in_height)

    scale_height = frame_height / in_height
    scale_width = frame_width / in_width

    resized_frame = cv2.resize(frame_copy, (in_width, in_height))
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    faces = cascade.detectMultiScale(gray_frame)
    bboxes = []
    for (x, y, w, h) in faces:
        x1, y1, x2, y2 = x, y, x + w, y + h
        bbox = [int(dim) for dim in (x1 * scale_width, y1 * scale_height, x2 * scale_width, y2 * scale_height)]
        bboxes.append(bbox)
        cv2.rectangle(frame_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    return frame_copy, bboxes

def detect_faces_dnn(net, frame, conf_threshold=0.7):
    frame_copy = frame.copy()
    frame_height, frame_width = frame_copy.shape[:2]
    blob = cv2.dnn.blobFromImage(frame_copy, 1.0, (300, 300), [104, 117, 123], False, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame_copy, bboxes

def detect_faces_dlib(detector, frame, in_height=300):
    frame_copy = frame.copy()
    frame_height, frame_width = frame_copy.shape[:2]
    in_width = int((frame_width / frame_height) * in_height)

    scale_height = frame_height / in_height
    scale_width = frame_width / in_width

    resized_frame = cv2.resize(frame_copy, (in_width, in_height))
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    face_rects = detector(rgb_frame, 0)
    bboxes = []
    for face_rect in face_rects:
        if isinstance(face_rect, dlib.rectangle):  # For HOG detector
            x1, y1, x2, y2 = face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()
        else:  # For MMOD detector
            x1, y1, x2, y2 = face_rect.rect.left(), face_rect.rect.top(), face_rect.rect.right(), face_rect.rect.bottom()
        bbox = [int(dim) for dim in (x1 * scale_width, y1 * scale_height, x2 * scale_width, y2 * scale_height)]
        bboxes.append(bbox)
        cv2.rectangle(frame_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    return frame_copy, bboxes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face detection")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"], help="Device to use")
    parser.add_argument("--net_type", type=str, default="caffe", choices=["caffe", "tf"], help="Type of network to run")
    args = parser.parse_args()

    if args.net_type == "caffe":
        model_file = "../src/models/trained_models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        config_file = "../src/models/trained_models/deploy.prototxt"
        net = cv2.dnn.readNetFromCaffe(config_file, model_file)
    else:
        model_file = "../src/models/trained_models/opencv_face_detector_uint8.pb"
        config_file = "../src/models/trained_models/opencv_face_detector.pbtxt"
        net = cv2.dnn.readNetFromTensorflow(model_file, config_file)

    if args.device == "cpu":
        net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    cap = cv2.VideoCapture(args.video if args.video else 0)

    output_folder = "output-dnn-videos"
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, os.path.basename(args.video)[:-4] + ".avi" if args.video else "grabbed_from_camera.avi")
    vid_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc("M", "J", "P", "G"), 25, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    frame_count = 0
    tt_haar, tt_dnn, tt_hog, tt_mmod = 0, 0, 0, 0

    while True:
        has_frame, frame = cap.read()
        if not has_frame:
            break

        frame_count += 1
        t = time.time()
        out_haar, _ = detect_faces_haar(face_cascade, frame)
        tt_haar += time.time() - t
        fps_haar = frame_count / tt_haar if tt_haar > 0 else 0

        t = time.time()
        out_dnn, _ = detect_faces_dnn(net, frame)
        tt_dnn += time.time() - t
        fps_dnn = frame_count / tt_dnn if tt_dnn > 0 else 0

        t = time.time()
        out_hog, _ = detect_faces_dlib(hog_detector, frame)
        tt_hog += time.time() - t
        fps_hog = frame_count / tt_hog if tt_hog > 0 else 0

        t = time.time()
        out_mmod, _ = detect_faces_dlib(mmod_detector, frame)
        tt_mmod += time.time() - t
        fps_mmod = frame_count / tt_mmod if tt_mmod > 0 else 0

        # Add text to display FPS and method name
        cv2.putText(out_haar, f"OpenCV Haar FPS: {fps_haar:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(out_dnn, f"OpenCV DNN {args.device.upper()} FPS: {fps_dnn:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(out_hog, f"Dlib HOG FPS: {fps_hog:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(out_mmod, f"Dlib MMOD {args.device.upper()} FPS: {fps_mmod:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        combined_top = np.hstack([out_haar, out_dnn])
        combined_bottom = np.hstack([out_hog, out_mmod])
        combined = np.vstack([combined_top, combined_bottom])


        cv2.imshow("Face Detection Comparison", combined)
        vid_writer.write(combined)

        if cv2.waitKey(5) == 27:
            break

    cv2.destroyAllWindows()
    vid_writer.release()