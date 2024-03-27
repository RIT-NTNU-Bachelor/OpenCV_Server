import cv2
import dlib
import time
import numpy as np
import matplotlib.pyplot as plt

haar_cascade = cv2.CascadeClassifier('../src/models/trained_models/haarcascade_frontalface_default.xml')
dnn_model = cv2.dnn.readNetFromCaffe('../src/models/trained_models/deploy.prototxt', '../src/models/trained_models/res10_300x300_ssd_iter_140000_fp16.caffemodel')
hog_detector = dlib.get_frontal_face_detector()
mmod_detector = dlib.cnn_face_detection_model_v1('../src/models/trained_models/mmod_human_face_detector.dat')

def benchmark_method(method, frame, iterations=1000):
    start = time.time()
    for _ in range(iterations):
        _ = method(frame)
    end = time.time()
    return (end - start) / iterations

def dnn_detection(frame):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 117, 123), False, False)
    dnn_model.setInput(blob)
    return dnn_model.forward()

# TODO: apply CUDA for GPU compiling and one without (to get with and without)
def mmod_detection(frame):
    # Convert to RGB as dlib expects images in rgb format
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mmod_detector(rgb_image, 1)

def haar_detection(img):
    return haar_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=10)

# Load and preprocess image
image = cv2.imread('../data/test_data/imgname')
image = cv2.resize(image, (300, 300))

# Convert to grayscale for Haar and HOG
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Benchmarking usage
dnn_times = [benchmark_method(dnn_detection, image) for _ in range(10)]
#mmod_cpu_times = [benchmark_method(mmod_detection, image) for _ in range(10)]
#mmod_gpu_times = [benchmark_method(mmod_detection, image) for _ in range(10)]
haar_times = [benchmark_method(haar_detection, gray_image) for _ in range(10)]
hog_times = [benchmark_method(hog_detector, gray_image) for _ in range(10)]

# Calculate average times
haar_avg = np.mean(haar_times)
dnn_avg = np.mean(dnn_times)
hog_avg = np.mean(hog_times)
#mmod_cpu_avg = np.mean(mmod_cpu_times)
#mmod_gpu_avg = np.mean(mmod_gpu_times)

# Print results
print(f"Average time (in seconds) for Haar: {haar_avg}")
print(f"Average time (in seconds) for DNN: {dnn_avg}")
print(f"Average time (in seconds) for HOG: {hog_avg}")
#print(f"Average time (in seconds) for MMOD (CPU): {mmod_cpu_avg}")
#print(f"Average time (in seconds) for MMOD (GPU): {mmod_gpu_avg}")

haar_fps = 1 / haar_avg
dnn_fps = 1 / dnn_avg
hog_fps = 1 / hog_avg
#mmod_cpu_fps = 1 / mmod_cpu_avg
#mmod_gpu_fps = 1 / mmod_gpu_fps

labels = ['Haar', 'DNN', 'HoG', 'MMOD (GPU)', 'MMOD (No-GPU)']
performance = [haar_fps, dnn_fps, hog_fps]
index = np.arange(len(labels))

# Plotting the bar graph
plt.bar(index, performance, alpha=0.8, color=['blue', 'orange', 'green', 'red', 'purple'])

# Add titles and labels
plt.title('Speed Comparison of Face Detection Methods')
plt.ylabel('Speed (FPS)')
plt.xlabel('Method')
plt.xticks(index, labels, rotation=30)
plt.ylim(0, max(performance) + 10)

# Display the values on the bars
for i in range(len(index)):
    plt.text(i, performance[i] + 5, str(round(performance[i], 2)), ha='center', va='bottom')

# Show plot
plt.tight_layout()
plt.show()
