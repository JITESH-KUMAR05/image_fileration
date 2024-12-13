import os
import shutil
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np

def select_folder():
    folder_path = filedialog.askdirectory()
    folder_entry.delete(0, tk.END)
    folder_entry.insert(0, folder_path)

def detect_person(image_path):
    # Load pre-trained model (e.g., YOLO)
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Load image
    img = cv2.imread(image_path)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Extract person features
    person_features = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == 0 and confidence > 0.5:  # Class ID 0 is for person
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                person_features.append((x, y, w, h))
    return person_features

def match_person(input_features, features):
    # Implement a method to compare features and determine if the person matches
    # For simplicity, we assume that if there are any features detected, it's a match
    if not input_features or not features:
        return False

    # Compare the bounding boxes (x, y, w, h) of the detected persons
    for (x1, y1, w1, h1) in input_features:
        for (x2, y2, w2, h2) in features:
            if abs(x1 - x2) < 50 and abs(y1 - y2) < 50 and abs(w1 - w2) < 50 and abs(h1 - h2) < 50:
                return True
    return False

def start_processing():
    folder_path = folder_entry.get()
    if not folder_path:
        print("No folder selected")
        return

    # this is the path to the input image
    input_image_path = "JITESH-min.jpg" 
    input_features = detect_person(input_image_path)
    if not input_features:
        print("No person detected in the input image")
        return

    new_folder = os.path.join(folder_path, "filtered_images")
    os.makedirs(new_folder, exist_ok=True)
    print(f"Created new folder: {new_folder}")

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            features = detect_person(image_path)
            if match_person(input_features, features):
                shutil.copy(image_path, new_folder)
                print(f"Copied {filename} to {new_folder}")

# Create the user interface
root = tk.Tk()
root.title("Image Folder Selector")

tk.Label(root, text="Folder Path:").pack()
folder_entry = tk.Entry(root, width=50)
folder_entry.pack()
tk.Button(root, text="Browse", command=select_folder).pack()
tk.Button(root, text="Start", command=start_processing).pack()

root.mainloop()