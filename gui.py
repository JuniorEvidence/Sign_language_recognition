import tkinter as tk
from tkinter import filedialog
from tkinter import *
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2

def load_model_from_json(json_path, weights_path):
    with open(json_path, "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights(weights_path)
    return model


top = tk.Tk()
top.geometry('800x600')
top.title('Sign Language Detector')
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

image_sign_language_model = load_model_from_json(r"C:\Users\LENEVO\OneDrive\Desktop\Assignment 2\model_a.json", r"C:\Users\LENEVO\OneDrive\Desktop\Assignment 2\model_weight.weights.h5")
video_sign_language_model = load_model(r"C:\Users\LENEVO\OneDrive\Desktop\Assignment 2\video_sign_language_model_wegith.weights.h5")


VALID_CLASSES = ["how are you", "what is your name", "who are you"]
IMAGE_CLASS_LABELS = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}
VIDEO_CLASS_LABELS = {0: 'how are you', 1: "what is your name", 2: 'who are you'}

def extract_frames(video_path, seq_length, img_size):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < seq_length:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (img_size, img_size))
        frames.append(frame)
    cap.release()
    frames = np.array(frames)
    if len(frames) < seq_length:
        frames = np.pad(frames, ((0, seq_length - len(frames)), (0, 0), (0, 0), (0, 0)), 'constant')
    return frames

def detect_image(file_path):
     image = cv2.imread(file_path)
     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     image = cv2.resize(image, (64, 64))  # Adjust size based on your image model's requirement
     image = image / 255.0
     image = np.expand_dims(image, axis=0)
     image = np.expand_dims(image, axis=-1)

     prediction = image_sign_language_model.predict(image)
     predicted_class = np.argmax(prediction, axis=1)
     result = IMAGE_CLASS_LABELS.get(predicted_class[0], "Unknown")

     print("Predicted Sign is: " + result)
     label1.configure(foreground="#011638", text=result)

def detect_video(file_path):
    seq_length = 30
    img_size = 64   

    frames = extract_frames(file_path, seq_length, img_size)
    frames = frames / 255.0
    frames = np.expand_dims(frames, axis=0)
    
    prediction = video_sign_language_model.predict(frames)
    predicted_class = np.argmax(prediction, axis=1)
    result = VIDEO_CLASS_LABELS.get(predicted_class[0], "Unknown")

    if result == "what is your name":
        result = 'who are you'
    elif result == 'who are you':
        result = "what is your name"

    if result not in VALID_CLASSES:
        result = "Invalid input, please upload valid data."

    print("Predicted Sign is: " + result)
    label1.configure(foreground="#011638", text=result)

def show_detect_button(file_path, file_type):
    if file_type == 'image':
        detect_b = Button(top, text="Detect Sign", command=lambda: detect_image(file_path), padx=10, pady=5)
    elif file_type == 'video':
        detect_b = Button(top, text="Detect Sign", command=lambda: detect_video(file_path), padx=10, pady=5)

    detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    detect_b.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')
        show_detect_button(file_path, 'image')
    except Exception as e:
        print(f"Error: {e}")

def upload_video():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = cv2.VideoCapture(file_path)
        ret, frame = uploaded.read()
        uploaded.release()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(frame)

        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')
        show_detect_button(file_path, 'video')
    except Exception as e:
        print(f"Error: {e}")

upload_image_button = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload_image_button.configure(background="#364156", foreground='white', font=('arial', 20, 'bold'))
upload_image_button.pack(side='bottom', pady=20)

upload_video_button = Button(top, text="Upload Video", command=upload_video, padx=10, pady=5)
upload_video_button.configure(background="#364156", foreground='white', font=('arial', 20, 'bold'))
upload_video_button.pack(side='bottom', pady=20)

sign_image.pack(side='bottom', expand='True')
label1.pack(side='bottom', expand='True')
heading = Label(top, text='Sign Language Detector', pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground="#364156")
heading.pack()
top.mainloop()