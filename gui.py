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

# Initialize the GUI application
top = tk.Tk()
top.geometry('800x600')
top.title('Alphabet Sign Language Detector')
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

# Load the model
image_sign_language_model = load_model_from_json(
    r"C:\Users\LENEVO\OneDrive\Desktop\7th sem\EMEL\model_a.json", 
    r"C:\Users\LENEVO\OneDrive\Desktop\7th sem\EMEL\model_weight.weights.h5"
)

# Define alphabet class labels
IMAGE_CLASS_LABELS = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 
                      10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 
                      18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

# Function to detect alphabet from an image
def detect_image(file_path):
    try:
        # Preprocess the image
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (48,48))  # Adjust size based on your model's requirement
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=-1)

        # Predict the class
        prediction = image_sign_language_model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)
        result = IMAGE_CLASS_LABELS.get(predicted_class[0], "Unknown")

        print("Predicted Alphabet is: " + result)
        label1.configure(foreground="#011638", text="Predicted Alphabet: " + result)
    except Exception as e:
        print(f"Error during detection: {e}")
        label1.configure(foreground="red", text="Error in processing the image.")

# Function to handle image upload
def upload_image():
    try:
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if not file_path:
            return  # If no file is selected, return early

        # Display the uploaded image
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')
        
        # Create and display the "Detect" button
        detect_b = Button(top, text="Detect Alphabet", command=lambda: detect_image(file_path), padx=10, pady=5)
        detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
        detect_b.place(relx=0.79, rely=0.46)
    except Exception as e:
        print(f"Error: {e}")

# Add a button for uploading images
upload_image_button = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload_image_button.configure(background="#364156", foreground='white', font=('arial', 20, 'bold'))
upload_image_button.pack(side='bottom', pady=20)

# Configure layout elements
sign_image.pack(side='bottom', expand='True')
label1.pack(side='bottom', expand='True')
heading = Label(top, text='Alphabet Sign Language Detector', pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground="#364156")
heading.pack()

# Run the GUI
top.mainloop()
