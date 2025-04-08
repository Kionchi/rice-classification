import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tkinter import Tk, filedialog
import os

def main():

    model_path = "C:\\Users\\Szymon\\Documents\\Rice_Classification\\Rice_Classification\\model_ryz.h5"  
    if not os.path.exists(model_path):
        print(f"Plik modelu '{model_path}' nie został znaleziony.")
        return

    print("Ładowanie modelu...")
    model = load_model(model_path)
    
    class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']  

    Tk().withdraw()  
    print("Wybierz plik obrazu...")
    image_path = filedialog.askopenfilename(title="Wybierz obraz ziarenka ryżu", filetypes=[("Pliki obrazów", "*.jpg;*.jpeg;*.png")])

    if not image_path:
        print("Nie wybrano pliku.")
        return

    print(f"Wybrano plik: {image_path}")

    print("Przetwarzanie obrazu...")
    img_size = (50, 50)  
    try:
        img = load_img(image_path, target_size=img_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  

    except Exception as e:
        print(f"Błąd podczas przetwarzania obrazu: {e}")
        return

    print("Klasyfikowanie...")
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class]

    print(f"Wynik: {class_names[predicted_class]} (Pewność: {confidence:.2f})")

if __name__ == "__main__":
    main()
