import os
from tensorflow import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
sns.set(style="whitegrid", color_codes=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator


import warnings
warnings.filterwarnings('ignore')

sciezka_danych = sciezka_danych = r'C:\Users\Szymon\Documents\Rice_Classification\Rice_Classification\Rice_Image_Dataset\Rice_Image_Dataset'


lista_obrazow = []
etykiety_obrazow = []

for folder in os.listdir(sciezka_danych):
    
    pelna_sciezka_folderu = os.path.join(sciezka_danych, folder)
    if not os.path.isdir(pelna_sciezka_folderu):
        continue
  
    for nazwa_pliku in os.listdir(pelna_sciezka_folderu):
        pelna_sciezka_obrazu = os.path.join(pelna_sciezka_folderu, nazwa_pliku)
        lista_obrazow.append(pelna_sciezka_obrazu)
        etykiety_obrazow.append(folder)
 
zestaw_danych = pd.DataFrame({'obraz': lista_obrazow, 'etykieta': etykiety_obrazow})

zestaw_danych.head()

os_y = sns.countplot(x=zestaw_danych.etykieta, palette="coolwarm")  

os_y.set_xlabel("Klasa")  
os_y.set_ylabel("Liczba próbek")  

plt.xticks(rotation=45)
plt.show()


rysunek = plt.figure(figsize=(16, 16))
uklad_siatki = GridSpec(5, 4, figure=rysunek)

for indeks, klasa in enumerate(zestaw_danych['etykieta'].unique()):
    obrazy_do_wyswietlenia = zestaw_danych[zestaw_danych['etykieta'] == klasa]['obraz'].values[:4]
    
    for pod_indeks, obraz_sciezka in enumerate(obrazy_do_wyswietlenia):
        os_podwykres = rysunek.add_subplot(uklad_siatki[indeks, pod_indeks])
        os_podwykres.imshow(plt.imread(obraz_sciezka))
        os_podwykres.axis('off')  
    
    os_podwykres.text(300, 100, klasa, fontsize=20, color='navy')

plt.show()

obrazy_train, obrazy_test, etykiety_train, etykiety_test = train_test_split(
    zestaw_danych['obraz'], zestaw_danych['etykieta'], test_size=0.2, random_state=42
)

zestaw_treningowy = pd.DataFrame({'obraz': obrazy_train, 'etykieta': etykiety_train})
zestaw_testowy = pd.DataFrame({'obraz': obrazy_test, 'etykieta': etykiety_test})

kod_etykiet = LabelEncoder()
etykiety_train = kod_etykiet.fit_transform(etykiety_train)
etykiety_test = kod_etykiet.transform(etykiety_test)


rozmiar_obrazu = (50, 50)
batch_rozmiar = 32

generator_danych = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='reflect'
)

train_generator = generator_danych.flow_from_dataframe(
    zestaw_treningowy,
    x_col='obraz',
    y_col='etykieta',
    target_size=rozmiar_obrazu,
    batch_size=batch_rozmiar,
    class_mode='categorical',
    shuffle=True
)

test_generator = generator_danych.flow_from_dataframe(
    zestaw_testowy,
    x_col='obraz',
    y_col='etykieta',
    target_size=rozmiar_obrazu,
    batch_size=batch_rozmiar,
    class_mode='categorical',
    shuffle=False
)

model_architektura = Sequential()
model_architektura.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(50, 50, 3)))
model_architektura.add(MaxPooling2D(pool_size=(2, 2)))
model_architektura.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model_architektura.add(MaxPooling2D(pool_size=(2, 2)))
model_architektura.add(Flatten())
model_architektura.add(Dense(128, activation='relu'))
model_architektura.add(Dense(5, activation='softmax'))

model_architektura.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

historia = model_architektura.fit(train_generator, epochs=5, validation_data=test_generator)

plt.figure(figsize=(10, 6))
plt.plot(historia.history['accuracy'], marker='o', label="Trening")
plt.plot(historia.history['val_accuracy'], marker='s', label="Walidacja")
plt.title("Porównanie dokładności", fontsize=15)
plt.xlabel("Epoka")
plt.ylabel("Dokładność")
plt.legend()
plt.show()

wyniki = model_architektura.evaluate(test_generator)
print(f'Dokładność modelu: {wyniki[1]:.2f}')

model_architektura.summary()

model_architektura.save('model_ryz.h5')
print("Model został zapisany!")

from sklearn.metrics import confusion_matrix
import seaborn as sns

predicted_labels = model_architektura.predict(test_generator)
predicted_labels = np.argmax(predicted_labels, axis=1)  


true_labels = test_generator.classes


conf_matrix = confusion_matrix(true_labels, predicted_labels)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=kod_etykiet.classes_, yticklabels=kod_etykiet.classes_)
plt.title('Macierz Pomyłek (Confusion Matrix)')
plt.xlabel('Przewidywane Etykiety')
plt.ylabel('Rzeczywiste Etykiety')
plt.show()


