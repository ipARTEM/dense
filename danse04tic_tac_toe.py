"""
Самостоятельно напишите нейронную сеть, которая может стать составной частью
системы бота для игры в "Крестики-нолики". Используя подготовленную базу изображений,
создайте и обучите нейронную сеть, распознающую две категории изображений: крестики и нолики.
Добейтесь точности распознавания более 95% (accuracy)
"""
import zipfile

# Подключение класса для создания нейронной сети прямого распространения
from tensorflow.keras.models import Sequential
# Подключение класса для создания полносвязного слоя
from tensorflow.keras.layers import Dense
# Подключение оптимизатора
from tensorflow.keras.optimizers import Adam
# Подключение утилит для to_categorical
from tensorflow.keras import utils
# Подключение библиотеки для загрузки изображений
from tensorflow.keras.preprocessing import image
# Подключение библиотеки для работы с массивами
import numpy as np
# Подключение модуля для работы с файлами
import os
# Подключение библиотек для отрисовки изображений
import matplotlib.pyplot as plt
from PIL import Image
# Вывод изображения в ноутбуке, а не в консоли или файле
# %matplotlib inline

# Загрузка датасета из облака
import gdown
gdown.download('https://storage.yandexcloud.net/aiueducation/Content/base/l3/hw_pro.zip', None, quiet=True)

file_zip = zipfile.ZipFile('hw_pro.zip')
file_zip.extractall('./')
file_zip.close()

# Путь к директории с базой
base_dir = 'hw_pro'
# Создание пустого списка для загрузки изображений обучающей выборки
x_train = []
# Создание списка для меток классов
y_train = []
# Задание высоты и ширины загружаемых изображений
img_height = 20
img_width = 20
# Перебор папок в директории базы
for patch in os.listdir(base_dir):
    # Перебор файлов в папках
    for img in os.listdir(base_dir + '/' + patch):
        # Добавление в список изображений текущей картинки
        x_train.append(image.img_to_array(image.load_img(base_dir + '/' + patch + '/' + img,
                                                         target_size=(img_height, img_width),
                                                         color_mode='grayscale')))
        # Добавление в массив меток, соответствующих классам
        if patch == '0':
            y_train.append(0)
        else:
            y_train.append(1)
# Преобразование в numpy-массив загруженных изображений и меток классов
x_train = np.array(x_train)
y_train = np.array(y_train)
# Вывод размерностей
print('Размер массива x_train', x_train.shape)
print('Размер массива y_train', y_train.shape)

np.where(y_train==0)[0].shape

np.where(y_train==1)[0].shape

# Вывод примера изображения из базы
plt.imshow(np.reshape(x_train[2], (20,20)), cmap='gray')
plt.show();

# Изменение формата входных картинок с 20*20 на 400*1
x_train= x_train.reshape(102,400)
# Выход новой размерности
print(x_train.shape)


# Нормирование входных картинок
# Преобразование x_train в число с плавающей точкой
x_train=x_train.astype('float32')
# Приведение значений к диапазону от 0 до 1
x_train=x_train/255
print(x_train)

# Преобразование ответов в формат one_hot_encoding
y_train=utils.to_categorical(y_train,2)

#Создание нейронной сети

# Создание сети прямого распространения
model=Sequential()
# Создание полносвязного слоя на 1000 нейронов и relu-активацией
model.add(Dense(1000, input_dim=400,activation='relu'))
# Создание полносвязного слоя с 2мя нейронами для каждого класса и softmax-активацией
model.add(Dense(2,activation='softmax'))
# Можно использовать если ответ 0 или 1
# model.add(Dense(1,activation='sigmoid'))
# Компиляция модели
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
# model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
# Обучение моделиTrue
model.fit(x_train,y_train,batch_size=10, epochs=20,verbose=1, shuffle=False)

# Вывод структуры модели
print()
print(model.summary)