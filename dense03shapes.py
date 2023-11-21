"""
Создайте систему компьютерного зрения, которая будет определять тип геометрической фигуры. Используя подготовленную базу и шаблон ноутбука проведите серию экспериментов по перебору гиперпараметров нейронной сети, распознающей три категории изображений (треугольник, круг, квадрат).

Поменяйте количество нейронов в сети, используя следующие значения:
один слой 10 нейронов
один слой 100 нейронов
один слой 5000 нейронов.
Поменяйте активационную функцию в скрытых слоях с relu на linear.
Поменяйте размеры batch_size:
10
100
1000
Выведите на экран получившиеся точности.
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
# Подключение библиотек для отрисовки изображений
import matplotlib.pyplot as plt
# Подключение модуля для работы с файлами
import os
# Вывод изображения в ноутбуке, а не в консоли или файле
# %matplotlib inline

import gdown

gdown.download('https://storage.yandexcloud.net/aiueducation/Content/base/l3/hw_light.zip', None, quiet=True)

file_zip = zipfile.ZipFile('hw_light.zip')
file_zip.extractall('./')
file_zip.close()

# Путь к директории с базой
base_dir = 'hw_light'
# base_dir = '.'
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
        elif patch == '3':
            y_train.append(1)
        else:
            y_train.append(2)

# Преобразование в numpy-массив загруженных изображений и меток классов
x_train = np.array(x_train)
y_train = np.array(y_train)
# Вывод размерностей
print('Размер массива x_train', x_train.shape)
print('Размер массива y_train', y_train.shape)

x_train[0].shape

plt.imshow(x_train[5], cmap='gray')

exl = 5

plt.imshow(np.reshape(x_train[exl], (20, 20)), cmap='gray')
plt.show()
print(y_train[exl])

x_shuffle = x_train.copy()
np.random.shuffle(x_shuffle)

# Создание полотна из десяти графиков
fig, axs = plt.subplots(1, 5, figsize=(10, 5))

# Проход и отрисовка по всем классам
for i in range(5):
    img = np.reshape(x_shuffle[i], (20, 20))
    axs[i].imshow(img, cmap='gray')

    # Вывод изображения
    plt.show()

# Изменение формата входных характеристик с 20*20 на 400*1
x_train = x_train.reshape(x_train.shape[0], 400)
# Вывод нового размера
print(x_train.shape)

x_train[0].dtype

# Нормирование входных картинок
# Преобразование x_train в числа с плавающей точкой (тип float)
x_train = x_train.astype('float32')
# Приведение значений к диапазону от 0 до 1
x_train = x_train / 255

# Вывод размерности
print(y_train.shape)

# Преобразование ответов в формат one_hot_encoding
y_train=utils.to_categorical(y_train,3)

# Вывод новой размерности
print(y_train.shape)

# Создание нейронной сети

# Список содержащий количество нейронов для экспериментов
list_neurons=[10,100,5000]

# Список, содержащий функции активации для экспериментов
list_activation=['relu','linear']

# Список, содержащий размер batch_size для экспериментов
list_batch=[10,100,1000]


# Список для сохранения точности сети при заданных параметрах
data_list=[]
# Перебор значений в списке с количеством нейронов
for neurons in list_neurons:
  # Перебор значений в списке с функцией активации
  for activation in list_activation:
    # Перебор значений в списке batch_size
    for batch in list_batch:
      # Создать сеть прямого распространения
      model=Sequential()
      model.add(Dense(neurons, input_dim=400, activation=activation))
      # Создание полносвязного слоя с тремя нейронами для каждого класса softmax-активацией
      model.add(Dense(3,activation='softmax'))
      # Компиляция модели
      model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001),metrics=['accuracy'])
      # Вывод структуры модели
      print(model.summary())
      # Вывод текущик параметров сети
      print(f'Neurons: {neurons}, activation: {activation}, batch_size: {batch}')

      # Обучение модели
      history= model.fit(x_train, y_train, batch_size=batch,epochs=10, verbose=1,shuffle=True)

      # Сохранение параметров и точность сети
      data_list.append(('Neurons: ',neurons, 'Activation: ', activation, 'Batch: ', batch, 'accuracy: ',
                        round(history.history['accuracy'][9],3)))


history.history

# Вывод результатов экспериментов
for i in data_list:
  print(i)
