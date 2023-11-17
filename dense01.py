
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow

print(tensorflow)

model = Sequential()

print('hello')

print(tensorflow.__version__)
# print(tensorflow.keras.models.__version__)
# print(tensorflow.keras.__version__)

model.add(Dense(32, input_dim=10))
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.summary()

model = Sequential()
model.add(Dense(32, input_dim=10))
model.add(Dense(5))
model.add(Dense(1))

model.summary()