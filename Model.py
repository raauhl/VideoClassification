from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, GRU, Input, TimeDistributed
from keras.utils import to_categorical
from keras.models import Model


import frameExtractor

(xTrain, yTrain) = frameExtractor.getData()
xTrain = xTrain / 255
#yTrain = to_categorical(yTrain)

#print(yTrain[0])
print(xTrain.shape)
print(yTrain.shape)

cnn = Sequential()
cnn.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(256,256,3), padding='same'))
cnn.add(MaxPooling2D(2))
cnn.add(Conv2D(16, kernel_size=3, activation='relu', padding='same'))
cnn.add(MaxPooling2D(2))
cnn.add(Conv2D(8, kernel_size=3, activation='relu', padding='same'))
cnn.add(MaxPooling2D(2))
cnn.add(Conv2D(8, kernel_size=3, activation='relu', padding='same'))
cnn.add(MaxPooling2D(2))
cnn.add(Flatten())
#cnn.add(Dense(1,activation='sigmoid'))
cnn.summary()
print('cnn built...')

"""
cnn = Sequential()
cnn.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(256,256,3), padding='same'))
cnn.add(MaxPooling2D(2))
cnn.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'))
cnn.add(MaxPooling2D(2))
cnn.add(Conv2D(16, kernel_size=3, activation='relu', padding='same'))
cnn.add(MaxPooling2D(2))
cnn.add(Conv2D(16, kernel_size=3, activation='relu', padding='same'))
cnn.add(MaxPooling2D(2))
cnn.add(Flatten())
#cnn.add(Dense(1,activation='sigmoid'))
cnn.summary()
print('cnn built...')
"""

rnn = Sequential()
rnn = GRU(40)
#rnn.summary()
print('rnn built...')

dense = Sequential()
dense.add(Dense(64,activation='relu'))
#dense.add(Dense(64,activation='relu'))
dense.add(Dense(1,activation='sigmoid'))
#dense.summary()
#print('dense built...')

main_input = Input(shape = (40, 256, 256, 3))    #input a sequence of 40 images
model = TimeDistributed(cnn)(main_input)         #this makes cnn run 40 times
model = rnn(model)
model = dense(model)

final_model = Model(inputs = main_input, outputs = model)
final_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
final_model.summary()

final_model.fit(xTrain, yTrain, batch_size=2, epochs=2)
print('done')



