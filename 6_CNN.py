import keras
print(keras.__version__)

from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical

'''
합성곱 신경망(Convnet, Convolutional Neural Network) 시작하기
url -> https://www.codeonweb.com/entry/f43a1869-4534-4560-868d-b560ffaa3eb1
'''

model = models.Sequential()
# Convolution layer     # input 28 x 28     # kernel 3 x 3
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))    # (image_height, image_width, image_channels)(배치 차원은 포함되지 않았습니다)과 같은 형태의 입력 텐서
# Pooling layer         # kernel 2 x 2
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# 모델 summary
model.summary()

# 1차원으로 평탄화
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# 모델 summary
model.summary()

# train, test split
(train_x, train_y), (test_x, test_y) = mnist.load_data()

train_x = train_x.reshape(60000, 28, 28, 1) #  = reshape(-1, 28, 28, 1)
train_x = train_x.astype('float32') / 255

test_x = test_x.reshape(10000, 28, 28, 1)   #  = reshape(-1, 28, 28, 1)
test_x = test_x.astype('float32') / 255

train_y = to_categorical(train_y)
test_y = to_categorical(test_y)

# 모델 학습
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_x, train_y, epochs=5, batch_size=32)

test_loss, test_acc = model.evaluate(test_x, test_y)

print('test_loss :', test_loss)
print('test_acc  :', test_acc)
