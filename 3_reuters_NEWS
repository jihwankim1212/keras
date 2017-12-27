
import keras
from keras.datasets import reuters  # 1986년에 발표한 짧은 뉴스 서비스와 그 주제로 된 세트, 46개의 주제
import numpy as np
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
import matplotlib.pyplot as plt
import copy

'''
뉴스 서비스 분류하기 : 다중 범주 분류하기 예제
url -> https://www.codeonweb.com/entry/4c71f7aa-5dd8-4efc-ba37-e4cd0359e8f6
로이터(Reuters) 뉴스 서비스를 46개의 상호 배타적인 주제로 분류
'''

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)   # num_words=10000 -> 훈련 데이터에서 가장 자주 쓰인 10,000 개의 단어만을 사용

print('len(train_data) :', len(train_data))     # 8982 개
print('len(test_data) :', len(test_data))       # 2246 개

print('train_data[10] :', train_data[10])

word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])    # ex -> ('fawc', 16260), ('degussa', 12089), ('woods', 8803)

# 3 index 씩 밀려있음
# Note that our indices were offset by 3
# because 0, 1 and 2 are reserved indices for "padding", "start of sequence", and "unknown".    # default = '?'로 설정
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

print('decoded_newswire :', decoded_newswire)
print('train_labels[10] :', train_labels[10])    # 0 ~ 45 주제 인덱스

# 벡터화
# len(sequences) x dimension(가장 자주 쓰인 단어 개수 10000개) matrix 생성
def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.   # set specific indices of results[i] to 1s   # 원 핫 인코딩
    return results

# Our vectorized training data
x_train = vectorize_sequences(train_data)
# Our vectorized test data
x_test = vectorize_sequences(test_data)

print('x_train.shape :', x_train.shape)
print('x_test.shape  :', x_test.shape)

# 원 핫 인코딩
def to_one_hot(labels, dimension=46):
    # len(labels) x dimension matrix 생성
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

# Our vectorized training labels
one_hot_train_labels = to_one_hot(train_labels)
# Our vectorized test labels
one_hot_test_labels = to_one_hot(test_labels)

print('train_labels :', train_labels)
print('one_hot_train_labels :', one_hot_train_labels)

# 위와 같은 내용의 함수 -> to_categorical = to_one_hot
# one_hot_train_labels = to_categorical(train_labels)
# one_hot_test_labels = to_categorical(test_labels)

'''
신경망 구성하기
'''
# 각 계층은 바로 앞 계층의 출력에서 제공되는 정보에만 접근할 수 있습니다.
# 하나의 계층이 분류 문제와 관련된 일부 정보를 떨구어 버린다면 이 정보는 절대로 다음 계층에 의해 복구되지 못 합니다
# 각 계층은 잠재적으로 "정보의 병목"이 될 수 있는 것이죠.
# 앞선 IMDB_binary_crossentropy 예제에서는 16차원의 중간 계층을 사용하였는데
# 16차원 공간은 46개의 서로 다른 범주를 분리하는 데에는 너무 제한되어 있을 수 있습니다.
# 64개로 구성해 보겠습니다

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))   # 46개의 상호 배타적인 주제로 분류, 46개의 점수의 총합은 1 (확률)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',  # 레이블의 실제 분포와 신경망의 출력의 확률 분포 간의 거리를 측정
              metrics=['accuracy'])

x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=10,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# 훈련 중 발생하는 모든 데이터를 포함하는 history 멤버
print('history.history.keys() :', history.history.keys())

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)
print('epochs :', epochs)

# 손실
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # clear figure

# 정확도
acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# epoch 8 이상 부터는 과적합으로 판단됨
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))   # 46개의 상호 배타적인 주제로 분류, 46개의 점수의 총합은 1 (확률)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',   # 두 확률 분포 사이의 거리
              metrics=['accuracy'])

model.fit(partial_x_train,
          partial_y_train,
          epochs=8,
          batch_size=512,
          validation_data=(x_val, y_val))

print('evaluate :', model.evaluate(x_test, one_hot_test_labels))

test_labels_copy = copy.copy(test_labels)
np.random.shuffle(test_labels_copy)
# 무작위 분류로 대비
print(float(np.sum(np.array(test_labels) == np.array(test_labels_copy))) / len(test_labels))

print('test_labels :', test_labels)
print('test_labels_copy :', test_labels_copy)

predictions = model.predict(x_test)
print('predictions.shape :', predictions.shape)                   # matrix
print('predictions[0].shape   :', predictions[0].shape)          # 각 엔트리의 길이는 46
print('predictions[0]         :', predictions[0])
print('np.sum(predictions[0]) :', np.sum(predictions[0]))        # 확률의 합은 1
print('np.argmax(predictions[0]) :', np.argmax(predictions[0])) # 가장 확률이 높은 index


'''
신경망 구성을 축소하여 modeling 
이제 46차원 보다 훨씬 적은, 예를 들어 4차원의 중간 계층을 사용하여 '정보 병목' 이 생기면 어떤 일이 발생하는지 대조
78% -> 70%
이런 하락은 대부분 많은 정보(46개 범주의 독립된 초평면을 복구하는 데 충분한 정보)를 너무 낮은 차원의 중간 계층 공간으로 압축하려고 한 데에서 기인한 것입니다. 
신경망은 8차원 표현으로 대부분의 필요한 정보를 우겨넣을 수는 있겠지만, 모든 정보는 아닙니다.
'''
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(4, activation='relu'))   # 64 -> 4로 축소
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(partial_x_train,
          partial_y_train,
          epochs=20,
          batch_size=128,
          validation_data=(x_val, y_val))

print('evaluate :', model.evaluate(x_test, one_hot_test_labels))
