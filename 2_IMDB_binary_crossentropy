
import keras
import numpy as np
from keras.datasets import imdb
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
import matplotlib.pyplot as plt

'''
영화 리뷰 분류 : 이진 분류 예제
url -> https://www.codeonweb.com/entry/c4161734-81b4-4e7c-a280-ae31f63b8261
영화 리뷰를 "긍정적" 리뷰와 "부정적" 리뷰로 분류하는 방법을 학습
IMDB 데이터셋 -> train 25000, test 25000, 50% 긍정리뷰 50% 부정리뷰로 구성
'''

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)  # num_words=10000 -> 훈련 데이터에서 가장 자주 쓰인 10,000 개의 단어만을 사용

print('train_data[0]   : ', train_data[0])      # 단어 인덱스의 리스트
print('train_labels[0] : ', train_labels[0])    # 0 부정적 1 긍정적
print('train_data.shape : ', train_data.shape)
print('test_data.shape  : ', test_data.shape)

print('max(sequence) : ', max([max(sequence) for sequence in train_data]))  # 10,000 개의 단어를 사용해서 인덱스가 9999까지

# word_index is a dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# print('word_index.items() : ', word_index.items())  # ex -> ('restrained', 6072), ('antecedent', 52870), ('gramm', 60397)

# We reverse it, mapping integer indices to words
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# 3 index 씩 밀려있음
# We decode the review; note that our indices were offset by 3
# because 0, 1 and 2 are reserved indices for "padding", "start of sequence", and "unknown".    # default = '?'로 설정
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

print(reverse_word_index.get(train_data[0][0] - 3, '?'))
print(reverse_word_index.get(train_data[0][1] - 3, '?'))
print(reverse_word_index.get(train_data[0][2] - 3, '?'))

print('decoded_review :', decoded_review)

# len(sequences) x dimension(가장 자주 쓰인 단어 개수 10000개) matrix 생성
def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s   # 원 핫 인코딩
    return results

# Our vectorized training data
x_train = vectorize_sequences(train_data)
# Our vectorized test data
x_test = vectorize_sequences(test_data)

print('x_train[0] :', x_train[0])

# 레이블을 벡터화
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

print('type(train_labels) :', type(train_labels))
print('type(y_train) :', type(y_train))

'''
신경망 구성하기
Dense(16, activation='relu') -> 각 Dense 계층에 넘겨지는 인수(16)은 계층의 "은닉 단위(hidden units)"의 갯수
relu 함수는 음수 값을 0으로 만들어 없애버리는 데에 반해 
sigmoid 함수는 임의의 값을 "찌그러뜨려" 0과 1 사이에 우겨 넣어서 확률로 해석할 수 있는 출력을 만들어 냅니다.
'''
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))    # 가장 자주 쓰인 단어 개수 10000개
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))    # sigmoid 활성화 함수 0 ~ 1 (0 부정적 , 1 긍정적)

# 1
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',   # binary_crossentropy 손실    # 확률 분포 간의 거리를 측정, 확률을 출력하는 모델을 다룰 때 binary_crossentropy는 보통 최선의 선택
              metrics=['accuracy'])           # 정확도(accuracy) 측정

# 2
# 하이퍼 파라미터 변경 가능
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 3
# loss나 metrics 인수
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

# 2와 3 같은 내용

# validation과 train split
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# 모델 학습
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=10,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# 훈련 중 발생하는 모든 데이터를 포함하는 history 멤버
history_dict = history.history
print('history_dict :', history_dict)
print('history_dict.keys() :', history_dict.keys())

acc = history.history['binary_accuracy']
val_acc = history.history['val_binary_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # clear figure

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# 그래프를 보니 4번째 epoch까지만 의미있는 것 같아 4번으로 실행
# 과적합
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4, batch_size=512)

print('results :', model.evaluate(x_test, y_test))
print('predict :', model.predict(x_test))

# 이진 분류(두 개의 출력 범주를 가지는) 문제에 있어서 신경망은 1개의 단위와 sigmoid 활성화를 사용하는 Dense 계층으로 끝나야 한다.
# 즉 신경망의 출력은 확률로 인코딩된 0과 1 사이의 스칼라이어야 합니다
# 스칼라 시그모이드 출력을 사용하는 이진 분류 문제에 대해 손실 함수는 binary_crossentropy를 사용
# rmsprop 최적화기는 일반적으로 어떤 문제인지를 불문하고 충분히 좋은 선택
