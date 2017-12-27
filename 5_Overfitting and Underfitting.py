import keras
from keras.datasets import imdb
import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt
from keras import regularizers

'''
과적합과 과소적합 (Overfitting and Underfitting)
https://www.codeonweb.com/entry/fca67c89-fc93-46c5-8cb9-1901935afd81

"최적화"는 훈련 데이터에 대해서 가능한 한 최고의 성능을 보이도록 모델을 조정하는 과정("기계 학습"의 그 "학습"입니다)
"일반화"는 훈련된 모델이 처음 보는 데이터에 얼마나 성능이 나오냐를 말함
과적합을 물리치는 과정을 정규화
'''

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)  # num_words=10000 -> 훈련 데이터에서 가장 자주 쓰인 10,000 개의 단어만을 사용

def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results

# Our vectorized training data
x_train = vectorize_sequences(train_data)
# Our vectorized test data
x_test = vectorize_sequences(test_data)
# Our vectorized labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

original_model = models.Sequential()
original_model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
original_model.add(layers.Dense(16, activation='relu'))
original_model.add(layers.Dense(1, activation='sigmoid'))

original_model.compile(optimizer='rmsprop',
                       loss='binary_crossentropy',
                       metrics=['acc'])

original_hist = original_model.fit(x_train, y_train,
                                   epochs=10,
                                   batch_size=512,
                                   validation_data=(x_test, y_test))

smaller_model = models.Sequential()
smaller_model.add(layers.Dense(4, activation='relu', input_shape=(10000,)))
smaller_model.add(layers.Dense(4, activation='relu'))
smaller_model.add(layers.Dense(1, activation='sigmoid'))

smaller_model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['acc'])

smaller_model_hist = smaller_model.fit(x_train, y_train,
                                       epochs=10,
                                       batch_size=512,
                                       validation_data=(x_test, y_test))

epochs = range(1, 11)
original_val_loss = original_hist.history['val_loss']
smaller_model_val_loss = smaller_model_hist.history['val_loss']

'''
원래의 신경망과 더 작은 신경망의 검증 손실을 비교해 봅시다. 
점은 더 작은 신경망의 검증 손실이고 십자는 원래의 신경망입니다
(더 낮은 검증 손실이 더 좋은 모델임을 의미합니다).
'''

# b+ is for "blue cross"
plt.plot(epochs, original_val_loss, 'b+', label='Original model')
# "bo" is for "blue dot"
plt.plot(epochs, smaller_model_val_loss, 'bo', label='Smaller model')
plt.title('1')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()

plt.show()

bigger_model = models.Sequential()
bigger_model.add(layers.Dense(512, activation='relu', input_shape=(10000,)))
bigger_model.add(layers.Dense(512, activation='relu'))
bigger_model.add(layers.Dense(1, activation='sigmoid'))

bigger_model.compile(optimizer='rmsprop',
                     loss='binary_crossentropy',
                     metrics=['acc'])

bigger_model_hist = bigger_model.fit(x_train, y_train,
                                     epochs=10,
                                     batch_size=512,
                                     validation_data=(x_test, y_test))

bigger_model_val_loss = bigger_model_hist.history['val_loss']

plt.clf()
plt.plot(epochs, original_val_loss, 'b+', label='Original model')
plt.plot(epochs, bigger_model_val_loss, 'bo', label='Bigger model')
plt.title('2')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()

plt.show()

'''
더 큰 신경망의 훈련 손실은 아주 빠르게 0에 가까워지죠. 
더 큰 용량의 신경망은 훈련 데이터를 더 빠르게 모델화 할 수 있지만(낮은 훈련 손실을 낳죠), 
거기에 과적합될 가능성도 더 큽니다 (훈련 손실과 검증 손실 사이의 큰 격차를 야기)
'''
original_train_loss = original_hist.history['loss']
bigger_model_train_loss = bigger_model_hist.history['loss']

plt.clf()
plt.plot(epochs, original_train_loss, 'b+', label='Original model')
plt.plot(epochs, bigger_model_train_loss, 'bo', label='Bigger model')
plt.title('3')
plt.xlabel('Epochs')
plt.ylabel('Training loss')
plt.legend()

plt.show()

'''
가중치 정규화를 추가하기

L2 정규화: 추가되는 비용이 가중치 계수의 제곱값에 비례(즉, 가중치의 "L2 norm"에 비례). 
L2 정규화는 신경망의 맥락에서는 가중치 감퇴(weight decay). 가중치 감퇴는 L2 정규화와 수학적으로 정확히 똑같습니다.
'''

# L2 정규화
l2_model = models.Sequential()
l2_model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
                          activation='relu', input_shape=(10000,)))
l2_model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
                          activation='relu'))
l2_model.add(layers.Dense(1, activation='sigmoid'))

# l2(0.001) -> 0.001 * weight_coefficient_value 계층의 가중치 행렬의 모든 계수가 신경망의 총 손실
l2_model.compile(optimizer='rmsprop',
                 loss='binary_crossentropy',
                 metrics=['acc'])

l2_model_hist = l2_model.fit(x_train, y_train,
                             epochs=10,
                             batch_size=512,
                             validation_data=(x_test, y_test))

l2_model_val_loss = l2_model_hist.history['val_loss']
'''
보시다시피, L2 정규화한 모델(점)은 원래의 모델(십자)보다 훨씬 과적합에 대한 저항력이 큽니다, 
두 모델이 동일한 갯수의 파라미터를 가지고 있음에도 말이죠.
'''
plt.clf()
plt.plot(epochs, original_val_loss, 'b+', label='Original model')
plt.plot(epochs, l2_model_val_loss, 'bo', label='L2-regularized model')
plt.title('4')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()

plt.show()

# dropout 적용
'''
신경망에서 가장 효과적이고 가장 널리 쓰이는 정규화 기술
훈련 중에 계층의 출력 특징 중 일부를 무작위로 "탈락"
보통은 0.2에서 0.5 사이의 값으로 설정
'''

dpt_model = models.Sequential()
dpt_model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
dpt_model.add(layers.Dropout(0.5))
dpt_model.add(layers.Dense(16, activation='relu'))
dpt_model.add(layers.Dropout(0.5))
dpt_model.add(layers.Dense(1, activation='sigmoid'))

dpt_model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])

dpt_model_hist = dpt_model.fit(x_train, y_train,
                               epochs=10,
                               batch_size=512,
                               validation_data=(x_test, y_test))

dpt_model_val_loss = dpt_model_hist.history['val_loss']

plt.clf()
plt.plot(epochs, original_val_loss, 'b+', label='Original model')
plt.plot(epochs, dpt_model_val_loss, 'bo', label='Dropout-regularized model')
plt.title('5')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()

plt.show()
