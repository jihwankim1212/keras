import keras
from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt

'''
url -> https://www.codeonweb.com/entry/c04908c2-26a9-4cee-8311-ec466a92eda2
보스턴 집값 예측하기 : 회귀 예제
'''
(train_data, train_targets), (test_data, test_targets) =  boston_housing.load_data()

print('train_data.shape :', train_data.shape)   # 404,  13
print('test_data.shape  :', test_data.shape)    # 102,  13

# print(train_data[:2])
# print(test_data[:2])

'''
features 13
1. 1인당 범죄율.
2. 25,000 평방 피트 이상으로 구획된 주거 용지의 비율.
3. 동네 별 비상업 용지 비율.
4. 찰스 강 가변수(강가에 접해 있으면 1, 아니면 0).
5. 질소 산화물 농도(천만분의 일 단위).
6. 주택 당 평균 방 갯수.
7. 1940년 이전에 지어진 자가 주택 비율.
8. 5개 보스턴 고용 센터까지의 가중 거리.
9. 방사형 고속도로에의 접근성 지수.
10. 10,000 달러 당 전체가치재산세율.
11. 동네 별 학생-교사 비.
12. Bk를 동네의 흑인 비율이라 할 때 1000 * (Bk - 0.63) ** 2
13. 인구 중 낮은 사회적 지위의 백분율.
'''

# 단위는 1 -> 1000
# print('train_targets :', train_targets)

# Normalization
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

print('mean :', mean)
print('std  :', std)

print('train_data :', train_data)
print('test_data  :', test_data)

def build_model():
    # Because we will need to instantiate
    # the same model multiple times,
    # we use a function to construct it.
    # 일반적으로 훈련 데이터가 적을 수록 과적합이 더 심하기 때문에 작은 신경망을 사용하는 것이 과적합을 완화하는 한 방법.
    model = models.Sequential()
    model.add(layers.Dense(64,
                           activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64,
                           activation='relu'))
    model.add(layers.Dense(1))  # 활성화 함수를 적용하면 출력값의 범위가 제한될 수 있습니다
                                # 여기서는 마지막 계층이 단순히 선형이기 때문에 신경망은 어떠한 범위의 예측 값도 학습할 수 있습니다.
    model.compile(optimizer='rmsprop',
                  loss='mse',           # 평균 제곱 오차는 예측과 목표 간 차이의 제곱으로 회귀 문제의 손실 함수
                  metrics=['mae'])      # 평균 절대 오차(Mean Absolute Error)는 예측과 목표 간 차이의 절대값
    return model

# K-fold 검증
# 사용 가능한 데이터를 K개의 부분으로 나누고(K는 보통 4나 5)
# K개의 동일한 모델을 인스턴스화 하여서 각 모델을 K-1 부분의 데이터로 훈련하고
# 나머지 한 부분의 데이터로 평가합니다.
k = 4   # K-fold 검증 계수
num_val_samples = len(train_data) // k
print('num_val_samples :', num_val_samples)
num_epochs = 100
all_scores = []

for i in range(k):
    print('processing fold #', i)
    # Prepare the validation data: data from partition # k
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    # Prepare the training data: data from all other partitions
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)

    # Build the Keras model (already compiled)
    model = build_model()
    # Train the model (in silent mode, verbose=0)
    model.fit(partial_train_data, partial_train_targets,
              epochs=num_epochs, batch_size=1, verbose=0)
    # Evaluate the model on the validation data
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

print('all_scores :', all_scores)
print('np.mean(all_scores) :', np.mean(all_scores))

# Some memory clean-up
K.clear_session()

num_epochs = 100
all_mae_histories = []
for i in range(k):
    print('processing fold #', i)
    # Prepare the validation data: data from partition # k
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    # Prepare the training data: data from all other partitions
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)

    # 위와 다른 수정 부분
    # Build the Keras model (already compiled)
    model = build_model()
    # Train the model (in silent mode, verbose=0)
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
    print('history.history.keys() :', history.history.keys())
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)
    test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
    print('test_mae_score :', test_mae_score)

average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

# 지수 이동 평균(exponential moving average)
def smooth_curve(points, factor=0.9):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.clf()
plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

# 80 epoch 이후에는 검증 평균 절대 오차가 유의미하게 개선되는 것을 멈춥니다 (과적합)
# Get a fresh, compiled model.
model = build_model()
# Train it on the entirety of the data.
model.fit(train_data, train_targets,
          epochs=80, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)

print('test_mae_score :', test_mae_score)

# 사용할 수 있는 데이터가 적을 때, K-Fold 검증은 모델을 평가할 때에 신뢰할 수 있는 좋은 방법입니다.
# 사용할 수 있는 데이터가 적을 때, 지나친 과적합을 방지하기 위해 적은 은닉 계층(보통 한개나 두개)을 사용한 작은 신경망 사용이 선호됩니다.
