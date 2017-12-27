
import keras
import numpy as np
from keras import layers
import random
import sys

'''
LSTM으로 텍스트 생성하기
url -> https://www.codeonweb.com/entry/b3b169b0-b604-4959-8ab4-4b3be4943663
19세기 후반 독일 철학자인 니체의 저작(영문 번역본)
'''
# txt load
path = keras.utils.get_file(
    'nietzsche.txt',
    origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')

# open 후 소문자 변환
text = open(path, encoding='utf8').read().lower()
print('\nCorpus length:', len(text))

# Length of extracted character sequences
maxlen = 60

# We sample a new sequence every `step` characters
step = 3

# This holds our extracted sequences
sentences = []

# This holds the targets (the follow-up characters)
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('Number of sequences:', len(sentences))

# List of unique characters in the corpus
chars = sorted(list(set(text)))

print('set(text)       :', set(text))       # collections of unique elements
print('list(set(text)) :', list(set(text))) # list로 구성
print('sorted(list(set(text))) :', sorted(list(set(text)))) # list sort

print('Unique characters:', len(chars))

# Dictionary mapping unique characters to their index in `chars`
char_indices = dict((char, chars.index(char)) for char in chars)
print('char_indices: ', char_indices)

# Next, one-hot encode the characters into binary arrays.
print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)   # (sequences, maxlen, unique_characters)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# LSTM 적용
model = keras.models.Sequential()
model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))       # input_shape  maxlen x len(chars) matrix
model.add(layers.Dense(len(chars), activation='softmax'))

# 옵티마이저 설정
optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)           # Normalization
    probas = np.random.multinomial(1, preds, 1)     # 가장 높은 값은 1 나머지는 0인 ndarray반환
    return np.argmax(probas)                       # 가장 높은 값의 index

for epoch in range(1, 2):
    print('epoch', epoch)
    # Fit the model for 1 epoch on the available training data
    model.fit(x, y,
              batch_size=128,
              epochs=1)

    # Select a text seed at random
    start_index = random.randint(0, len(text) - maxlen - 1)     # 랜덤으로 text 중 선택
    generated_text = text[start_index: start_index + maxlen]
    print('--- Generating with seed: "' + generated_text + '"')

    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print('------ temperature:', temperature)
        print('generated_text :', generated_text)
        # sys.stdout.write(generated_text)

        # We generate 30 characters
        for i in range(30):
            sampled = np.zeros((1, maxlen, len(chars)))     # sentence 1개, maxlen, len(chars) 3차원
            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1.

            # 모델 예측 (다음 글자 예측)
            preds = model.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = chars[next_index]

            # 예측한 단어를 generated_text에 붙여줌
            generated_text += next_char
            generated_text = generated_text[1:]

            # print('next_char :', next_char)
            # sys.stdout.write(next_char)
            # sys.stdout.flush()
        print()
