# -*- coding: utf-8 -*-
"""
Enhanced Character-Level Text Generation using RNN
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout # pyright: ignore[reportMissingImports]

# 🔁 CHANGED training text (more diversity)
text = "Artificial intelligence enables machines to learn patterns from data"

chars = sorted(list(set(text)))
char_to_index = {c: i for i, c in enumerate(chars)}
index_to_char = {i: c for i, c in enumerate(chars)}

seq_length = 6          # 🔁 increased sequence length
sequences = []
labels = []

for i in range(len(text) - seq_length):
    sequences.append([char_to_index[c] for c in text[i:i + seq_length]])
    labels.append(char_to_index[text[i + seq_length]])

X = np.array(sequences)
y = np.array(labels)

X_one_hot = tf.one_hot(X, len(chars))
y_one_hot = tf.one_hot(y, len(chars))

text_len = 60           # 🔁 generate more characters

# 🔁 MODEL ARCHITECTURE CHANGED
model = Sequential()
model.add(SimpleRNN(
    64,                         # 🔁 hidden units changed
    input_shape=(seq_length, len(chars)),
    activation='tanh',          # 🔁 relu → tanh
    return_sequences=False
))
model.add(Dropout(0.3))         # 🔁 NEW dropout layer
model.add(Dense(len(chars), activation='softmax'))

# 🔁 Optimizer learning rate changed
optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 🔁 Reduced epochs for faster training
model.fit(X_one_hot, y_one_hot, epochs=60, verbose=2)

# 🔁 Changed starting sequence
start_seq = "Artificial intelligence "

generated_text = start_seq

for _ in range(text_len):
    x = np.array([[char_to_index[c] for c in generated_text[-seq_length:]]])
    x_one_hot = tf.one_hot(x, len(chars))
    preds = model.predict(x_one_hot, verbose=0)

    # 🔁 Added temperature sampling
    temperature = 0.8
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    next_index = np.random.choice(len(chars), p=preds[0])
    generated_text += index_to_char[next_index]

print("\nGenerated Text:")
print(generated_text)
