import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Dummy EHR data
import numpy as np
X = np.random.rand(1000, 100)
y = np.random.randint(0, 2, size=(1000,))

model = Sequential([
    Embedding(input_dim=1000, output_dim=64),
    LSTM(32),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=5)
