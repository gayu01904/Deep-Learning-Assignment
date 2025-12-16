# -*- coding: utf-8 -*-
"""
Further Modified AlexNet
"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam

class AlexNet(Sequential):
    def __init__(self, input_shape, num_classes):
        super().__init__()

        # First Convolutional Layer
        self.add(Conv2D(
            64,                     # 🔁 CHANGED: 96 → 64 filters
            kernel_size=(11, 11),
            strides=4,
            padding='valid',
            activation='relu',
            input_shape=input_shape
        ))
        self.add(BatchNormalization())
        self.add(MaxPooling2D(pool_size=(3, 3), strides=2))

        # Second Convolutional Layer
        self.add(Conv2D(
            192,                    # 🔁 CHANGED: 256 → 192 filters
            kernel_size=(5, 5),
            padding='same',
            activation='relu'
        ))
        self.add(BatchNormalization())
        self.add(MaxPooling2D(pool_size=(3, 3), strides=2))

        # Third, Fourth, Fifth Convolutional Layers
        self.add(Conv2D(384, (3, 3), activation='relu', padding='same'))
        self.add(Conv2D(256, (3, 3), activation='relu', padding='same'))  # 🔁 384 → 256
        self.add(Conv2D(128, (3, 3), activation='relu', padding='same'))  # 🔁 256 → 128
        self.add(MaxPooling2D(pool_size=(3, 3), strides=2))

        # Flatten
        self.add(Flatten())

        # Fully Connected Layers
        self.add(Dense(2048, activation='relu'))   # 🔁 4096 → 2048
        self.add(Dropout(0.6))
        self.add(Dense(1024, activation='relu'))   # 🔁 4096 → 1024
        self.add(Dropout(0.6))

        # Output Layer
        self.add(Dense(num_classes, activation='softmax'))

# Model creation
model = AlexNet(input_shape=(224, 224, 3), num_classes=10)

# 🔁 NEW: Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),   # 🔁 Explicit optimizer
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display model architecture
model.summary()
