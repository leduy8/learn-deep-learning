import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

import numpy as np

from sklearn.metrics import confusion_matrix

from create_participants import fake_data
from plot_confusion_matrix import plot_confusion_matrix

# * Disable GPU for tensorflow
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# * Create dataset of 2100 participants
scale_train_samples = []
train_labels = []

scale_train_samples, train_labels = fake_data()

# for i in scale_train_samples:
#     print(i)

scale_test_samples = []
test_labels = []

scale_test_samples, test_labels = fake_data()

# * Using GPU for processing
# physical_devices = tf.config.experimental.list_physical_devices("GPU")
# #print("Num GPUs Available: ", len(physical_devices))
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# * Build a Sequential Model
model = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax'),
])

# model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss="sparse_categorical_crossentropy", metrics=['accuracy'])

model.fit(x=scale_train_samples, y=train_labels, validation_split=0.1,
          batch_size=10, epochs=30, shuffle=True, verbose=0)

# * Make neural network for inference for predictions on test dataset
predictions = model.predict(x=scale_test_samples, batch_size=10, verbose=0)

# * Round predictions for most probable prediction
rounded_predictions = np.argmax(predictions, axis=-1)

# for i in rounded_predictions:
#     print(i)

# * Create confusion matrix to visualize predictions accuracy
cm = confusion_matrix(y_true=test_labels, y_pred=rounded_predictions)
cm_plot_labels = ['no_side_effects', 'had_side_effects']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
