from create_participants import fake_data
from create_model import create_model
from train_model import train_model
from predict_model import predict_model

import tensorflow
from tensorflow import keras
from tensorflow.keras.models import load_model, model_from_json

import os
from pathlib import Path
Path("./models").mkdir(parents=True, exist_ok=True)

# * Disable GPU for tensorflow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# * Using GPU for processing
# physical_devices = tf.config.experimental.list_physical_devices("GPU")
# #print("Num GPUs Available: ", len(physical_devices))
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# * Create dataset of 2100 participants
# scale_train_samples = []
# train_labels = []
# scale_train_samples, train_labels = fake_data()

# scale_test_samples = []
# test_labels = []
# scale_test_samples, test_labels = fake_data()

# model = create_model()

# train_model(model, scale_train_samples, train_labels)

# predict_model(model, scale_test_samples, test_labels)

# * Save model(the architecture, the weights, the optimizer, the state of the optimizer, the learning rate, the loss, etc.) to a .h5 file
# ? If found a model, delete it and save a new one
# if os.path.isfile("models/medical_trial_model.h5") is True:
#     os.remove("models/medical_trial_model.h5")
# model.save(r"models/medical_trial_model.h5")

model = load_model(r"models/medical_trial_model.h5")

# * Save model architecture as json
json_string = model.to_json()
# * Load json architecture to model
# ? Only load the architecture. Weight, optimizer, etc is not included
model_architecture = model_from_json(json_string)

# * Save model weights to a .h5 file
# ? If found the model's weight, delete it and save a new one
if os.path.isfile("models/medical_trial_model_weights.h5") is True:
    os.remove("models/medical_trial_model_weights.h5")
model.save(r"models/medical_trial_model_weights.h5")

# * Load weights to a existed model, with SAME architecture
model_architecture.load_weights(r'models/medical_trial_model_weights.h5')
model_architecture.summary()
