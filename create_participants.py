import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler


def fake_data():
    train_samples = []
    train_labels = []

    for i in range(50):
        # * 5% young ppl experient side effect
        random_younger = randint(13, 64)
        train_samples.append(random_younger)
        train_labels.append(1)

        # * 5% older ppl don't experient side effect
        random_older = randint(65, 100)
        train_samples.append(random_older)
        train_labels.append(0)

    for i in range(1000):
        # * 95% young ppl don't experient side effect
        random_younger = randint(13, 64)
        train_samples.append(random_younger)
        train_labels.append(0)

        # * 95% older ppl experient side effect
        random_older = randint(65, 100)
        train_samples.append(random_older)
        train_labels.append(1)

    # * Data Processing
    train_samples = np.array(train_samples)
    train_labels = np.array(train_labels)
    train_samples, train_labels = shuffle(train_samples, train_labels)

    # * Scale down input from range 13-100 to 0-1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scale_train_samples = scaler.fit_transform(train_samples.reshape(-1, 1))
    return scale_train_samples, train_labels
