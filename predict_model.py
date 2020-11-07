import numpy as np
from sklearn.metrics import confusion_matrix
from plot_confusion_matrix import plot_confusion_matrix


def predict_model(model, scale_test_samples, test_labels):
    # * Make neural network for inference for predictions on test dataset
    predictions = model.predict(x=scale_test_samples, batch_size=10, verbose=0)

    # * Round predictions for most probable prediction
    rounded_predictions = np.argmax(predictions, axis=-1)

    # * Create confusion matrix to visualize predictions accuracy
    cm = confusion_matrix(y_true=test_labels, y_pred=rounded_predictions)
    cm_plot_labels = ['no_side_effects', 'had_side_effects']
    plot_confusion_matrix(cm=cm, classes=cm_plot_labels,
                          title='Confusion Matrix')
