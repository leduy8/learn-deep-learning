def train_model(model, scale_train_samples, train_labels):
    model.fit(x=scale_train_samples, y=train_labels, validation_split=0.1,
              batch_size=10, epochs=30, shuffle=True, verbose=0)
