
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from utils import load_genre_folder
from model import build_cnn
import argparse

def load_dataset(data_dir='data/genres', sr=22050, duration=30, n_mels=128):
    genres = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,d))])
    X = []
    y = []
    for idx, g in enumerate(genres):
        folder = os.path.join(data_dir, g)
        print('Loading genre:', g)
        S_list = load_genre_folder(folder, sr=sr, duration=duration, n_mels=n_mels)
        for S in S_list:
            X.append(S)
            y.append(idx)
    X = np.array(X)
    y = np.array(y)
    return X, y, genres

def main(args):
    X, y, genres = load_dataset(data_dir=args.data_dir, sr=args.sr, duration=args.duration, n_mels=args.n_mels)
    # normalize and reshape
    X = (X - X.min()) / (X.max() - X.min() + 1e-6)
    X = X[..., np.newaxis].astype('float32')
    y_cat = to_categorical(y, num_classes=len(genres))

    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1765, random_state=42, stratify=np.argmax(y_train, axis=1))

    model = build_cnn(input_shape=X_train.shape[1:], num_classes=len(genres))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss'),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4)
    ]

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks)
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test accuracy:', test_acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/genres')
    parser.add_argument('--sr', type=int, default=22050)
    parser.add_argument('--duration', type=int, default=30)
    parser.add_argument('--n_mels', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    main(args)
