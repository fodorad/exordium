import numpy as np
import tensorflow as tf


def extract_feature_maps(X: np.ndarray,
                         batch_size: int = 10,
                         model=tf.keras.applications.vgg19.VGG19,
                         preprocess_input=tf.keras.applications.vgg19.preprocess_input):
    assert X.ndim == 4 or X.ndim == 5
    assert X.shape[-1] == 3
    assert X.shape[-2] == X.shape[-3]
    X_preprocessed = np.zeros_like(X, dtype=np.float32)
    if X.ndim == 4: # (B, H, W, C)
        print('Preprocessing input...')
        for i in range(X.shape[0]): 
            X_preprocessed[i,...] = preprocess_input(X[i,j,...].astype(np.float32))
        print('Loading model...')
        inp = tf.keras.layers.Input(shape=(X.shape[1],X.shape[2],3))
        base_model = model(weights='imagenet', include_top=False)
        x = base_model(inp)
    elif X.ndim == 5: # (B, T, H, W, C)
        print('Preprocessing input...')
        for i in range(X.shape[0]): 
            for j in range(X.shape[1]):
                X_preprocessed[i,j,...] = preprocess_input(X[i,j,...].astype(np.float32))
        print('Loading model...')
        inp = tf.keras.layers.Input(shape=(X.shape[1],X.shape[2],X.shape[3],3))
        base_model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False)
        x = tf.keras.layers.TimeDistributed(base_model)(inp)
    else:
        raise ValueError(f'Only (B,H,W,C) or (B,T,H,W,C) input shapes are expected. Got {X.shape}')
    model = tf.keras.models.Model(inputs=inp, outputs=x)
    model.summary()
    print('Extract feature vectors...')
    preds = model.predict(X_preprocessed, batch_size=batch_size, verbose=1)
    return preds

