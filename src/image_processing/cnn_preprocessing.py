import tensorflow as tf

def cnn_preprocessing(x_train,x_test):
    x_train = x_train[..., tf.newaxis]     # add channel dimension
    x_test = x_test[..., tf.newaxis]  
    return x_train, x_test