import keras.backend as K

if K.backend() == 'tensorflow':
    from .tensorflow_backend import *

    rnn = lambda *args, **kwargs: K.rnn(*args, **kwargs) + ([],)
