from keras.models import Model
from keras.layers import Input, Dense, Flatten

def get_model():

    # Model parameters
    rows, cols = 28, 28
    input_shape = (rows, cols, 1)

    nb_classes = 10

    hidden_size = 128

    inp = Input(shape=input_shape)
    flat = Flatten()(inp)
    hidden_1 = Dense(hidden_size, activation='sigmoid')(flat)
    hidden_2 = Dense(hidden_size, activation='sigmoid')(hidden_1)
    out = Dense(nb_classes, activation='softmax')(hidden_2)

    model = Model(inputs=inp, outputs=out)

    print(model.summary())

    return model


if __name__ == '__main__':

    model = get_model()
