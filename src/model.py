from keras import Sequential
from keras.layers import Dense
from keras.losses import mean_squared_error
from keras.optimizers import Adam

from K import K


class Model:

    @staticmethod
    def build_model():
        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=K.STATE_SIZE))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(K.ACTION_SIZE, activation='linear'))
        model.compile(loss=mean_squared_error, optimizer=Adam(lr=K.LEARNING_RATE))
        return model

if __name__ == '__main__':
    from keras.utils import plot_model
    plot_model(Model.build_model(), to_file='results/model_diagram.png')