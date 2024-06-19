from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GRU
from tensorflow.keras.optimizers import Adam

def build_model(currency, x_train, y_train, cols_y):
    model = Sequential()

    model.add(GRU(units=60, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(0.2))

    model.add(GRU(units=60, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(GRU(units=60, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(GRU(units=60))
    model.add(Dropout(0.2))

    model.add(Dense(units=len(cols_y)))

    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(x_train, y_train, epochs=120, steps_per_epoch=40)
    model.save(f"./model/{currency}_gru.h5")

    print(f"Train {currency} currency with GRU Model successfully!")
    print("==================================================")
    print(" ")

    return model