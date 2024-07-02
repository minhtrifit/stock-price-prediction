import yfinance as yf
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from module import lstm_model
from module import rnn_model
from module import gru_model

currencies = ["BTC-USD", "ETH-USD", "ADA-USD"]
start = "05/03/2009"
end = dt.datetime.now().strftime("%d/%m/%Y")
test_size = 60
pre_day = 30
models = ["1: LSTM", "2: RNN", "3: GRU"]

def process_data(df, ma_1, ma_2, ma_3):
    scala_x = MinMaxScaler(feature_range=(0, 1))
    scala_y = MinMaxScaler(feature_range=(0, 1))

    cols_x = [
        "H-L",
        "O-C",
        f"SMA_{ma_1}",
        f"SMA_{ma_2}",
        f"SMA_{ma_3}",
        f"SD_{ma_1}",
        f"SD_{ma_3}",
    ]
    cols_y = ["Close"]

    scaled_data_x = scala_x.fit_transform(df[cols_x].values.reshape(-1, len(cols_x)))
    scaled_data_y = scala_y.fit_transform(df[cols_y].values.reshape(-1, len(cols_y)))

    x_total = []
    y_total = []

    for i in range(pre_day, len(df)):
        x_total.append(scaled_data_x[i - pre_day : i])
        y_total.append(scaled_data_y[i])

    x_train = np.array(x_total[: len(x_total) - test_size])
    x_test = np.array(x_total[len(x_total) - test_size :])

    y_train = np.array(y_total[: len(y_total) - test_size])
    y_test = np.array(y_total[len(y_total) - test_size :])

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    return scala_x, scala_y, x_train, x_test, y_train, y_test, cols_x, cols_y

def price_plot(currency, df, predict_prices):
    real_price = df[len(df) - test_size :]["Close"].values.reshape(-1, 1)
    real_price = np.array(real_price)
    real_price = real_price.reshape(real_price.shape[0], 1)

    plt.plot(real_price, color="red", label=f"Real {currency} Prices")
    plt.plot(predict_prices, color="blue", label=f"Predicted {currency} Prices")

    plt.title(f"{currency} Prices")

    plt.xlabel("Time")
    plt.ylabel("Stock Prices")

    plt.ylim(bottom=0)

    plt.savefig(f"./image/{currency}.png")

def make_prediction(df, model, cols_x, scala_x, scala_y):
    x_predict = df[len(df) - pre_day :][cols_x].values.reshape(-1, len(cols_x))
    x_predict = scala_x.transform(x_predict)
    x_predict = np.array(x_predict)
    x_predict = x_predict.reshape(1, x_predict.shape[0], len(cols_x))

    prediction = model.predict(x_predict)
    prediction = scala_y.inverse_transform(prediction)

    print(prediction)

def training_model(currency, model_type):
    # Load data
    Ticker = yf.Ticker(currency)
    df = Ticker.history(period="max")
    df = pd.DataFrame(df)
    print(df)

    df["H-L"] = df["High"] - df["Low"]
    df["O-C"] = df["Open"] - df["Close"]

    ma_1 = 7
    ma_2 = 14
    ma_3 = 21

    df[f"SMA_{ma_1}"] = df["Close"].rolling(window=ma_1).mean()
    df[f"SMA_{ma_2}"] = df["Close"].rolling(window=ma_2).mean()
    df[f"SMA_{ma_3}"] = df["Close"].rolling(window=ma_3).mean()
    df[f"SD_{ma_1}"] = df["Close"].rolling(window=ma_1).std()
    df[f"SD_{ma_3}"] = df["Close"].rolling(window=ma_3).std()

    df.dropna(inplace=True)

    df.to_csv(f"./dataset/{currency}.csv")
    print(f"Loading {currency} data successfully!")

    # Process Data
    scala_x, scala_y, x_train, x_test, y_train, y_test, cols_x, cols_y = process_data(df, ma_1, ma_2, ma_3)

    # Build Model
    if model_type == "LSTM":
        model = lstm_model.build_model(currency, x_train, y_train, cols_y)
    if model_type == "RNN":
        model = rnn_model.build_model(currency, x_train, y_train, cols_y)
    if model_type == "GRU":
        model = gru_model.build_model(currency, x_train, y_train, cols_y)

    # Testing
    predict_prices = model.predict(x_test)
    predict_prices = scala_y.inverse_transform(predict_prices)

    # Ploting the Stat
    price_plot(currency, df, predict_prices)

    # Make Prediction
    make_prediction(df, model, cols_x, scala_x, scala_y)

def choose_model(number):
    if number == "1":
        return "LSTM"
    if number == "2":
        return "RNN"
    if number == "3":
        return "GRU"
    else:
        return "none"

print("Choose traning model:")
for model in models:
    print(model)
print("")
print("You choose:")
model_choose = input()
model_type = choose_model(model_choose)

if model_type != "none":
    for currency in currencies:
        training_model(currency, model_type)
    
    print("Train model done!")
