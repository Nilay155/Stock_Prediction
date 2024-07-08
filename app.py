import numpy as np
import pandas as pd
import pandas_datareader as pdr
import streamlit as st
import yfinance as yf
import seaborn as sns
import datetime as dt
from keras.models import load_model
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM
from ta import momentum
import matplotlib.pyplot as plt

start = '2014-01-01'
stop = '2024-07-09'

st.title('Stock Mark')

# Button for navigation
selected_page = st.radio("Select a Page", ["Home", "Educational Resources"])

# Page content
if selected_page == "Home":
    st.title("Welcome to the Stock Market Web Page!")
    # Home page content

elif selected_page == "Educational Resources":
    st.title("Educational Resources")
    st.write("Expand your knowledge about investing, trading strategies, and market analysis with these resources:")

    # List of articles
    st.subheader("Articles")
    st.markdown("- [Introduction to Stock Market](https://www.example.com/article1)")
    st.markdown("- [Technical Analysis Basics](https://www.example.com/article2)")
    st.markdown("- [Fundamental Analysis Guide](https://www.example.com/article3)")

    # List of tutorials
    st.subheader("Tutorials")
    st.markdown("- [Stock Market Investing 101](https://www.example.com/tutorial1)")
    st.markdown("- [Options Trading Strategies](https://www.example.com/tutorial2)")
    st.markdown("- [Introduction to Forex Trading](https://www.example.com/tutorial3)")

    # List of videos
    st.subheader("Videos")
    st.markdown("- [Candlestick Charting Explained](https://www.example.com/video1)")
    st.markdown("- [Risk Management Techniques](https://www.example.com/video2)")
    st.markdown("- [Value Investing Principles](https://www.example.com/video3)")


company = st.text_input('Enter Stock','META')

data = yf.download(company, start=start, end=stop)

st.subheader('Date from 2014 - 2023')
st.write(data.describe())
st.write(data.tail(60))

st.subheader('Stock Trends')
figure = plt.figure(figsize= (10,5))
plt.plot(data['Close'])
st.pyplot(figure)

# Features button
if st.button('Features'):
    # Plotting different graphs with colors and styles
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    
    # Graph 1: Line plot
    axs[0, 0].plot(data['Close'], color='blue')
    axs[0, 0].set_title('Line Plot')
    
    # Graph 2: Scatter plot
    axs[0, 1].scatter(data.index, data['Close'], color='red', marker='o')
    axs[0, 1].set_title('Scatter Plot')
    
    # Graph 3: Bar plot
    axs[1, 0].bar(data.index, data['Volume'], color='green')
    axs[1, 0].set_title('Bar Plot')
    
    # Graph 4: Area plot
    axs[1, 1].fill_between(data.index, data['Close'], color='orange', alpha=0.5)
    axs[1, 1].set_title('Area Plot')
    
    # Adjusting the layout
    plt.tight_layout()
    
    # Display the graphs
    st.pyplot(fig)


# Create candlestick chart
fig = go.Figure(data=go.Candlestick(x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close']))

# Customize chart layout
fig.update_layout(title='Stock Price Candlestick Chart',
                  xaxis_title='Date',
                  yaxis_title='Price')

# Display the chart
st.plotly_chart(fig)


# Calculate moving average
data['MA'] = data['Close'].rolling(window=20).mean()

# Plot moving average
plt.plot(data.index, data['Close'], label='Close Price')
plt.plot(data.index, data['MA'], label='Moving Average')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Price with Moving Average')
plt.legend()
st.pyplot(plt)


# User input for multiple stock symbols
symbols = st.text_input('Enter Stock Symbols (comma-separated)', 'AAPL,GOOGL,MSFT')

# Split symbols and retrieve data for each stock
stock_symbols = [symbol.strip() for symbol in symbols.split(',')]
stock_data = yf.download(stock_symbols, start=start, end=stop)

# Plot stock prices for multiple stocks
for symbol in stock_symbols:
    plt.plot(stock_data.index, stock_data['Close'][symbol], label=symbol)

plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Prices Comparison')
plt.legend()
st.pyplot(plt)

# Streamlit app setup
st.title("Personalized Watchlists")

# Watchlist management
watchlist = []

# Add stock to watchlist
st.header("Add Stock to Watchlist")

stock_symbol = st.text_input("Stock Symbol", value="AAPL")

if st.button("Add to Watchlist"):
    watchlist.append(stock_symbol)
    st.success(f"Added {stock_symbol} to your watchlist!")

# Display watchlist
st.header("Your Watchlist")

if len(watchlist) > 0:
    for symbol in watchlist:
        st.write(symbol)
else:
    st.info("Your watchlist is empty.")

# Historical Returns Calculation
returns = data['Close'].pct_change()
returns_cumulative = (returns + 1).cumprod() - 1

# Historical Returns Plot
if st.button('Historical Returns'):
    plt.plot(returns_cumulative)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.title('Historical Cumulative Returns')
    st.pyplot()

# Calculate RSI
data['RSI'] = momentum.RSIIndicator(data['Close']).rsi()

# Plot RSI
# Relative Strength Index (RSI)
delta = data['Close'].diff()
gain = delta.mask(delta < 0, 0)
loss = -delta.mask(delta > 0, 0)
average_gain = gain.rolling(window=14).mean()
average_loss = loss.rolling(window=14).mean()
rs = average_gain / average_loss
rsi = 100 - (100 / (1 + rs))

if st.button('Relative Strength Index (RSI)'):
    fig, ax = plt.subplots()
    ax.plot(data.index, rsi)
    ax.axhline(30, color='red', linestyle='--')
    ax.axhline(70, color='red', linestyle='--')
    ax.set_xlabel('Date')
    ax.set_ylabel('RSI')
    ax.set_title('Relative Strength Index (RSI)')
    st.pyplot(fig)

# Bollinger Bands

rolling_mean = data['Close'].rolling(window=20).mean()
rolling_std = data['Close'].rolling(window=20).std()
upper_band = rolling_mean + (2 * rolling_std)
lower_band = rolling_mean - (2 * rolling_std)
if st.button('Bollinger Bands'):
    fig, ax = plt.subplots()
    ax.plot(data.index, data['Close'], label='Close Price')
    ax.plot(data.index, upper_band, label='Upper Band')
    ax.plot(data.index, lower_band, label='Lower Band')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title('Bollinger Bands')
    ax.legend()
    st.pyplot(fig)

 

if st.button('Stock Price Forecast'):
    
    data = yf.download(company, start=start, end=stop)

    # Prepare
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    prediction_days = 60

    X_train = []
    y_train = []

    for x in range(prediction_days, len(scaled_data)):
        X_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    # Model
    model = load_model('model.h5')

    # Test Model
    test_start = dt.datetime(2023, 5, 1)
    test_stop = dt.datetime.now()

    test_data = yf.download(company, start=test_start, end=test_stop)
    actual_prices = test_data['Close'].values

    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    # Predict Next Day
    real_data = [model_inputs[len(model_inputs) - prediction_days:len(model_inputs-1), 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    prediction = prediction[0][0]
    
    st.write('THE PREDICTED PRICE FOR THE FUTURE :',prediction)

stock_data = pd.DataFrame({
    'Symbol': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB'],
    'Name': ['Apple Inc.', 'Alphabet Inc.', 'Microsoft Corporation', 'Amazon.com Inc.', 'Facebook Inc.'],
    'MarketCap': [2265, 1869, 1779, 1697, 1048],
    'Industry': ['Technology', 'Technology', 'Technology', 'Retail', 'Technology'],
    'PE_Ratio': [31.98, 29.32, 38.78, 66.45, 24.59]
})


st.title("Advanced Stock Screener")

industry = st.selectbox("Industry", sorted(stock_data['Industry'].unique()))
min_market_cap = st.number_input("Minimum Market Cap", min_value=0)
max_pe_ratio = st.number_input("Maximum P/E Ratio", min_value=0)

filtered_data = stock_data[
    (stock_data['Industry'] == industry) &
    (stock_data['MarketCap'] >= min_market_cap) &
    (stock_data['PE_Ratio'] <= max_pe_ratio)
]

if len(filtered_data) > 0:
    st.dataframe(filtered_data)
else:
    st.info("No stocks match the selected criteria.")



