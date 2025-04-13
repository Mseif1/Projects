import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    stock_data = stock.history(period="max").reset_index()
    return stock_data

def get_revenue_data(url, table_class):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    tables = soup.find_all("table", class_=table_class)
    
    if tables:
        df = pd.read_html(str(tables[0]))[0]
        return df
    else:
        return pd.DataFrame()

tesla_stock = get_stock_data("TSLA")
print(tesla_stock.head())

tesla_revenue_url = "https://www.macrotrends.net/stocks/charts/TSLA/tesla/revenue"
tesla_revenue = get_revenue_data(tesla_revenue_url, "historical_data_table")
print(tesla_revenue.tail())

gme_stock = get_stock_data("GME")
print(gme_stock.head())

gme_revenue_url = "https://www.macrotrends.net/stocks/charts/GME/gamestop/revenue"
gme_revenue = get_revenue_data(gme_revenue_url, "historical_data_table")
print(gme_revenue.tail())

def make_graph(stock_data, title):
    plt.figure(figsize=(10,5))
    sns.lineplot(data=stock_data, x=stock_data['Date'], y="Close", label=title)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Stock Price (USD)")
    plt.legend()
    plt.show()

make_graph(tesla_stock, "Tesla Stock Price Over Time")

make_graph(gme_stock, "GameStop Stock Price Over Time")