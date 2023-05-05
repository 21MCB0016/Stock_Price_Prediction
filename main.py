from datetime import datetime, timedelta
from yahoofinancials import YahooFinancials
from prediction import Predictor
import pandas as pd
import pymongo


class Stock:
    def __init__(self, db_uri, db_name, db_collection, tickers):
        self.client = pymongo.MongoClient(db_uri)
        self.db = self.client[db_name]
        self.collection = self.db[db_collection]
        self.tickers = tickers
        self.update_stock_data()

    def update_stock_data(self):
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=2191)).strftime('%Y-%m-%d')
        yahoo_financials = YahooFinancials(self.tickers)
        stock_data = yahoo_financials.get_historical_price_data(start_date, end_date, "daily")
        if stock_data:
            for i in self.tickers:
                prices = stock_data[i]["prices"]
                data = pd.DataFrame(prices)[["close", "formatted_date", "adjclose", "low", "high", "open", "volume"]]
                data["ticker"] = i
                data["date"] = pd.to_datetime(data["formatted_date"])
                data = data[["ticker", "date", "close", "adjclose", "low", "high", "open", "volume"]]
                self.collection.insert_many(data.to_dict(orient="records"))
                self.collection.update_many({"ticker": i},
                                            {"$set": {"last_updated": datetime.now().strftime('%Y-%m-%d')}},
                                            upsert=True)

        print("Stock data updated successfully")


if __name__ == "__main__":
    # Define the MongoDB URI and database name
    db_uri = "mongodb://localhost:27017"
    db_name = "stock_data"
    db_collection = "bank_stocks"
    tickers = ["JPM", "BAC", "WFC", "C", "MS", "GS", "BK", "USB", "PNC", "COF", "AXP", "BLK", "SGI", "MET", "AIG"]
    # Create a new Stock instance
    stock = Stock(db_uri, db_name, db_collection, tickers)
    # Create a new Prediction instance
    prediction = Predictor(db_uri, db_name, db_collection, tickers)
