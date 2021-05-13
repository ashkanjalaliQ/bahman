import pandas as pd
import yfinance as yf

class Finance:
    def __init__(self, symbol):
        #self.symbol = symbol
        pass

    def get_history(self, date_range, period, symbol):
        history = yf.Ticker(symbol)
        history = history.history(period=period, start=date_range)
        history['Date'] = history.index
        history = history.set_index(pd.DatetimeIndex(history['Date'].values))

        return history

    '''def get_symbol_name(self):
        return self.symbol'''
