import os

import pandas as pd
import csv
import requests
from .symbols_data import SYMBOLS
from .tse_settings import TSE_TICKER_EXPORT_DATA_ADDRESS
from .translations import HISTORY_FIELD_MAPPINGS


class Stock_Data:
    def __init__(self, symbol_name):
        self.symbol_name = symbol_name

    def __get_symbol_id(self):
        return SYMBOLS[self.symbol_name]

    def __get_symbol_url(self):
        return TSE_TICKER_EXPORT_DATA_ADDRESS.format(self.__get_symbol_id())

    def __request_url(self):
        return requests.get(self.__get_symbol_url())

    def __save_csv(self):
        with open(f'{self.symbol_name}.csv', 'w') as f:
            writer = csv.writer(f)
            for line in self.__request_url().iter_lines():
                writer.writerow(line.decode('utf-8').split(','))

    def __rename_columns(self, csv_file):
        return csv_file.rename(columns=HISTORY_FIELD_MAPPINGS)

    def __reverse(self, csv_file):
        return csv_file.iloc[::-1]

    def __normalization_file(self):
        csv_file = pd.read_csv(f'{self.symbol_name}.csv', encoding='utf-8')
        csv_file = self.__rename_columns(csv_file)
        csv_file = self.__reverse(csv_file)
        csv_file['Date'] = pd.to_datetime(csv_file['<DTYYYYMMDD>'], format='%Y%m%d')
        csv_file.set_index('Date', inplace=True)
        csv_file.drop(["<DTYYYYMMDD>"], axis = 1, inplace = True)

        return csv_file

    def __delete_file(self):
        try:
            os.remove(f'{self.symbol_name}.csv')
        except:
            print('OSError: The file could not be deleted.')

    def get_data(self):
        self.__save_csv()
        data = self.__normalization_file()
        data.to_csv(f'{self.symbol_name}.csv')
        self.__delete_file()

        return data



