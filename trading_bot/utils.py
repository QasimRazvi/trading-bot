import os
import math
import logging

import pandas as pd
from pandas import DataFrame, to_datetime, read_json
import numpy as np

import keras.backend as K

# Formats Position
format_position = lambda price: ('-$' if price < 0 else '+$') + '{0:.2f}'.format(abs(price))

# Formats Currency
format_currency = lambda price: '${0:.2f}'.format(abs(price))


def show_train_result(result, val_position, initial_offset):
    """ Displays training results
    """
    if val_position == initial_offset or val_position == 0.0:
        logging.info('Episode {}/{} - Train Position: {}  Val Position: USELESS  Train Loss: {:.4f}'
                     .format(result[0], result[1], format_position(result[2]), result[3]))
    else:
        logging.info('Episode {}/{} - Train Position: {}  Val Position: {}  Train Loss: {:.4f})'
                     .format(result[0], result[1], format_position(result[2]), format_position(val_position),
                             result[3], ))


def show_eval_result(model_name, profit, initial_offset):
    """ Displays eval results
    """
    if profit == initial_offset or profit == 0.0:
        logging.info('{}: USELESS\n'.format(model_name))
    else:
        logging.info('{}: {}\n'.format(model_name, format_position(profit)))


def get_stock_data(stock_file):
    """Reads stock data from csv file
    """
    df = pd.read_csv(stock_file)
    return list(df['Adj Close'])


def get_json_data(stock_file):
    """
    Reads data from json file (nested array)
    returns list
    """
    # read json retrieved from binance
    # convert to dataframe and add column names
    pair = read_json(stock_file, orient='values')
    cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    pair.columns = cols

    # ensure floats
    pairdata = pair.astype(dtype={'open': 'float', 'high': 'float',
                                  'low': 'float', 'close': 'float', 'volume': 'float'})
    # convert timestamps
    pairdata['date'] = to_datetime(pairdata['date'],
                                   unit='ms',
                                   utc=True,
                                   infer_datetime_format=True)
    return list(pairdata['close'])


def switch_k_backend_device():
    """ Switches `keras` backend from GPU to CPU if required.

    Faster computation on CPU (if using tensorflow-gpu).
    """
    if K.backend() == "tensorflow":
        logging.debug("switching to TensorFlow for CPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
