import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import datetime
import os 
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols
import argparse
import datetime
import random


def get_data(symbols, 
             add_ref=True,
             data_source='yahoo',
             price='Adj Close',
             start='1/21/2010', 
             end='4/15/2016'):
    """Read stock data (adjusted close) for given symbols from."""
    
    if add_ref and 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    df = web.DataReader(symbols, 
                        data_source=data_source,
                        start=start, 
                        end=end)
    
    return df[price]

def compute_daily_returns(df):
    """Compute and return the daily return values."""
    # Note: Returned DataFrame must have the same number of rows
    daily_returns = (df / df.shift(1)) - 1 
    daily_returns.ix[0,:] = 0 
    return daily_returns

def fill_missing_values(df_data,is_null_max_perc=0.3):
    """Fill missing values in data frame, in place."""
    df_data = df_data[df_data.columns[df_data.isnull().mean() < is_null_max_perc]]
    df_data.fillna(method='ffill',inplace=True)
    df_data.fillna(method='backfill',inplace=True)
    return df_data

def cumulative_returns(df):
    return df/df.iloc[0,:] - 1 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--window-size', type=int, default=30)
    parser.add_argument('--file-name', type=str, default='top_cumulative_return.csv')
    parser.add_argument('--max-stocks', type=int, default=100)
    parser.add_argument('--top_k', type=int, default=10)
    args = parser.parse_args()
    
    # print parameters
    print('-' * 30)
    print('Parameters .')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)
    
    ##
    nasdaq_symbols = get_nasdaq_symbols()
    stock_list = list(nasdaq_symbols.index)
    random.shuffle(stock_list)
    if args.max_stocks is not None and args.max_stocks > 0:
        stock_list = stock_list[:args.max_stocks]
        
    
    end_time = datetime.date.today().strftime("%m/%d/%Y")
    start_time = (datetime.date.today() + datetime.timedelta(days=-1*args.window_size)).strftime("%m/%d/%Y")
    
    df = fill_missing_values(get_data(stock_list, start=start_time, end=end_time))
    
    cum_df = cumulative_returns(df)
    
    top_k = pd.DataFrame(cum_df.iloc[len(cum_df) - 1,:].sort_values(ascending=False)[:args.top_k])
    
    print(top_k)
    
    top_k.to_csv(args.file_name)
    
    
    
    
    
    
    