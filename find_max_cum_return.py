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
import logging
import time 

def get_logger(log_dir, name):
    """Get a `logging.Logger` instance that prints to the console
    and an auxiliary file.
    Args:
        log_dir (str): Directory in which to create the log file.
        name (str): Name to identify the logs.
    Returns:
        logger (logging.Logger): Logger instance for logging events.
    """
    class StreamHandlerWithTQDM(logging.Handler):
        """Let `logging` print without breaking `tqdm` progress bars.
        See Also:
            > https://stackoverflow.com/questions/38543506
        """
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Log everything (i.e., DEBUG level and above) to a file
    log_path = os.path.join(log_dir, f'{name}.txt')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Log everything except DEBUG level (i.e., INFO level and above) to console
    console_handler = StreamHandlerWithTQDM()
    console_handler.setLevel(logging.INFO)

    # Create format for the logs
    file_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                       datefmt='%m.%d.%y %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    console_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                          datefmt='%m.%d.%y %H:%M:%S')
    console_handler.setFormatter(console_formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_data(symbols, 
             add_ref=True,
             data_source='quandl', #'yahoo' 
             price='AdjClose', #price='Adj Close',
             start='1/21/2010', 
             end='4/15/2016'):
    """Read stock data (adjusted close) for given symbols from."""
    if type(symbols) == list:
        symbols = list(set(symbols))
    if data_source.lower()=='quandl':
      df = web.DataReader(symbols, 'quandl', start=start, end=end , api_key=os.getenv('QUANDL_API_KEY'))
      return df[price]
    else:
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
    log = get_logger('.', 'find_max_cum_ret')
    nasdaq_symbols = get_nasdaq_symbols()
    stock_list = list(nasdaq_symbols.index)
    random.shuffle(stock_list)
    if args.max_stocks is not None and args.max_stocks > 0:
        stock_list = stock_list[:args.max_stocks]
        
    
    end_time = datetime.date.today().strftime("%m/%d/%Y")
    start_time = (datetime.date.today() + datetime.timedelta(days=-1*args.window_size)).strftime("%m/%d/%Y")
    
    # beffering
    BUFF_SIZE = 50
    n_batch = len(stock_list) // BUFF_SIZE
    cum_df = None 
    
    for i in range(n_batch):
        batch = stock_list[i*BUFF_SIZE:(i+1)*BUFF_SIZE]
        try:
            _df = fill_missing_values(get_data(batch, start=start_time, end=end_time))
            _cum_df = cumulative_returns(_df)
            if cum_df is None:
                cum_df = _cum_df
            else:
                cum_df = pd.concat([cum_df,_cum_df],axis=1) 
                # top-k 
                top_k = pd.DataFrame(cum_df.iloc[len(cum_df) - 1,:].sort_values(ascending=False)[:args.top_k])
                print("***********",i+1,"/",n_batch)
                print(top_k)
                top_k.to_csv(args.file_name)
        except Exception as err:
            log.error('<<Error>>::'+str(err), exc_info=True)
    
    
    ##################
    seconds = time.time() - start_time_millis
    mins = seconds / 60
    hours = mins / 60
    days = hours / 24
    print("------>>>>>>> elapsed seconds: " + str(seconds))
    print("------>>>>>>> elapsed minutes: " + str(mins))
    print("------>>>>>>> elapsed hours: " + str(hours))
    print("------>>>>>>> elapsed days: " + str(days))
