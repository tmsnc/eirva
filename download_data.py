import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import pickle as pkl
import os
import sys
from pathlib import Path

# adding Pushover module
dir_to_add = str(Path('E:\Google Drive\Python\Tools'))
sys.path.append(dir_to_add)
from pyPushover import pushover_send

os.chdir('E:/Google Drive/3. UM MADS/2022 Winter/SIADS 697 698 Capstone/eirva/eirva')

try:
    df_csi = pd.read_csv("2846_holdings.csv", skiprows=2)

    csi_tickers = df_csi.loc[df_csi['Ticker'].str.contains('[0-9]+$', regex=True), 'Ticker']
    csi_tickers = csi_tickers.map(lambda x: '{:06d}'.format(int(x)))
    csi_tickers = csi_tickers.map(lambda x: ['{}.SS'.format(x) if x[0]=='6' else '{}.SZ'.format(x)][0])
    csi_list = csi_tickers.to_list()
    yf_tickers = yf.Tickers(csi_list)

    # Run and save111111111
    df_5y = yf_tickers.download(period='5y', interval='1d', threads=True)
    pkl.dump(df_5y, open('.\data\df_5y_{}.pkl'.format(datetime.datetime.strftime(datetime.datetime.today(), '%Y%m%d')), 'wb'))

    df_info_raw = pd.DataFrame({ticker: tick_obj.info for ticker, tick_obj in yf_tickers.tickers.items()})
    pkl.dump(df_info_raw, open('.\data\df_info_raw_{}.pkl'.format(datetime.datetime.strftime(datetime.datetime.today(), '%Y%m%d')), 'wb'))

    ### Send notification
    pushover_send('Success: CSI300 data downloaded', 'yfinance CSI300')
    print('Data downloaded!')

except Exception as e:
    pushover_send('Failed: CSI300 data cannot download, error: {}'.format(str(e)), 'yfinance CSI300')
    print('Something went wrong...')
