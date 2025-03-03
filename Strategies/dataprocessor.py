# Backtesting Algorithm Refactor - Class Design
import MetaTrader5 as mt5
import pandas as pd
import datetime
import requests
import time
from datetime import datetime


class DataProcessor:
    """Handles data retrieval from MT5, Binance, and local sources, along with data optimization and exchange rates."""
    #
    def __init__(self,period_info,settings):
        self.period_info= period_info
        self.settings = settings  # Store settings for use across methods

    def get_data(self):
        symbol= self.settings['symbol']
        interval=self.settings['interval']
        lw_interval= self.settings['lw_interval']
        exchange = self.settings['broker_info']['exchange']
        broker_info = {
            "IC": (
                "ICMarketsSC-Demo",
                   "C:/Program Files/MetaTrader 5 IC Markets Global/terminal64.exe"),
            "Tickmill": (
            , "Tickmill-Demo",
                "C:/Program Files/Tickmill MT5 Terminal/terminal64.exe")
        }

        if exchange in broker_info:
            login, password, server, path = broker_info[exchange]
            if not self.start_mt5(login, password, server, path):
                print(f"Failed to connect to broker for {symbol}.")
                return None, None, None, None

        exchange_symbol = self.get_exchange_rate_symbol(symbol, exchange)

        if exchange in ['IC', 'Tickmill']:
            data, data_lw = self.get_mt5_data(symbol, interval, lw_interval, self.period_info)
            exchange_data, exchange_data_lw = self.get_mt5_data(exchange_symbol, interval, lw_interval,
                                                                self.period_info) if exchange_symbol else (None, None)

        elif exchange == 'binance':
            fetch_method = self.get_db_binance_data if self.settings['broker_info']['db'] else self.get_binance_data_direct
            data, data_lw = fetch_method(symbol, interval, lw_interval, self.period_info['start_date'],
                                         self.period_info['end_date'], self.settings['broker_info']['market_type'])

            if data.empty:
                print('Incomplete Binance data for input.')
                return None, None, None, None
            exchange_data, exchange_data_lw = fetch_method(exchange_symbol, interval, lw_interval,
                                                           self.period_info['start_date'], self.period_info['end_date'],
                                                           self.settings['broker_info'][
                                                               'market_type']) if exchange_symbol else (None, None)

        else:
            print(f"Invalid exchange: {exchange} for symbol {symbol}.")
            return None, None, None, None
        print('Data of higher timeframe:\n', data)
        print('Data of lower timeframe:\n', data_lw)

        return data, data_lw, exchange_data, exchange_data_lw
    def start_mt5(self, login, password, server, path):
        # Initialize MT5 connection
        if not mt5.initialize(path=path):
            error_code = mt5.last_error()
            print(f"initialize() failed, error code = {error_code}")
            if error_code[0] == -6:
                print("Check your credentials and server name.")
            return False

        # Connect to the specified trading account
        authorized = mt5.login(login, password=password, server=server)
        if not authorized:
            error_code = mt5.last_error()
            print(f"login() failed, error code = {error_code}")
            if error_code[0] == -6:
                print("Authorization failed: Verify login ID, password, and server.")
            return False

        #     print("MT5 initialized and logged in successfully!")
        return True


    def get_timeframe(self, interval):
        timeframe_mapping = {
            '1m': mt5.TIMEFRAME_M1,
            '2m': mt5.TIMEFRAME_M2,
            '3m': mt5.TIMEFRAME_M3,
            '4m': mt5.TIMEFRAME_M4,
            '5m': mt5.TIMEFRAME_M5,
            '6m': mt5.TIMEFRAME_M6,
            '10m': mt5.TIMEFRAME_M10,
            '12m': mt5.TIMEFRAME_M12,
            '15m': mt5.TIMEFRAME_M15,
            '20m': mt5.TIMEFRAME_M20,
            '30m': mt5.TIMEFRAME_M30,
            '1h': mt5.TIMEFRAME_H1,
            '2h': mt5.TIMEFRAME_H2,
            '3h': mt5.TIMEFRAME_H3,
            '4h': mt5.TIMEFRAME_H4,
            '6h': mt5.TIMEFRAME_H6,
            '8h': mt5.TIMEFRAME_H8,
            '12h': mt5.TIMEFRAME_H12,
            '1day': mt5.TIMEFRAME_D1,
            '1week': mt5.TIMEFRAME_W1,
            '1month': mt5.TIMEFRAME_MN1
        }
        return timeframe_mapping.get(interval, None)

    def get_mt5_data(self, symbol, interval, lw_interval, backtest):
        interval_minutes = {
            '1m': 1, '5m': 5, '15m': 15,
            '30m': 30, '1h': 60, '2h': 120, '4h': 240, '12h': 720, '1day': 1440,
        }

        minutes_per_candle = interval_minutes.get(interval, None)
        if minutes_per_candle is None:
            raise ValueError("Unsupported interval")

        # Convert start_date and end_date from string to datetime objects
        start_date = datetime.strptime(backtest.get('start_date', False), '%Y-%m-%d')
        end_date = datetime.strptime(backtest.get('end_date', False), '%Y-%m-%d')
        end_date = end_date  # Reset seconds and microseconds for all intervals

        data = pd.DataFrame(mt5.copy_rates_range(symbol,
                                                 self.get_timeframe(interval),
                                                 start_date,
                                                 end_date))
        data = data.iloc[:, :6]
        data.columns = ['time', 'Open', 'High', 'Low', 'Close', 'Volume']
        data.set_index('time', inplace=True)
        data.index = pd.to_datetime(data.index, unit='s')
        data = data.astype(float)
        # Localize to GMT+3 and convert to GMT
        data.index = data.index.tz_localize('Etc/GMT-3').tz_convert('GMT')

        ## get lower dataframe
        data_lw = pd.DataFrame(mt5.copy_rates_range(symbol,
                                                    self.get_timeframe(lw_interval),
                                                    start_date,
                                                    end_date))
        data_lw = data_lw.iloc[:, :6]
        data_lw.columns = ['time', 'Open', 'High', 'Low', 'Close', 'Volume']
        data_lw.set_index('time', inplace=True)
        data_lw.index = pd.to_datetime(data_lw.index, unit='s')
        data_lw = data_lw.astype(float)
        # Localize to GMT+3 and convert to GMT
        data_lw.index = data_lw.index.tz_localize('Etc/GMT-3').tz_convert('GMT')

        return data, data_lw

    def get_binance_data_direct(self,
            symbol, interval, lw_interval, start_datetime, end_datetime, market_type, retries=3, retry_delay=5
    ):
        BINANCE_BASE_URL_SPOT = 
        BINANCE_BASE_URL_FUTURES = 
        if market_type == 'spot':
            exchange_url = BINANCE_BASE_URL_SPOT
        elif market_type == 'future':
            exchange_url = BINANCE_BASE_URL_FUTURES

        # Ensure datetime consistency
        start_datetime = pd.to_datetime(start_datetime).tz_localize('UTC')
        end_datetime = pd.to_datetime(end_datetime).tz_localize('UTC')
        start = int(start_datetime.timestamp() * 1000)
        end = int(end_datetime.timestamp() * 1000)

        def fetch_data(exchange_url, interval, start, end):
            klines = []
            while True:
                try:
                    new_klines = requests.get(
                        exchange_url,
                        params={
                            'symbol': symbol,
                            'interval': interval,
                            'startTime': start,
                            'endTime': end,
                            'limit': 1000
                        }
                    ).json()
                    if not new_klines:
                        break
                    klines.extend(new_klines)
                    start = new_klines[-1][0] + 1
                    if start >= end:
                        break
                except Exception as e:
                    nonlocal retries
                    retries -= 1
                    print(f"An error occurred: {e}. Retrying in {retry_delay} seconds...")
                    if retries == 0:
                        print("Maximum retries reached. Exiting.")
                        break
                    time.sleep(retry_delay)

            if not klines:
                return pd.DataFrame()

            df = pd.DataFrame(klines, columns=[
                'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Close Time', 'Quote Asset Volume', 'Number of Trades',
                'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
            ])

            df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms', utc=True)
            df = df[['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume']]
            df.columns = ['time', 'Open', 'High', 'Low', 'Close', 'Volume']
            df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(
                float)
            df.set_index('time', inplace=True)
            df.sort_index(inplace=True)
            df = df[~df.index.duplicated(keep='first')]
            return df

        # Fetch data for the main interval
        df = fetch_data(exchange_url, interval, start, end)
        # Fetch data for the lower interval
        df_lw = fetch_data(exchange_url, lw_interval, start, end)

        # Check for missing data in the main interval
        full_period = pd.date_range(start=start_datetime, end=end_datetime, freq=interval)
        missing_period = full_period.difference(df.index)

        if not missing_period.empty:
            print("Some data is missing for the requested period. Returning empty DataFrames.")
            return pd.DataFrame(), pd.DataFrame()

        return df, df_lw


    def optimize_financial_data(self, df):
        for col in df.columns:
            col_type = df[col].dtypes

            if col_type == 'float64':
                # Convert to float32 for columns with decimal values
                df[col] = df[col].astype('float32')

        return df

    def get_exchange_rate_symbol(self, symbol, exchange='IC'):
        usd_priced_symbols = ["USTEC", "US30", "DE40", "US500", "SPX500", "XAUUSD", "XAGUSD", "WTI", "BRENT"]

        # If the symbol is in the list of USD priced symbols, return nothing
        if symbol in usd_priced_symbols:
            return

        if exchange == 'IC' or exchange == 'Tickmill':
            account_currency = 'USD'
        elif exchange == 'binance':
            account_currency = 'USDT'
        # Extract the base and quote currencies from the symbol
        base_currency = symbol[:3]  # First three characters (e.g., "EUR" from "EURGBP")
        quote_currency = symbol[3:]  # Last three characters (e.g., "GBP" from "EURGBP")

        if account_currency == quote_currency:
            # If the account currency is the same as the quote currency, return nothing
            return
        else:
            # Determine the conversion pair symbol
            conversion_pair = quote_currency + account_currency  # e.g., "GBPUSD"
            if mt5.symbol_select(conversion_pair):
                return conversion_pair
            else:
                # Try the reverse pair (e.g., "USDGBP")
                conversion_pair = account_currency + quote_currency
                if mt5.symbol_select(conversion_pair):
                    return conversion_pair

    def get_exchange_rate(self, df, exchange_df,  df_minute, exchange_df_minute, symbol, exchange='IC'):
        # List of known symbols that are priced in USD by default (like indices, commodities, etc.)
        usd_priced_symbols = ["USTEC", "US30", "DE40", "US500", "SPX500", "XAUUSD", "XAGUSD", "WTI", "BRENT"]

        # If the symbol is in the list of USD-priced symbols, set exchange rate to 1
        if symbol in usd_priced_symbols:
            df['exchange_rate'] = 1.0
            df_minute['exchange_rate'] = 1.0
            return df, df_minute

        # Extract the base and quote currencies from the symbol
        base_currency = symbol[:3]  # First three characters (e.g., "EUR" from "EURGBP")
        quote_currency = symbol[3:]  # Last three characters (e.g., "GBP" from "EURGBP")

        if exchange=='IC' or exchange=='Tickmill':
            account_currency= 'USD'
        elif exchange== 'binance':
            account_currency= 'USDT'

        if account_currency == quote_currency:
            # If the account currency is the same as the quote currency, the exchange rate is 1
            df['exchange_rate'] = 1.0
            df_minute['exchange_rate'] = 1.0
            return df, df_minute
        else:

            conversion_pair = quote_currency + account_currency  # e.g., "GBPUSD"

            if mt5.symbol_select(conversion_pair):

                df['exchange_rate'] = exchange_df['Close']
                df_minute['exchange_rate'] = exchange_df_minute['Close']
                return df, df_minute
            else:
                # Try the reverse pair (e.g., "USDGBP")
                conversion_pair = account_currency + quote_currency
                if mt5.symbol_select(conversion_pair):
                    df['exchange_rate'] = 1/exchange_df['Close']
                    df_minute['exchange_rate'] = 1/exchange_df_minute['Close']
                    return df, df_minute

        # If no valid conversion pair is found, return df without modifying
        return df, df_minute



# Next Step: Implement TradeManager with position sizing, trade entry/exit logic, and Martingale strategy.





