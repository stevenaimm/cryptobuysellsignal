# Start of trading_bot.py
import pandas as pd
import numpy as np
from binance.client import Client
import time
# import os # For API key management if used: os.environ.get('BINANCE_API_KEY')

# Initialize Binance client globally
# For public data endpoints, API key/secret are not mandatory.
# For trading or private endpoints, you would use:
# client = Client(os.environ.get('BINANCE_API_KEY'), os.environ.get('BINANCE_API_SECRET'))
client = None # Initialize client as None initially

# Configuration defaults (can be overridden in run_bot or process_symbol)
DEFAULT_DATA_LIMIT = 250
DEFAULT_SMA_SHORT_WINDOW = 50
DEFAULT_SMA_LONG_WINDOW = 200
DEFAULT_RSI_WINDOW = 14
DEFAULT_RSI_OVERSOLD = 30
DEFAULT_RSI_OVERBOUGHT = 70

def get_historical_klines(symbol, interval, limit=DEFAULT_DATA_LIMIT):
    """
    Fetches historical k-line (candlestick) data for a given symbol.
    """
    try:
        # print(f"Fetching {limit} klines for {symbol}, interval {interval}")
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        if not klines:
            print(f"Warning: No k-line data received for {symbol} with interval {interval}.")
            return None
        # ... (rest of the function remains the same)
        columns = [
            'kline_open_time', 'open', 'high', 'low', 'close', 'volume',
            'kline_close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
            'ignore'
        ]
        df = pd.DataFrame(klines, columns=columns)

        numeric_cols = ['open', 'high', 'low', 'close', 'volume',
                        'quote_asset_volume', 'taker_buy_base_asset_volume',
                        'taker_buy_quote_asset_volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce') # Coerce errors to NaN

        df['kline_open_time'] = pd.to_datetime(df['kline_open_time'], unit='ms')
        df['kline_close_time'] = pd.to_datetime(df['kline_close_time'], unit='ms')

        df.set_index('kline_open_time', inplace=True)

        # Check for NaNs introduced by coerce
        if df[numeric_cols].isnull().any().any():
            print(f"Warning: NaN values found in numeric columns for {symbol} after conversion. Data might be corrupted.")
            # Potentially drop rows with NaNs if appropriate, or handle as error
            # df.dropna(subset=numeric_cols, inplace=True)
        return df

    except Exception as e: # More specific exceptions can be caught, e.g., BinanceAPIException
        print(f"Error fetching k-lines for {symbol}: {e}")
        return None

def calculate_sma(data, window):
    """
    Calculates the Simple Moving Average (SMA).
    """
    if data is None or not isinstance(data, pd.Series):
        # print("Error: Input data for SMA must be a Pandas Series.")
        return pd.Series(dtype='float64')
    if len(data) < window:
        # print(f"Warning: Not enough data (len {len(data)}) to calculate SMA with window {window}.")
        return pd.Series(dtype='float64')
    return data.rolling(window=window, min_periods=window).mean() # Ensure full window for SMA

def calculate_rsi(data, window=DEFAULT_RSI_WINDOW):
    """
    Calculates the Relative Strength Index (RSI).
    """
    if data is None or not isinstance(data, pd.Series):
        # print("Error: Input data for RSI must be a Pandas Series.")
        return pd.Series(dtype='float64')
    if len(data) < window + 1:
        # print(f"Warning: Not enough data (len {len(data)}) for RSI with window {window}.")
        return pd.Series(dtype='float64')

    delta = data.diff()
    if delta is None: return pd.Series(dtype='float64')

    # Correctly handle NaNs from .diff() and ensure they propagate
    # so that gain/loss series start with NaN if delta did.
    gain = delta.copy()
    gain[delta <= 0] = 0.0  # Set non-positive changes to 0, keeps NaNs as NaNs
    gain[delta.isnull()] = np.nan # Explicitly ensure NaNs in delta are NaNs in gain

    loss = -delta.copy() # Make losses positive
    loss[delta >= 0] = 0.0  # Set non-negative changes to 0, keeps NaNs as NaNs
    loss[delta.isnull()] = np.nan # Explicitly ensure NaNs in delta are NaNs in loss

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    # Avoid division by zero if loss is 0 (which can happen if price never drops in the window)
    rs = avg_gain / avg_loss.replace(0, 0.000001) # Replace 0 loss with a tiny number
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(data, sma_short_window, sma_long_window, rsi_window, rsi_oversold, rsi_overbought):
    """
    Generates trading signals. Parameters are now explicit.
    """
    # Initialize signals Series
    signals = pd.Series(index=data.index, dtype=str, name='signal')

    sma_short_col = f'sma_{sma_short_window}'
    sma_long_col = f'sma_{sma_long_window}'
    rsi_col = f'rsi_{rsi_window}'

    if sma_short_col not in data.columns or sma_long_col not in data.columns or rsi_col not in data.columns:
        print(f"Error: Required indicator columns ({sma_short_col}, {sma_long_col}, {rsi_col}) not found.")
        # If critical columns are missing, return 'Hold' for all entries.
        # Ensure the Series has the correct index even when returning early.
        signals = signals.reindex(data.index).fillna('Hold')
        return signals

    # Shifted SMAs for crossover detection (needed for entry signals)
    # Ensure these new columns are added to a copy if data modification is not desired upstream,
    # or ensure that the DataFrame passed is okay to be modified.
    # For this function, it's usually fine to modify the input `data` DataFrame.
    prev_sma_short_col = f'prev_{sma_short_col}'
    prev_sma_long_col = f'prev_{sma_long_col}'
    data[prev_sma_short_col] = data[sma_short_col].shift(1)
    data[prev_sma_long_col] = data[sma_long_col].shift(1)

    # Conditions for entering a new Long/Short position (crossover based)
    enter_long_crossover = (data[sma_short_col] > data[sma_long_col]) & (data[prev_sma_short_col] <= data[prev_sma_long_col])
    enter_short_crossover = (data[sma_short_col] < data[sma_long_col]) & (data[prev_sma_short_col] >= data[prev_sma_long_col])

    # Conditions for holding/maintaining a position (RSI checks & SMA state)
    rsi_ok_for_long = data[rsi_col] < rsi_overbought
    rsi_ok_for_short = data[rsi_col] > rsi_oversold

    # General market conditions based on SMAs (not crossover, but current state)
    sma_is_bullish = data[sma_short_col] > data[sma_long_col]
    sma_is_bearish = data[sma_short_col] < data[sma_long_col]

    # Iterate and apply stateful logic
    # This loop is necessary because the signal at time t depends on signal at t-1
    # Initialize the first signal as 'Hold' as there's no previous signal to maintain.
    if len(data) > 0:
        signals.iloc[0] = 'Hold' # Explicitly set first signal to Hold

    for i in range(len(data)): # Start from 0 to set initial signal based on entry, or from 1 if first is always Hold
        current_signal_decision = 'Hold' # Default for current iteration

        previous_signal_state = signals.iloc[i-1] if i > 0 else 'Hold'

        # Check for new Long entry
        if enter_long_crossover.iloc[i] and rsi_ok_for_long.iloc[i]:
            current_signal_decision = 'Long'
        # Check for new Short entry
        elif enter_short_crossover.iloc[i] and rsi_ok_for_short.iloc[i]:
            current_signal_decision = 'Short'
        # Else, if already in a Long position (based on previous_signal_state), check if it should be maintained
        elif previous_signal_state == 'Long':
            if sma_is_bullish.iloc[i] and rsi_ok_for_long.iloc[i]:
                current_signal_decision = 'Long'  # Maintain Long
            # else: current_signal_decision remains 'Hold' (implicit exit from Long)
        # Else, if already in a Short position, check if it should be maintained
        elif previous_signal_state == 'Short':
            if sma_is_bearish.iloc[i] and rsi_ok_for_short.iloc[i]:
                current_signal_decision = 'Short' # Maintain Short
            # else: current_signal_decision remains 'Hold' (implicit exit from Short)

        signals.iloc[i] = current_signal_decision

    return signals

def process_symbol(symbol, interval,
                   data_limit=DEFAULT_DATA_LIMIT,
                   sma_short_window=DEFAULT_SMA_SHORT_WINDOW,
                   sma_long_window=DEFAULT_SMA_LONG_WINDOW,
                   rsi_window=DEFAULT_RSI_WINDOW,
                   rsi_oversold=DEFAULT_RSI_OVERSOLD,
                   rsi_overbought=DEFAULT_RSI_OVERBOUGHT):
    """
    Fetches data, calculates indicators using configurable parameters, and generates signal.
    """
    print(f"Processing {symbol} with SMA {sma_short_window}/{sma_long_window} and RSI {rsi_window} ({rsi_oversold}/{rsi_overbought})")
    data = get_historical_klines(symbol, interval, limit=data_limit)

    if data is None or data.empty:
        print(f"Could not fetch sufficient data for {symbol}. Skipping.")
        return "No signal"

    # Calculate indicators with dynamic column names
    sma_short_col = f'sma_{sma_short_window}'
    sma_long_col = f'sma_{sma_long_window}'
    rsi_col = f'rsi_{rsi_window}'

    data[sma_short_col] = calculate_sma(data['close'], sma_short_window)
    data[sma_long_col] = calculate_sma(data['close'], sma_long_window)
    data[rsi_col] = calculate_rsi(data['close'], rsi_window)

    data.dropna(inplace=True)

    if data.empty:
        print(f"Not enough data for {symbol} after indicator calculation. Skipping.")
        return "No signal"

    data['signal'] = generate_signals(data, sma_short_window, sma_long_window, rsi_window, rsi_oversold, rsi_overbought)

    latest_signal = data['signal'].iloc[-1] if not data.empty else "No signal"

    # Display relevant columns for the latest data
    display_cols = ['close', sma_short_col, sma_long_col, rsi_col, 'signal']
    print(f"Latest data for {symbol} (last 3 entries):\n{data[display_cols].tail(3)}")
    print(f"Latest signal for {symbol}: {latest_signal}")
    return latest_signal

def run_bot(symbols_to_trade, trade_interval_str='1h', bot_config=None):
    """
    Main bot loop that processes symbols continuously.

    :param symbols_to_trade: A list of trading symbols (e.g., ['XRPUSDT', 'ADAUSDT'])
    :param trade_interval_str: String representation of the interval (e.g., '1h', '4h', '1d')
    :param bot_config: Dictionary to override default strategy parameters.
    """
    global client # Indicate that we are using and potentially modifying the global client

    if client is None:
        print("Initializing Binance Client...")
        try:
            # If API keys were configured (e.g., via os.environ), they would be passed here:
            # import os # Make sure os is imported if using os.environ
            # api_key = os.environ.get('BINANCE_API_KEY')
            # api_secret = os.environ.get('BINANCE_API_SECRET')
            # if api_key and api_secret:
            #    client = Client(api_key, api_secret)
            # else:
            #    client = Client() # For public data only if keys are not found
            client = Client() # Using public client for now
            client.ping() # Verify connection
            print("Binance Client initialized and connection successful.")
        except Exception as e:
            print(f"Error: Failed to initialize or connect with Binance Client: {e}")
            print("Please check your internet connection. If using API keys, verify their validity and permissions.")
            print("Bot cannot continue without a working client. Exiting.")
            return # Exit if client can't be initialized

    if bot_config is None:
        bot_config = {}

    interval_map = {
        '1m': Client.KLINE_INTERVAL_1MINUTE, '5m': Client.KLINE_INTERVAL_5MINUTE,
        '15m': Client.KLINE_INTERVAL_15MINUTE, '30m': Client.KLINE_INTERVAL_30MINUTE,
        '1h': Client.KLINE_INTERVAL_1HOUR, '4h': Client.KLINE_INTERVAL_4HOUR,
        '1d': Client.KLINE_INTERVAL_1DAY,
    }
    binance_interval = interval_map.get(trade_interval_str.lower())
    if not binance_interval:
        print(f"Error: Invalid trade interval: {trade_interval_str}. Valid options: {list(interval_map.keys())}")
        return

    print(f"Starting trading bot for symbols: {symbols_to_trade} with data interval: {trade_interval_str}")
    print(f"Bot will re-evaluate signals every 5 minutes. Press Ctrl+C to stop.")

    # Apply global config or defaults from bot_config or constants
    config = {
        'data_limit': bot_config.get('data_limit', DEFAULT_DATA_LIMIT),
        'sma_short_window': bot_config.get('sma_short_window', DEFAULT_SMA_SHORT_WINDOW),
        'sma_long_window': bot_config.get('sma_long_window', DEFAULT_SMA_LONG_WINDOW),
        'rsi_window': bot_config.get('rsi_window', DEFAULT_RSI_WINDOW),
        'rsi_oversold': bot_config.get('rsi_oversold', DEFAULT_RSI_OVERSOLD),
        'rsi_overbought': bot_config.get('rsi_overbought', DEFAULT_RSI_OVERBOUGHT),
    }

    try:
        while True:
            print(f"\n--- Starting new cycle at {time.strftime('%Y-%m-%d %H:%M:%S')} ---")
            for symbol in symbols_to_trade:
                # process_symbol uses the global client implicitly via get_historical_klines
                current_signal = process_symbol(symbol, binance_interval,
                                                data_limit=config['data_limit'],
                                                sma_short_window=config['sma_short_window'],
                                                sma_long_window=config['sma_long_window'],
                                                rsi_window=config['rsi_window'],
                                                rsi_oversold=config['rsi_oversold'],
                                                rsi_overbought=config['rsi_overbought'])
                # The process_symbol function already prints the latest signal and data.
                # print(f"Action/Consideration for {symbol}: {current_signal}") # This is redundant
                print("-" * 70) # Separator between symbols

            print(f"--- Cycle complete. Waiting for 5 minutes before next run. ---")
            time.sleep(300)  # Sleep for 300 seconds (5 minutes)

    except KeyboardInterrupt:
        print("\nBot stopped by user (Ctrl+C). Exiting gracefully.")
    except Exception as e:
        print(f"An unexpected error occurred during bot operation: {e}")
        # Consider adding more specific error handling or logging here for robustness
    finally:
        print("Trading bot has shut down.")


if __name__ == '__main__':
    # Define symbols and interval for the bot
    symbols = ['XRPUSDT', 'ADAUSDT']

    # Set the trade_interval to '5m' to align with the 5-minute execution cycle of run_bot()
    trade_interval = '5m'
    print(f"Bot configured to use '{trade_interval}' k-line data for analysis.")

    # --- Configuration for Strategy Parameters ---
    # The bot will run every 5 minutes. The `trade_interval` above defines the timeframe of
    # the candles (e.g., '5m' candles).
    # You can customize the strategy parameters (SMA windows, RSI settings, etc.)
    # by passing a 'bot_config' dictionary to run_bot.
    # If bot_config is None or an empty dictionary, the global DEFAULT_ values at the
    # top of the script will be used for those parameters not in bot_config.

    # Example: Using default indicator parameters with the 5m interval
    # For a 5-minute interval, default SMAs (e.g., DEFAULT_SMA_SHORT_WINDOW=50, DEFAULT_SMA_LONG_WINDOW=200) mean looking at:
    # SMA 50 on 5m chart: 50 * 5m = 250 minutes = ~4.17 hours
    # SMA 200 on 5m chart: 200 * 5m = 1000 minutes = ~16.67 hours
    # Ensure DEFAULT_DATA_LIMIT (currently 250) is sufficient for the longest window. For SMA 200, it is.

    active_config = {} # An empty dict means use all DEFAULT_ values for indicators

    # Example: Custom configuration for a 5-minute strategy (uncomment and modify to use)
    # active_config = {
    #     'data_limit': 300,        # Fetch more data if needed for very long SMAs or initial buffer
    #     'sma_short_window': 12,   # e.g., 1-hour equivalent SMA (12 * 5m = 60m)
    #     'sma_long_window': 72,    # e.g., 6-hour equivalent SMA (72 * 5m = 360m)
    #     'rsi_window': 14,         # Standard RSI period
    #     # rsi_oversold and rsi_overbought will use defaults if not specified here
    # }

    if active_config: # Check if dictionary is not empty
        print(f"Using custom configuration for some/all strategy parameters: {active_config}")
        # Parameters not in active_config will still use DEFAULT_ values via .get() in run_bot
    else:
        print("Using default global configuration for all strategy parameters.")

    run_bot(symbols, trade_interval, bot_config=active_config)
# End of trading_bot.py
