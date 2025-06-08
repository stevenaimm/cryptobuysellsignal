# Binance Trading Signal Bot

This Python bot generates trading signals (Long/Short/Hold) for specified cryptocurrency pairs on Binance. It uses a common trading strategy based on Simple Moving Average (SMA) crossovers confirmed by the Relative Strength Index (RSI).

## Features

*   Fetches historical k-line (candlestick) data from Binance.
*   Calculates technical indicators:
    *   Simple Moving Averages (SMA)
    *   Relative Strength Index (RSI)
*   Generates trading signals based on a configurable strategy.
*   Supports multiple trading pairs.
*   Configurable parameters for indicators and strategy.
*   Includes unit tests for core logic.

## Trading Strategy

The bot employs the following strategy:

1.  **Simple Moving Average (SMA) Crossover:**
    *   Uses two SMAs: a short-period SMA and a long-period SMA (e.g., 50-period and 200-period).
    *   A potential **Long** signal is identified when the short-period SMA crosses above the long-period SMA.
    *   A potential **Short** signal is identified when the short-period SMA crosses below the long-period SMA.

2.  **Relative Strength Index (RSI) Confirmation:**
    *   The RSI is used to filter SMA crossover signals and avoid entries in potentially overbought or oversold market conditions.
    *   For a **Long** signal (SMA short > SMA long):
        *   The entry is confirmed if the RSI is *below* an overbought threshold (e.g., RSI < 70).
        *   The signal remains 'Long' as long as SMA short > SMA long and RSI does not become overbought.
    *   For a **Short** signal (SMA short < SMA long):
        *   The entry is confirmed if the RSI is *above* an oversold threshold (e.g., RSI > 30).
        *   The signal remains 'Short' as long as SMA short < SMA long and RSI does not become oversold.
    *   If RSI conditions are not met at the time of an SMA crossover, or if they cease to be met while a position is notionally held, the signal reverts to 'Hold'.

## Setup and Installation

1.  **Clone the repository (if applicable):**
    ```bash
    # git clone <repository-url>
    # cd <repository-name>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    Make sure you have `pip` installed. The required libraries are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
    This will install `python-binance` for interacting with the Binance API and `pandas` for data manipulation.

## How to Run the Bot

The main script is `trading_bot.py`. You can run it from the root directory of the project.

```bash
python trading_bot.py
```

The script will then process the default symbols (`XRPUSDT`, `ADAUSDT`) with the default 1-hour interval and strategy parameters, printing the latest signal for each.

### Configuration

You can customize the bot's behavior by modifying the parameters in the `if __name__ == '__main__':` block of `trading_bot.py`:

*   **`symbols`**: A list of trading symbols to process (e.g., `['BTCUSDT', 'ETHUSDT']`).
*   **`trade_interval`**: The k-line interval for analysis (e.g., `'15m'`, `'4h'`, `'1d'`).
*   **`custom_config`**: A dictionary to override default strategy parameters:
    *   `data_limit`: Number of k-lines to fetch (default: 250).
    *   `sma_short_window`: Period for the short SMA (default: 50).
    *   `sma_long_window`: Period for the long SMA (default: 200).
    *   `rsi_window`: Period for the RSI (default: 14).
    *   `rsi_oversold`: RSI threshold for oversold conditions (default: 30).
    *   `rsi_overbought`: RSI threshold for overbought conditions (default: 70).

**Example of running with custom configuration (modify `trading_bot.py`):**
```python
if __name__ == '__main__':
    symbols = ['BTCUSDT', 'ETHUSDT']
    trade_interval = '4h'

    custom_config = {
        'data_limit': 300,
        'sma_short_window': 20,
        'sma_long_window': 100,
        'rsi_window': 10,
        'rsi_oversold': 25,
        'rsi_overbought': 75,
    }
    # To run with default settings:
    # run_bot(symbols, trade_interval)
    # To run with custom settings:
    run_bot(symbols, trade_interval, bot_config=custom_config)
```

## Running Unit Tests

Unit tests are located in the `tests` directory. To run the tests, navigate to the root directory of the project and use the following command:

```bash
python -m unittest tests.test_trading_bot
```
Alternatively, you can run:
```bash
python -m unittest discover tests
```

## Important Considerations

*   **Signal Generation Only**: This bot **generates trading signals** based on historical data and technical analysis. It **does not execute trades** automatically. Any trading decisions should be made after careful consideration and risk assessment.
*   **API Key Security**:
    *   The current version of the bot primarily uses public Binance API endpoints to fetch k-line data, which do not require API keys.
    *   If you intend to extend this bot for actual trading or to access private account information, you will need to provide API keys.
    *   **Never hardcode API keys directly into the script.** Use environment variables or a secure configuration file. The `trading_bot.py` script includes comments indicating where API keys would be used if needed:
        ```python
        # For trading or private endpoints, you would use:
        # client = Client(os.environ.get('BINANCE_API_KEY'), os.environ.get('BINANCE_API_SECRET'))
        ```
*   **Risk Management**: Trading cryptocurrencies involves significant risk. Always use sound risk management practices. This bot is provided for educational and informational purposes only.
*   **Strategy Effectiveness**: The effectiveness of any trading strategy can vary depending on market conditions. Backtest thoroughly and use paper trading before risking real capital.
*   **Rate Limits**: Be mindful of Binance API rate limits if you modify the bot to fetch data more frequently or for many symbols.

## Disclaimer

This software is provided "as is", without warranty of any kind, express or implied. The authors or contributors are not responsible for any financial losses or other damages incurred from the use of this software.
