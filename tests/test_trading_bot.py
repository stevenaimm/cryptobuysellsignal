import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to the Python path to import trading_bot
# This allows running tests from the 'tests' directory or the root project directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Attempt to import functions and defaults from trading_bot
try:
    from trading_bot import calculate_sma, calculate_rsi, generate_signals, \
        DEFAULT_SMA_SHORT_WINDOW, DEFAULT_SMA_LONG_WINDOW, DEFAULT_RSI_WINDOW, \
        DEFAULT_RSI_OVERSOLD, DEFAULT_RSI_OVERBOUGHT
except ImportError as e:
    print(f"Error: Could not import from trading_bot: {e}. Ensure trading_bot.py is in the parent directory and readable.")
    # Define defaults here if import fails, for tests to be parsable by unittest discovery,
    # though they will likely fail if the functions themselves can't be imported.
    DEFAULT_SMA_SHORT_WINDOW = 50
    DEFAULT_SMA_LONG_WINDOW = 200
    DEFAULT_RSI_WINDOW = 14
    DEFAULT_RSI_OVERSOLD = 30
    DEFAULT_RSI_OVERBOUGHT = 70
    # Define dummy functions if import fails
    def calculate_sma(data, window): return pd.Series(dtype='float64')
    def calculate_rsi(data, window): return pd.Series(dtype='float64')
    def generate_signals(data, sma_short_w, sma_long_w, rsi_w, rsi_os, rsi_ob): return pd.Series(dtype='str')


class TestTradingBotIndicators(unittest.TestCase):

    def test_calculate_sma(self):
        data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name="close") # Added name to series

        # Test with window 3
        expected_sma3 = pd.Series([np.nan, np.nan, 2.0, 3.0, 4.0, 5.0], index=data.index, name="close")
        sma3 = calculate_sma(data, 3)
        pd.testing.assert_series_equal(sma3, expected_sma3, check_dtype=False)

        # Test with window larger than data length
        # The function returns an empty Series (no index, no name) in this case.
        expected_sma7 = pd.Series(dtype='float64')
        sma7 = calculate_sma(data, 7)
        pd.testing.assert_series_equal(sma7, expected_sma7, check_dtype=False)

        # Test with empty series
        empty_data = pd.Series(dtype='float64', name="close")
        expected_empty_sma = pd.Series(dtype='float64') # Empty series also expected here
        empty_sma = calculate_sma(empty_data, 3)
        pd.testing.assert_series_equal(empty_sma, expected_empty_sma, check_dtype=False)

        # Test with None input
        none_sma = calculate_sma(None, 3)
        pd.testing.assert_series_equal(none_sma, expected_empty_sma, check_dtype=False)


    def test_calculate_rsi(self):
        data = pd.Series(
            [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64],
            name="close"
        )
        rsi_window = DEFAULT_RSI_WINDOW

        rsi_values = calculate_rsi(data, rsi_window)

        self.assertEqual(rsi_values.isnull().sum(), rsi_window, f"First {rsi_window} RSI values should be NaN")

        valid_rsi_values = rsi_values.dropna()
        if not valid_rsi_values.empty:
            self.assertTrue(all(val >= 0 and val <= 100 for val in valid_rsi_values), "RSI values should be between 0 and 100")

        short_data = pd.Series([1.0, 2.0, 3.0], name="close")
        expected_short_rsi = pd.Series(dtype='float64')
        short_rsi = calculate_rsi(short_data, rsi_window)
        pd.testing.assert_series_equal(short_rsi, expected_short_rsi, check_dtype=False)

        empty_data = pd.Series(dtype='float64', name="close")
        expected_empty_rsi = pd.Series(dtype='float64')
        empty_rsi = calculate_rsi(empty_data, rsi_window)
        pd.testing.assert_series_equal(empty_rsi, expected_empty_rsi, check_dtype=False)

        none_rsi = calculate_rsi(None, rsi_window)
        pd.testing.assert_series_equal(none_rsi, expected_empty_rsi, check_dtype=False)


class TestTradingBotSignals(unittest.TestCase):

    def setUp(self):
        self.dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06'])

        self.sma_short_w = DEFAULT_SMA_SHORT_WINDOW
        self.sma_long_w = DEFAULT_SMA_LONG_WINDOW
        self.rsi_w = DEFAULT_RSI_WINDOW
        self.rsi_os = DEFAULT_RSI_OVERSOLD
        self.rsi_ob = DEFAULT_RSI_OVERBOUGHT

        self.sma_short_col = f'sma_{self.sma_short_w}'
        self.sma_long_col = f'sma_{self.sma_long_w}'
        self.rsi_col = f'rsi_{self.rsi_w}'

    def _create_test_df(self, sma_short_vals, sma_long_vals, rsi_vals):
        df = pd.DataFrame(index=self.dates[:len(sma_short_vals)])
        df[self.sma_short_col] = sma_short_vals
        df[self.sma_long_col]  = sma_long_vals
        df[self.rsi_col]       = rsi_vals
        return df

    def test_generate_signals_long(self):
        df = self._create_test_df(
            sma_short_vals = [49, 49.5, 51, 52, 53, 54],
            sma_long_vals  = [50, 50,   50, 50, 50, 50],
            rsi_vals       = [60, 60,   60, 60, 60, 60]
        )
        signals = generate_signals(df.copy(), self.sma_short_w, self.sma_long_w, self.rsi_w, self.rsi_os, self.rsi_ob)
        expected_signals = pd.Series(['Hold', 'Hold', 'Long', 'Long', 'Long', 'Long'], index=df.index, name='signal')
        pd.testing.assert_series_equal(signals, expected_signals, check_dtype=False)

    def test_generate_signals_short(self):
        df = self._create_test_df(
            sma_short_vals = [51, 50.5, 49, 48, 47, 46],
            sma_long_vals  = [50, 50,   50, 50, 50, 50],
            rsi_vals       = [40, 40,   40, 40, 40, 40]
        )
        signals = generate_signals(df.copy(), self.sma_short_w, self.sma_long_w, self.rsi_w, self.rsi_os, self.rsi_ob)
        expected_signals = pd.Series(['Hold', 'Hold', 'Short', 'Short', 'Short', 'Short'], index=df.index, name='signal')
        pd.testing.assert_series_equal(signals, expected_signals, check_dtype=False)

    def test_generate_signals_rsi_blocks_long(self):
        df = self._create_test_df(
            sma_short_vals = [49, 49.5, 51, 52, 53, 54],
            sma_long_vals  = [50, 50,   50, 50, 50, 50],
            rsi_vals       = [self.rsi_ob, self.rsi_ob, self.rsi_ob, self.rsi_ob, self.rsi_ob, self.rsi_ob]
        )
        signals = generate_signals(df.copy(), self.sma_short_w, self.sma_long_w, self.rsi_w, self.rsi_os, self.rsi_ob)
        expected_signals = pd.Series(['Hold', 'Hold', 'Hold', 'Hold', 'Hold', 'Hold'], index=df.index, name='signal')
        pd.testing.assert_series_equal(signals, expected_signals, check_dtype=False)

    def test_generate_signals_rsi_blocks_short(self):
        df = self._create_test_df(
            sma_short_vals = [51, 50.5, 49, 48, 47, 46],
            sma_long_vals  = [50, 50,   50, 50, 50, 50],
            rsi_vals       = [self.rsi_os, self.rsi_os, self.rsi_os, self.rsi_os, self.rsi_os, self.rsi_os]
        )
        signals = generate_signals(df.copy(), self.sma_short_w, self.sma_long_w, self.rsi_w, self.rsi_os, self.rsi_ob)
        expected_signals = pd.Series(['Hold', 'Hold', 'Hold', 'Hold', 'Hold', 'Hold'], index=df.index, name='signal')
        pd.testing.assert_series_equal(signals, expected_signals, check_dtype=False)

    def test_generate_signals_no_crossover(self):
        df = self._create_test_df(
            sma_short_vals = [51, 52, 53, 54, 55, 56],
            sma_long_vals  = [50, 50, 50, 50, 50, 50],
            rsi_vals       = [50, 50, 50, 50, 50, 50]
        )
        signals = generate_signals(df.copy(), self.sma_short_w, self.sma_long_w, self.rsi_w, self.rsi_os, self.rsi_ob)
        expected_signals = pd.Series(['Hold', 'Hold', 'Hold', 'Hold', 'Hold', 'Hold'], index=df.index, name='signal')
        pd.testing.assert_series_equal(signals, expected_signals, check_dtype=False)

    def test_generate_signals_missing_columns(self):
        df = pd.DataFrame(index=self.dates[:3])
        signals = generate_signals(df.copy(), self.sma_short_w, self.sma_long_w, self.rsi_w, self.rsi_os, self.rsi_ob)
        expected_signals = pd.Series(['Hold', 'Hold', 'Hold'], index=df.index, name='signal') # Expect 'Hold' due to missing cols
        pd.testing.assert_series_equal(signals, expected_signals, check_dtype=False)

if __name__ == '__main__':
    unittest.main(verbosity=2)
