import os
import sys
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time  # Import time module
from datetime import time as dt_time  
from ib_insync import *
import logging
import pytz
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
import warnings
from joblib import Parallel, delayed
from itertools import combinations

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# Strategy parameters
TIMEFRAME_OPTIONS = ['1 min', '5 mins', '15 mins', '30 mins', '1 hour', '1 day']
TIMEFRAME_MAPPING = {
    '1 min': ('1 D', '1 min'),      # Adjusted durations to match IB API limits
    '5 mins': ('2 D', '5 mins'),
    '15 mins': ('5 D', '15 mins'),
    '30 mins': ('10 D', '30 mins'),
    '1 hour': ('30 D', '1 hour'),
    '1 day': ('2 Y', '1 day')
}

# Strategy parameters
formation_period_bars = 252  # Number of bars in the formation period
trading_period_bars = 84     # Number of bars in the trading period
max_holding_period_bars = 10000  # Max holding period in bars
num_pairs = 500             # Number of pairs to select
z_entry_threshold = 2      # Z-score entry threshold for entry
z_exit_threshold = 0.5      # Z-score exit threshold for exit
stop_loss_threshold = 0.5   # Stop loss threshold as a decimal (e.g., 0.5 for 50%)
risk_per_trade_pct = 0.01    # Risk per trade as percentage (1%)
fee_per_share = 0.005        # Fee per share ($0.005 per share)

# List of symbols
symbols = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'JNJ', 'V', 'JPM',
    'WMT', 'PG', 'MA', 'UNH', 'DIS', 'NVDA', 'HD', 'PYPL', 'BAC', 'VZ',
    'ADBE', 'CMCSA', 'NFLX', 'XOM', 'INTC', 'T', 'CSCO', 'PFE', 'KO',
    'MRK', 'ABBV', 'PEP', 'ABT', 'CRM', 'ACN', 'MDT', 'COST', 'WFC', 'TMO',
    'DHR', 'AMGN', 'QCOM', 'TXN', 'NEE', 'ORCL', 'UPS', 'BMY', 'MS', 'LIN',
    'EURUSD','USDJPY','XAUUSD','GBPUSD'
]

# Timezone
timezone = pytz.timezone('US/Eastern')

# Initialize global variables
positions = {}
contracts = {}
initial_balance = 100000  # Placeholder, will be updated from IBKR
selected_pairs = []       # Store selected pairs
trading_end_time = None   # End time of the trading period

# Market open and close times
market_open_time = dt_time(9, 30)   # Market opens at 9:30 AM
market_close_time = dt_time(16, 0)  # Market closes at 4:00 PM

# Trading time constraints
market_open_timer = '00:10'      # Time after market open to start trading (HH:MM format)
market_close_timer = '00:10'     # Time before market close to stop trading (HH:MM format)

async def async_input(prompt):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, input, prompt)

async def main():
    # Initialize IB
    ib = IB()
    try:
        await ib.connectAsync('127.0.0.1', 7497, clientId=1)
        logger.info("Connected to Interactive Brokers.")
    except Exception as e:
        logger.error(f"Failed to connect to Interactive Brokers: {e}")
        sys.exit(1)

    # Synchronize the client
    ib.reqMarketDataType(1)  # Request live data
    await asyncio.sleep(1)  # Give some time for synchronization
    logger.info("Synchronization complete")

    # Retrieve account balance
    account_summary = await ib.accountSummaryAsync()
    net_liquidation = float([v.value for v in account_summary if v.tag == 'NetLiquidation'][0])
    global initial_balance
    initial_balance = net_liquidation
    logger.info(f"Account balance retrieved: ${initial_balance:.2f}")

    # Prompt user for timeframe
    print("Select the timeframe:")
    for idx, tf in enumerate(TIMEFRAME_OPTIONS):
        print(f"{idx + 1}: {tf}")
    timeframe_index = await async_input("Enter the number corresponding to the desired timeframe (e.g., 1): ")
    timeframe = TIMEFRAME_OPTIONS[int(timeframe_index) - 1]

    # Mapping for duration and bar size
    durationStr_default, barSizeSetting = TIMEFRAME_MAPPING[timeframe]
    bar_duration_seconds = get_timeframe_duration_in_seconds(timeframe)

    # Create contracts
    for symbol in symbols:
        contracts[symbol] = Stock(symbol, 'SMART', 'USD')
    logger.info("Qualifying contracts...")
    await ib.qualifyContractsAsync(*contracts.values())
    logger.info("Contracts qualified.")

    # Session folder for logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_folder = f'Sessions/TradingSession_{timestamp}'
    os.makedirs(session_folder, exist_ok=True)

    # Initialize a CSV file for signals
    csv_file = os.path.join(session_folder, f"trade_history_{timestamp}.csv")
    with open(csv_file, 'w') as f:
        f.write('Entry Time,Exit Time,Position,Stock1,Stock2,Quantity1,Quantity2,Entry Price1,Entry Price2,Exit Price1,Exit Price2,Profit1,Profit2,Fees1,Fees2,Net Profit,Exit Reason\n')

    # Initialize formation and trading periods
    global trading_end_time
    formation_start_time = None
    formation_end_time = None
    trading_start_time = None
    trading_end_time = None

    # Run the trading loop
    try:
        while True:
            if not is_market_open():
                logger.info("Market is closed. Waiting for market to open...")
                await asyncio.sleep(60)
                continue

            current_time = datetime.now(timezone)

            # Check if we need to perform pair analysis (either at the start or after trading period ends)
            if trading_end_time is None or current_time >= trading_end_time:
                # Perform pair analysis
                formation_end_time = current_time - timedelta(seconds=1 * bar_duration_seconds)
                formation_start_time = formation_end_time - timedelta(seconds=formation_period_bars * bar_duration_seconds)
                # Ensure formation_start_time is within market hours
                if formation_start_time.time() < market_open_time:
                    formation_start_time = formation_start_time.replace(hour=market_open_time.hour, minute=market_open_time.minute, second=0)
                logger.info(f"Performing pair analysis for period {formation_start_time} to {formation_end_time}")

                # Fetch historical data for formation period
                historical_data = await fetch_historical_data_period(ib, formation_start_time, formation_end_time, barSizeSetting)
                if not historical_data:
                    logger.error("No historical data fetched for any symbol. Retrying after waiting.")
                    await asyncio.sleep(60)
                    continue

                # Analyze pairs using cointegration-based selection
                selected_pairs = analyze_pairs(historical_data)
                logger.info(f"Selected {len(selected_pairs)} pairs for trading.")

                # Set up trading period
                trading_start_time = current_time
                trading_end_time = trading_start_time + timedelta(seconds=trading_period_bars * bar_duration_seconds)
                logger.info(f"Trading period until {trading_end_time}")

                # Close any existing positions before starting new trading period
                await close_all_positions(ib, csv_file)

                # Clear positions
                positions.clear()

            # During trading period, manage positions
            if current_time < trading_end_time:
                await run_trading_logic(ib, timeframe, barSizeSetting, csv_file, selected_pairs)
            else:
                logger.info("Trading period ended. Proceeding to next formation period.")
                # Close all positions at the end of trading period
                await close_all_positions(ib, csv_file)
                selected_pairs = []

            # Wait for the next bar
            logger.info(f"Waiting for {bar_duration_seconds} seconds until next bar.")
            await asyncio.sleep(bar_duration_seconds)
    except KeyboardInterrupt:
        logger.info("\nTrading loop interrupted by user.")
        close_positions = await async_input("Do you want to close all open positions? (y/n): ")
        if close_positions.lower() == 'y':
            await close_all_positions(ib, csv_file)
            logger.info("All positions closed.")
        else:
            logger.info("Positions left open.")
    finally:
        ib.disconnect()
        logger.info("Disconnected from Interactive Brokers.")

def is_market_open():
    now = datetime.now(timezone).time()
    return market_open_time <= now <= market_close_time

def get_timeframe_duration_in_seconds(timeframe):
    mapping = {
        '1 min': 60,
        '5 mins': 300,
        '15 mins': 900,
        '30 mins': 1800,
        '1 hour': 3600,
        '1 day': 86400
    }
    return mapping.get(timeframe, 60)

async def fetch_historical_data_period(ib, start_time, end_time, barSizeSetting):
    logger.info("Fetching historical data for formation period...")
    tasks = []
    duration = end_time - start_time
    # Adjust durationStr to match IB API requirements
    durationStr = f"{int(duration.total_seconds())} S"
    max_duration = '30 D' if barSizeSetting != '1 day' else '2 Y'  # Adjust as per IB API limitations
    if duration > timedelta(days=365):  # Limit to max allowed by IB API
        durationStr = max_duration
    for symbol, contract in contracts.items():
        tasks.append(fetch_historical_data(ib, contract, '', durationStr, barSizeSetting, start_time, end_time))
    results = await asyncio.gather(*tasks)
    historical_data = {symbol: data for symbol, data in results if data is not None}
    logger.info("Historical data fetching completed.")
    return historical_data

async def fetch_historical_data(ib, contract, endDateTime, durationStr, barSizeSetting, start_time, end_time):
    symbol = contract.symbol
    try:
        logger.info(f"Fetching data for {symbol}")
        bars = await ib.reqHistoricalDataAsync(
            contract,
            endDateTime=endDateTime,
            durationStr=durationStr,
            barSizeSetting=barSizeSetting,
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1,
            keepUpToDate=False,
            timeout=60
        )
        if bars:
            df = util.df(bars)
            df.set_index('date', inplace=True)
            df.index = pd.to_datetime(df.index, utc=True).tz_convert('US/Eastern')
            df = df.loc[start_time:end_time]
            df = df.drop_duplicates()
            num_bars = len(df)
            logger.info(f"Data fetched for {symbol} from {start_time} to {end_time} ({num_bars} bars)")
            if num_bars == 0:
                return symbol, None
            return symbol, df
        else:
            logger.warning(f"No data returned for {symbol}")
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
    return symbol, None

def analyze_pairs(historical_data):
    logger.info("Preparing price series for pair analysis.")
    # Prepare price series
    prices = pd.DataFrame({symbol: data['close'] for symbol, data in historical_data.items()})
    prices = prices.dropna(axis=1, how='any')

    valid_tickers = prices.columns
    logger.info(f"Number of valid tickers: {len(valid_tickers)}")
    ticker_pairs = list(combinations(valid_tickers, 2))
    logger.info(f"Analyzing {len(ticker_pairs)} pairs.")

    pairs_info = []
    required_data_points = int(formation_period_bars * 0.8)  # 80% of formation period

    def analyze_pair(pair):
        stock1, stock2 = pair
        series1 = prices[stock1].dropna()
        series2 = prices[stock2].dropna()

        # Find the intersection of dates
        common_dates = series1.index.intersection(series2.index)
        series1_common = series1.loc[common_dates]
        series2_common = series2.loc[common_dates]

        # Ensure sufficient overlapping data
        if len(series1_common) < required_data_points:
            return None

        # Check if either series is constant
        if series1_common.nunique() < 2 or series2_common.nunique() < 2:
            return None

        try:
            coint_t, p_value, _ = coint(series1_common, series2_common)
        except Exception as e:
            logger.warning(f"Cointegration test failed for {stock1} and {stock2}: {e}")
            return None

        if p_value < 0.05:
            try:
                X = sm.add_constant(series2_common)
                model = sm.OLS(series1_common, X)
                result = model.fit()
                if len(result.params) >=2:
                    beta = result.params.iloc[1]
                else:
                    return None
            except Exception:
                return None

            spread = series1_common - beta * series2_common
            spread_mean = spread.mean()
            spread_std = spread.std()

            if spread_std == 0 or np.isnan(spread_std):
                return None

            # Calculate half-life of mean reversion
            spread_lag = spread.shift(1)
            spread_diff = spread - spread_lag
            spread_lag2 = spread_lag.dropna()
            spread_diff2 = spread_diff.dropna()
            if len(spread_lag2) > 0:
                beta_hr = np.polyfit(spread_lag2, spread_diff2, 1)[0]
                if beta_hr != 0:
                    half_life = -np.log(2) / beta_hr
                else:
                    half_life = np.nan
            else:
                half_life = np.nan

            # Ensure half-life is positive and reasonable
            if half_life > 0 and half_life < (formation_period_bars / 2):
                # Calculate in-sample Sharpe Ratio
                z_score = (spread - spread_mean) / spread_std
                in_sample_return = -np.diff(z_score) * z_score[:-1]
                if np.std(in_sample_return) != 0:
                    in_sample_sr = np.mean(in_sample_return) / np.std(in_sample_return)
                else:
                    in_sample_sr = 0

                return {
                    'stock1': stock1,
                    'stock2': stock2,
                    'beta': beta,
                    'half_life': half_life,
                    'p_value': p_value,
                    'in_sample_sr': in_sample_sr,
                    'spread_mean': spread_mean,
                    'spread_std': spread_std
                }
        return None

    results = Parallel(n_jobs=-1)(delayed(analyze_pair)(pair) for pair in ticker_pairs)
    pairs_info = [res for res in results if res is not None]

    # Select top N pairs based on in-sample SR
    pairs_info_sorted = sorted(pairs_info, key=lambda x: x['in_sample_sr'], reverse=True)
    selected_pairs = pairs_info_sorted[:num_pairs]
    logger.info("Pair analysis completed.")
    return selected_pairs

async def run_trading_logic(ib, timeframe, barSizeSetting, csv_file, selected_pairs):
    global positions
    # Fetch the latest bar data for selected symbols
    symbols_to_fetch = set()
    for pair in selected_pairs:
        symbols_to_fetch.update([pair['stock1'], pair['stock2']])
    tasks = []
    for symbol in symbols_to_fetch:
        contract = contracts[symbol]
        tasks.append(fetch_latest_bar(ib, contract, barSizeSetting))
    results = await asyncio.gather(*tasks)
    latest_prices = {symbol: data['close'].iloc[-1] for symbol, data in results if data is not None}

    if not latest_prices:
        logger.warning("No latest price data fetched.")
        return

    # Update existing positions
    logger.info("Updating existing positions...")
    await update_positions(ib, latest_prices, csv_file)

    # Generate new signals
    logger.info("Generating new trading signals...")
    await generate_signals(ib, selected_pairs, latest_prices, csv_file)

async def fetch_latest_bar(ib, contract, barSizeSetting):
    symbol = contract.symbol
    try:
        logger.info(f"Fetching latest bar for {symbol}")
        bars = await ib.reqHistoricalDataAsync(
            contract,
            endDateTime='',
            durationStr='1 D',
            barSizeSetting=barSizeSetting,
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1,
            keepUpToDate=False
        )
        if bars:
            df = util.df(bars)
            df.set_index('date', inplace=True)
            df.index = pd.to_datetime(df.index, utc=True).tz_convert('US/Eastern')
            df = df.iloc[-1:]
            logger.info(f"Latest bar fetched for {symbol}")
            return symbol, df
        else:
            logger.warning(f"No data returned for {symbol}")
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
    return symbol, None

async def update_positions(ib, latest_prices, csv_file):
    global positions
    positions_to_remove = []
    for pair_key in list(positions.keys()):
        position = positions[pair_key]
        stock1 = position['stock1']
        stock2 = position['stock2']
        beta = position['beta']

        # Check if trade is allowed at current time
        current_time = datetime.now(timezone)
        if not is_trade_allowed(current_time):
            continue  # Skip updating positions outside trading hours

        if stock1 in latest_prices and stock2 in latest_prices:
            price1 = latest_prices[stock1]
            price2 = latest_prices[stock2]

            spread = price1 - beta * price2
            spread_mean = position['spread_mean']
            spread_std = position['spread_std']
            z_score = (spread - spread_mean) / spread_std

            holding_period = position['holding_period'] + 1
            position['holding_period'] = holding_period

            entry_spread = position['entry_spread']
            spread_pct_change = abs((spread - entry_spread) / entry_spread)

            exit_signal = False
            exit_reason = ''

            # Exit conditions
            if (position['type'] == 'long' and z_score >= -z_exit_threshold) or \
               (position['type'] == 'short' and z_score <= z_exit_threshold):
                exit_signal = True
                exit_reason = 'Exit Signal'

            elif holding_period >= max_holding_period_bars:
                exit_signal = True
                exit_reason = 'Max Holding Period Reached'

            elif spread_pct_change > stop_loss_threshold:
                exit_signal = True
                exit_reason = 'Stop Loss Hit'

            if exit_signal:
                logger.info(f"Closing position for {stock1} and {stock2} due to {exit_reason}")
                # Close position
                await close_position(ib, position, price1, price2, exit_reason, csv_file)
                positions_to_remove.append(pair_key)
            else:
                logger.info(f"Position for {stock1} and {stock2} remains open.")
        else:
            logger.warning(f"Price data not available for {stock1} or {stock2}")

    for pair_key in positions_to_remove:
        del positions[pair_key]

async def generate_signals(ib, selected_pairs, latest_prices, csv_file):
    global positions
    for pair in selected_pairs:
        stock1 = pair['stock1']
        stock2 = pair['stock2']
        beta = pair['beta']
        spread_mean = pair['spread_mean']
        spread_std = pair['spread_std']

        pair_key = f"{stock1}_{stock2}"
        if pair_key in positions:
            continue  # Position already open

        # Check if trade is allowed at current time
        current_time = datetime.now(timezone)
        if not is_trade_allowed(current_time):
            continue  # Skip generating signals outside trading hours

        if stock1 in latest_prices and stock2 in latest_prices:
            price1 = latest_prices[stock1]
            price2 = latest_prices[stock2]

            spread = price1 - beta * price2
            z_score = (spread - spread_mean) / spread_std

            if z_score > z_entry_threshold:
                # Short spread
                per_trade_notional = initial_balance * risk_per_trade_pct
                quantity1 = per_trade_notional / price1
                quantity2 = beta * quantity1

                if quantity1 < 1 or abs(quantity2) < 1:
                    continue

                quantity1 = int(quantity1)
                quantity2 = int(quantity2)

                position = {
                    'type': 'short',
                    'entry_time': datetime.now(timezone),
                    'stock1': stock1,
                    'stock2': stock2,
                    'quantity1': quantity1,
                    'quantity2': quantity2,
                    'entry_price1': price1,
                    'entry_price2': price2,
                    'beta': beta,
                    'spread_mean': spread_mean,
                    'spread_std': spread_std,
                    'entry_spread': spread,
                    'holding_period': 0,
                    'csv_file': csv_file,
                    'entry_z': z_score,
                    'entry_fees1': quantity1 * fee_per_share,
                    'entry_fees2': abs(quantity2) * fee_per_share
                }
                logger.info(f"Opening short position for {stock1} and {stock2}")
                await open_position(ib, position, csv_file)
                positions[pair_key] = position
            elif z_score < -z_entry_threshold:
                # Long spread
                per_trade_notional = initial_balance * risk_per_trade_pct
                quantity1 = per_trade_notional / price1
                quantity2 = beta * quantity1

                if quantity1 < 1 or abs(quantity2) < 1:
                    continue

                quantity1 = int(quantity1)
                quantity2 = int(quantity2)

                position = {
                    'type': 'long',
                    'entry_time': datetime.now(timezone),
                    'stock1': stock1,
                    'stock2': stock2,
                    'quantity1': quantity1,
                    'quantity2': quantity2,
                    'entry_price1': price1,
                    'entry_price2': price2,
                    'beta': beta,
                    'spread_mean': spread_mean,
                    'spread_std': spread_std,
                    'entry_spread': spread,
                    'holding_period': 0,
                    'csv_file': csv_file,
                    'entry_z': z_score,
                    'entry_fees1': quantity1 * fee_per_share,
                    'entry_fees2': abs(quantity2) * fee_per_share
                }
                logger.info(f"Opening long position for {stock1} and {stock2}")
                await open_position(ib, position, csv_file)
                positions[pair_key] = position

def is_trade_allowed(current_datetime):
    # Parse Market Open and Close Timers
    market_open_delay = timedelta(hours=int(market_open_timer.split(':')[0]), minutes=int(market_open_timer.split(':')[1]))
    market_close_advance = timedelta(hours=int(market_close_timer.split(':')[0]), minutes=int(market_close_timer.split(':')[1]))

    market_open_today = datetime.combine(current_datetime.date(), market_open_time)
    market_open_with_delay = (market_open_today + market_open_delay).time()
    market_close_today = datetime.combine(current_datetime.date(), market_close_time)
    market_close_with_advance = (market_close_today - market_close_advance).time()

    if market_open_with_delay <= current_datetime.time() <= market_close_with_advance:
        return True
    else:
        return False

async def open_position(ib, position, csv_file):
    stock1 = position['stock1']
    stock2 = position['stock2']
    quantity1 = position['quantity1']
    quantity2 = position['quantity2']
    position_type = position['type']

    contract1 = contracts[stock1]
    contract2 = contracts[stock2]

    if position_type == 'short':
        order1 = MarketOrder('SELL', quantity1)
        order2 = MarketOrder('BUY', abs(quantity2))
    else:
        order1 = MarketOrder('BUY', quantity1)
        order2 = MarketOrder('SELL', abs(quantity2))

    trade1 = ib.placeOrder(contract1, order1)
    trade2 = ib.placeOrder(contract2, order2)

    await asyncio.sleep(1)  # Give time for orders to be processed

    filled1 = await wait_for_fill(ib, trade1)
    filled2 = await wait_for_fill(ib, trade2)

    if filled1 and filled2:
        fills1 = trade1.fills
        fills2 = trade2.fills

        actual_qty1 = sum(fill.execution.shares for fill in fills1)
        actual_price1 = sum(fill.execution.shares * fill.execution.price for fill in fills1) / actual_qty1
        actual_time1 = fills1[-1].execution.time.astimezone(timezone)

        actual_qty2 = sum(fill.execution.shares for fill in fills2)
        actual_price2 = sum(fill.execution.shares * fill.execution.price for fill in fills2) / actual_qty2
        actual_time2 = fills2[-1].execution.time.astimezone(timezone)

        # Get fees from fills if available; else use estimated fees
        fees1 = sum(fill.commissionReport.commission for fill in fills1)
        fees2 = sum(fill.commissionReport.commission for fill in fills2)

        position.update({
            'actual_entry_time1': actual_time1,
            'actual_entry_time2': actual_time2,
            'actual_quantity1': actual_qty1,
            'actual_quantity2': actual_qty2,
            'actual_entry_price1': actual_price1,
            'actual_entry_price2': actual_price2,
            'entry_fees1': fees1 if fees1 else position['entry_fees1'],
            'entry_fees2': fees2 if fees2 else position['entry_fees2']
        })

        # Log to CSV
        log_entry_signal(csv_file, position)
        logger.info(f"Opened {position_type} position: {stock1} and {stock2}")
    else:
        logger.error(f"Failed to open position for {stock1} and {stock2}")

async def close_position(ib, position, price1, price2, exit_reason, csv_file):
    stock1 = position['stock1']
    stock2 = position['stock2']
    quantity1 = position['quantity1']
    quantity2 = position['quantity2']
    position_type = position['type']

    contract1 = contracts[stock1]
    contract2 = contracts[stock2]

    if position_type == 'short':
        order1 = MarketOrder('BUY', quantity1)
        order2 = MarketOrder('SELL', abs(quantity2))
    else:
        order1 = MarketOrder('SELL', quantity1)
        order2 = MarketOrder('BUY', abs(quantity2))

    trade1 = ib.placeOrder(contract1, order1)
    trade2 = ib.placeOrder(contract2, order2)

    await asyncio.sleep(1)  # Give time for orders to be processed

    filled1 = await wait_for_fill(ib, trade1)
    filled2 = await wait_for_fill(ib, trade2)

    if filled1 and filled2:
        fills1 = trade1.fills
        fills2 = trade2.fills

        actual_exit_qty1 = sum(fill.execution.shares for fill in fills1)
        actual_exit_price1 = sum(fill.execution.shares * fill.execution.price for fill in fills1) / actual_exit_qty1
        actual_exit_time1 = fills1[-1].execution.time.astimezone(timezone)

        actual_exit_qty2 = sum(fill.execution.shares for fill in fills2)
        actual_exit_price2 = sum(fill.execution.shares * fill.execution.price for fill in fills2) / actual_exit_qty2
        actual_exit_time2 = fills2[-1].execution.time.astimezone(timezone)

        # Get fees from fills
        fees1 = sum(fill.commissionReport.commission for fill in fills1)
        fees2 = sum(fill.commissionReport.commission for fill in fills2)

        # Update position with exit details
        position.update({
            'actual_exit_time1': actual_exit_time1,
            'actual_exit_time2': actual_exit_time2,
            'actual_exit_price1': actual_exit_price1,
            'actual_exit_price2': actual_exit_price2,
            'exit_fees1': fees1,
            'exit_fees2': fees2,
            'exit_reason': exit_reason
        })

        # Log trade to CSV
        update_csv_entry(csv_file, position)
        logger.info(f"Closed {position_type} position: {stock1} and {stock2} due to {exit_reason}")
    else:
        logger.error(f"Failed to close position for {stock1} and {stock2}")

def log_entry_signal(csv_file, position):
    data = {
        'Entry Time': position['entry_time'],
        'Exit Time': '',
        'Position': position['type'].capitalize(),
        'Stock1': position['stock1'],
        'Stock2': position['stock2'],
        'Quantity1': position.get('actual_quantity1', position['quantity1']),
        'Quantity2': position.get('actual_quantity2', position['quantity2']),
        'Entry Price1': position.get('actual_entry_price1', position['entry_price1']),
        'Entry Price2': position.get('actual_entry_price2', position['entry_price2']),
        'Exit Price1': '',
        'Exit Price2': '',
        'Profit1': '',
        'Profit2': '',
        'Fees1': position.get('entry_fees1', 0),
        'Fees2': position.get('entry_fees2', 0),
        'Net Profit': '',
        'Exit Reason': ''
    }
    with open(csv_file, 'a') as f:
        f.write(','.join(map(str, data.values())) + '\n')

def update_csv_entry(csv_file, position):
    # Prepare data to write
    data = {
        'Entry Time': position['entry_time'],
        'Exit Time': datetime.now(timezone),
        'Position': position['type'].capitalize(),
        'Stock1': position['stock1'],
        'Stock2': position['stock2'],
        'Quantity1': position.get('actual_quantity1', position['quantity1']),
        'Quantity2': position.get('actual_quantity2', position['quantity2']),
        'Entry Price1': position.get('actual_entry_price1', position['entry_price1']),
        'Entry Price2': position.get('actual_entry_price2', position['entry_price2']),
        'Exit Price1': position.get('actual_exit_price1', ''),
        'Exit Price2': position.get('actual_exit_price2', ''),
        'Profit1': '',
        'Profit2': '',
        'Fees1': '',
        'Fees2': '',
        'Net Profit': '',
        'Exit Reason': position.get('exit_reason', '')
    }

    # Calculate profits and fees
    try:
        quantity1 = float(data['Quantity1'])
        quantity2 = float(data['Quantity2'])
        entry_price1 = float(data['Entry Price1'])
        entry_price2 = float(data['Entry Price2'])
        exit_price1 = float(data['Exit Price1']) if data['Exit Price1'] != '' else 0
        exit_price2 = float(data['Exit Price2']) if data['Exit Price2'] != '' else 0

        # Calculate profits
        if position['type'] == 'long':
            # Long spread: Buy stock1, Sell stock2
            profit1 = (exit_price1 - entry_price1) * quantity1
            profit2 = (entry_price2 - exit_price2) * (-quantity2)
        else:
            # Short spread: Sell stock1, Buy stock2
            profit1 = (entry_price1 - exit_price1) * quantity1
            profit2 = (exit_price2 - entry_price2) * quantity2

        # Fees
        total_fees1 = position.get('entry_fees1', 0) + position.get('exit_fees1', 0)
        total_fees2 = position.get('entry_fees2', 0) + position.get('exit_fees2', 0)

        net_profit = profit1 + profit2 - total_fees1 - total_fees2

        # Update data
        data['Profit1'] = round(profit1, 2)
        data['Profit2'] = round(profit2, 2)
        data['Fees1'] = round(total_fees1, 2)
        data['Fees2'] = round(total_fees2, 2)
        data['Net Profit'] = round(net_profit, 2)
    except Exception as e:
        logger.error(f"Error calculating profits: {e}")

    # Write to CSV
    with open(csv_file, 'a') as f:
        f.write(','.join(map(str, data.values())) + '\n')

async def wait_for_fill(ib, trade, timeout=10):
    start_time = time.time()
    while True:
        if trade.isDone():
            return True
        if time.time() - start_time > timeout:
            return False
        await asyncio.sleep(1)

async def close_all_positions(ib, csv_file):
    global positions
    if positions:
        logger.info("Closing all open positions...")
        latest_prices = {}
        tasks = []
        for position in positions.values():
            stock1 = position['stock1']
            stock2 = position['stock2']
            if stock1 not in latest_prices:
                tasks.append(fetch_latest_price(ib, stock1))
            if stock2 not in latest_prices:
                tasks.append(fetch_latest_price(ib, stock2))
        results = await asyncio.gather(*tasks)
        latest_prices.update({symbol: price for symbol, price in results if price is not None})

        positions_to_remove = []
        for pair_key in list(positions.keys()):
            position = positions[pair_key]
            stock1 = position['stock1']
            stock2 = position['stock2']
            if stock1 in latest_prices and stock2 in latest_prices:
                price1 = latest_prices[stock1]
                price2 = latest_prices[stock2]
                await close_position(ib, position, price1, price2, "Strategy Exit", csv_file)
                positions_to_remove.append(pair_key)
            else:
                logger.warning(f"Price data not available for {stock1} or {stock2}")
        for pair_key in positions_to_remove:
            del positions[pair_key]
    else:
        logger.info("No open positions to close.")

async def fetch_latest_price(ib, symbol):
    contract = contracts[symbol]
    try:
        [ticker] = await ib.reqTickersAsync(contract)
        return symbol, ticker.marketPrice()
    except Exception as e:
        logger.error(f"Error fetching latest price for {symbol}: {e}")
    return symbol, None

if __name__ == '__main__':
    asyncio.run(main())