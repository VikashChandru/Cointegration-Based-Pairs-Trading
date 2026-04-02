import os
import sys
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from datetime import time as dt_time
from ib_insync import *
import logging
import pytz
from statsmodels.tsa.stattools import coint, adfuller, zivot_andrews
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


# ─────────────────────────────────────────────────────────────────────────────
# Strategy parameters
# ─────────────────────────────────────────────────────────────────────────────
TIMEFRAME_OPTIONS = ['1 min', '5 mins', '15 mins', '30 mins', '1 hour', '1 day']
TIMEFRAME_MAPPING = {
    '1 min':   ('1 D',  '1 min'),
    '5 mins':  ('2 D',  '5 mins'),
    '15 mins': ('5 D',  '15 mins'),
    '30 mins': ('10 D', '30 mins'),
    '1 hour':  ('30 D', '1 hour'),
    '1 day':   ('2 Y',  '1 day'),
}

formation_period_bars   = 100
trading_period_bars     = 30
max_holding_period_bars = 500
num_pairs               = 20

FORMATION_BARS = {
    '1 min':   100,
    '5 mins':  100,
    '15 mins': 100,
    '30 mins': 100,
    '1 hour':  120,
    '1 day':   252,
}
TRADING_BARS = {
    '1 min':   30,
    '5 mins':  30,
    '15 mins': 30,
    '30 mins': 30,
    '1 hour':  60,
    '1 day':   63,
}

z_entry_threshold      = 2.0
z_exit_threshold       = 0.5
stop_loss_threshold    = 0.5
risk_per_trade_pct     = 0.01
fee_per_share          = 0.005

# ── RELAXED validation thresholds ────────────────────────────────────────────
MIN_OUT_SAMPLE_SHARPE  = -2.0    # Very relaxed (was -0.5)
MAX_BETA_STD           = 5.0     # Very relaxed (was 2.0)
MIN_HALF_LIFE          = 1

# ── Fallback / forced-pair settings ──────────────────────────────────────────
# If fewer than FORCE_PAIR_THRESHOLD pairs pass the full filter pipeline,
# the bot will supplement with FORCED pairs selected purely on correlation.
FORCE_PAIR_THRESHOLD   = 3       # Minimum pairs before force-mode activates
FORCE_NUM_PAIRS        = 5       # How many forced pairs to add
FORCED_PAIRS_FLAG      = "*** FILTER BYPASSED — FORCED PAIR (correlation only) ***"

# Well-known high-correlation equity pairs to try first in force mode
CANDIDATE_FORCED_PAIRS = [
    ('AAPL', 'MSFT'),
    ('GOOGL', 'META'),
    ('JPM',  'BAC'),
    ('XOM',  'CVX'),   # Note: CVX not in default universe — handled gracefully
    ('V',    'MA'),
    ('KO',   'PEP'),
    ('AMZN', 'NFLX'),
    ('NVDA', 'INTC'),
    ('PFE',  'MRK'),
    ('ABBV', 'AMGN'),
    ('ADBE', 'CRM'),
    ('QCOM', 'TXN'),
    ('UNH',  'ABT'),
    ('DIS',  'CMCSA'),
    ('WMT',  'COST'),
]


# ─────────────────────────────────────────────────────────────────────────────
# Universe
# ─────────────────────────────────────────────────────────────────────────────
EQUITY_SYMBOLS = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'JNJ', 'V',   'JPM',
    'WMT',  'PG',   'MA',   'UNH',   'DIS',  'NVDA', 'HD',  'PYPL', 'BAC',
    'VZ',   'ADBE', 'CMCSA','NFLX',  'XOM',  'INTC', 'T',   'CSCO', 'PFE',
    'KO',   'MRK',  'ABBV', 'PEP',   'ABT',  'CRM',  'ACN', 'MDT',  'COST',
    'WFC',  'TMO',  'DHR',  'AMGN',  'QCOM', 'TXN',  'NEE', 'ORCL', 'UPS',
    'BMY',  'MS',   'LIN',
]

FOREX_SYMBOLS = {
    'EURUSD': ('EUR', 'USD'),
    'USDJPY': ('USD', 'JPY'),
    'GBPUSD': ('GBP', 'USD'),
}

COMMODITY_SYMBOLS = {
    'XAUUSD': dict(symbol='GC', secType='CONTFUT', exchange='COMEX', currency='USD'),
}

symbols = EQUITY_SYMBOLS + list(FOREX_SYMBOLS.keys()) + list(COMMODITY_SYMBOLS.keys())

timezone = pytz.timezone('US/Eastern')

positions        = {}
contracts        = {}
initial_balance  = 100000
selected_pairs   = []
trading_end_time = None

market_open_time   = dt_time(9, 30)
market_close_time  = dt_time(16, 0)
market_open_timer  = '00:10'
market_close_timer = '00:10'

MARKET_HOURS_OVERRIDE = False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
async def async_input(prompt):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, input, prompt)


def build_contract(symbol: str):
    if symbol in FOREX_SYMBOLS:
        base, quote = FOREX_SYMBOLS[symbol]
        return Forex(pair=f'{base}{quote}')
    if symbol in COMMODITY_SYMBOLS:
        kw = COMMODITY_SYMBOLS[symbol]
        return Contract(**kw)
    return Stock(symbol, 'SMART', 'USD')


def now_eastern() -> datetime:
    return datetime.now(pytz.utc).astimezone(timezone)


def is_market_open(override: bool = False) -> bool:
    if override:
        return True
    now = now_eastern()
    if now.weekday() >= 5:
        return False
    t = now.time()
    return market_open_time <= t <= market_close_time


def get_timeframe_duration_in_seconds(timeframe):
    mapping = {
        '1 min':   60,
        '5 mins':  300,
        '15 mins': 900,
        '30 mins': 1800,
        '1 hour':  3600,
        '1 day':   86400,
    }
    return mapping.get(timeframe, 60)


def is_trade_allowed(current_datetime: datetime) -> bool:
    market_open_delay    = timedelta(
        hours=int(market_open_timer.split(':')[0]),
        minutes=int(market_open_timer.split(':')[1])
    )
    market_close_advance = timedelta(
        hours=int(market_close_timer.split(':')[0]),
        minutes=int(market_close_timer.split(':')[1])
    )
    base_date              = current_datetime.date()
    market_open_with_delay = (datetime.combine(base_date, market_open_time)  + market_open_delay).time()
    market_close_with_adv  = (datetime.combine(base_date, market_close_time) - market_close_advance).time()
    t = current_datetime.astimezone(timezone).replace(tzinfo=None).time()
    return market_open_with_delay <= t <= market_close_with_adv


# ─────────────────────────────────────────────────────────────────────────────
# IB duration helpers
# ─────────────────────────────────────────────────────────────────────────────
IB_MAX_DURATION = {
    '1 secs':  '1800 S', '5 secs':  '3600 S', '10 secs': '3600 S',
    '15 secs': '14400 S','30 secs': '28800 S',
    '1 min':   '1 D',   '2 mins':  '2 D',    '3 mins':  '1 W',
    '5 mins':  '1 W',   '10 mins': '1 W',    '15 mins': '2 W',
    '20 mins': '2 W',   '30 mins': '1 M',    '1 hour':  '1 M',
    '2 hours': '1 M',   '3 hours': '1 M',    '4 hours': '1 M',
    '8 hours': '1 M',   '1 day':   '1 Y',    '1 week':  '2 Y',
    '1 month': '2 Y',
}

IB_DURATION_DAYS = {
    '1800 S': 0.02, '3600 S': 0.04, '14400 S': 0.17, '28800 S': 0.33,
    '1 D': 1, '2 D': 2, '1 W': 7, '2 W': 14,
    '1 M': 30, '1 Y': 365, '2 Y': 730,
}


def get_ib_duration(barSizeSetting: str, num_bars: int) -> str:
    bar_seconds = {
        '1 secs': 1, '5 secs': 5, '10 secs': 10, '15 secs': 15,
        '30 secs': 30, '1 min': 60, '2 mins': 120, '3 mins': 180,
        '5 mins': 300, '10 mins': 600, '15 mins': 900, '20 mins': 1200,
        '30 mins': 1800, '1 hour': 3600, '2 hours': 7200, '3 hours': 10800,
        '4 hours': 14400, '8 hours': 28800, '1 day': 86400,
        '1 week': 604800, '1 month': 2592000,
    }.get(barSizeSetting, 60)

    needed_seconds = int(num_bars * bar_seconds * 1.4)
    needed_days    = needed_seconds / 86400

    max_dur  = IB_MAX_DURATION.get(barSizeSetting, '1 D')
    max_days = IB_DURATION_DAYS.get(max_dur, 1)

    if needed_days <= max_days:
        if needed_days >= 1:
            return f"{max(1, int(needed_days))} D"
        else:
            return f"{needed_seconds} S"
    else:
        return max_dur


# ─────────────────────────────────────────────────────────────────────────────
# Data fetching
# ─────────────────────────────────────────────────────────────────────────────
async def fetch_historical_data_period(ib, start_time, end_time, barSizeSetting):
    logger.info("Fetching historical data for formation period...")
    durationStr = get_ib_duration(barSizeSetting, formation_period_bars)
    logger.info(f"Using durationStr={durationStr} for barSize={barSizeSetting}")

    tasks = []
    for symbol, contract in contracts.items():
        tasks.append(
            fetch_historical_data(ib, contract, symbol, '', durationStr,
                                  barSizeSetting, start_time, end_time)
        )
    results = await asyncio.gather(*tasks)
    historical_data = {symbol: data for symbol, data in results if data is not None}
    logger.info(f"Historical data fetching completed. Got data for {len(historical_data)} symbols.")
    return historical_data


async def fetch_historical_data(ib, contract, symbol, endDateTime, durationStr,
                                barSizeSetting, start_time, end_time):
    if symbol in FOREX_SYMBOLS:
        what_to_show = 'MIDPOINT'
    elif symbol in COMMODITY_SYMBOLS:
        what_to_show = 'TRADES'
    else:
        what_to_show = 'TRADES'
    try:
        logger.info(f"Fetching data for {symbol}")
        bars = await ib.reqHistoricalDataAsync(
            contract,
            endDateTime=endDateTime,
            durationStr=durationStr,
            barSizeSetting=barSizeSetting,
            whatToShow=what_to_show,
            useRTH=(symbol not in FOREX_SYMBOLS),
            formatDate=1,
            keepUpToDate=False,
            timeout=60
        )
        if bars:
            df = util.df(bars)
            df.set_index('date', inplace=True)
            df.index = pd.to_datetime(df.index, utc=True).tz_convert('US/Eastern')
            df = df.drop_duplicates()
            if len(df) == 0:
                return symbol, None
            logger.info(f"Data fetched for {symbol}: {len(df)} bars")
            return symbol, df
        else:
            logger.warning(f"No data returned for {symbol}")
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
    return symbol, None


async def fetch_latest_bar(ib, contract, symbol, barSizeSetting):
    if symbol in FOREX_SYMBOLS:
        what_to_show = 'MIDPOINT'
    elif symbol in COMMODITY_SYMBOLS:
        what_to_show = 'TRADES'
    else:
        what_to_show = 'TRADES'
    duration_for_one = get_ib_duration(barSizeSetting, 5)
    try:
        logger.info(f"Fetching latest bar for {symbol}")
        bars = await ib.reqHistoricalDataAsync(
            contract,
            endDateTime='',
            durationStr=duration_for_one,
            barSizeSetting=barSizeSetting,
            whatToShow=what_to_show,
            useRTH=(symbol not in FOREX_SYMBOLS),
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


async def fetch_latest_price(ib, symbol):
    contract = contracts[symbol]
    try:
        [ticker] = await ib.reqTickersAsync(contract)
        return symbol, ticker.marketPrice()
    except Exception as e:
        logger.error(f"Error fetching latest price for {symbol}: {e}")
    return symbol, None


# ─────────────────────────────────────────────────────────────────────────────
# Pair analysis  —  full 8-stage filter + forced-pair fallback
# ─────────────────────────────────────────────────────────────────────────────
def build_forced_pair(stock1, stock2, prices):
    """
    Build a minimal pair dict using only OLS regression — no statistical
    tests required.  Tagged clearly so downstream logging can flag it.
    """
    s1 = prices[stock1].dropna()
    s2 = prices[stock2].dropna()
    common = s1.index.intersection(s2.index)
    s1, s2 = s1.loc[common], s2.loc[common]

    if len(s1) < 10:
        return None

    try:
        X    = sm.add_constant(s2)
        beta = sm.OLS(s1, X).fit().params.iloc[1]
    except Exception:
        return None

    spread       = s1 - beta * s2
    spread_mean  = spread.mean()
    spread_std   = spread.std()

    if spread_std == 0 or np.isnan(spread_std):
        return None

    # Quick half-life estimate (may be NaN — that's fine for forced pairs)
    try:
        lag  = spread.shift(1).dropna()
        diff = (spread - spread.shift(1)).dropna()
        idx  = lag.index.intersection(diff.index)
        b    = np.polyfit(lag.loc[idx].values, diff.loc[idx].values, 1)[0]
        half_life = -np.log(2) / b if b < 0 else 50.0
    except Exception:
        half_life = 50.0

    return {
        'stock1':        stock1,
        'stock2':        stock2,
        'beta':          beta,
        'half_life':     half_life,
        'eg_p_value':    np.nan,
        'adf_stat':      np.nan,
        'adf_p':         np.nan,
        'za_stat':       np.nan,
        'za_p':          np.nan,
        'out_sample_sr': 0.0,
        'beta_std':      np.nan,
        'spread_mean':   spread_mean,
        'spread_std':    spread_std,
        'forced':        True,       # ← key flag
    }


def analyze_pairs(historical_data):
    """
    Select cointegrated pairs using a multi-stage filter.
    If fewer than FORCE_PAIR_THRESHOLD pairs survive, supplement with
    correlation-based forced pairs (filter bypassed — logged clearly).
    """
    logger.info("Preparing price series for pair analysis.")
    prices = pd.DataFrame(
        {symbol: data['close'] for symbol, data in historical_data.items()}
    ).dropna(axis=1, how='any')

    valid_tickers = prices.columns.tolist()
    ticker_pairs  = list(combinations(valid_tickers, 2))
    logger.info(f"Number of valid tickers: {len(valid_tickers)}")
    logger.info(f"Analyzing {len(ticker_pairs)} pairs.")

    required_data_points = 10

    def analyze_pair(pair):
        stock1, stock2 = pair

        s1 = prices[stock1].dropna()
        s2 = prices[stock2].dropna()
        common_dates = s1.index.intersection(s2.index)
        s1 = s1.loc[common_dates]
        s2 = s2.loc[common_dates]

        if len(s1) < required_data_points:
            return None
        if s1.nunique() < 2 or s2.nunique() < 2:
            return None

        # ── 1. ENGLE-GRANGER cointegration test (p ≤ 0.10) ───────────────────
        try:
            eg_stat, eg_p_value, _ = coint(s1, s2)
        except Exception:
            return None

        if eg_p_value > 0.10:
            return None

        # ── 2. TRAIN / TEST SPLIT  (70 / 30) ─────────────────────────────────
        split    = int(len(s1) * 0.7)
        s1_train = s1.iloc[:split];  s1_test = s1.iloc[split:]
        s2_train = s2.iloc[:split];  s2_test = s2.iloc[split:]

        if len(s1_train) < 10 or len(s1_test) < 5:
            return None

        # ── 3. OLS HEDGE RATIO ────────────────────────────────────────────────
        try:
            X_train = sm.add_constant(s2_train)
            model   = sm.OLS(s1_train, X_train).fit()
            beta    = model.params.iloc[1]
        except Exception:
            return None

        spread_train = s1_train - beta * s2_train
        spread_test  = s1_test  - beta * s2_test

        # ── 4. ADF TEST on training residuals ─────────────────────────────────
        try:
            adf_stat, adf_p, *_ = adfuller(spread_train, autolag='AIC')
        except Exception:
            return None

        if adf_p > 0.10:
            return None

        # ── 5. ZIVOT-ANDREWS (only for ≥ 50 bars; soft fail) ─────────────────
        za_stat, za_p = np.nan, 1.0
        if len(spread_train) >= 50:
            try:
                za_stat, za_p, *_ = zivot_andrews(
                    spread_train, trim=0.15, maxlag=None,
                    regression='c', autolag='AIC'
                )
            except Exception:
                za_p = 1.0
        if len(spread_train) >= 50 and za_p > 0.15:
            return None

        # ── 6. HALF-LIFE FILTER ───────────────────────────────────────────────
        spread_lag  = spread_train.shift(1).dropna()
        spread_diff = (spread_train - spread_train.shift(1)).dropna()
        idx = spread_lag.index.intersection(spread_diff.index)
        spread_lag  = spread_lag.loc[idx]
        spread_diff = spread_diff.loc[idx]

        if len(spread_lag) < 10:
            return None

        try:
            beta_hr   = np.polyfit(spread_lag.values, spread_diff.values, 1)[0]
            half_life = -np.log(2) / beta_hr if beta_hr < 0 else np.nan
        except Exception:
            return None

        max_half_life = formation_period_bars
        if np.isnan(half_life) or not (MIN_HALF_LIFE <= half_life <= max_half_life):
            return None

        # ── 7. OUT-OF-SAMPLE SHARPE ───────────────────────────────────────────
        spread_mean = spread_train.mean()
        spread_std  = spread_train.std()

        if spread_std == 0 or np.isnan(spread_std):
            return None

        z_test    = (spread_test - spread_mean) / spread_std
        signal    = -np.sign(z_test.values[:-1])
        raw_ret   = signal * np.diff(z_test.values)
        sig_chg   = np.diff(np.concatenate([[0], signal])) != 0
        costs     = sig_chg[:-1] * fee_per_share * 2
        ret_net   = raw_ret - costs

        if len(ret_net) < 10 or np.std(ret_net) == 0:
            return None

        out_sample_sr = np.mean(ret_net) / np.std(ret_net)
        if out_sample_sr < MIN_OUT_SAMPLE_SHARPE:
            return None

        # ── 8. BETA STABILITY ─────────────────────────────────────────────────
        rolling_window = max(5, int(len(s1_train) * 0.25))
        rolling_betas  = []
        for i in range(rolling_window, len(s1_train) + 1):
            try:
                Xr = sm.add_constant(s2_train.iloc[i - rolling_window:i])
                yr = s1_train.iloc[i - rolling_window:i]
                rolling_betas.append(sm.OLS(yr, Xr).fit().params.iloc[1])
            except Exception:
                continue

        if len(rolling_betas) < 3:
            return None

        beta_std = np.std(rolling_betas)
        if beta_std > MAX_BETA_STD:
            return None

        return {
            'stock1':        stock1,
            'stock2':        stock2,
            'beta':          beta,
            'half_life':     half_life,
            'eg_p_value':    eg_p_value,
            'adf_stat':      adf_stat,
            'adf_p':         adf_p,
            'za_stat':       za_stat,
            'za_p':          za_p,
            'out_sample_sr': out_sample_sr,
            'beta_std':      beta_std,
            'spread_mean':   spread_mean,
            'spread_std':    spread_std,
            'forced':        False,
        }

    # ── Parallel full-filter evaluation ──────────────────────────────────────
    results    = Parallel(n_jobs=-1, prefer='threads')(
        delayed(analyze_pair)(pair) for pair in ticker_pairs
    )
    pairs_info = [r for r in results if r is not None]
    pairs_info_sorted = sorted(pairs_info, key=lambda x: x['out_sample_sr'], reverse=True)
    selected   = pairs_info_sorted[:num_pairs]

    logger.info(f"Full-filter: {len(selected)} pairs selected "
                f"(from {len(pairs_info)} that passed all filters).")

    # ── FORCED PAIR FALLBACK ──────────────────────────────────────────────────
    if len(selected) < FORCE_PAIR_THRESHOLD:
        logger.warning(
            f"⚠️  Only {len(selected)} pairs survived full filter "
            f"(threshold={FORCE_PAIR_THRESHOLD}). "
            f"Activating FORCED PAIR mode — filter criteria BYPASSED."
        )
        existing_pairs = {(p['stock1'], p['stock2']) for p in selected}
        forced_count   = 0

        for s1, s2 in CANDIDATE_FORCED_PAIRS:
            if forced_count >= FORCE_NUM_PAIRS:
                break
            if (s1, s2) in existing_pairs or (s2, s1) in existing_pairs:
                continue
            if s1 not in prices.columns or s2 not in prices.columns:
                logger.warning(f"  Forced pair {s1}/{s2} skipped — not in price data.")
                continue

            fp = build_forced_pair(s1, s2, prices)
            if fp is None:
                logger.warning(f"  Could not build forced pair {s1}/{s2}.")
                continue

            selected.append(fp)
            existing_pairs.add((s1, s2))
            forced_count += 1
            logger.warning(
                f"  {FORCED_PAIRS_FLAG}\n"
                f"  Forced pair added: {s1}/{s2}  "
                f"beta={fp['beta']:.4f}  HL={fp['half_life']:.1f}"
            )

        logger.warning(
            f"⚠️  Final pair count after forced additions: {len(selected)} "
            f"({forced_count} forced / filter-bypassed)."
        )

    # ── Log top-5 for visibility ──────────────────────────────────────────────
    for p in selected[:5]:
        tag = " [FORCED/BYPASSED]" if p.get('forced') else ""
        logger.info(
            f"  {p['stock1']}/{p['stock2']}{tag}  "
            f"EG_p={p['eg_p_value'] if not np.isnan(p['eg_p_value']) else 'N/A'}  "
            f"ADF_p={p['adf_p'] if not np.isnan(p['adf_p']) else 'N/A'}  "
            f"HL={p['half_life']:.1f}  OOS_SR={p['out_sample_sr']:.3f}"
        )

    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Trading logic
# ─────────────────────────────────────────────────────────────────────────────
async def run_trading_logic(ib, timeframe, barSizeSetting, csv_file, selected_pairs):
    global positions
    symbols_to_fetch = set()
    for pair in selected_pairs:
        symbols_to_fetch.update([pair['stock1'], pair['stock2']])

    tasks = []
    for symbol in symbols_to_fetch:
        contract = contracts[symbol]
        tasks.append(fetch_latest_bar(ib, contract, symbol, barSizeSetting))
    results = await asyncio.gather(*tasks)

    latest_prices = {}
    for symbol, data in results:
        if data is not None:
            latest_prices[symbol] = data['close'].iloc[-1]

    if not latest_prices:
        logger.warning("No latest price data fetched.")
        return

    logger.info("Updating existing positions...")
    await update_positions(ib, latest_prices, csv_file)

    logger.info("Generating new trading signals...")
    await generate_signals(ib, selected_pairs, latest_prices, csv_file)


async def generate_signals(ib, selected_pairs, latest_prices, csv_file):
    global positions
    for pair in selected_pairs:
        stock1      = pair['stock1']
        stock2      = pair['stock2']
        beta        = pair['beta']
        spread_mean = pair['spread_mean']
        spread_std  = pair['spread_std']
        pair_key    = f"{stock1}_{stock2}"
        is_forced   = pair.get('forced', False)

        if pair_key in positions:
            continue

        current_time   = now_eastern()
        is_forex_pair  = (stock1 in FOREX_SYMBOLS or stock2 in FOREX_SYMBOLS or
                          stock1 in COMMODITY_SYMBOLS or stock2 in COMMODITY_SYMBOLS)
        if not is_forex_pair and not is_trade_allowed(current_time):
            continue

        if stock1 not in latest_prices or stock2 not in latest_prices:
            continue

        price1  = latest_prices[stock1]
        price2  = latest_prices[stock2]
        spread  = price1 - beta * price2
        z_score = (spread - spread_mean) / spread_std

        per_trade_notional = initial_balance * risk_per_trade_pct
        quantity1 = max(1, int(per_trade_notional / price1))
        quantity2 = max(1, int(per_trade_notional / price2))

        if z_score > z_entry_threshold:
            pos_type = 'short'
        elif z_score < -z_entry_threshold:
            pos_type = 'long'
        else:
            continue

        # ── Prominent logging when filter was bypassed ────────────────────────
        if is_forced:
            logger.warning(
                f"\n{'='*70}\n"
                f"  ORDER PLACED — {FORCED_PAIRS_FLAG}\n"
                f"  Pair    : {stock1} / {stock2}\n"
                f"  Type    : {pos_type.upper()}\n"
                f"  Z-score : {z_score:.4f}\n"
                f"  NOTE    : Statistical validation was SKIPPED for this pair.\n"
                f"            Entry is based on correlation/OLS hedge ratio only.\n"
                f"{'='*70}"
            )
        else:
            logger.info(f"Opening {pos_type} position: {stock1}/{stock2}  z={z_score:.2f}")

        position = {
            'type':          pos_type,
            'entry_time':    now_eastern(),
            'stock1':        stock1,
            'stock2':        stock2,
            'quantity1':     quantity1,
            'quantity2':     quantity2,
            'entry_price1':  price1,
            'entry_price2':  price2,
            'beta':          beta,
            'spread_mean':   spread_mean,
            'spread_std':    spread_std,
            'entry_spread':  spread,
            'holding_period': 0,
            'csv_file':      csv_file,
            'entry_z':       z_score,
            'entry_fees1':   quantity1 * fee_per_share,
            'entry_fees2':   quantity2 * fee_per_share,
            'forced':        is_forced,
        }
        await open_position(ib, position, csv_file)
        positions[pair_key] = position


async def update_positions(ib, latest_prices, csv_file):
    global positions
    positions_to_remove = []

    for pair_key in list(positions.keys()):
        position  = positions[pair_key]
        stock1    = position['stock1']
        stock2    = position['stock2']
        beta      = position['beta']
        is_forced = position.get('forced', False)

        current_time = now_eastern()
        if not is_trade_allowed(current_time):
            continue

        if stock1 not in latest_prices or stock2 not in latest_prices:
            logger.warning(f"Price data not available for {stock1} or {stock2}")
            continue

        price1      = latest_prices[stock1]
        price2      = latest_prices[stock2]
        spread      = price1 - beta * price2
        spread_mean = position['spread_mean']
        spread_std  = position['spread_std']
        z_score     = (spread - spread_mean) / spread_std

        position['holding_period'] += 1
        holding_period    = position['holding_period']
        entry_spread      = position['entry_spread']
        spread_pct_change = abs((spread - entry_spread) / entry_spread) if entry_spread != 0 else 0

        exit_signal = False
        exit_reason = ''

        if (position['type'] == 'long'  and z_score >= -z_exit_threshold) or \
           (position['type'] == 'short' and z_score <=  z_exit_threshold):
            exit_signal = True
            exit_reason = 'Exit Signal'
        elif holding_period >= max_holding_period_bars:
            exit_signal = True
            exit_reason = 'Max Holding Period Reached'
        elif spread_pct_change > stop_loss_threshold:
            exit_signal = True
            exit_reason = 'Stop Loss Hit'

        if exit_signal:
            forced_tag = " [FORCED/BYPASSED PAIR]" if is_forced else ""
            logger.info(f"Closing {stock1}/{stock2}{forced_tag} — {exit_reason}")
            await close_position(ib, position, price1, price2, exit_reason, csv_file)
            positions_to_remove.append(pair_key)
        else:
            forced_tag = " [FORCED]" if is_forced else ""
            logger.info(f"Position {stock1}/{stock2}{forced_tag} open  z={z_score:.2f}")

    for pair_key in positions_to_remove:
        del positions[pair_key]


# ─────────────────────────────────────────────────────────────────────────────
# Order management
# ─────────────────────────────────────────────────────────────────────────────
async def open_position(ib, position, csv_file):
    stock1      = position['stock1']
    stock2      = position['stock2']
    quantity1   = position['quantity1']
    quantity2   = position['quantity2']
    pos_type    = position['type']
    contract1   = contracts[stock1]
    contract2   = contracts[stock2]
    is_forced   = position.get('forced', False)

    def make_order(action, qty, price, symbol):
        is_fx = symbol in FOREX_SYMBOLS or symbol in COMMODITY_SYMBOLS
        if is_fx:
            o = LimitOrder(action, qty, round(price, 5))
            o.tif = 'GTC'
            return o
        else:
            o = MarketOrder(action, qty)
            o.tif = 'GTC'
            return o

    if pos_type == 'short':
        order1 = make_order('SELL', quantity1, position['entry_price1'], stock1)
        order2 = make_order('BUY',  quantity2, position['entry_price2'], stock2)
    else:
        order1 = make_order('BUY',  quantity1, position['entry_price1'], stock1)
        order2 = make_order('SELL', quantity2, position['entry_price2'], stock2)

    bypass_note = f" *** FILTER BYPASSED ***" if is_forced else ""
    logger.info(f">>> PLACING ORDER{bypass_note}: {order1.action} {quantity1} {stock1}")
    logger.info(f">>> PLACING ORDER{bypass_note}: {order2.action} {quantity2} {stock2}")

    trade1 = ib.placeOrder(contract1, order1)
    trade2 = ib.placeOrder(contract2, order2)

    def onError(trade, message):
        logger.error(f"!!! IB ORDER ERROR for {trade.contract.symbol}: {message}")

    trade1.errorEvent += onError
    trade2.errorEvent += onError

    await asyncio.sleep(1)

    filled1 = await wait_for_fill(ib, trade1)
    filled2 = await wait_for_fill(ib, trade2)

    if filled1 and filled2:
        fills1 = trade1.fills
        fills2 = trade2.fills

        actual_qty1   = sum(f.execution.shares for f in fills1)
        actual_price1 = (sum(f.execution.shares * f.execution.price for f in fills1) / actual_qty1
                         if actual_qty1 else position['entry_price1'])
        actual_time1  = fills1[-1].execution.time.astimezone(timezone) if fills1 else now_eastern()

        actual_qty2   = sum(f.execution.shares for f in fills2)
        actual_price2 = (sum(f.execution.shares * f.execution.price for f in fills2) / actual_qty2
                         if actual_qty2 else position['entry_price2'])
        actual_time2  = fills2[-1].execution.time.astimezone(timezone) if fills2 else now_eastern()

        fees1 = sum(f.commissionReport.commission for f in fills1)
        fees2 = sum(f.commissionReport.commission for f in fills2)

        position.update({
            'actual_quantity1':   actual_qty1,
            'actual_quantity2':   actual_qty2,
            'actual_entry_price1': actual_price1,
            'actual_entry_price2': actual_price2,
            'actual_entry_time1':  actual_time1,
            'actual_entry_time2':  actual_time2,
            'entry_fees1': fees1,
            'entry_fees2': fees2,
        })

        bypass_note = f" *** FILTER BYPASSED / FORCED PAIR ***" if is_forced else ""
        logger.info(
            f"✅ ORDER FILLED{bypass_note}: {stock1}/{stock2}  "
            f"{pos_type.upper()}  "
            f"qty1={actual_qty1} @{actual_price1:.4f}  "
            f"qty2={actual_qty2} @{actual_price2:.4f}"
        )

        # ── Log entry to CSV ──────────────────────────────────────────────────
        log_entry_signal(csv_file, position)

    else:
        status1 = trade1.orderStatus.status
        status2 = trade2.orderStatus.status
        logger.error(
            f"❌ ORDER NOT FILLED: {stock1}/{stock2}  "
            f"status: {stock1}={status1}, {stock2}={status2}"
        )


async def close_position(ib, position, price1, price2, exit_reason, csv_file):
    stock1    = position['stock1']
    stock2    = position['stock2']
    quantity1 = position['quantity1']
    quantity2 = position['quantity2']
    pos_type  = position['type']
    contract1 = contracts[stock1]
    contract2 = contracts[stock2]

    def make_close_order(action, qty, price, symbol):
        is_fx = symbol in FOREX_SYMBOLS or symbol in COMMODITY_SYMBOLS
        if is_fx:
            o = LimitOrder(action, qty, round(price, 5))
            o.tif = 'GTC'
            return o
        else:
            return MarketOrder(action, qty)

    exit_price1 = position.get('actual_entry_price1', position['entry_price1'])
    exit_price2 = position.get('actual_entry_price2', position['entry_price2'])

    if pos_type == 'short':
        order1 = make_close_order('BUY',  quantity1,      exit_price1, stock1)
        order2 = make_close_order('SELL', abs(quantity2), exit_price2, stock2)
    else:
        order1 = make_close_order('SELL', quantity1,      exit_price1, stock1)
        order2 = make_close_order('BUY',  abs(quantity2), exit_price2, stock2)

    trade1 = ib.placeOrder(contract1, order1)
    trade2 = ib.placeOrder(contract2, order2)
    await asyncio.sleep(1)

    filled1 = await wait_for_fill(ib, trade1)
    filled2 = await wait_for_fill(ib, trade2)

    if filled1 and filled2:
        fills1 = trade1.fills
        fills2 = trade2.fills

        actual_exit_qty1   = sum(f.execution.shares for f in fills1)
        actual_exit_price1 = (sum(f.execution.shares * f.execution.price for f in fills1) / actual_exit_qty1
                              if actual_exit_qty1 else exit_price1)
        actual_exit_time1  = fills1[-1].execution.time.astimezone(timezone) if fills1 else now_eastern()

        actual_exit_qty2   = sum(f.execution.shares for f in fills2)
        actual_exit_price2 = (sum(f.execution.shares * f.execution.price for f in fills2) / actual_exit_qty2
                              if actual_exit_qty2 else exit_price2)
        actual_exit_time2  = fills2[-1].execution.time.astimezone(timezone) if fills2 else now_eastern()

        fees1 = sum(f.commissionReport.commission for f in fills1)
        fees2 = sum(f.commissionReport.commission for f in fills2)

        position.update({
            'actual_exit_time1':  actual_exit_time1,
            'actual_exit_time2':  actual_exit_time2,
            'actual_exit_price1': actual_exit_price1,
            'actual_exit_price2': actual_exit_price2,
            'exit_fees1':  fees1,
            'exit_fees2':  fees2,
            'exit_reason': exit_reason,
        })
        update_csv_entry(csv_file, position)
        logger.info(f"Closed {pos_type} position: {stock1}/{stock2} — {exit_reason}")
    else:
        logger.error(f"Failed to close position for {stock1}/{stock2}")


async def wait_for_fill(ib, trade, timeout=10):
    start = time.time()
    while True:
        if trade.isDone():
            return True
        if time.time() - start > timeout:
            return False
        await asyncio.sleep(1)


async def close_all_positions(ib, csv_file):
    global positions
    if not positions:
        logger.info("No open positions to close.")
        return

    logger.info("Closing all open positions...")
    unique_symbols = set()
    for pos in positions.values():
        unique_symbols.update([pos['stock1'], pos['stock2']])

    tasks  = [fetch_latest_price(ib, s) for s in unique_symbols]
    results = await asyncio.gather(*tasks)
    latest_prices = {sym: price for sym, price in results if price is not None}

    for pair_key in list(positions.keys()):
        pos    = positions[pair_key]
        s1, s2 = pos['stock1'], pos['stock2']
        if s1 in latest_prices and s2 in latest_prices:
            await close_position(ib, pos, latest_prices[s1], latest_prices[s2],
                                 "Strategy Exit", csv_file)
        else:
            logger.warning(f"Price data not available for {s1} or {s2}")

    positions.clear()


# ─────────────────────────────────────────────────────────────────────────────
# CSV logging
# ─────────────────────────────────────────────────────────────────────────────
def log_entry_signal(csv_file, position):
    is_forced   = position.get('forced', False)
    forced_note = 'FILTER_BYPASSED' if is_forced else 'VALIDATED'
    data = {
        'Entry Time':      position.get('actual_entry_time1', position['entry_time']),
        'Exit Time':       '',
        'Position':        position['type'].capitalize(),
        'Stock1':          position['stock1'],
        'Stock2':          position['stock2'],
        'Quantity1':       position.get('actual_quantity1',    position['quantity1']),
        'Quantity2':       position.get('actual_quantity2',    position['quantity2']),
        'Entry Price1':    position.get('actual_entry_price1', position['entry_price1']),
        'Entry Price2':    position.get('actual_entry_price2', position['entry_price2']),
        'Exit Price1':     '',
        'Exit Price2':     '',
        'Profit1':         '',
        'Profit2':         '',
        'Fees1':           position.get('entry_fees1', 0),
        'Fees2':           position.get('entry_fees2', 0),
        'Net Profit':      '',
        'Exit Reason':     '',
        'Filter Status':   forced_note,   # NEW column — clearly marks bypassed orders
    }
    with open(csv_file, 'a') as f:
        f.write(','.join(map(str, data.values())) + '\n')


def update_csv_entry(csv_file, position):
    is_forced   = position.get('forced', False)
    forced_note = 'FILTER_BYPASSED' if is_forced else 'VALIDATED'
    data = {
        'Entry Time':      position.get('actual_entry_time1', position['entry_time']),
        'Exit Time':       now_eastern(),
        'Position':        position['type'].capitalize(),
        'Stock1':          position['stock1'],
        'Stock2':          position['stock2'],
        'Quantity1':       position.get('actual_quantity1',    position['quantity1']),
        'Quantity2':       position.get('actual_quantity2',    position['quantity2']),
        'Entry Price1':    position.get('actual_entry_price1', position['entry_price1']),
        'Entry Price2':    position.get('actual_entry_price2', position['entry_price2']),
        'Exit Price1':     position.get('actual_exit_price1',  ''),
        'Exit Price2':     position.get('actual_exit_price2',  ''),
        'Profit1':         '',
        'Profit2':         '',
        'Fees1':           '',
        'Fees2':           '',
        'Net Profit':      '',
        'Exit Reason':     position.get('exit_reason', ''),
        'Filter Status':   forced_note,
    }

    try:
        qty1         = float(data['Quantity1'])
        qty2         = float(data['Quantity2'])
        entry_price1 = float(data['Entry Price1'])
        entry_price2 = float(data['Entry Price2'])
        exit_price1  = float(data['Exit Price1'])  if data['Exit Price1']  != '' else 0.0
        exit_price2  = float(data['Exit Price2'])  if data['Exit Price2']  != '' else 0.0

        if position['type'] == 'long':
            profit1 = (exit_price1 - entry_price1) * qty1
            profit2 = (entry_price2 - exit_price2) * abs(qty2)
        else:
            profit1 = (entry_price1 - exit_price1) * qty1
            profit2 = (exit_price2 - entry_price2) * abs(qty2)

        total_fees1 = position.get('entry_fees1', 0) + position.get('exit_fees1', 0)
        total_fees2 = position.get('entry_fees2', 0) + position.get('exit_fees2', 0)
        net_profit  = profit1 + profit2 - total_fees1 - total_fees2

        data['Profit1']    = round(profit1,    2)
        data['Profit2']    = round(profit2,    2)
        data['Fees1']      = round(total_fees1, 2)
        data['Fees2']      = round(total_fees2, 2)
        data['Net Profit'] = round(net_profit,  2)
    except Exception as e:
        logger.error(f"Error calculating profits: {e}")

    with open(csv_file, 'a') as f:
        f.write(','.join(map(str, data.values())) + '\n')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
async def main():
    ib = IB()
    try:
        await ib.connectAsync('127.0.0.1', 7497, clientId=1)
        logger.info("Connected to Interactive Brokers.")
    except Exception as e:
        logger.error(f"Failed to connect to Interactive Brokers: {e}")
        sys.exit(1)

    ib.reqMarketDataType(1)
    await asyncio.sleep(1)
    logger.info("Synchronization complete")

    account_summary = await ib.accountSummaryAsync()
    net_liquidation = float(
        [v.value for v in account_summary if v.tag == 'NetLiquidation'][0]
    )
    global initial_balance
    initial_balance = net_liquidation
    logger.info(f"Account balance: ${initial_balance:.2f}")

    # Timeframe selection
    print("Select the timeframe:")
    for idx, tf in enumerate(TIMEFRAME_OPTIONS):
        print(f"{idx + 1}: {tf}")
    timeframe_index   = await async_input("Enter the number corresponding to the desired timeframe: ")
    timeframe         = TIMEFRAME_OPTIONS[int(timeframe_index) - 1]
    _, barSizeSetting = TIMEFRAME_MAPPING[timeframe]
    bar_duration_seconds = get_timeframe_duration_in_seconds(timeframe)

    global formation_period_bars, trading_period_bars
    formation_period_bars = FORMATION_BARS.get(timeframe, 100)
    trading_period_bars   = TRADING_BARS.get(timeframe, 30)
    logger.info(f"Timeframe={timeframe}  formation={formation_period_bars} bars  trading={trading_period_bars} bars")

    current_et = now_eastern()
    logger.info(f"Current time (US/Eastern): {current_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    global MARKET_HOURS_OVERRIDE
    override_input = await async_input(
        "Bypass market-hours check? (y = run anytime, n = only during NYSE hours) [n]: "
    )
    MARKET_HOURS_OVERRIDE = override_input.strip().lower() == 'y'
    if MARKET_HOURS_OVERRIDE:
        logger.info("Market-hours override ENABLED.")

    for symbol in symbols:
        contracts[symbol] = build_contract(symbol)

    logger.info("Qualifying contracts...")
    valid_symbols = []
    for symbol, contract in list(contracts.items()):
        try:
            qualified = await ib.qualifyContractsAsync(contract)
            if qualified:
                contracts[symbol] = qualified[0]
                valid_symbols.append(symbol)
            else:
                logger.warning(f"Could not qualify {symbol}; skipping.")
                del contracts[symbol]
        except Exception as e:
            logger.warning(f"Error qualifying {symbol}: {e}; skipping.")
            del contracts[symbol]
    logger.info(f"Contracts qualified: {len(valid_symbols)} symbols active.")

    timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_folder = f'Sessions/TradingSession_{timestamp}'
    os.makedirs(session_folder, exist_ok=True)

    csv_file = os.path.join(session_folder, f"trade_history_{timestamp}.csv")
    with open(csv_file, 'w') as f:
        f.write('Entry Time,Exit Time,Position,Stock1,Stock2,Quantity1,Quantity2,'
                'Entry Price1,Entry Price2,Exit Price1,Exit Price2,'
                'Profit1,Profit2,Fees1,Fees2,Net Profit,Exit Reason,Filter Status\n')

    global trading_end_time
    formation_start_time = None
    formation_end_time   = None
    trading_start_time   = None
    trading_end_time     = None

    try:
        while True:
            if not is_market_open(override=MARKET_HOURS_OVERRIDE):
                logger.info("Market is closed. Waiting...")
                await asyncio.sleep(60)
                continue

            current_time = now_eastern()

            if trading_end_time is None or current_time >= trading_end_time:
                formation_end_time   = current_time - timedelta(seconds=bar_duration_seconds)
                formation_start_time = formation_end_time - timedelta(
                    seconds=formation_period_bars * bar_duration_seconds * 3
                )
                logger.info(f"Formation period: {formation_start_time} → {formation_end_time}")

                historical_data = await fetch_historical_data_period(
                    ib, formation_start_time, formation_end_time, barSizeSetting
                )
                if not historical_data:
                    logger.error("No historical data. Retrying in 60 s.")
                    await asyncio.sleep(60)
                    continue

                selected_pairs = analyze_pairs(historical_data)

                # ── Summary banner ────────────────────────────────────────────
                n_forced    = sum(1 for p in selected_pairs if p.get('forced'))
                n_validated = len(selected_pairs) - n_forced
                logger.info(
                    f"\n{'='*60}\n"
                    f"  PAIR SELECTION SUMMARY\n"
                    f"  Total pairs selected : {len(selected_pairs)}\n"
                    f"  Fully validated      : {n_validated}\n"
                    f"  Filter BYPASSED      : {n_forced}  ← orders will be flagged\n"
                    f"{'='*60}"
                )

                trading_start_time = current_time
                trading_end_time   = trading_start_time + timedelta(
                    seconds=trading_period_bars * bar_duration_seconds
                )
                logger.info(f"Trading period until {trading_end_time}")

                await close_all_positions(ib, csv_file)
                positions.clear()

            if current_time < trading_end_time:
                await run_trading_logic(ib, timeframe, barSizeSetting, csv_file, selected_pairs)
            else:
                logger.info("Trading period ended.")
                await close_all_positions(ib, csv_file)
                selected_pairs = []

            logger.info(f"Waiting {bar_duration_seconds} s until next bar.")
            await asyncio.sleep(bar_duration_seconds)

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user.")
        close_yn = await async_input("Close all open positions? (y/n): ")
        if close_yn.lower() == 'y':
            await close_all_positions(ib, csv_file)
            logger.info("All positions closed.")
    finally:
        ib.disconnect()
        logger.info("Disconnected from Interactive Brokers.")


if __name__ == '__main__':
    asyncio.run(main())