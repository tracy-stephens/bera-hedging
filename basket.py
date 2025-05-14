import ccxt
import time
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.decomposition import PCA
import cvxpy as cp


def hedge_basket_universe():
    hyperliquid = ccxt.hyperliquid()
    markets = hyperliquid.load_markets()

    filters = {
        'swap': True,
        'active': True,
    }

    info = {}
    for symbol, details in tqdm(markets.items()):
        fits_filters = True
        for k, v in filters.items():
            if details.get(k) != v:
                fits_filters = False
                break
        if fits_filters:
            symbol_info = {
                'dayNtlVlm': float(details['info']['dayNtlVlm']),
                'openInterest': float(details['info']['openInterest']),
                'midPx': float(details['info']['midPx']),
                'maxLeverage': float(details['info']['maxLeverage']),
            }
            symbol_info['openInterestUsd'] = symbol_info['openInterest'] * symbol_info['midPx']
            info[symbol.replace('/USDC:USDC', '')] = symbol_info
        
    df = pd.DataFrame(info).T.sort_values('dayNtlVlm', ascending=False)
    return df


def historical_funding_rate(ticker, start=None, end=None):
    hyperliquid = ccxt.hyperliquid()
    all_funding_rates = []
    while start < end:
        funding_rate_history = hyperliquid.fetchFundingRateHistory(symbol=f"{ticker}/USDC:USDC", since=start, limit=500)
        if not funding_rate_history:
            break
        all_funding_rates.extend(funding_rate_history)
        start = funding_rate_history[-1]['timestamp'] + 1  # Move start to the next timestamp

    result = {entry['timestamp']: entry['fundingRate'] for entry in all_funding_rates}
    return result


def historical_close_volume(ticker, start=None, end=None):
    hyperliquid = ccxt.hyperliquid()
    all_data = []
    while start < end:
        ohlcv_data = hyperliquid.fetch_ohlcv(symbol=f"{ticker}/USDC:USDC", timeframe='1h', since=start, limit=500)
        if not ohlcv_data:
            break
        all_data.extend(ohlcv_data)
        start = ohlcv_data[-1][0] + 1  # Move start to the next timestamp

    result = {entry[0]: {'close': entry[4], 'volume': entry[5]} for entry in all_data}  # Extract timestamp, close price, and volume
    return result


def get_historical_data(ticker, start, end):
    funding_rate = historical_funding_rate(ticker, start, end)
    close_volume = historical_close_volume(ticker, start, end)

    df = pd.DataFrame({
        'funding': funding_rate,
        'close': {timestamp: data['close'] for timestamp, data in close_volume.items()},
        'volume': {timestamp: data['volume'] for timestamp, data in close_volume.items()},
    }).sort_index()
    df.index = pd.to_datetime(df.index, unit='ms', utc=True)
    df = df.resample('h').last()    
    return df.dropna(how='all')


def compute_max_weights(notional_df, basket_size, max_pct_daily_volume=0.05):
    daily_volume = notional_df.mean() * 24
    max_allocation_usd = daily_volume * max_pct_daily_volume
    max_weights = (max_allocation_usd / basket_size).fillna(1.0)
    return max_weights[notional_df.columns], daily_volume


def compute_funding_vector(funding_df):
    return -(funding_df.mean() * 24 * 365).values  # annualized funding cost


def get_market_component(asset_returns, bera_returns):
    pca = PCA(n_components=1)
    pca_factors = pca.fit_transform(asset_returns)
    pc1 = pd.Series(pca_factors[:, 0], index=asset_returns.index)

    # Project BERA onto PC1
    bera_returns, pc1 = bera_returns.align(pc1, axis=0)
    coef = np.dot(bera_returns, pc1) / np.dot(pc1, pc1)
    bera_market_component = coef * pc1

    return asset_returns, bera_market_component


def optimize_weights(asset_returns, bera_target, funding_vector, max_weights, lambda_funding=0.0001):
    R = asset_returns.values
    r_bera = bera_target.values
    n_assets = R.shape[1]
    w = cp.Variable(n_assets)

    r_portfolio = R @ w
    tracking_error = cp.sum_squares(r_portfolio - r_bera)
    funding_penalty = funding_vector @ w

    objective = cp.Minimize(tracking_error + lambda_funding * funding_penalty)
    constraints = [cp.sum(w) == 1, w >= 0, w <= max_weights.values]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    return pd.Series(w.value, index=asset_returns.columns)


def run_portfolio_optimization(
        asset_returns, 
        benchmark_returns, 
        notional_df, 
        funding_df,
        basket_size=5_000_000, 
        max_pct_daily_volume=0.05,
        lambda_funding=0.1
    ):
    asset_cols = asset_returns.columns

    # Constraints
    max_weights, daily_volume = compute_max_weights(notional_df, basket_size, max_pct_daily_volume)
    funding_vector = compute_funding_vector(funding_df)

    # Market component of BERA
    cleaned_returns, benchmark_market_component = get_market_component(asset_returns, benchmark_returns)

    # Optimize
    weights = optimize_weights(cleaned_returns, benchmark_market_component, funding_vector, max_weights, lambda_funding)
    filtered_weights = weights[weights > 1e-3]

    # Results DataFrame
    summary = pd.DataFrame({
        'weights': filtered_weights,
        'funding rate (ann.)': -pd.Series(funding_vector, index=asset_cols),
        'pct_daily_volume ($5m Basket)': (weights * basket_size) / daily_volume
    }).dropna().sort_values(by='weights', ascending=False).round(3)

    return summary


class HedgeBasket:
    def __init__(
            self, 
            ticker, 
            timestamp,
            basket_size=5_000_000,
            max_pct_daily_volume=0.05,
            lambda_funding=0.01
        ):
        self.ticker = ticker
        self.timestamp = timestamp
        self.data_start = self.timestamp - 30*24*60*60*1000

        self.basket_size = basket_size
        self.max_pct_daily_volume = max_pct_daily_volume
        self.lambda_funding = lambda_funding

        self._universe = None
        self._benchmark_data = None
        self._asset_data = None
        self._optimized_weights = None

        self.asset_funding = None
        self.asset_close = None
        self.asset_volume_usd = None

    @property
    def universe(self):
        if self._universe is None:
            self.universe_screener = hedge_basket_universe(self.timestamp)
            symbols = self.universe_screener.query(
                f'openInterestUsd > 10000000' + \
                f' and maxLeverage > 5' + \
                f' and dayNtlVlm > 5000000'
            ).index.to_list()
            self._universe = [i for i in symbols if i != self.ticker]
        return self._universe
    
    @property
    def benchmark_data(self):
        if self._benchmark_data is None:
            self._benchmark_data = get_historical_data(self.ticker, self.data_start, self.timestamp)
            self.benchmark_returns = self._benchmark_data['close'].pct_change(fill_method=None)
        return self._benchmark_data
    
    @property
    def asset_data(self):
        if self._asset_data is None:
            self.universe
            data = {}
            start = int(self.benchmark_data.index[0].timestamp() * 1000)
            for symbol in tqdm(self._universe):
                data[symbol] = get_historical_data(symbol, start, self.timestamp)
                time.sleep(2)
            self._asset_data = data
        
        self.asset_funding = pd.DataFrame({symbol: data['funding'] for symbol, data in self._asset_data.items()})
        self.asset_close = pd.DataFrame({symbol: data['close'] for symbol, data in self._asset_data.items()})
        self.asset_returns = self.asset_close.pct_change()
        asset_volume = pd.DataFrame({symbol: data['volume'] for symbol, data in self._asset_data.items()})
        self.asset_volume_usd = asset_volume * self.asset_close
        return self._asset_data

    @property
    def optimized_weights(self):
        if self._optimized_weights is None:
            if self.asset_data is None:
                self.universe
                self.benchmark_data
                self.asset_data
            benchmark, asset = self.benchmark_returns.dropna().align(self.asset_returns.dropna(), join='inner', axis=0)

            opt_weights = run_portfolio_optimization(
                asset_returns=asset,
                benchmark_returns=benchmark,
                notional_df=self.asset_volume_usd,
                funding_df=self.asset_funding,
                basket_size=self.basket_size, 
                max_pct_daily_volume=self.max_pct_daily_volume,
                lambda_funding=self.lambda_funding
            )
            self._optimized_weights = opt_weights
            self.basket_returns = self.asset_returns[list(opt_weights.index)].sum(axis=1)
            self.correlation = self.basket_returns.corr(self.benchmark_returns)
        return self._optimized_weights


if __name__ == "__main__":

    timestamp = 1740787200000

    hedge_basket = HedgeBasket(ticker='BERA', timestamp=timestamp) 
    print(hedge_basket.universe)
    print(hedge_basket.benchmark_data)
    hedge_basket.asset_data
    #print(hedge_basket.close)

    print(hedge_basket.optimized_weights)

    pass