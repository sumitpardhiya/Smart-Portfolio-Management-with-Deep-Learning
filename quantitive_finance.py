import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Layer, MultiHeadAttention, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import tensorflow as tf


warnings.filterwarnings(action="ignore")

tf.random.set_seed(42)
np.random.seed(42)


def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    return pd.Series(rsi, index=series.index)

def download_data(tickers, fx_tickers, start_date, end_date):
 
    print("Downloading historical data...")
    all_symbols = list(tickers.keys()) + list(fx_tickers.keys())
    
    prices_series = []
    volumes_series = []
 
    all_available_indices = []

    for symbol in all_symbols:
        try:
            ticker_obj = yf.Ticker(symbol)
            data = ticker_obj.history(start=start_date, end=end_date)
            
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)

            if not data.empty:
                current_price_series = None
                if 'Adj Close' in data.columns:
                    current_price_series = data['Adj Close'].rename(symbol)
                elif 'Close' in data.columns:
                    current_price_series = data['Close'].rename(symbol)
                else:
                    print(f"Warning: Neither 'Adj Close' nor 'Close' found for {symbol}. Skipping.")
                    continue
                
                if current_price_series is not None and not current_price_series.empty:
                    prices_series.append(current_price_series)
                    all_available_indices.append(current_price_series.index)

                    if 'Volume' in data.columns:
                        volumes_series.append(data['Volume'].rename(symbol))
                    else:
                        volumes_series.append(pd.Series(0.0, index=data.index).rename(symbol))
                        print(f"Warning: Volume data not found for {symbol}. Setting to 0.")
                else:
                    print(f"Warning: Price data for {symbol} is empty after column selection. Skipping.")
            else:
                print(f"Warning: No data downloaded for {symbol}. Skipping.")
        except Exception as e:
            print(f"Error downloading data for {symbol}: {e}. Skipping.")

    if not prices_series:
        print("Error: No price data successfully downloaded for any ticker. Returning empty DataFrames.")
        return pd.DataFrame(columns=all_symbols), pd.DataFrame(columns=all_symbols)

    df_prices = pd.concat(prices_series, axis=1)
    df_volumes = pd.concat(volumes_series, axis=1)

    master_index = all_available_indices[0]
    for idx in all_available_indices[1:]:
        master_index = master_index.union(idx)
    master_index = master_index.sort_values() 

    df_prices = df_prices.reindex(master_index)
    df_volumes = df_volumes.reindex(master_index)

    df_prices = df_prices.reindex(columns=all_symbols)
    df_volumes = df_volumes.reindex(columns=all_symbols)

    print("Data download complete.")
    return df_prices, df_volumes

def calculate_features(df_prices, df_volumes):
    print("Calculating financial features...")
    df_returns = np.log(df_prices / df_prices.shift(1))

    features_list = []

    for ticker in df_prices.columns:
        asset_features = pd.DataFrame(index=df_prices.index)
        asset_features[f'{ticker}_log_return'] = df_returns[ticker]
        
        asset_features[f'{ticker}_ma_5d'] = df_returns[ticker].rolling(window=5).mean()
        asset_features[f'{ticker}_ma_21d'] = df_returns[ticker].rolling(window=21).mean()
        
        asset_features[f'{ticker}_vol_21d'] = df_returns[ticker].rolling(window=21).std()

        asset_features[f'{ticker}_rsi_14d'] = calculate_rsi(df_prices[ticker], window=14)
        
        if ticker in df_volumes.columns and not df_volumes[ticker].empty:
            asset_features[f'{ticker}_volume'] = df_volumes[ticker]
        else:
            asset_features[f'{ticker}_volume'] = 0.0 
            print(f"Warning: Volume data not available for {ticker}. Setting to 0.")

        features_list.append(asset_features)

    all_features = pd.concat(features_list, axis=1)
    print("Feature calculation complete.")
    return all_features

class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=None): 
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config

class PositionalEmbedding(Layer):
 
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.output_dim = output_dim
        self.position_embedding_layer = Dense(output_dim)

    def call(self, inputs):
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        position_embedding = self.position_embedding_layer(tf.expand_dims(tf.cast(positions, dtype=tf.float32), axis=-1))

        return inputs + tf.expand_dims(position_embedding, axis=0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "output_dim": self.output_dim,
        })
        return config

def build_lstm_model(input_shape, num_assets):
    print("Building LSTM model...")
    model_input = Input(shape=input_shape)
    x = LSTM(50, return_sequences=True, activation='tanh')(model_input)
    x = Dropout(0.2)(x)
    x = LSTM(50, return_sequences=False, activation='tanh')(x)
    x = Dropout(0.2)(x)
    model_output = Dense(num_assets, activation='linear')(x) 
    model = Model(inputs=model_input, outputs=model_output, name="LSTM_Model")
    print("LSTM model built.")
    return model

def build_transformer_model(input_shape, num_assets, num_heads=8, model_dim=64, ff_dim=128, dropout_rate=0.1):

    print("Building Transformer model...")
    sequence_length = input_shape[0]
    num_features_per_timestep = input_shape[1]

    inputs = Input(shape=input_shape) 
    x = Dense(model_dim, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)

    x = PositionalEmbedding(sequence_length, model_dim)(x)

    x = TransformerBlock(model_dim, num_heads, ff_dim, rate=dropout_rate)(x)
    x = TransformerBlock(model_dim, num_heads, ff_dim, rate=dropout_rate)(x) 

    x = tf.keras.layers.GlobalAveragePooling1D()(x) 

    model_output = Dense(num_assets, activation='linear')(x)
    model = Model(inputs=inputs, outputs=model_output, name="Transformer_Model")
    print("Transformer model built.")
    return model

def train_model(model, X_train, y_train, X_val, y_val, model_name, hyperparameters):
    print(f"Training {model_name} model...")
    model.compile(optimizer=Adam(learning_rate=hyperparameters['learning_rate']), loss='mse')
    
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=hyperparameters['early_stopping_patience'], 
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=hyperparameters['epochs'],
        batch_size=hyperparameters['batch_size'],
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )
    print(f"{model_name} model training complete.")
    return model, history

def run_backtest(model, model_name, all_features_scaled, original_returns, num_assets, asset_tickers_list, initial_portfolio_value, transaction_cost_bps, lookback_window, test_start_date, test_end_date):
    print(f"Running backtest for {model_name}...")
    
    transaction_cost = transaction_cost_bps / 10000.0
    
    strategy_daily_returns = []
    portfolio_weights_history = [] 
   
    current_weights = np.ones(num_assets) / num_assets

    idx_start_in_features_scaled = all_features_scaled.index.get_loc(pd.Timestamp(test_start_date))

    for i in range(idx_start_in_features_scaled, len(all_features_scaled)):
        current_date_for_prediction = all_features_scaled.index[i]
        
        if i < lookback_window:
            strategy_daily_returns.append(0.0)
            portfolio_weights_history.append(current_weights)
            continue
        input_sequence = all_features_scaled.iloc[i - lookback_window : i].values.reshape(1, lookback_window, -1)

        predicted_raw = model.predict(input_sequence, verbose=0)
        
        if predicted_raw.ndim == 2 and predicted_raw.shape[0] == 1:
            predicted_returns = predicted_raw[0]
        elif predicted_raw.ndim == 1 and predicted_raw.shape[0] == num_assets:
            predicted_returns = predicted_raw 
        else:
            print(f"Warning: Unexpected prediction shape for {model_name} at {current_date_for_prediction}. "
                  f"Expected shape compatible with (1, {num_assets}), got {predicted_raw.shape}. "
                  "Using zeros for prediction.")
            predicted_returns = np.zeros(num_assets)

        predicted_returns_series = pd.Series(predicted_returns, index=asset_tickers_list)
  
        covariance_start_date = all_features_scaled.index[i - lookback_window]
        covariance_end_date = all_features_scaled.index[i-1] 
        
        historical_returns_for_cov = original_returns.loc[covariance_start_date:covariance_end_date]
        
        if len(historical_returns_for_cov) < lookback_window:
            print(f"Skipping prediction for {current_date_for_prediction} due to insufficient data for covariance.")
            strategy_daily_returns.append(0.0) 
            portfolio_weights_history.append(current_weights)
            continue

        sample_cov_matrix = risk_models.sample_cov(historical_returns_for_cov, frequency=252)
        
        new_weights = None
        try:
            ef = EfficientFrontier(predicted_returns_series, sample_cov_matrix, verbose=False)
            ef.add_constraint(lambda w: w >= 0) 
            ef.add_constraint(lambda w: np.sum(w) == 1)
            
            raw_weights = ef.max_sharpe()
            new_weights_dict = ef.clean_weights(raw_weights, cutoff=1e-6)
            new_weights = np.array([new_weights_dict.get(ticker, 0.0) for ticker in asset_tickers_list])
            
            if np.sum(new_weights) > 0:
                new_weights = new_weights / np.sum(new_weights) 
            else: 
                new_weights = np.ones(num_assets) / num_assets
            
        except Exception as e:
            new_weights = np.ones(num_assets) / num_assets
            
        if new_weights is None or np.any(new_weights < -1e-9) or abs(np.sum(new_weights) - 1) > 1e-6:
             new_weights = np.ones(num_assets) / num_assets

        actual_day_returns = original_returns.loc[current_date_for_prediction]
        
        daily_portfolio_return_pre_cost = np.dot(new_weights, actual_day_returns.values)
      
        turnover = np.sum(np.abs(new_weights - current_weights)) / 2 
        cost_in_return = turnover * transaction_cost
        
        daily_portfolio_return_net_cost = daily_portfolio_return_pre_cost - cost_in_return
        strategy_daily_returns.append(daily_portfolio_return_net_cost)

        current_weights = new_weights

        portfolio_weights_history.append(current_weights)
        
    print(f"Backtest for {model_name} complete.")

    test_period_dates = original_returns.loc[test_start_date:test_end_date].index
    
    if len(strategy_daily_returns) != len(test_period_dates):
        print(f"Warning: Mismatch in length of daily returns ({len(strategy_daily_returns)}) and expected test dates ({len(test_period_dates)}). Adjusting.")
        min_len = min(len(strategy_daily_returns), len(test_period_dates))
        strategy_daily_returns = strategy_daily_returns[:min_len]
        test_period_dates = test_period_dates[:min_len]
        portfolio_weights_history = portfolio_weights_history[:min_len]

    portfolio_returns_series = pd.Series(strategy_daily_returns, index=test_period_dates)
    cumulative_values = (1 + portfolio_returns_series).cumprod() * initial_portfolio_value
 
    portfolio_weights_df = pd.DataFrame(portfolio_weights_history, index=test_period_dates, columns=asset_tickers_list)
    
    return cumulative_values, portfolio_weights_df

def evaluate_portfolio(portfolio_values, initial_value):
 
    print("Evaluating portfolio performance...")
    portfolio_returns = portfolio_values.pct_change().dropna()

    annualized_return = (1 + portfolio_returns).prod()**(252/len(portfolio_returns)) - 1

    annualized_volatility = portfolio_returns.std() * np.sqrt(252)

    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else np.nan

    peak = portfolio_values.expanding(min_periods=1).max()
    drawdown = (portfolio_values - peak) / peak
    max_drawdown = drawdown.min()

    calmar_ratio = annualized_return / abs(max_drawdown) if abs(max_drawdown) != 0 else np.nan

    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else np.nan
    sortino_ratio = (annualized_return - 0) / downside_std if downside_std > 0 else np.nan 
    metrics = {
        "Annualized Return (%)": annualized_return * 100,
        "Annualized Volatility (%)": annualized_volatility * 100,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown (%)": max_drawdown * 100,
        "Calmar Ratio": calmar_ratio,
        "Sortino Ratio": sortino_ratio,
    }
    print("Performance evaluation complete.")
    return metrics


def main():
    asset_tickers = {
        "^GSPC": "S&P 500",
        "^FTSE": "FTSE 100",
        "^N225": "Nikkei 225",
        "EEM": "MSCI Emerging Mkts ETF",
        "GC=F": "Gold Price (Futures Proxy)",
        "TLT": "U.S. 10Y Treasury Bond (ETF Proxy)" 
    }
    asset_tickers_list = list(asset_tickers.keys())

    fx_tickers = {
        "GBPUSD=X": "GBP/USD",
        "JPY=X": "JPY/USD",
    }

    overall_start_date = "2009-01-01" 
    overall_end_date = "2020-12-31"

    train_start_date = "2010-01-01"
    train_end_date = "2017-12-31"
    test_start_date = "2018-01-01"
    test_end_date = "2020-12-31"
    
    lookback_window = 60 
    transaction_cost_bps = 10 
    initial_portfolio_value = 100

    model_hyperparameters = {
        'learning_rate': 0.001,
        'epochs': 100,
        'batch_size': 32,
        'early_stopping_patience': 10,
        'validation_split': 0.1 
    }

    prices_raw, volumes_raw = download_data(asset_tickers, fx_tickers, overall_start_date, overall_end_date)

    print("Saving raw downloaded data...")
    prices_raw.to_csv("downloaded_prices_raw.csv")
    volumes_raw.to_csv("downloaded_volumes_raw.csv")
    print("Raw downloaded data saved.")

    df_prices = prices_raw.filter(items=asset_tickers_list).copy()
    df_volumes = volumes_raw.filter(items=asset_tickers_list).copy()
    df_fx = prices_raw.filter(items=list(fx_tickers.keys())).copy()

    if df_prices.empty or df_fx.empty:
        print("Error: Could not download all required data. Please check tickers and date ranges.")
        return

    print("Converting non-USD indices to USD...")
    if "^FTSE" in df_prices.columns and "GBPUSD=X" in df_fx.columns:
        df_prices["^FTSE"] = df_prices["^FTSE"] * df_fx["GBPUSD=X"].reindex(df_prices.index, method='ffill')
    if "^N225" in df_prices.columns and "JPY=X" in df_fx.columns:
        df_prices["^N225"] = df_prices["^N225"] * df_fx["JPY=X"].reindex(df_prices.index, method='ffill')
    print("FX conversion complete.")

    df_prices.ffill(inplace=True)
    df_volumes.ffill(inplace=True) 
    df_prices.dropna(inplace=True) 
    df_volumes.dropna(inplace=True)

    full_features_df = calculate_features(df_prices, df_volumes)
    full_features_df.ffill(inplace=True) 
    full_features_df.dropna(inplace=True)

    common_index = df_prices.index.intersection(full_features_df.index)
    df_prices = df_prices.loc[common_index]
    full_features_df = full_features_df.loc[common_index]
    df_volumes = df_volumes.loc[common_index] 
    
    original_log_returns = np.log(df_prices / df_prices.shift(1)).dropna()
    
    common_index_for_returns = original_log_returns.index.intersection(full_features_df.index)
    original_log_returns = original_log_returns.loc[common_index_for_returns]
    full_features_df = full_features_df.loc[common_index_for_returns]

    train_features = full_features_df.loc[train_start_date:train_end_date]
    train_returns = original_log_returns.loc[train_start_date:train_end_date]

    test_start_date_for_seq_creation = (pd.Timestamp(test_start_date) - pd.Timedelta(days=lookback_window * 1.5)) 
 
    backtest_features_df_raw = full_features_df.loc[test_start_date_for_seq_creation:test_end_date]

    scaler = StandardScaler()
    scaler.fit(train_features)
    
    train_features_scaled = scaler.transform(train_features)
    backtest_features_scaled = scaler.transform(backtest_features_df_raw)

    train_features_scaled_df = pd.DataFrame(train_features_scaled, index=train_features.index, columns=train_features.columns)
    backtest_features_scaled_df = pd.DataFrame(backtest_features_scaled, index=backtest_features_df_raw.index, columns=backtest_features_df_raw.columns)

    def create_sequences(features_df, returns_df, lookback):
        X, y = [], []
        for i in range(lookback, len(features_df)):
            X.append(features_df.iloc[i - lookback : i].values)
            y.append(returns_df.iloc[i].values)
        return np.array(X), np.array(y)

    X_train_seq, y_train_seq = create_sequences(train_features_scaled_df, train_returns, lookback_window)

    val_split_idx = int(len(X_train_seq) * (1 - model_hyperparameters['validation_split']))
    X_val_seq = X_train_seq[val_split_idx:]
    y_val_seq = y_train_seq[val_split_idx:]
    X_train_seq = X_train_seq[:val_split_idx]
    y_train_seq = y_train_seq[:val_split_idx]

    input_shape = (lookback_window, X_train_seq.shape[2]) 
    num_assets = len(asset_tickers_list)


    lstm_model = build_lstm_model(input_shape, num_assets)
    lstm_model, _ = train_model(lstm_model, X_train_seq, y_train_seq, X_val_seq, y_val_seq, "LSTM", model_hyperparameters)


    tf.keras.utils.get_custom_objects().update({"TransformerBlock": TransformerBlock})
    tf.keras.utils.get_custom_objects().update({"PositionalEmbedding": PositionalEmbedding})
    
    transformer_model = build_transformer_model(input_shape, num_assets) 
    transformer_model, _ = train_model(transformer_model, X_train_seq, y_train_seq, X_val_seq, y_val_seq, "Transformer", model_hyperparameters)

    print("\nSaving trained models...")
    lstm_model.save("lstm_model.h5")
    transformer_model.save("transformer_model.h5")
    print("Trained models saved.")


    lstm_portfolio_values, lstm_weights = run_backtest(
        lstm_model, 
        "LSTM Strategy", 
        backtest_features_scaled_df, 
        original_log_returns,      
        num_assets, 
        asset_tickers_list, 
        initial_portfolio_value, 
        transaction_cost_bps, 
        lookback_window,
        test_start_date,           
        test_end_date              
    )
    
    transformer_portfolio_values, transformer_weights = run_backtest(
        transformer_model, 
        "Transformer Strategy", 
        backtest_features_scaled_df, 
        original_log_returns, 
        num_assets, 
        asset_tickers_list, 
        initial_portfolio_value, 
        transaction_cost_bps, 
        lookback_window,
        test_start_date, 
        test_end_date
    )

    ew_portfolio_returns_raw = original_log_returns.loc[test_start_date:test_end_date].mean(axis=1)
    ew_portfolio_values = (1 + ew_portfolio_returns_raw).cumprod() * initial_portfolio_value

    static_mvo_train_returns = original_log_returns.loc[train_start_date:train_end_date]
    static_mvo_expected_returns = static_mvo_train_returns.mean() * 252
    static_mvo_covariance = static_mvo_train_returns.cov() * 252

    static_mvo_weights = None
    try:
        ef_static = EfficientFrontier(static_mvo_expected_returns, static_mvo_covariance, verbose=False)
        ef_static.add_constraint(lambda w: w >= 0)
        ef_static.add_constraint(lambda w: np.sum(w) == 1)
        raw_weights = ef_static.max_sharpe()
        static_mvo_weights_dict = ef_static.clean_weights(raw_weights, cutoff=1e-6)
        static_mvo_weights = np.array([static_mvo_weights_dict.get(ticker, 0.0) for ticker in asset_tickers_list])
        if np.sum(static_mvo_weights) > 0:
            static_mvo_weights /= np.sum(static_mvo_weights)
        else:
            static_mvo_weights = np.ones(num_assets) / num_assets
    except Exception as e:
        print(f"Static MVO optimization failed: {e}. Using equal weights for static MVO.")
        static_mvo_weights = np.ones(num_assets) / num_assets

    static_mvo_daily_returns = np.dot(original_log_returns.loc[test_start_date:test_end_date].values, static_mvo_weights)
    static_mvo_portfolio_values = (1 + pd.Series(static_mvo_daily_returns, index=original_log_returns.loc[test_start_date:test_end_date].index)).cumprod() * initial_portfolio_value

    lstm_metrics = evaluate_portfolio(lstm_portfolio_values, initial_portfolio_value)
    transformer_metrics = evaluate_portfolio(transformer_portfolio_values, initial_portfolio_value)
    ew_metrics = evaluate_portfolio(ew_portfolio_values, initial_portfolio_value)
    static_mvo_metrics = evaluate_portfolio(static_mvo_portfolio_values, initial_portfolio_value)

    print("\nSaving portfolio performance data to CSV files...")
    lstm_portfolio_values.to_csv("lstm_portfolio_values.csv")
    transformer_portfolio_values.to_csv("transformer_portfolio_values.csv")
    ew_portfolio_values.to_csv("ew_portfolio_values.csv")
    static_mvo_portfolio_values.to_csv("static_mvo_portfolio_values.csv")
    print("Portfolio performance data saved.")

    plt.figure(figsize=(14, 8))
    plt.plot(lstm_portfolio_values, label='LSTM Strategy')
    plt.plot(transformer_portfolio_values, label='Transformer Strategy')
    plt.plot(ew_portfolio_values, label='Equal Weight (EW) Benchmark', linestyle='--')
    plt.plot(static_mvo_portfolio_values, label='Static MVO Benchmark', linestyle=':')
    
    plt.title('Cumulative Portfolio Value (2018-2020)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    print("Saving cumulative portfolio value chart...")
    plt.savefig("cumulative_portfolio_value.png")
    print("Chart saved.")
    plt.show()

if __name__ == "__main__":
    main()