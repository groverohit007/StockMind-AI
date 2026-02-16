# Advanced Multi-Timeframe Stock Prediction System
# Add these enhanced functions to your logic.py

import pandas as pd
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator, CCIIndicator, AroonIndicator
from ta.volatility import AverageTrueRange, BollingerBands, KeltnerChannel, DonchianChannel
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, MFIIndicator, VolumeWeightedAveragePrice
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# PART 1: ADVANCED FEATURE ENGINEERING
# ============================================================================

def create_advanced_features(df, timeframe='1d'):
    """
    Create comprehensive technical features for robust predictions.
    
    Args:
        df: DataFrame with OHLCV data
        timeframe: '1h', '1d', '1w', '1mo' for time-aware features
    
    Returns:
        DataFrame with 80+ technical features
    """
    df = df.copy()
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    
    # ==================== TREND INDICATORS ====================
    
    # Moving Averages (Multiple Timeframes)
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'SMA_{period}'] = SMAIndicator(close, window=period).sma_indicator()
        df[f'EMA_{period}'] = EMAIndicator(close, window=period).ema_indicator()
    
    # Moving Average Convergence/Divergence
    macd = MACD(close)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Average Directional Index (Trend Strength)
    adx = ADXIndicator(high, low, close, window=14)
    df['ADX'] = adx.adx()
    df['ADX_Pos'] = adx.adx_pos()
    df['ADX_Neg'] = adx.adx_neg()
    
    # Commodity Channel Index
    df['CCI'] = CCIIndicator(high, low, close, window=20).cci()
    
    # Aroon Indicator
    aroon = AroonIndicator(close, window=25)
    df['Aroon_Up'] = aroon.aroon_up()
    df['Aroon_Down'] = aroon.aroon_down()
    df['Aroon_Indicator'] = aroon.aroon_indicator()
    
    # ==================== MOMENTUM INDICATORS ====================
    
    # RSI (Multiple Timeframes)
    for period in [7, 14, 21]:
        df[f'RSI_{period}'] = RSIIndicator(close, window=period).rsi()
    
    # Stochastic Oscillator
    stoch = StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()
    
    # Williams %R
    df['Williams_R'] = WilliamsRIndicator(high, low, close, lbp=14).williams_r()
    
    # Rate of Change
    for period in [9, 12, 25]:
        df[f'ROC_{period}'] = ROCIndicator(close, window=period).roc()
    
    # ==================== VOLATILITY INDICATORS ====================
    
    # Bollinger Bands
    bb = BollingerBands(close, window=20, window_dev=2)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df['BB_Mid'] = bb.bollinger_mavg()
    df['BB_Width'] = (df['BB_High'] - df['BB_Low']) / df['BB_Mid']
    df['BB_Position'] = (close - df['BB_Low']) / (df['BB_High'] - df['BB_Low'])
    
    # Average True Range
    df['ATR'] = AverageTrueRange(high, low, close, window=14).average_true_range()
    df['ATR_Percent'] = (df['ATR'] / close) * 100
    
    # Keltner Channels
    keltner = KeltnerChannel(high, low, close, window=20)
    df['Keltner_High'] = keltner.keltner_channel_hband()
    df['Keltner_Low'] = keltner.keltner_channel_lband()
    df['Keltner_Mid'] = keltner.keltner_channel_mband()
    
    # Donchian Channels
    donchian = DonchianChannel(high, low, close, window=20)
    df['Donchian_High'] = donchian.donchian_channel_hband()
    df['Donchian_Low'] = donchian.donchian_channel_lband()
    
    # Historical Volatility
    df['HV_10'] = close.pct_change().rolling(10).std() * np.sqrt(252)
    df['HV_30'] = close.pct_change().rolling(30).std() * np.sqrt(252)
    
    # ==================== VOLUME INDICATORS ====================
    
    # On-Balance Volume
    df['OBV'] = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    df['OBV_Change'] = df['OBV'].pct_change()
    
    # Chaikin Money Flow
    df['CMF'] = ChaikinMoneyFlowIndicator(high, low, close, volume, window=20).chaikin_money_flow()
    
    # Money Flow Index
    df['MFI'] = MFIIndicator(high, low, close, volume, window=14).money_flow_index()
    
    # Volume Weighted Average Price (if intraday)
    if timeframe in ['1h', '15m', '30m']:
        try:
            df['VWAP'] = VolumeWeightedAveragePrice(high, low, close, volume).volume_weighted_average_price()
        except:
            df['VWAP'] = close  # Fallback
    
    # Volume Indicators
    df['Volume_SMA_20'] = volume.rolling(20).mean()
    df['Volume_Ratio'] = volume / df['Volume_SMA_20']
    df['Volume_Change'] = volume.pct_change()
    
    # ==================== PRICE PATTERNS ====================
    
    # Price Changes
    df['Returns'] = close.pct_change()
    df['Returns_5'] = close.pct_change(5)
    df['Returns_10'] = close.pct_change(10)
    df['Returns_20'] = close.pct_change(20)
    
    # High-Low Range
    df['HL_Range'] = (high - low) / close
    df['HL_Range_MA'] = df['HL_Range'].rolling(14).mean()
    
    # Gap Analysis
    df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    
    # Price Position
    df['Price_Position'] = (close - low) / (high - low)
    
    # Distance from Moving Averages
    for period in [20, 50, 200]:
        df[f'Dist_SMA_{period}'] = (close - df[f'SMA_{period}']) / df[f'SMA_{period}']
    
    # ==================== ADVANCED PATTERNS ====================
    
    # Support/Resistance Levels
    df['Swing_High'] = high.rolling(5, center=True).max()
    df['Swing_Low'] = low.rolling(5, center=True).min()
    
    # Trend Detection
    df['Higher_High'] = (high > high.shift(1)).astype(int)
    df['Lower_Low'] = (low < low.shift(1)).astype(int)
    
    # Momentum Acceleration
    df['Momentum'] = close.diff(10)
    df['Momentum_Change'] = df['Momentum'].diff()
    
    # Volatility Clustering
    df['Volatility_Cluster'] = df['Returns'].rolling(20).std()
    df['Volatility_Change'] = df['Volatility_Cluster'].pct_change()
    
    # ==================== TIME-BASED FEATURES ====================
    
    # Day of Week, Hour (if applicable)
    if hasattr(df.index, 'dayofweek'):
        df['DayOfWeek'] = df.index.dayofweek
        df['IsMonday'] = (df['DayOfWeek'] == 0).astype(int)
        df['IsFriday'] = (df['DayOfWeek'] == 4).astype(int)
    
    if timeframe in ['1h', '15m', '30m']:
        if hasattr(df.index, 'hour'):
            df['Hour'] = df.index.hour
            df['IsMarketOpen'] = ((df['Hour'] >= 9) & (df['Hour'] < 16)).astype(int)
            df['IsFirstHour'] = ((df['Hour'] >= 9) & (df['Hour'] < 10)).astype(int)
            df['IsLastHour'] = ((df['Hour'] >= 15) & (df['Hour'] < 16)).astype(int)
    
    # ==================== CROSS-INDICATOR SIGNALS ====================
    
    # Golden/Death Cross
    df['Golden_Cross'] = ((df['SMA_50'] > df['SMA_200']) & 
                          (df['SMA_50'].shift(1) <= df['SMA_200'].shift(1))).astype(int)
    df['Death_Cross'] = ((df['SMA_50'] < df['SMA_200']) & 
                         (df['SMA_50'].shift(1) >= df['SMA_200'].shift(1))).astype(int)
    
    # MACD Crossovers
    df['MACD_Cross_Up'] = ((df['MACD'] > df['MACD_Signal']) & 
                           (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))).astype(int)
    df['MACD_Cross_Down'] = ((df['MACD'] < df['MACD_Signal']) & 
                             (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))).astype(int)
    
    # RSI Divergence (Simplified)
    df['RSI_Oversold'] = (df['RSI_14'] < 30).astype(int)
    df['RSI_Overbought'] = (df['RSI_14'] > 70).astype(int)
    
    # Volume Surge
    df['Volume_Surge'] = (df['Volume_Ratio'] > 2).astype(int)
    
    # Fill NaN with forward/backward fill
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Drop any remaining NaN
    df = df.dropna()
    
    return df


# ============================================================================
# PART 2: MULTI-TIMEFRAME PREDICTION ENGINE
# ============================================================================

class MultiTimeframePredictionEngine:
    """
    Advanced prediction engine that creates separate models for different timeframes.
    """
    
    def __init__(self, ticker):
        self.ticker = ticker
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        
    def create_target(self, df, horizon, threshold=0.02):
        """
        Create target variable based on future returns.
        
        Args:
            df: DataFrame with price data
            horizon: Number of periods to look ahead
            threshold: Minimum % move to classify as BUY/SELL
        
        Returns:
            Target series (0=SELL, 1=HOLD, 2=BUY)
        """
        future_return = df['Close'].shift(-horizon) / df['Close'] - 1
        
        # Multi-class classification
        target = pd.Series(1, index=df.index)  # Default: HOLD
        target[future_return > threshold] = 2  # BUY
        target[future_return < -threshold] = 0  # SELL
        
        return target
    
    def create_stacked_ensemble(self):
        """
        Create a sophisticated stacked ensemble model.
        """
        # Level 0 models (Base learners)
        base_models = [
            ('rf', RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )),
            ('gb', GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=7,
                min_samples_split=20,
                min_samples_leaf=10,
                subsample=0.8,
                random_state=42
            )),
            ('hgb', HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.05,
                max_depth=7,
                random_state=42
            )),
            ('et', ExtraTreesClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )),
            ('ada', AdaBoostClassifier(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42
            ))
        ]
        
        # Level 1 model (Meta-learner)
        meta_model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        
        # Stacking Classifier
        stacking_model = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
            cv=5,
            n_jobs=-1
        )
        
        return stacking_model
    
    def train_timeframe_model(self, df, timeframe, horizon):
        """
        Train model for specific timeframe.
        
        Args:
            df: DataFrame with features
            timeframe: '1h', '1d', '1w', '1mo'
            horizon: Periods ahead to predict
        """
        print(f"Training {timeframe} model (horizon={horizon})...")
        
        # Create target
        df['Target'] = self.create_target(df, horizon)
        
        # Remove rows where target is NaN (last horizon rows)
        df = df[:-horizon].copy()
        
        # Get feature columns (exclude OHLCV and Target)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Target', 
                       'Adj Close', 'Dividends', 'Stock Splits']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols]
        y = df['Target']
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Train/Test Split (80/20, time-series aware)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        scaler = RobustScaler()  # Better for outliers
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = self.create_stacked_ensemble()
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        y_pred_proba_test = model.predict_proba(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'precision': precision_score(y_test, y_pred_test, average='weighted'),
            'recall': recall_score(y_test, y_pred_test, average='weighted'),
            'f1': f1_score(y_test, y_pred_test, average='weighted')
        }
        
        # Try to calculate AUC (for binary/multi-class)
        try:
            metrics['auc'] = roc_auc_score(y_test, y_pred_proba_test, 
                                          multi_class='ovr', average='weighted')
        except:
            metrics['auc'] = None
        
        # Cross-validation score
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=tscv, scoring='accuracy')
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        # Feature importance (from RandomForest in ensemble)
        try:
            rf_model = model.named_estimators_['rf']
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.feature_importance[timeframe] = feature_importance
        except:
            pass
        
        # Store model and scaler
        self.models[timeframe] = {
            'model': model,
            'feature_cols': feature_cols,
            'horizon': horizon
        }
        self.scalers[timeframe] = scaler
        self.performance_metrics[timeframe] = metrics
        
        print(f"✅ {timeframe} Model Trained:")
        print(f"   Train Accuracy: {metrics['train_accuracy']:.3f}")
        print(f"   Test Accuracy: {metrics['test_accuracy']:.3f}")
        print(f"   CV Score: {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")
        print(f"   F1 Score: {metrics['f1']:.3f}")
        
        return model, scaler, metrics
    
    def predict_all_timeframes(self, current_data):
        """
        Make predictions for all trained timeframes.
        
        Args:
            current_data: DataFrame with latest data
        
        Returns:
            Dictionary with predictions for each timeframe
        """
        predictions = {}
        
        for timeframe, model_dict in self.models.items():
            model = model_dict['model']
            feature_cols = model_dict['feature_cols']
            scaler = self.scalers[timeframe]
            
            # Get latest features
            X_latest = current_data[feature_cols].iloc[-1:]
            
            # Handle inf/nan
            X_latest = X_latest.replace([np.inf, -np.inf], np.nan)
            X_latest = X_latest.fillna(0)
            
            # Scale
            X_scaled = scaler.transform(X_latest)
            
            # Predict
            pred_class = model.predict(X_scaled)[0]
            pred_proba = model.predict_proba(X_scaled)[0]
            
            # Map to signal
            signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            signal = signal_map[pred_class]
            
            # Confidence is the probability of the predicted class
            confidence = pred_proba[pred_class]
            
            predictions[timeframe] = {
                'signal': signal,
                'confidence': confidence,
                'probabilities': {
                    'SELL': pred_proba[0],
                    'HOLD': pred_proba[1],
                    'BUY': pred_proba[2]
                },
                'metrics': self.performance_metrics[timeframe]
            }
        
        return predictions


# ============================================================================
# PART 3: MAIN PREDICTION FUNCTION (Replace existing train_consensus_model)
# ============================================================================

def train_multi_timeframe_model(ticker, use_cache=True):
    """
    Train comprehensive multi-timeframe prediction models.
    
    Args:
        ticker: Stock symbol
        use_cache: Use cached models if available
    
    Returns:
        prediction_engine: Trained MultiTimeframePredictionEngine
        all_predictions: Dict with predictions for all timeframes
        processed_data: Dict with processed DataFrames for each timeframe
    """
    
    engine = MultiTimeframePredictionEngine(ticker)
    processed_data = {}
    
    # Define timeframes and their configurations
    timeframe_configs = {
        'hourly': {
            'interval': '1h',
            'period': '1mo',
            'horizon': 24,  # Predict 24 hours ahead
            'label': '24-Hour'
        },
        'daily': {
            'interval': '1d',
            'period': '2y',
            'horizon': 5,  # Predict 5 days ahead
            'label': 'Weekly'
        },
        'weekly': {
            'interval': '1d',
            'period': '5y',
            'horizon': 20,  # Predict ~1 month ahead (20 trading days)
            'label': 'Monthly'
        },
        'monthly': {
            'interval': '1d',
            'period': '10y',
            'horizon': 60,  # Predict ~3 months ahead (60 trading days)
            'label': 'Quarterly'
        }
    }
    
    # Train models for each timeframe
    for tf_name, config in timeframe_configs.items():
        try:
            # Fetch data
            data = get_data(ticker, period=config['period'], interval=config['interval'])
            
            if data is None or len(data) < 100:
                print(f"⚠️ Insufficient data for {tf_name} model")
                continue
            
            # Create features
            processed = create_advanced_features(data, timeframe=config['interval'])
            
            if len(processed) < config['horizon'] + 50:
                print(f"⚠️ Not enough data after feature engineering for {tf_name}")
                continue
            
            # Train model
            engine.train_timeframe_model(
                processed,
                timeframe=tf_name,
                horizon=config['horizon']
            )
            
            processed_data[tf_name] = processed
            
        except Exception as e:
            print(f"❌ Error training {tf_name} model: {str(e)}")
            continue
    
    # Make predictions with all models
    all_predictions = {}
    
    for tf_name in engine.models.keys():
        if tf_name in processed_data:
            preds = engine.predict_all_timeframes(processed_data[tf_name])
            all_predictions.update(preds)
    
    return engine, all_predictions, processed_data


# ============================================================================
# PART 4: ENHANCED PREDICTION DISPLAY FUNCTION
# ============================================================================

def get_multi_timeframe_predictions(ticker):
    """
    Get comprehensive predictions for all timeframes.
    
    Returns:
        Dictionary with detailed predictions and metrics
    """
    
    print(f"\n🤖 Training Multi-Timeframe AI Models for {ticker}...")
    print("=" * 60)
    
    engine, predictions, processed_data = train_multi_timeframe_model(ticker)
    
    # Format output
    result = {
        'ticker': ticker,
        'timestamp': pd.Timestamp.now(),
        'predictions': predictions,
        'engine': engine,
        'processed_data': processed_data
    }
    
    # Summary
    print("\n📊 PREDICTION SUMMARY")
    print("=" * 60)
    
    for tf, pred in predictions.items():
        print(f"\n{tf.upper()}:")
        print(f"  Signal: {pred['signal']} (Confidence: {pred['confidence']*100:.1f}%)")
        print(f"  Probabilities: BUY={pred['probabilities']['BUY']*100:.1f}% | "
              f"HOLD={pred['probabilities']['HOLD']*100:.1f}% | "
              f"SELL={pred['probabilities']['SELL']*100:.1f}%")
        print(f"  Model Accuracy: {pred['metrics']['test_accuracy']*100:.1f}%")
    
    return result
