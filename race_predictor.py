import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import time
import lightgbm as lgb

def clean_order(order):
    """着順を数値に変換する関数"""
    try:
        return float(order)
    except (ValueError, TypeError):
        return 99

def process_passing_order(passing):
    """通過順位を処理する関数"""
    try:
        return int(passing.split('-')[-1])
    except:
        return 0

def remove_outliers(df, column):
    """外れ値を除去する関数"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))]
    return df

def enhance_features(df):
    """特徴量を強化する関数"""
    try:
        # 1. 過去のレース成績（馬名でソート）
        df = df.sort_values(['馬名']).reset_index(drop=True)
        df['前走着順'] = df.groupby('馬名')['着順'].shift(1)
        df['前々走着順'] = df.groupby('馬名')['着順'].shift(2)
        df['平均着順'] = df.groupby('馬名')['着順'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
        
        # 2. タイムに関する特徴
        df['タイム_秒'] = df['タイム'].apply(
            lambda x: sum(float(i) * 60 ** idx for idx, i in enumerate(reversed(str(x).split(':')))) 
            if pd.notna(x) and str(x).count(':') == 1 else np.nan
        )
        df['前走タイム'] = df.groupby('馬名')['タイム_秒'].shift(1)
        df['前々走タイム'] = df.groupby('馬名')['タイム_秒'].shift(2)
        df['平均タイム'] = df.groupby('馬名')['タイム_秒'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
        
        # 3. 上りタイムの特徴
        df['前走上り'] = df.groupby('馬名')['上り'].shift(1)
        df['前々走上り'] = df.groupby('馬名')['上り'].shift(2)
        df['平均上り'] = df.groupby('馬名')['上り'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
        
        # 4. 着差に関する特徴
        df['前走着差'] = df.groupby('馬名')['着差'].shift(1)
        df['前々走着差'] = df.groupby('馬名')['着差'].shift(2)
        
        # 5. コース適性
        df['馬場適性'] = df.groupby(['馬名', '馬場'])['着順'].transform('mean')
        df['距離適性'] = df.groupby(['馬名', '距離'])['着順'].transform('mean')
        df['コース適性'] = df.groupby(['馬名', '場名', '距離'])['着順'].transform('mean')
        
        # 6. 騎手関連の特徴
        df['騎手勝率'] = df.groupby('騎手')['着順'].transform(lambda x: (x == 1).mean())
        df['騎手連対率'] = df.groupby('騎手')['着順'].transform(lambda x: (x <= 2).mean())
        df['騎手複勝率'] = df.groupby('騎手')['着順'].transform(lambda x: (x <= 3).mean())
        
        # 7. 馬体重関連の特徴
        df['馬体重変化率'] = pd.to_numeric(df['増減'], errors='coerce') / pd.to_numeric(df['馬体重'], errors='coerce')
        df['体重/斤量比'] = pd.to_numeric(df['馬体重'], errors='coerce') / pd.to_numeric(df['斤量'], errors='coerce')
        df['標準体重との差'] = df.groupby('馬名')['馬体重'].transform(
            lambda x: pd.to_numeric(x, errors='coerce') - pd.to_numeric(x, errors='coerce').mean()
        )
        
        print("特徴量の生成が完了しました")
        return df
        
    except Exception as e:
        print(f"特徴量生成中にエラーが発生しました: {str(e)}")
        raise

def improve_preprocessing(df, is_new_race=False):
    """データ前処理を改善する関数"""
    if not is_new_race:
        # 既存レースデータの処理
        df['タイム_秒'] = df['タイム'].apply(
            lambda x: sum(float(i) * 60 ** idx for idx, i in enumerate(reversed(str(x).split(':')))) 
            if pd.notna(x) else np.nan
        )
    
    # 1. 外れ値の処理
    numerical_columns = ['馬体重', '増減']
    for col in numerical_columns:
        if col in df.columns:
            df = remove_outliers(df, col)
    
    # 2. 欠損値の処理
    for col in numerical_columns:
        if col in df.columns:
            df[col] = df.groupby(['場名', '距離'])[col].transform(
                lambda x: x.fillna(x.mean())
            )
    
    # 3. カテゴリカル変数の欠損値処理
    categorical_columns = ['場名', '種別', '回り', '天候', '馬場', '性別']
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    return df

def convert_margin_to_numeric(margin):
    """着差を数値に変換する関数"""
    if pd.isna(margin) or margin == '':
        return np.nan
    
    # 特殊な着差表記を数値に変換
    margin_dict = {
        'クビ': 0.3,
        'ハナ': 0.2,
        'アタマ': 0.4,
        '大': 3.0,
    }
    
    try:
        # 既に数値の場合
        return float(margin)
    except (ValueError, TypeError):
        # 特殊表記の場合
        if margin in margin_dict:
            return margin_dict[margin]
        
        # 1.1/4のような表記の場合
        try:
            if '/' in margin:
                parts = margin.split('/')
                if len(parts) == 2:
                    whole = float(parts[0].split('.')[0])
                    frac = float(parts[0].split('.')[1]) if '.' in parts[0] else 0
                    decimal = float(parts[0].split('.')[1]) / float(parts[1]) if '.' in parts[0] else float(parts[0]) / float(parts[1])
                    return whole + frac + decimal
            return float(margin)
        except:
            return np.nan

def load_and_prepare_data(directory_path):
    """データ読み込みと前処理を行う関数"""
    # CSVファイルの実際のカラム名に合わせて修正
    usecols = [
        '着順', '着差', '場名', '種別', '距離', '回り', '天候', '馬場',
        '枠番', '馬番', '馬名', '性別', '年齢', '斤量', 'タイム', '通過',
        '上り', '馬体重', '増減', '騎手'
    ]
    
    all_files = glob.glob(os.path.join(directory_path, "**/*.csv"), recursive=True)
    df = pd.concat(
        (pd.read_csv(f, usecols=usecols) for f in all_files),
        ignore_index=True
    )
    
    print(f"読み込んだファイル数: {len(all_files)}")
    print(f"総データ行数: {len(df)}")
    
    # 着差を数値に変換
    df['着差'] = df['着差'].apply(convert_margin_to_numeric)
    
    # その他の数値型への変換
    numeric_columns = ['枠番', '馬番', '年齢', '斤量', '馬体重', '増減', '距離']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 基本的な前処理
    df['着順'] = df['着順'].apply(clean_order)
    df = df[df['着順'] < 99]
    
    # 特徴量の強化
    df = enhance_features(df)
    
    # カテゴリカル変数のエンコーディング
    le = LabelEncoder()
    categorical_columns = ['場名', '種別', '回り', '天候', '馬場', '性別', '騎手']
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col].astype(str))
    
    # 使用する特徴量を数値型とカテゴリカル型に分類
    numeric_features = [
        '枠番', '馬番', '年齢', '斤量', '距離',
        '馬体重', '増減', '前走着順', '前々走着順',
        '平均着順', '前走タイム', '前々走タイム', '平均タイム',
        '前走上り', '前々走上り', '平均上り', '前走着差',
        '前々走着差', '馬場適性', '距離適性', 'コース適性',
        '騎手勝率', '騎手連対率', '騎手複勝率', '馬体重変化率',
        '体重/斤量比', '標準体重との差'
    ]
    
    categorical_features = [
        '場名', '種別', '回り', '天候', '馬場', '性別'
    ]
    
    features = numeric_features + categorical_features
    
    # 欠損値の処理（数値型とカテゴリカル型で別々に処理）
    for col in numeric_features:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # 無限値をNaNに変換し、その後中央値で埋める
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in numeric_features:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    X = df[features]
    y = df['着順']
    
    print("データの前処理が完了しました")
    return X, y, df

def train_advanced_model(X, y):
    """モデルの学習を行う関数"""
    # データの分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # スケーリング
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 特徴量の重み付け
    feature_weights = {
        '前走着順': 1.5,
        '前々走着順': 1.2,
        '平均着順': 1.5,
        '騎手勝率': 1.5,
        '馬場適性': 1.5,
        '距離適性': 1.8,
        'コース適性': 1.5,
        '馬体重': 1.2,
        '増減': 1.2,
    }
    
    # 重み付けの適用
    sample_weights = np.ones(len(X_train))
    for feature, weight in feature_weights.items():
        if feature in X.columns:
            feature_idx = list(X.columns).index(feature)
            X_train_scaled[:, feature_idx] *= weight
            X_test_scaled[:, feature_idx] *= weight
    
    # LightGBMモデル
    print("\nLightGBMモデルのチューニング中...")
    lgb_model = LGBMRegressor(
        objective='regression',
        metric='rmse',
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        learning_rate=0.1,
        n_estimators=200,
        importance_type='gain'
    )
    
    # LightGBMの学習
    lgb_model.fit(
        X_train_scaled, 
        y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test_scaled, y_test)],
        eval_metric='rmse',
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=0)  # 進捗表示を無効化
        ]
    )
    
    # 特徴量の重要度を表示
    print("\nLightGBMの特徴量重要度:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': lgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance.head(10))
    
    # XGBoostモデル
    print("\nXGBoostモデルのチューニング中...")
    xgb_model = XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse',
        colsample_bytree=0.8,
        subsample=0.8,
        learning_rate=0.1,
        n_estimators=200,
        importance_type='gain',
        early_stopping_rounds=50
    )
    
    # XGBoostの学習
    xgb_model.fit(
        X_train_scaled, 
        y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False
    )
    
    # XGBoostの特徴量重要度を表示
    print("\nXGBoostの特徴量重要度:")
    xgb_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(xgb_importance.head(10))
    
    # RandomForestモデル
    print("\nRandomForestモデルのチューニング中...")
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        n_jobs=-1  # 並列処理を有効化
    )
    
    # RandomForestの学習
    rf_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    
    # モデルを辞書に格納
    best_models = {
        'LightGBM': lgb_model,
        'XGBoost': xgb_model,
        'RandomForest': rf_model
    }
    
    return best_models, scaler, X_test_scaled, y_test

def validate_model(X, y, model):
    """時系列クロスバリデーションを実装する関数"""
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = np.sqrt(mean_squared_error(y_test, y_pred))
        scores.append(score)
    
    return np.mean(scores), np.std(scores)

def predict_new_race(model, scaler, new_race_data, historical_data):
    """新しいレースを予測する関数"""
    try:
        # CSVファイルの列名を確認
        print("\n読み込んだデータの列名:")
        print(new_race_data.columns.tolist())
        
        # 列名のクリーニング（空白文字の削除）
        new_race_data.columns = new_race_data.columns.str.strip()
        
        # 数値型への変換
        numeric_columns = {
            '枠番': '枠番',
            '馬番': '馬番',
            '年齢': '年齢',
            '斤量': '斤量',
            '馬体重': '馬体重',
            '増減': '増減',
            '距離': '距離'
        }
        
        for col in numeric_columns.keys():
            if col in new_race_data.columns:
                new_race_data[col] = pd.to_numeric(new_race_data[col], errors='coerce')
        
        # 必要なカラムの確認と追加
        new_race_data['着順'] = np.nan
        new_race_data['着差'] = np.nan
        new_race_data['タイム'] = np.nan
        new_race_data['通過'] = np.nan
        new_race_data['上り'] = np.nan
        
        # 過去データと結合して特徴量を作成
        combined_data = pd.concat([historical_data, new_race_data], ignore_index=True)
        processed_data = enhance_features(combined_data)
        
        # 新しいレースのデータのみを抽出（最後の18行）
        processed_data = processed_data.tail(len(new_race_data)).copy()
        
        # カテゴリカル変数のエンコーディング
        le = LabelEncoder()
        categorical_columns = ['場名', '種別', '回り', '天候', '馬場', '性別']
        for col in categorical_columns:
            if col in processed_data.columns:
                processed_data[col] = le.fit_transform(processed_data[col].astype(str))
        
        # 使用する特徴量
        numeric_features = [
            '枠番', '馬番', '年齢', '斤量', '距離',
            '馬体重', '増減', '前走着順', '前々走着順',
            '平均着順', '前走タイム', '前々走タイム', '平均タイム',
            '前走上り', '前々走上り', '平均上り', '前走着差',
            '前々走着差', '馬場適性', '距離適性', 'コース適性',
            '騎手勝率', '騎手連対率', '騎手複勝率', '馬体重変化率',
            '体重/斤量比', '標準体重との差'
        ]
        
        categorical_features = [
            '場名', '種別', '回り', '天候', '馬場', '性別'
        ]
        
        features = numeric_features + categorical_features
        
        # 欠損値の処理
        for col in numeric_features:
            if col in processed_data.columns:
                processed_data[col] = processed_data[col].fillna(processed_data[col].median())
        
        for col in categorical_features:
            if col in processed_data.columns:
                processed_data[col] = processed_data[col].fillna(processed_data[col].mode()[0])
        
        # 無限値の処理
        processed_data = processed_data.replace([np.inf, -np.inf], np.nan)
        for col in numeric_features:
            if col in processed_data.columns:
                processed_data[col] = processed_data[col].fillna(processed_data[col].median())
        
        # 特徴量の選択
        X_new = processed_data[features].copy()
        
        # スケーリング
        X_new_scaled = scaler.transform(X_new)
        
        # 予測
        predictions = model.predict(X_new_scaled)
        
        # 結果をデータフレームにまとめる
        results = pd.DataFrame({
            '馬名': new_race_data['馬名'].values,
            '予測着順': predictions,
            '枠番': new_race_data['枠番'].values,
            '馬番': new_race_data['馬番'].values,
            '性別': new_race_data['性別'].values,
            '年齢': new_race_data['年齢'].values,
            '斤量': new_race_data['斤量'].values,
            '馬体重': new_race_data['馬体重'].values,
            '増減': new_race_data['増減'].values
        }).sort_values('予測着順')
        
        # 予測着順を1-18の範囲にスケーリング
        num_horses = len(predictions)
        min_pred = results['予測着順'].min()
        max_pred = results['予測着順'].max()
        results['予測着順'] = ((results['予測着順'] - min_pred) / 
                          (max_pred - min_pred) * (num_horses - 1) + 1).round(2)
        
        # 上位3頭に印をつける
        results['印'] = ''
        results.iloc[0:3, results.columns.get_loc('印')] = ['◎', '○', '▲']
        
        # 結果の表示形式を整える
        results = results.round({
            '予測着順': 2,
            '枠番': 0,
            '馬番': 0,
            '年齢': 0,
            '斤量': 1,
            '馬体重': 0,
            '増減': 0
        })
        
        return results
        
    except Exception as e:
        print(f"予測処理中にエラーが発生しました: {str(e)}")
        print("処理中のデータ形状:")
        print(f"new_race_data: {new_race_data.shape}")
        print(f"processed_data: {processed_data.shape if 'processed_data' in locals() else 'Not created'}")
        print(f"X_new: {X_new.shape if 'X_new' in locals() else 'Not created'}")
        print("\n列名の確認:")
        print(f"new_race_data columns: {new_race_data.columns.tolist()}")
        if 'processed_data' in locals():
            print(f"processed_data columns: {processed_data.columns.tolist()}")
        raise

def debug_csv_contents(file_path):
    """CSVファイルの内容を確認するための関数"""
    try:
        # ファイルの存在確認
        if not os.path.exists(file_path):
            print(f"ファイルが見つかりません: {file_path}")
            return
        
        # CSVファイルの内容を確認
        df = pd.read_csv(file_path)
        print("\nCSVファイルの情報:")
        print(f"列名: {df.columns.tolist()}")
        print("\n最初の数行:")
        print(df.head().to_string())
        print(f"\n行数: {len(df)}")
        
    except Exception as e:
        print(f"CSVファイル確認中にエラーが発生: {str(e)}")

if __name__ == "__main__":
    try:
        # データの読み込みと前処理
        directory_path = 'data'
        start_time = time.time()
        
        X, y, df = load_and_prepare_data(directory_path)
        
        # モデルの訓練
        best_models, scaler, X_test, y_test = train_advanced_model(X, y)
        
        # 処理時間の表示
        end_time = time.time()
        print(f"\n処理時間: {end_time - start_time:.2f}秒")
        
        # 新しいレースの予測
        new_race_file = 'new_race.csv'
        if os.path.exists(new_race_file):
            print("\n新しいレースの予測結果:")
            
            # CSVファイルを読み込む
            new_race_data = pd.read_csv(new_race_file)
            
            # 各モデルの予測結果を表示
            for name, model in best_models.items():
                print(f"\n{name}モデルによる予測:")
                predictions = predict_new_race(model, scaler, new_race_data, df)
                print(predictions.to_string(index=False))
                
        # CSVファイルの内容を確認
        new_race_file = 'new_race.csv'
        if os.path.exists(new_race_file):
            debug_csv_contents(new_race_file)
                
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        print("\n新しいレースのCSVファイルの列名を確認してください。")
        print("必要な列: 馬名, 枠番, 馬番, 性齢, 斤量, 騎手, 馬体重, 増減, 場名, 種別, 距離, 回り, 天候, 馬場")