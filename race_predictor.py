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
import difflib
import traceback

def clean_order(order):
    """着順を数値に変換する関数"""
    try:
        # 文字列の場合は数値に変換
        if isinstance(order, str):
            # 中止、失格、取消などの場合
            if any(x in order for x in ['中', '失', '取', '除']):
                return 99
            # 数値として扱える場合
            return float(order)
        # 既に数値の場合
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

def save_jockey_names(historical_data):
    """騎手名の一覧をファイルに保存する関数"""
    jockey_names = historical_data['騎手'].unique()  # 騎手名のユニークなリストを取得
    with open('jockey_names.txt', 'w', encoding='utf-8') as file:
        for name in jockey_names:
            file.write(name + '\n')

def calculate_jockey_stats(historical_data):
    """騎手の成績を計算する関数"""
    # 騎手名の一覧をファイルに保存
    #save_jockey_names(historical_data)
    
    # 着順に基づいて1着、2着以内、3着以内のフラグを作成
    historical_data = historical_data.copy()
    historical_data['１着'] = (historical_data['着順'] == 1).astype(int)
    historical_data['２着以内'] = (historical_data['着順'] <= 2).astype(int)
    historical_data['３着以内'] = (historical_data['着順'] <= 3).astype(int)
    
    # 騎手ごとの統計を計算
    jockey_stats = historical_data.groupby('騎手').agg({
        '着順': ['count', 'mean'],
        '１着': ['sum', lambda x: x.sum() / len(x)],  # 勝率
        '２着以内': ['sum', lambda x: x.sum() / len(x)],  # 連対率
        '３着以内': ['sum', lambda x: x.sum() / len(x)]   # 複勝率
    }).round(3)
    
    jockey_stats.columns = [
        'レース数', '平均着順', '勝利数', '勝率',
        '連対数', '連対率', '複勝数', '複勝率'
    ]
    
    return jockey_stats

def preprocess_race_data(data):
    """レースデータの前処理を行う関数"""
    processed_data = data.copy()
    
    # 必須でないカラムの初期化
    optional_columns = ['タイム', '着差', '上り', '通過']
    for col in optional_columns:
        if col not in processed_data.columns:
            processed_data[col] = np.nan
    
    # タイムの処理（文字列から秒数への変換）
    def convert_time_to_seconds(time_str):
        try:
            if pd.isna(time_str):
                return np.nan
            # 1:33.9 形式を秒数に変換
            if isinstance(time_str, str):
                minutes, seconds = time_str.split(':')
                return float(minutes) * 60 + float(seconds)
            return float(time_str)
        except:
            return np.nan
    
    # タイムを秒数に変換
    processed_data['タイム'] = processed_data['タイム'].apply(convert_time_to_seconds)
    
    # 数値型への変換
    numeric_columns = {
        '着順': float,
        '枠番': int,
        '馬番': int,
        '年齢': int,
        '斤量': float,
        '馬体重': float,
        '増減': float,
        '距離': float,
        '着差': float,
        '上り': float
    }
    
    # その他の数値カラムを変換（存在するカラムのみ）
    for col, dtype in numeric_columns.items():
        if col in processed_data.columns:
            processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce').astype(dtype)
    
    # 欠損値の処理
    processed_data = processed_data.replace([np.inf, -np.inf], np.nan)
    
    return processed_data

def create_race_features(df):
    """モデルで使用する特徴量のリストを返す関数"""
    base_features = [
        '馬番', '斤量', '馬体重', '増減',
        '騎手勝率', '騎手複勝率', '騎手平均着順',
        '平均着順', '平均タイム', '平均上り',
        'コース適性', '距離適性', '馬場適性'
    ]
    
    # 利用可能な特徴量のみを返す
    available_features = [f for f in base_features if f in df.columns]
    print(f"\n利用可能な特徴量: {available_features}")
    
    return available_features

def calculate_horse_stats(historical_data):
    """馬の統計情報を計算する関数"""
    try:
        # 馬の平均タイム
        horse_time = historical_data.groupby('馬名')['タイム_秒'].mean().reset_index()
        horse_time.columns = ['馬名', '平均タイム']
        
        # 馬の平均上り
        horse_last = historical_data.groupby('馬名')['上り'].mean().reset_index()
        horse_last.columns = ['馬名', '平均上り']
        
        # 馬の平均着順
        horse_rank = historical_data.groupby('馬名')['着順'].mean().reset_index()
        horse_rank.columns = ['馬名', '平均着順']
        
        # 馬の統計を結合
        horse_stats = horse_time.merge(horse_last, on='馬名', how='outer').merge(horse_rank, on='馬名', how='outer')
        
        return horse_stats
    except Exception as e:
        print(f"馬の統計計算中にエラーが発生: {str(e)}")
        raise

def merge_horse_stats(data, horse_stats):
    """馬の統計情報をデータフレームにマージする関数"""
    try:
        # 新しいデータに馬の統計を結合
        merged_data = data.merge(horse_stats, on='馬名', how='left')
        
        # 統計のない馬は平均値で補完
        for col in ['平均タイム', '平均上り', '平均着順']:
            if col in merged_data.columns:
                merged_data[col] = merged_data[col].fillna(horse_stats[col].mean())
        
        return merged_data
    except Exception as e:
        print(f"馬の統計マージ中にエラーが発生: {str(e)}")
        raise

def enhance_features(data, historical_data=None):
    try:
        processed_data = data.copy()
        
        if historical_data is not None:
            print(f"\n処理対象レース種別: {processed_data['種別'].iloc[0]}")
            
            # 馬の過去成績を計算
            horse_stats = calculate_horse_stats(historical_data)
            processed_data = merge_horse_stats(processed_data, horse_stats)
            
            # 騎手の成績を計算
            jockey_stats = calculate_jockey_stats(historical_data)
            
            # 騎手の統計を結合
            for stat in ['勝率', '複勝率', '平均着順']:
                col_name = f'騎手{stat}'
                if col_name in jockey_stats.columns:
                    processed_data = processed_data.merge(
                        jockey_stats[['騎手', col_name]], 
                        on='騎手', 
                        how='left'
                    )
                    processed_data[col_name] = processed_data[col_name].fillna(jockey_stats[col_name].mean())
            
            # コース適性、距離適性、馬場適性を計算
            for feature in ['場名', '距離', '馬場']:
                # データ型を事前に統一
                if feature == '距離':
                    # 距離を数値型に変換
                    historical_data[feature] = pd.to_numeric(historical_data[feature].astype(str).replace(r'[^\d.]', '', regex=True), errors='coerce')
                    processed_data[feature] = pd.to_numeric(processed_data[feature].astype(str).replace(r'[^\d.]', '', regex=True), errors='coerce')
                
                # 平均着順を計算
                feature_rank = historical_data.groupby(['馬名', feature])['着順'].mean().reset_index()
                feature_rank.columns = ['馬名', feature, f'{feature}_平均着順']
                
                # レース数を計算
                feature_count = historical_data.groupby(['馬名', feature])['着順'].count().reset_index()
                feature_count.columns = ['馬名', feature, f'{feature}_レース数']
                
                # 統計を結合
                feature_stats = feature_rank.merge(feature_count, on=['馬名', feature])
                
                # 適性を計算
                feature_stats[f'{feature}_適性'] = 1 / (feature_stats[f'{feature}_平均着順'] * 
                                                    np.log1p(feature_stats[f'{feature}_レース数']))
                
                # 新しいデータに結合（データ型を明示的に指定）
                if feature == '距離':
                    feature_stats['距離'] = feature_stats['距離'].astype(float)
                    processed_data['距離'] = processed_data['距離'].astype(float)
                
                processed_data = processed_data.merge(
                    feature_stats[['馬名', feature, f'{feature}_適性']], 
                    on=['馬名', feature], 
                    how='left'
                )
                
                # 欠損値を平均値で補完
                processed_data[f'{feature}_適性'] = processed_data[f'{feature}_適性'].fillna(
                    feature_stats[f'{feature}_適性'].mean()
                )
            
            # カラム名を変更
            processed_data = processed_data.rename(columns={
                '場名_適性': 'コース適性',
                '距離_適性': '距離適性',
                '馬場_適性': '馬場適性'
            })
            
            # NaNを平均値で補完
            numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if processed_data[col].isna().any():
                    processed_data[col] = processed_data[col].fillna(processed_data[col].mean())
            
            # 前走の距離を取得と距離変化の計算
            processed_data['前走距離'] = historical_data['距離'].shift(1)
            processed_data['距離変化'] = processed_data.apply(
                lambda x: '延長' if x['距離'] > x['前走距離'] else ('短縮' if x['距離'] < x['前走距離'] else '同じ'),
                axis=1
            )
        
        return processed_data
        
    except Exception as e:
        print(f"特徴量生成中にエラーが発生: {str(e)}")
        traceback.print_exc()
        raise

def improve_preprocessing(df, is_new_race=False):
    """データ前処理を改善する関数"""
    try:
        if not is_new_race:
            # タイムを秒数に変換
            df['タイム_秒'] = df['タイム'].apply(
                lambda x: sum(float(i) * 60 ** idx for idx, i in enumerate(reversed(str(x).split(':')))) 
                if pd.notna(x) else np.nan
            )
            
            # 着差を数値に変換
            df['着差'] = df['着差'].apply(convert_margin_to_numeric)
            
            # 上りを数値に変換
            df['上り'] = pd.to_numeric(df['上り'], errors='coerce')
        
        # 騎手名の処理: 半角大文字、スペースなしに変換
        if '騎手' in df.columns:
            df['騎手'] = df['騎手'].str.replace(' ', '').str.upper()
        
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
        
        # 4. 距離を数値型に変換
        df['距離'] = pd.to_numeric(df['距離'], errors='coerce')
        
        return df
        
    except Exception as e:
        print(f"前処理中にエラーが発生しました: {str(e)}")
        raise

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
    try:
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"ディレクトリが見つかりません: {directory_path}")
        
        all_files = glob.glob(os.path.join(directory_path, "**/*.csv"), recursive=True)
        if not all_files:
            raise FileNotFoundError(f"CSVファイルが見つかりません: {directory_path}")
        
        # CSVファイルの検索
        usecols = [
            '着順', '着差', '場名', '種別', '距離', '回り', '天候', '馬場',
            '枠番', '馬番', '馬名', '性別', '年齢', '斤量', 'タイム', '通過',
            '上り', '馬体重', '増減', '騎手'
        ]
        
        # データフレームのリストを作成と結合
        dfs = []
        for file in all_files:
            try:
                df = pd.read_csv(file, usecols=usecols)
                dfs.append(df)
            except Exception as e:
                print(f"警告: {os.path.basename(file)} の読込みに失敗しました")
                continue
        
        if not dfs:
            raise ValueError("読み込み可能なCSVファイルがありません")
        
        # データフレームの結合
        df = pd.concat(dfs, ignore_index=True)
        
        # 障害レースを除外
        df = df[~df['種別'].str.contains('障', na=False)]
        print(f"障害レース除外後のデータ数: {len(df)}行")
        
        print(f"データセット（前処理前）: {len(df)}行")
        
        # 種別を明示的に判定
        df['種別'] = df['種別'].apply(
            lambda x: '芝' if '芝' in str(x) else 'ダート' if 'ダート' in str(x) else x
        )
        
        # 芝とダートのデータ数を表示
        print(f"芝レース数: {len(df[df['種別'] == '芝'])}行")
        print(f"ダートレース数: {len(df[df['種別'] == 'ダート'])}行")
        
        # 基本的な前処理
        df = improve_preprocessing(df)
        
        # 着順の前処理
        df['着順'] = df['着順'].apply(clean_order)
        df = df[df['着順'] != 99]  # 失格、中止などを除外
        df = df.dropna(subset=['着順'])  # 着順のNaNを除外
        
        # 特徴量の生成（種別とに分けて処理）
        df_turf = df[df['種別'] == '芝'].copy()
        df_dirt = df[df['種別'] == 'ダート'].copy()
        
        # それぞれの種別で特徴量を生成
        if len(df_turf) > 0:
            df_turf = enhance_features(df_turf, df_turf.copy())
        if len(df_dirt) > 0:
            df_dirt = enhance_features(df_dirt, df_dirt.copy())
        
        # データを再結合
        df = pd.concat([df_turf, df_dirt], ignore_index=True)
        
        # NaNを含む行を削除
        df = df.dropna()
        
        # 必要な特徴量の選択
        features = create_race_features(df)
        
        # 特徴量とターゲットの分離
        X = df[features].copy()
        y = df['着順'].copy()
        
        # 最終的なNaNチェック
        if X.isna().any().any():
            print("警告: 特徴量にNaNが含まれています")
            print(X.isna().sum()[X.isna().sum() > 0])
            X = X.fillna(X.mean())
        
        print(f"データセット（前処理後）: {len(df)}行")
        print(f"\n利用可能な特徴量: {features}")
        
        return X, y, df
        
    except Exception as e:
        print(f"データ準備中にエラーが発生しました: {str(e)}")
        raise

def train_advanced_model(X, y):
    """モデルの学習を行う関数"""
    # データの分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # スケーリング
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n=== モデル学習開始 ===")
    
    # 特徴量の重み付け
    feature_weights = {
        '馬番': 0.8,
        '斤量': 0.5,
        '馬体重': 0.3,
        '増減': 0.3,
        '騎手勝率': 0.7,
        '騎手複勝率': 0.7,
        '騎手平均着順': 0.5,
        '前走着順': 0.5,
        '前走タイム': 0.7,
        '前走上り': 0.7,
        '前走着差': 0.7,
        '前々走着順': 0.5,
        '前々走イム': 0.5,
        '前々走上り': 0.5,
        '前々走着差': 0.5,
        '平均着順': 0.4,
        '平均タイム': 0.4,
        '平均上り': 0.5,
        'コース適性': 0.7,
        '距離適性': 0.9,
        '馬場適性': 0.7,
        '距離変化': 0.6,  # 新たに追加した特徴量の重み
        # の特徴量も必要に応じて追加
    }
    
    # LightGBMモデル
    print("LightGBMモデルの学習中...")
    lgb_model = LGBMRegressor(
        objective='regression',
        metric='rmse',
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        learning_rate=0.1,
        n_estimators=200,
        importance_type='gain',
        verbose=-1
    )
    
    # 特徴量の重みを適用
    sample_weights = np.ones(len(X_train))
    for feature, weight in feature_weights.items():
        if feature in X_train.columns:
            sample_weights *= (1 + weight * np.abs(X_train[feature]))
    
    # 重みを正規化
    sample_weights = sample_weights / sample_weights.mean()
    
    # LightGBMの学習（重み付き）
    lgb_model.fit(
        X_train_scaled, 
        y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test_scaled, y_test)],
        eval_metric='rmse',
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=0)
        ]
    )
    
    # 特徴量の重要度を表示
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': lgb_model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    print("\nLightGBMの特徴量重要度:")
    print(feature_importance.to_string())
    
    # XGBoostモデル（同様に重み付け）
    print("\nXGBoostモデルの学習中...")
    xgb_model = XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse',
        colsample_bytree=0.8,
        subsample=0.8,
        learning_rate=0.1,
        n_estimators=200,
        importance_type='gain',
        early_stopping_rounds=50,
        verbosity=0
    )
    
    xgb_model.fit(
        X_train_scaled, 
        y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False
    )
    
    # RandomForestモデル（同様に重み付け）
    print("RandomForestモデルの学習中...")
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        n_jobs=-1,
        verbose=0
    )
    
    rf_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    
    print("=== モデル学習完了 ===\n")
    
    # モデルを辞書に格納
    best_models = {
        'LightGBM': lgb_model,
        'XGBoost': xgb_model,
        'RandomForest': rf_model
    }
    
    return best_models, scaler, X_test_scaled, y_test, feature_importance

def validate_model(X, y, model):
    """時系列ロスバリデーションを実装する関数"""
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

def predict_new_race(model, scaler, new_race_data, training_data, model_name):
    """新しいレースの予測を行う関数"""
    try:
        # 重複行の削除（完全な重複のみ）
        new_race_data = new_race_data.drop_duplicates(subset=['馬名'], keep='first')
        
        # 必須カラムの存在確認と欠損値���処理
        required_columns = ['馬名', '馬番', '騎手', '馬体重', '増減']
        for col in required_columns:
            if col not in new_race_data.columns:
                raise ValueError(f"必要なカラム '{col}' が不足しています")
            
            # 数値型カラムの欠損値を0で補完
            if col in ['馬体重', '増減', '馬番']:
                new_race_data[col] = pd.to_numeric(new_race_data[col], errors='coerce').fillna(0)
            # 文字列型カラムの欠損値を'不明'で補完
            elif col == '騎手':
                new_race_data[col] = new_race_data[col].fillna('不明')
        
        processed_data = enhance_features(new_race_data, training_data)
        required_features = create_race_features(processed_data)
        X_new = processed_data[required_features].copy()
        X_new = X_new.fillna(X_new.mean())
        X_new_scaled = scaler.transform(X_new)
        predictions = model.predict(X_new_scaled)
        predictions = np.clip(predictions, 1, 18)
        
        # 結果をデータフレームに格納
        results = pd.DataFrame({
            '馬名': new_race_data['馬名'],
            '予測着順': predictions,
            '馬番': new_race_data['馬番'],
            '騎手': new_race_data['騎手'],
            '馬体重': new_race_data['馬体重'],
            '増減': new_race_data['増減']
        })
        
        # 数値型カラムの安全な変換
        for col in ['馬体重', '増減', '馬番']:
            results[col] = pd.to_numeric(results[col], errors='coerce').fillna(0).astype(float)
        
        results = results.sort_values('予測着順')
        results['予測着順'] = results['予測着順'].round(1)
        results.index = range(1, len(results) + 1)
        
        return results, predictions
        
    except Exception as e:
        print(f"\n=== {model_name}モデルでエラーが発生しました ===")
        print(f"エラー内容: {str(e)}")
        raise

def ensemble_predictions(predictions_dict, weights=None):
    """アンサンブル学習の結果を計算する関数"""
    if weights is None:
        weights = {
            'LightGBM': 0.4,
            'XGBoost': 0.35,
            'RandomForest': 0.25
        }
    
    weighted_predictions = np.zeros_like(list(predictions_dict.values())[0])
    for model_name, predictions in predictions_dict.items():
        weighted_predictions += predictions * weights[model_name]
    
    return weighted_predictions

def debug_csv_contents(file_path):
    """CSVファイルの内容を確認するための関数"""
    try:
        # ファイルの存在確認
        if not os.path.exists(file_path):
            print(f"ファイルが見つかりません: {file_path}")
            return
        
        # CSVファイルの内容を確
        df = pd.read_csv(file_path)
        print("\nCSVファイルの情報:")
        print(f"列名: {df.columns.tolist()}")
        print("\n最初の数行:")
        print(df.head().to_string())
        print(f"\n行数: {len(df)}")
        
    except Exception as e:
        print(f"CSVファイル確認中にエラーが発生: {str(e)}")

def format_race_results(results_df):
    """レース結果を整形して表示する関数"""
    try:
        def str_width(s):
            """文字列の表示幅を計算（全角2、半角1として計算）"""
            width = 0
            for c in str(s):
                if ord(c) < 128:  # ASCII文字（半角）
                    width += 1
                else:  # その他（全角と仮定）
                    width += 2
            return width

        def pad_string(s, width):
            """文字列を指定した表示幅に調整（全角/半角を考慮）"""
            current_width = str_width(s)
            if current_width < width:
                return s + ' ' * (width - current_width)
            return s

        # 馬名の最大幅を計算（最低20文字分は確保）
        max_horse_name_width = max(
            max(str_width(name) for name in results_df['馬名']),
            20  # 最小幅（全角10文字分）
        ) + 2  # 余白として2文字分追加
        
        # 列幅の設定（全角文字を考慮）
        column_widths = {
            '馬名': max_horse_name_width,
            '予測着順': 10,   # 全角5文字分
            '馬番': 8,       # 全角4文字分
            '騎手': 20,      # 全角10文字分
            '馬体重': 10,     # 全角5文字分
            '増減': 8        # 全角4文字分
        }
        
        # データの整形
        formatted_df = results_df.copy()
        formatted_df['予測着順'] = formatted_df['予測着順'].map('{:.1f}'.format)
        
        # 数値カラムの安全な変換とフォーマット
        for col, fmt in [('馬体重', '{:.0f}'), ('増減', '{:+.0f}'), ('馬番', '{:.0f}')]:
            formatted_df[col] = pd.to_numeric(formatted_df[col], errors='coerce').fillna(0)
            formatted_df[col] = formatted_df[col].map(fmt.format)
        
        # ヘッダーの作成
        header = "".join(pad_string(col, column_widths[col]) for col in column_widths.keys())
        separator = "-" * str_width(header)
        
        # 各行のデータを整形
        rows = []
        for idx, row in formatted_df.iterrows():
            formatted_row = "".join(
                pad_string(str(row[col]), column_widths[col])
                for col in column_widths.keys()
            )
            rows.append(f"{idx:2d}  {formatted_row}")
        
        # 結果を結合
        return "\n".join([header, separator] + rows)
        
    except Exception as e:
        print(f"フォーマット中にエラーが発生しました: {str(e)}")
        raise

def main():
    try:
        directory_path = os.path.join(os.path.dirname(__file__), "data", "training")
        X, y, df = load_and_prepare_data(directory_path)
        best_models, scaler, X_test_scaled, y_test, feature_importance = train_advanced_model(X, y)
        
        new_race_file = "data/prediction/new_race.csv"
        try:
            # 新しいレースデータの読み込み
            new_race_data = pd.read_csv(new_race_file)
            
            # 必須カラムの確認
            required_columns = ['馬名', '馬番', '騎手', '馬体重', '増減']
            missing_columns = [col for col in required_columns if col not in new_race_data.columns]
            if missing_columns:
                raise ValueError(f"予測用データに必要なカラムが不足しています: {missing_columns}")
            
            new_race_data = improve_preprocessing(new_race_data, is_new_race=True)
            
            print("\n" + "="*80)
            print("各モデルの予測結果")
            print("="*80)
            
            predictions_dict = {}
            for model_name, model in best_models.items():
                results, predictions = predict_new_race(
                    model=model,
                    scaler=scaler,
                    new_race_data=new_race_data,
                    training_data=df,
                    model_name=model_name
                )
                predictions_dict[model_name] = predictions
                
                print(f"\n【{model_name}の予測】")
                print("-"*80)
                formatted_results = results[['馬名', '予測着順', '馬番', '騎手', '馬体重', '増減']].copy()
                print(format_race_results(formatted_results))
            
            # アンサンブル学習の結果を計算
            ensemble_preds = ensemble_predictions(predictions_dict)
            
            # アンサンブル結果をデータフレームに格納
            ensemble_results = pd.DataFrame({
                '馬名': new_race_data['馬名'],
                '予測着順': ensemble_preds,
                '馬番': new_race_data['馬番'],
                '騎手': new_race_data['騎手'],
                '馬体重': new_race_data['馬体重'],
                '増減': new_race_data['増減']
            })
            
            ensemble_results = ensemble_results.sort_values('予測着順')
            ensemble_results.index = range(1, len(ensemble_results) + 1)
            
            print("\n" + "="*80)
            print("最終予測（アンサンブル学習の結果）")
            print("="*80)
            
            formatted_ensemble = ensemble_results[['馬名', '予測着順', '馬番', '騎手', '馬体重', '増減']].copy()
            print(format_race_results(formatted_ensemble))
            print("\n")
            
        except FileNotFoundError:
            print(f"\nエラー: 予測用のファイル '{new_race_file}' が見つかりません。")
        except ValueError as ve:
            print(f"\nエラー: {str(ve)}")
            
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")

if __name__ == "__main__":
    main()