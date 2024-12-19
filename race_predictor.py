# -*- coding: utf-8 -*-

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
import re
from tqdm import tqdm

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
        # 基本情報
        '馬番', '斤量', '馬体重', '増減',
        
        # 統計情報
        '平均着順', '平均タイム', '平均上り',
        
        # コース適性
        'コース_中京', 'コース_中山', 'コース_京都', 
        'コース_函館', 'コース_小倉', 'コース_新潟', 
        'コース_札幌', 'コース_東京', 'コース_福島', 
        'コース_阪神',
        
        # 距離適性
        '距離_短距離', '距離_マイル', '距離_中距離', '距離_長距離',
        
        # 馬場適性
        '馬場_不良', '馬場_稍重', '馬場_良', '馬場_重'
    ]
    
    # 利用可能な特徴量のみを返す
    available_features = [f for f in base_features if f in df.columns]
    print(f"\n利用可能な特徴量: {available_features}")
    
    return available_features

def calculate_horse_stats(historical_data):
    """馬の統計情報を計算する関数"""
    try:
        if '馬名' not in historical_data.columns:
            raise ValueError("データに'馬名'カラムが含まれていません")
        
        # 基本統計の計算
        stats = []
        
        # 馬の平均タイム
        if 'タイム_秒' in historical_data.columns:
            horse_time = historical_data.groupby('馬名')['タイム_秒'].mean()
            stats.append(pd.DataFrame({'馬名': horse_time.index, '平均タイム': horse_time.values}))
        
        # 馬の平均上り
        if '上り' in historical_data.columns:
            horse_last = historical_data.groupby('馬名')['上り'].mean()
            stats.append(pd.DataFrame({'馬名': horse_last.index, '平均上り': horse_last.values}))
        
        # 馬の平均着順
        if '着順' in historical_data.columns:
            horse_rank = historical_data.groupby('馬名')['着順'].mean()
            stats.append(pd.DataFrame({'馬名': horse_rank.index, '平均着順': horse_rank.values}))
        
        # コース適性（コースごとの平均着順）
        if all(col in historical_data.columns for col in ['場名', '着順']):
            course_stats = historical_data.pivot_table(
                values='着順',
                index='馬名',
                columns='場名',
                aggfunc='mean',
                fill_value=None
            ).add_prefix('コース_')
            stats.append(course_stats.reset_index())
        
        # 距離適性（距離帯ごとの平均着順）
        if all(col in historical_data.columns for col in ['距離', '着順']):
            historical_data['距離帯'] = pd.cut(
                historical_data['距離'],
                bins=[0, 1400, 1800, 2200, float('inf')],
                labels=['短距離', 'マイル', '中距離', '長距離']
            )
            distance_stats = historical_data.pivot_table(
                values='着順',
                index='馬名',
                columns='距離帯',
                aggfunc='mean',
                fill_value=None
            ).add_prefix('距離_')
            stats.append(distance_stats.reset_index())
        
        # 馬場適性（馬場状態ごとの平均着順）
        if all(col in historical_data.columns for col in ['馬場', '着順']):
            track_stats = historical_data.pivot_table(
                values='着順',
                index='馬名',
                columns='馬場',
                aggfunc='mean',
                fill_value=None
            ).add_prefix('馬場_')
            stats.append(track_stats.reset_index())
        
        # 全ての統計を結合
        if not stats:
            return pd.DataFrame({'馬名': historical_data['馬名'].unique()})
        
        result = stats[0]
        for df in stats[1:]:
            result = result.merge(df, on='馬名', how='outer')
        
        print(f"生成された特徴量: {result.columns.tolist()}")
        return result
        
    except Exception as e:
        print(f"馬の統計計算中にエラー: {str(e)}")
        traceback.print_exc()
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

def merge_jockey_stats(data, jockey_stats):
    """騎手の統計情報をデータフレームにマージする関数"""
    try:
        # jockey_statsのインデックスをリセット
        jockey_stats = jockey_stats.reset_index()
        
        # 新しいデータに騎手の統計を結合
        merged_data = data.merge(
            jockey_stats[['騎手', 'レース数', '平均着順', '勝率', '複勝率']], 
            on='騎手',
            how='left'
        )
        
        # 統計のない騎手は平均値で補完
        for col in ['平均着順', '勝率', '複勝率']:
            if col in merged_data.columns:
                merged_data[col] = merged_data[col].fillna(jockey_stats[col].mean())
        
        # カラム名を変更
        merged_data = merged_data.rename(columns={
            '勝率': '騎手勝率',
            '複勝率': '騎手複勝率',
            '平均着順': '騎手平均着順'
        })
        
        return merged_data
        
    except Exception as e:
        print(f"騎手統計マージ中にエラーが発生: {str(e)}")
        raise

def enhance_features(data, historical_data=None, all_courses=None, is_new_race=False):
    """特徴量を生成する関数"""
    try:
        processed_data = data.copy()
        
        if historical_data is not None:
            print("\n特徴量生成の進捗状況:")
            
            # コース適性の計算（最適化版）
            print("コース適性の計算中...")
            
            # 使用するコースのリストを決定
            if all_courses is None:
                unique_courses = processed_data['場名'].unique()
            else:
                unique_courses = all_courses
            
            # 馬ごとのコース成績を事前計算
            horse_course_stats = {}
            for horse in tqdm(processed_data['馬名'].unique(), desc="馬ごとの成績集計"):
                horse_history = historical_data[historical_data['馬名'] == horse]
                horse_course_stats[horse] = {}
                
                for course in unique_courses:
                    course_history = horse_history[horse_history['場名'] == course]
                    
                    if len(course_history) == 0 or is_new_race:
                        horse_course_stats[horse][course] = 0
                        continue
                    
                    if '着順' not in course_history.columns:
                        print(f"警告: {horse}の過去データに着順情報がありません")
                        horse_course_stats[horse][course] = 0
                        continue
                    
                    # 着順の平均を計算
                    avg_rank = course_history['着順'].apply(
                        lambda x: 18 if isinstance(x, str) and '着' not in x else float(str(x).replace('着', ''))
                    ).mean()
                    
                    # 1-3着の割合を計算（新しいレースの場合はスキップ）
                    if not is_new_race:
                        top3_rate = len(course_history[course_history['着順'].isin(['1着', '2着', '3着'])]) / len(course_history)
                    else:
                        top3_rate = 0
                    
                    # スコアの計算
                    score = (1 / avg_rank * 0.7) + (top3_rate * 0.3) if not is_new_race else 0
                    horse_course_stats[horse][course] = score
            
            # 事前計算した成績をデータフレームに適用
            for course in tqdm(unique_courses, desc="コース特徴量の生成"):
                col_name = f'コース_{course}'
                processed_data[col_name] = processed_data['馬名'].map(
                    lambda x: horse_course_stats.get(x, {}).get(course, 0)
                )
            
            # すべてのコース列が存在することを確認
            all_course_columns = [f'コース_{course}' for course in all_courses] if all_courses else []
            for col in all_course_columns:
                if col not in processed_data.columns:
                    processed_data[col] = 0
            
            # 距離適性の計算も同様に修正
            print("\n距離適性の計算中...")
            distance_ranges = {
                '短距離': (0, 1400),
                'マイル': (1401, 1800),
                '中距離': (1801, 2400),
                '長距離': (2401, float('inf'))
            }
            
            # 馬ごとの距離成績を事前計算
            horse_distance_stats = {}
            for horse in tqdm(processed_data['馬名'].unique(), desc="馬ごとの距離成績集計"):
                horse_history = historical_data[historical_data['馬名'] == horse]
                horse_distance_stats[horse] = {}
                
                for distance_type, (min_dist, max_dist) in distance_ranges.items():
                    if is_new_race:
                        horse_distance_stats[horse][distance_type] = 0
                        continue
                        
                    distance_history = horse_history[
                        (horse_history['距離'] >= min_dist) & 
                        (horse_history['距離'] <= max_dist)
                    ]
                    
                    if len(distance_history) == 0 or '着順' not in distance_history.columns:
                        horse_distance_stats[horse][distance_type] = 0
                        continue
                    
                    avg_rank = distance_history['着順'].apply(
                        lambda x: 18 if isinstance(x, str) and '着' not in x else float(str(x).replace('着', ''))
                    ).mean()
                    
                    top3_rate = len(distance_history[distance_history['着順'].isin(['1着', '2着', '3着'])]) / len(distance_history)
                    
                    score = (1 / avg_rank * 0.5) + (top3_rate * 0.5)
                    horse_distance_stats[horse][distance_type] = score
            
            # 事前計算した距離成績をデータフレームに適用
            for distance_type in tqdm(distance_ranges.keys(), desc="距離特徴量の生成"):
                col_name = f'距離_{distance_type}'
                processed_data[col_name] = processed_data['馬名'].map(
                    lambda x: horse_distance_stats[x][distance_type]
                )
            
            print("\n特徴量生成完了")
            
        return processed_data
        
    except Exception as e:
        print(f"特徴量生成中にエラー: {str(e)}")
        traceback.print_exc()
        raise

def improve_preprocessing(df, is_new_race=False):
    """データ前処理を改善る関数"""
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
                df[col] = df.groupby(['馬名', '距離'])[col].transform(
                    lambda x: x.fillna(x.mean())
                )
        
        # 3. カテゴリ変数の欠損値処理
        categorical_columns = ['場名', '種別', '回り', '天候', '馬場', '性別']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0])
        
        # 4. 距離を数値型に変換
        df['距離'] = pd.to_numeric(df['距離'], errors='coerce')
        
        return df
        
    except Exception as e:
        print(f"前処理中にエラーが発生しました: {str(e)}")
        traceback.print_exc()  # スタックトレースを出力
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
            '上り', '馬体重', '増減', '騎手', '開催日'  # '開催日'を追加
        ]
        
        # まず1つ目のファイルの列名を確認
        first_file = all_files[0]
        try:
            df_sample = pd.read_csv(first_file, nrows=1, encoding='utf-8')
        except UnicodeDecodeError:
            df_sample = pd.read_csv(first_file, nrows=1, encoding='cp932')
        
        print("\n利用可能な列名:")
        print(df_sample.columns.tolist())
        
        # データフレームのリスト作成と結合
        dfs = []
        for file in all_files:
            try:
                try:
                    df = pd.read_csv(file, encoding='utf-8')
                    df['file_path'] = file
                except UnicodeDecodeError:
                    df = pd.read_csv(file, encoding='cp932')
                    df['file_path'] = file
                dfs.append(df)
            except Exception as e:
                print(f"警告: {os.path.basename(file)} 読み込みに失敗しました: {str(e)}")
                continue
        
        if not dfs:
            raise ValueError("読み込み可能なCSVファイルがありません")
        
        # データフレームの結合
        df = pd.concat(dfs, ignore_index=True)
        
        # 障害レースを除外
        df = df[~df['種別'].str.contains('障', na=False)]
        print(f"障害レース除外後のデータ数: {len(df)}行")
        
        print(f"データセット（前処理）: {len(df)}行")
        
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
        
        print(f"\nデータ前処理後の行数: {len(df)}行")
        
        # 特徴量の生成（種別ごとに分けて処理）
        df_turf = df[df['種別'] == '芝'].copy()
        df_dirt = df[df['種別'] == 'ダート'].copy()
        
        # それぞれの種別で特徴量を生成
        processed_dfs = []
        if len(df_turf) > 0:
            df_turf = enhance_features(df_turf, df_turf.copy())
            processed_dfs.append(df_turf)
        if len(df_dirt) > 0:
            df_dirt = enhance_features(df_dirt, df_dirt.copy())
            processed_dfs.append(df_dirt)
        
        # データを結合
        if processed_dfs:
            df = pd.concat(processed_dfs, ignore_index=True)
        else:
            raise ValueError("処理可能なデータがありません")
        
        print(f"特徴量生成後の行数: {len(df)}行")
        
        # 必要な特徴量の選択
        features = create_race_features(df)
        
        # 特徴量とターゲットの分離
        X = df[features].copy()
        y = df['着順'].copy()
        
        # NaNの処理を改善
        # 1. 馬体重と増減の処理
        X['馬体重'] = X['馬体重'].fillna(X['馬体重'].mean())
        X['増減'] = X['増減'].fillna(0)  # 増減不明は0として扱う
        
        # 2. コース適性の処理
        course_columns = [col for col in X.columns if col.startswith('コース_')]
        for col in course_columns:
            X[col] = X[col].fillna(0)  # 未経験のコースは0として扱う
        
        # 3. 距離適性の処理
        distance_columns = [col for col in X.columns if col.startswith('距離_')]
        for col in distance_columns:
            X[col] = X[col].fillna(0)  # 未経験の距離は0として扱う
        
        # 4. 馬場適性の処理
        ground_columns = [col for col in X.columns if col.startswith('馬場_')]
        for col in ground_columns:
            X[col] = X[col].fillna(0)  # 未経験の馬場状態は0として扱う
        
        # 5. 基本統計量の処理
        stat_columns = ['平均着順', '平均タイム', '平均上り']
        for col in stat_columns:
            if col in X.columns:
                X[col] = X[col].fillna(X[col].mean())
        
        print("\n欠損値処理後のNaN数:")
        print(X.isna().sum()[X.isna().sum() > 0])
        
        # 残りのNaNがある場合の処理
        remaining_nan_cols = X.columns[X.isna().any()].tolist()
        if remaining_nan_cols:
            print(f"\n警告: 以下の列にまだNaNが残っています: {remaining_nan_cols}")
            # 残りのNaNを0で埋める
            X = X.fillna(0)
        
        print(f"\n最終的なデータセットの行数: {len(X)}行")
        print(f"利用可能な特徴量: {features}")
        
        if len(X) == 0:
            raise ValueError("有効なデータがありません。")
        
        return X, y, df
        
    except Exception as e:
        print(f"データ読み込み中にエラーが発生しました: {str(e)}")
        traceback.print_exc()
        raise

def train_advanced_model(X, y):
    """モデルの学習を行う関数"""
    try:
        # コース名の列を保存
        course_columns = [col for col in X.columns if col.startswith('コース_')]
        all_courses = [col.replace('コース_', '') for col in course_columns]
        
        # 特徴量の重み付け
        feature_weights = {
            # 基本情報（中程度の重み）
            '馬番': 0.6,
            '斤量': 0.7,
            '馬体重': 0.8,
            '増減': 0.6,
            
            # 騎手関連（高い重み）
            '騎手勝率': 0.9,
            '騎手複勝率': 0.9,
            '騎手平均着順': 0.8,
            
            # 馬の基本統計（高い重み）
            '平均タイム': 1.0,
            '平均上り': 0.9,
            '平均着順': 0.9,
            
            # 前走情報（非常に高い重み）
            '前走順': 1.0,
            '前走着差': 0.9,
            '前走上り': 0.9,
            '前走タイム': 0.9,
            
            # 前前走情報（高い重み）
            '前前走着順': 0.8,
            '前前走着差': 0.7,
            '前前走上り': 0.7,
            '前前走タイム': 0.7,
            
            # コース適性（中程度の重み）
            'コース_': 0.7,
            
            # 距離適性（中程度の重み）
            '距離_': 0.8,
            
            # 馬場適性（中程度の重み）
            '馬場_': 0.6
        }
        
        # 特徴量の重み付けを適用
        weighted_X = X.copy()
        for feature in X.columns:
            for pattern, weight in feature_weights.items():
                if feature.startswith(pattern) or feature == pattern:
                    weighted_X[feature] *= weight
                    break
        
        # データの分割
        X_train, X_test, y_train, y_test = train_test_split(
            weighted_X, y, test_size=0.2, random_state=42
        )
        
        # スケーリング
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # LightGBMモデルのパラメータ
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'early_stopping_rounds': 50,
            'verbose': -1
        }
        
        print("LightGBMモデルの学習中...")
        lgb_model = lgb.train(
            lgb_params,
            lgb.Dataset(X_train_scaled, y_train),
            valid_sets=[lgb.Dataset(X_test_scaled, y_test)],
            num_boost_round=1000
        )
        
        # 特徴量の重要度を計算
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': lgb_model.feature_importance()
        }).sort_values('importance', ascending=False)
        
        print("\nLightGBM特徴量重度:")
        print(feature_importance)
        
        # モデルを辞書に格納
        best_models = {
            'LightGBM': lgb_model,
            'RandomForest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ).fit(X_train_scaled, y_train)
        }
        
        return best_models, scaler, X_test_scaled, y_test, feature_importance, all_courses
        
    except Exception as e:
        print(f"\n=== LightGBMモデルでエラーが発生しました ===")
        print(f"エラー内容: {str(e)}")
        raise

def validate_model(X, y, model):
    """時系列スバリデーションを実装る関数"""
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

def predict_new_race(model, scaler, new_race_data, training_data, model_name, all_courses):
    """新しいレースの予測を行う関数"""
    try:
        # 特徴量の生成
        features = enhance_features(new_race_data, new_race_data.copy(), all_courses, is_new_race=True)
        
        # 複行の削除（完全な重複のみ）
        new_race_data = new_race_data.drop_duplicates(subset=['馬名'], keep='first')
        
        # 必須カラムの存在確認と欠損値処理
        required_columns = ['馬名', '馬番', '騎手', '馬体重', '増減']
        for col in required_columns:
            if col not in new_race_data.columns:
                raise ValueError(f"必要なカラム '{col}' が不足しています")
            
            # 数値型カラムの欠損値を0で補完
            if col in ['馬体重', '増減', '馬番']:
                new_race_data[col] = pd.to_numeric(new_race_data[col], errors='coerce').fillna(0)
            # 文字列型カラムの損値を'不明'で補完
            elif col == '騎手':
                new_race_data[col] = new_race_data[col].fillna('不明')
        
        processed_data = enhance_features(new_race_data, training_data, all_courses)
        required_features = create_race_features(processed_data)
        X_new = processed_data[required_features].copy()
        X_new = X_new.fillna(X_new.mean())
        X_new_scaled = scaler.transform(X_new)
        predictions = model.predict(X_new_scaled)
        predictions = np.clip(predictions, 1, 18)
        
        # 果をデータフレームに格納
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
        print(f"\n=== {model_name}デルでエラーが発生しました ===")
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
        
        # CSVファイルの内容を確認
        df = pd.read_csv(file_path)
        print("\nCSVファイルの報:")
        print(f"列名: {df.columns.tolist()}")
        print("\n数行:")
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
            """文字列を指定した表示幅に調整（全角/半角を考��）"""
            current_width = str_width(s)
            if current_width < width:
                return s + ' ' * (width - current_width)
            return s

        # 馬名の最大幅を計算（最低20文字分は確保）
        max_horse_name_width = max(
            max(str_width(name) for name in results_df['馬名']),
            20  # 最小幅（全角10文字分）
        ) + 2  # 余白として2文字分追加
        
        # 列設定（全角文字を考慮）
        column_widths = {
            '馬名': max_horse_name_width,
            '予測着順': 10,   # 全角5文字分
            '馬番': 8,       # 全角4文字
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
        
        # ���ッダーの作成
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

def analyze_jockey_performance(training_data, course_name):
    """騎手のコース適性を分析する関数"""
    try:
        # 対象コスのデータのみを抽出
        course_data = training_data[training_data['場名'] == course_name].copy()
        
        # 騎手ごとの成績を計算
        jockey_stats = pd.DataFrame()
        
        # 基本統計
        basic_stats = course_data.groupby('騎手').agg({
            '着順': ['count', 'mean', 'std'],
            'タイム_秒': ['mean', 'std']
        }).round(2)
        
        # 1着、2着、3着の回数
        for rank in [1, 2, 3]:
            rank_counts = course_data[course_data['着順'] == rank]['騎手'].value_counts()
            basic_stats[f'{rank}着回数'] = rank_counts
        
        # 複勝率と勝率を計算
        basic_stats['複勝回数'] = basic_stats['1着回数'] + basic_stats['2着回数'] + basic_stats['3着回数']
        basic_stats['勝率'] = (basic_stats['1着回数'] / basic_stats[('着順', 'count')]).round(3)
        basic_stats['複勝率'] = (basic_stats['複勝回数'] / basic_stats[('着順', 'count')]).round(3)
        
        # 直近の調子（過去3ヶ月）を考慮
        recent_data = course_data[course_data['日付'] >= (pd.Timestamp.now() - pd.DateOffset(months=3))]
        recent_stats = recent_data.groupby('騎手')['着順'].agg(['count', 'mean']).round(2)
        basic_stats['直近平均着順'] = recent_stats['mean']
        basic_stats['直近騎乗回数'] = recent_stats['count']
        
        # コース適性スコアの算（0-100のスコアル）
        weights = {
            '勝率': 0.4,
            '複勝率': 0.3,
            '着順_mean': 0.2,
            '直近平均着順': 0.1
        }
        
        # 各標を0-1にスケーリング
        stats_for_scoring = pd.DataFrame({
            '勝率': basic_stats['勝率'],
            '複勝率': basic_stats['複勝率'],
            '着順_mean': 1 / basic_stats[('着順', 'mean')],  # 着順は逆数を取る
            '直近平均着順': 1 / basic_stats['直近平均着順']  # 着順は逆数を取る
        })
        
        # min-maxスケーリング
        stats_scaled = (stats_for_scoring - stats_for_scoring.min()) / (stats_for_scoring.max() - stats_for_scoring.min())
        
        # 総合スアの計算
        course_aptitude = sum(stats_scaled[col] * weight for col, weight in weights.items()) * 100
        basic_stats['コース適性'] = course_aptitude.round(1)
        
        # 果の整形
        result = pd.DataFrame({
            '騎乗回数': basic_stats[('着順', 'count')],
            '1着': basic_stats['1着回数'],
            '2着': basic_stats['2着回数'],
            '3着': basic_stats['3着回数'],
            '勝率': (basic_stats['勝率'] * 100).round(1),
            '複勝率': (basic_stats['複勝率'] * 100).round(1),
            '平均着順': basic_stats[('着順', 'mean')],
            '直近平均着順': basic_stats['直近平均着順'],
            'コース適性': basic_stats['コース適性']
        })
        
        # 適性スコアで降順ソート
        result = result.sort_values('コース適性', ascending=False)
        
        # 出力
        print(f"\n=== {course_name}における騎手実績 ===")
        print("\n【凡例】")
        print("・コース適性: 0-100のスコア（勝率、複勝率、平均着順、直近成績を考慮）")
        print("・勝率/複勝率: パーセンテージ表示")
        print("・直近平均着順: 過去3ヶ月の平均着順\n")
        
        pd.set_option('display.max_rows', None)
        print(result.to_string(float_format=lambda x: '{:.1f}'.format(x)))
        
        return result
        
    except Exception as e:
        print(f"騎手分析中にエラーが発生しました: {str(e)}")
        raise

def save_feature_stats(stats, filename):
    """特徴量の統計情報を保存する関数"""
    stats.to_pickle(f'stats/{filename}.pkl')

def load_feature_stats(filename):
    """保存された特徴量の統計情報を読み込む関数"""
    return pd.read_pickle(f'stats/{filename}.pkl')

def get_last_race_stats(historical_data):
    """前データを取得する関数"""
    try:
        # ディレクトリ名から日付を抽出して追加
        def extract_date_from_path(path):
            # パスから年月日を抽出（例: "2024/01/14" -> "20240114"）
            match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', path)
            if match:
                return f"{match.group(1)}{match.group(2)}{match.group(3)}"
            return None
        
        if 'file_path' in historical_data.columns:
            historical_data['日付'] = historical_data['file_path'].apply(extract_date_from_path)
            historical_data = historical_data.sort_values('日付', ascending=True)
        
        # 馬ごとの前走データを取得
        last_race = historical_data.groupby('馬名').agg({
            '着順': 'last',
            'タイム_秒': 'last',
            '上り': 'last',
            '着差': 'last',
            '距離': 'last'
        }).reset_index()
        
        # カラム名を変更
        last_race = last_race.rename(columns={
            '着順': '前走着順',
            'タイム_秒': '前走タイム',
            '上り': '前走上り',
            '着差': '前走着差',
            '距離': '前走距離'
        })
        
        # 前々走データも取得
        second_last_race = historical_data.groupby('馬名').agg({
            '着順': lambda x: x.iloc[-2] if len(x) > 1 else None,
            'タイム_秒': lambda x: x.iloc[-2] if len(x) > 1 else None,
            '上り': lambda x: x.iloc[-2] if len(x) > 1 else None,
            '着差': lambda x: x.iloc[-2] if len(x) > 1 else None
        }).reset_index()
        
        # カラム名を変更
        second_last_race = second_last_race.rename(columns={
            '着順': '前々走着順',
            'タイム_秒': '前々走タイム',
            '上り': '前々走上り',
            '着差': '前々走着差'
        })
        
        # 前走と前々走のデータを合
        race_stats = last_race.merge(second_last_race, on='馬名', how='left')
        
        # 距離変化を計算（現在の距離 - 前走距離）
        if '距離' in historical_data.columns:
            current_distance = historical_data.groupby('馬名')['距離'].last()
            race_stats['距離変化'] = current_distance - race_stats['前走距離']
        
        return race_stats
        
    except Exception as e:
        print(f"前走データ取得中にエラー: {str(e)}")
        raise

def merge_last_race_stats(data, last_race_stats):
    """前走データをデータフレームにマージする関数"""
    try:
        # 新しいデータに前走データを結合
        merged_data = data.merge(last_race_stats, on='馬名', how='left')
        
        # 前走データの欠損値を処理
        numeric_columns = [
            '前走着順', '前走タイム', '前走上り', '前走着差',
            '前々走着順', '前々走タイム', '前々走上り', '前々走着差',
            '距離変化'
        ]
        
        for col in numeric_columns:
            if col in merged_data.columns:
                # 欠損値を平均値で補完
                merged_data[col] = merged_data[col].fillna(last_race_stats[col].mean())
        
        return merged_data
        
    except Exception as e:
        print(f"前走データマージ中にエラー: {str(e)}")
        raise

def calculate_course_performance(row, historical_data, course):
    """コースごとの成績を計算する関数"""
    try:
        # tqdmを使用してプログレスバーを表示
        tqdm.pandas(desc=f"コース {course} の処理中")
        
        # 該当馬の過去レースを抽出
        horse_history = historical_data[
            (historical_data['馬名'] == row['馬名']) & 
            (historical_data['場名'] == course)
        ]
        
        if len(horse_history) == 0:
            return 0  # コース実績なし
        
        # 着順の平均を計算（着外は18着として扱う）
        avg_rank = horse_history['着順'].apply(
            lambda x: 18 if isinstance(x, str) and '着' not in x else float(str(x).replace('着', ''))
        ).mean()
        
        # 1-3着の割合を計算
        top3_rate = len(horse_history[horse_history['着順'].isin(['1着', '2着', '3着'])]) / len(horse_history)
        
        # スコアの計算（着順の平均と1-3着率を考慮）
        score = (1 / avg_rank * 0.7) + (top3_rate * 0.3)
        return score
        
    except Exception as e:
        print(f"コース適性計算中にエラー: {str(e)}")
        return 0

def calculate_distance_performance(row, historical_data, min_dist, max_dist):
    """距離帯ごとの成績を計算する関数"""
    try:
        # tqdmを使用してプログレスバーを表示
        tqdm.pandas(desc=f"距離帯 {min_dist}-{max_dist} の処理中")
        
        # 該当馬の過去レースを抽出
        horse_history = historical_data[
            (historical_data['馬名'] == row['馬名']) & 
            (historical_data['距離'] >= min_dist) & 
            (historical_data['距離'] <= max_dist)
        ]
        
        if len(horse_history) == 0:
            return 0  # 距離実績なし
        
        # 着順の平均を計算（着外は18着として扱う）
        avg_rank = horse_history['着順'].apply(
            lambda x: 18 if isinstance(x, str) and '着' not in x else float(str(x).replace('着', ''))
        ).mean()
        
        # 1-3着の割合を計算
        top3_rate = len(horse_history[horse_history['着順'].isin(['1着', '2着', '3着'])]) / len(horse_history)
        
        # タイム指数の計算（オプション）
        if 'タイム' in horse_history.columns:
            time_index = 1 / horse_history['タイム'].mean() if horse_history['タイム'].mean() > 0 else 0
        else:
            time_index = 0
        
        # スコアの計算（着順の平均、1-3着率、タイム指数を考慮）
        score = (1 / avg_rank * 0.5) + (top3_rate * 0.5)
        return score
        
    except Exception as e:
        print(f"距離��性計算中にエラー: {str(e)}")
        return 0

def calculate_ground_performance(row, historical_data, ground_condition):
    """馬場状態ごとの成績を計算する関数"""
    try:
        # tqdmを使用してプログレスバーを表示
        tqdm.pandas(desc=f"馬場状態 {ground_condition} の処理中")
        
        # 該当馬の過去レースを抽出
        horse_history = historical_data[
            (historical_data['馬名'] == row['馬名']) & 
            (historical_data['馬場'] == ground_condition)
        ]
        
        if len(horse_history) == 0:
            return 0  # 馬場状態での実績なし
        
        # 着順の平均を計算（着外は18着として扱う）
        avg_rank = horse_history['着順'].apply(
            lambda x: 18 if isinstance(x, str) and '着' not in x else float(str(x).replace('着', ''))
        ).mean()
        
        # 1-3着の割合を計算
        top3_rate = len(horse_history[horse_history['着順'].isin(['1着', '2着', '3着'])]) / len(horse_history)
        
        # スコアの計算
        score = (1 / avg_rank * 0.6) + (top3_rate * 0.4)
        return score
        
    except Exception as e:
        print(f"馬場適性計算中にエラー: {str(e)}")
        return 0

def main():
    try:
        directory_path = os.path.join(os.path.dirname(__file__), "data", "training")
        X, y, df = load_and_prepare_data(directory_path)
        best_models, scaler, X_test_scaled, y_test, feature_importance, all_courses = train_advanced_model(X, y)
        
        new_race_file = "data/prediction/new_race.csv"
        try:
            # 新いレースデータの読み込み
            new_race_data = pd.read_csv(new_race_file)
            
            # 必須カラムの存在確認
            required_columns = ['馬名', '馬番', '騎手', '馬体重', '増減']
            missing_columns = [col for col in required_columns if col not in new_race_data.columns]
            if missing_columns:
                raise ValueError(f"予測用データに必要なカラムが不足ています: {missing_columns}")
            
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
                    model_name=model_name,
                    all_courses=all_courses
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