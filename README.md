# keiba

生成AIを用いて作成した競馬データの解析プログラムです。

## ディレクトリ構成

```
keiba/
├── data/
│   ├── prediction/
│   ├── training/
├── get_data.py
├── race_predictor.py
├── get_new_race_data.py
├── README.md
```

## 仮想環境有効化

```bash
python -m venv venv
source venv/bin/activate
```

## データ取得

```bash
python get_data.py
```
## データ解析

```bash
python race_predictor.py
```

## 近日開催レース情報の取得方法

近日開催予定のレース情報を取得して、予測用のCSVファイルを作成するには以下のコマンドを実行します：

```
python get_new_race_data.py
```

デフォルトでは、現在の年月のレース情報を取得し、`data/prediction/new_race.csv`に保存します。

### オプション

- `--year`: 取得する年（例: 2024）
- `--month`: 取得する月（例: 6）
- `--race_id`: 特定のレースIDを指定する場合
- `--output`: 出力ファイルパス

### 例

特定のレースIDを指定して情報を取得する：
```
python get_new_race_data.py --race_id 202406020611
```

特定の年月のレース情報を取得する：
```
python get_new_race_data.py --year 2024 --month 7
```

出力先を変更する：
```
python get_new_race_data.py --output data/prediction/my_race.csv
```
