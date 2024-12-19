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
