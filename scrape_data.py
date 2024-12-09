from bs4 import BeautifulSoup
import pandas as pd
import os
import re

def parse_race_results(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # レース結果テーブルを取得
    result_table = soup.find('table', class_='race_table_01')
    if not result_table:
        return None
    
    results = []
    
    # テーブルの行を処理
    rows = result_table.find_all('tr')[1:]  # ヘッダー行をスキップ
    for row in rows:
        cols = row.find_all('td')
        if len(cols) < 10:
            continue
            
        # 馬名を取得
        horse_name = cols[3].find('a').text.strip() if cols[3].find('a') else ''
        
        # 性別と年齢を分離
        sex_age = cols[4].text.strip()
        sex = sex_age[0] if sex_age else ''
        age = sex_age[1:] if len(sex_age) > 1 else ''
        
        # 通過順位を取得
        passing_order = cols[10].text.strip() if len(cols) > 10 else ''
        
        # 馬体重を取得・パース
        weight_text = cols[14].text.strip() if len(cols) > 14 else ''
        weight_match = re.match(r'(\d+)\(([-+]?\d+)\)', weight_text)
        if weight_match:
            weight = weight_match.group(1)
            weight_diff = weight_match.group(2)
        else:
            weight = ''
            weight_diff = ''
        
        result = {
            '着順': cols[0].text.strip(),
            '枠番': cols[1].text.strip(),
            '馬番': cols[2].text.strip(),
            '馬名': horse_name,
            '性別': sex,
            '年齢': age,
            '斤量': cols[5].text.strip(),
            '騎手': cols[6].find('a').text.strip() if cols[6].find('a') else '',
            'タイム': cols[7].text.strip(),
            '着差': cols[8].text.strip(),
            '通過': passing_order,
            '上り': cols[11].text.strip() if len(cols) > 11 else '',
            '単勝': cols[12].text.strip() if len(cols) > 12 else '',
            '人気': cols[13].text.strip() if len(cols) > 13 else '',
            '馬体重': weight,
            '増減': weight_diff
        }
        results.append(result)
    
    return pd.DataFrame(results)

def process_html_file(input_path):
    # 入力ファイルの名前から拡張子を除いた部分を取得
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join('data', f'{base_name}.csv')
    
    # HTMLファイルを読み込み
    with open(input_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # データフレームを作成
    df = parse_race_results(html_content)
    
    # CSVファイルに出力
    if df is not None:
        os.makedirs('data', exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"データを{output_path}に出力しました。")
    else:
        print("レース結果の取得に失敗しました。")

# 使用例
input_file = 'data/202306010101.html'
process_html_file(input_file)