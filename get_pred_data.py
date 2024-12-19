import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_race_data(url):
    # ヘッダーを設定
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # URLからデータを取得
    response = requests.get(url, headers=headers)
    response.encoding = response.apparent_encoding  # 文字コードを自動判別
    soup = BeautifulSoup(response.text, 'html.parser')

    # データを格納するリスト
    race_data = []

    # レース情報を抽出
    for row in soup.select('table tr'):
        cols = row.find_all('td')
        if len(cols) > 0:
            # 各列のデータが存在するか確認し、存在しない場合はスペースを代入
            horse_name = cols[1].text.strip() if len(cols) > 1 else ' '
            course_name = cols[2].text.strip() if len(cols) > 2 else ' '
            race_type = cols[3].text.strip() if len(cols) > 3 else ' '
            distance = cols[4].text.strip() if len(cols) > 4 else ' '
            direction = cols[5].text.strip() if len(cols) > 5 else ' '
            weather = cols[6].text.strip() if len(cols) > 6 else ' '
            track_condition = cols[7].text.strip() if len(cols) > 7 else ' '
            frame_number = cols[8].text.strip() if len(cols) > 8 else ' '
            horse_number = cols[9].text.strip() if len(cols) > 9 else ' '
            sex = cols[10].text.strip() if len(cols) > 10 else ' '
            age = cols[11].text.strip() if len(cols) > 11 else ' '
            weight = cols[12].text.strip() if len(cols) > 12 else ' '
            body_weight = cols[13].text.strip() if len(cols) > 13 else ' '
            weight_change = cols[14].text.strip() if len(cols) > 14 else ' '
            jockey = cols[15].text.strip() if len(cols) > 15 else ' '

            race_data.append([horse_name, course_name, race_type, distance, direction,
                              weather, track_condition, frame_number, horse_number,
                              sex, age, weight, body_weight, weight_change, jockey])

    # DataFrameに変換
    columns = ['馬名', '場名', '種別', '距離', '回り', '天候', '馬場', '枠番', '馬番', 
               '性別', '年齢', '斤量', '馬体重', '増減', '騎手']
    df = pd.DataFrame(race_data, columns=columns)

    # CSVファイルに保存
    df.to_csv('data/prediction/new_race.csv', index=False, encoding='utf-8-sig')
    print("データが new_race.csv に保存されました。")

if __name__ == "__main__":
    url = "https://race.netkeiba.com/race/shutuba.html?race_id=202406050811"
    scrape_race_data(url)
