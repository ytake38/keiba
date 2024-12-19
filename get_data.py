import requests
from bs4 import BeautifulSoup
import pandas as pd
import gc
import re
import time
import pickle
from pathlib import Path
from typing import List, Optional

def parse_race_results(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # レース情報を取得
    race_data = {
        '場名': '',
        '種別': '',
        '距離': '',
        '回り': '',
        '天候': '',
        '馬場': ''
    }
    
    try:
        # レース条件を取得
        race_data_span = soup.find('dl', class_='racedata').find('span')
        if race_data_span:
            race_details = race_data_span.text.strip()
            # 例: "ダ右1200m / 天候 : 晴 / ダート : 良 / 発走 : 09:55"
            
            # 種別、距離、回りを取得
            course_match = re.match(r'([芝ダ])([右左外内])?(\d+)m', race_details)
            if course_match:
                race_data['種別'] = 'ダート' if course_match.group(1) == 'ダ' else '芝'
                race_data['回り'] = course_match.group(2) if course_match.group(2) else ''
                race_data['距離'] = course_match.group(3)
            
            # 天候を取得
            weather_match = re.search(r'天候\s*:\s*(\S+)', race_details)
            if weather_match:
                race_data['天候'] = weather_match.group(1)
            
            # 馬場状態を取得
            track_match = re.search(r'[芝ダート]\s*:\s*(\S+)', race_details)
            if track_match:
                race_data['馬場'] = track_match.group(1)
        
        # 競馬場名を取得
        place_text = soup.find('p', class_='smalltxt')
        if place_text:
            place_match = re.search(r'\d+回(\S+)\d+日', place_text.text)
            if place_match:
                race_data['場名'] = place_match.group(1)

    except Exception as e:
        print(f"レース情報の取得でエラー: {e}")
    
    # レース結果テーブルを取得
    try:
        result_table = soup.find('table', class_='race_table_01')
        if not result_table:
            print("レース結果テーブルが見つかりません")
            return None
        
        results = []
        rows = result_table.find_all('tr')[1:]  # ヘッダー行をスキップ
        
        for row in rows:
            try:
                cols = row.find_all('td')
                if len(cols) < 10:
                    continue
                
                # 馬名を取得
                horse_name_elem = cols[3].find('a')
                horse_name = horse_name_elem.text.strip() if horse_name_elem else ''
                
                # 性別と年齢を分離
                sex_age = cols[4].text.strip()
                sex = sex_age[0] if sex_age else ''
                age = sex_age[1:] if len(sex_age) > 1 else ''
                
                # 騎手を取得
                jockey_elem = cols[6].find('a')
                jockey = jockey_elem.text.strip() if jockey_elem else ''
                
                # 通過順位を取得
                passing_order = cols[10].text.strip() if len(cols) > 10 else ''
                
                # 馬体重を取得・パース
                weight_text = cols[14].text.strip() if len(cols) > 14 else ''
                weight_match = re.match(r'(\d+)\(([-+]?\d+)\)', weight_text)
                weight = weight_match.group(1) if weight_match else ''
                weight_diff = weight_match.group(2) if weight_match else ''
                
                result = {
                    '場名': race_data['場名'],
                    '種別': race_data['種別'],
                    '距離': race_data['距離'],
                    '回り': race_data['回り'],
                    '天候': race_data['天候'],
                    '馬場': race_data['馬場'],
                    '着順': cols[0].text.strip(),
                    '枠番': cols[1].text.strip(),
                    '馬番': cols[2].text.strip(),
                    '馬名': horse_name,
                    '性別': sex,
                    '年齢': age,
                    '斤量': cols[5].text.strip(),
                    '騎手': jockey,
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
                
            except Exception as e:
                print(f"行の処理でエラー: {e}")
                continue
        
        if not results:
            print("有効なレース結果が見つかりません")
            return None
            
        return pd.DataFrame(results)
        
    except Exception as e:
        print(f"レース結果テーブルの処理でエラー: {e}")
        return None

class NetkeibaClient:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
        }
        self.CALENDAR_URL = "https://race.netkeiba.com/top/calendar.html?year={year}&month={month}"
        self.RACEDATE_URL = "https://db.netkeiba.com/race/list/{racedate}"
        self.RACE_URL = "https://db.netkeiba.com/race/{race_id}"
        
    def _make_request(self, url: str) -> Optional[str]:
        """URLにリクエストを送信してレスポンスを取得する"""
        try:
            response = requests.get(url, headers=self.headers)
            response.encoding = 'EUC-JP'
            if response.status_code == 200:
                return response.text
            else:
                print(f"\nError {response.status_code}: {url}")
                return None
        except Exception as e:
            print(f"\nRequest failed: {url}")
            print(f"Error: {e}")
            return None
        finally:
            time.sleep(1)  # アクセス間隔を設ける

    def get_calendar_dates(self, year: int, month: int) -> List[str]:
        """指定年月のレース開催日一覧を取得する"""
        url = self.CALENDAR_URL.format(year=year, month=month)
        html = self._make_request(url)
        if not html:
            return []
        
        soup = BeautifulSoup(html, "html.parser")
        table_data = soup.find("table", class_="Calendar_Table")
        if not table_data:
            return []
            
        link_set = table_data.find_all("a")
        date_id_list = [
            re.search(r"\d+$", link.get("href")).group()
            for link in link_set
            if link.get("href") and re.search(r"\d+$", link.get("href"))
        ]
        return date_id_list

    def get_race_ids_by_date(self, date_id: str) -> List[str]:
        """指定開催日のレースID一覧を取得する"""
        url = self.RACEDATE_URL.format(racedate=date_id)
        html = self._make_request(url)
        if not html:
            return []
        
        soup = BeautifulSoup(html, "html.parser")
        table_data = soup.find_all('dl', class_="race_top_data_info fc")
        race_ids = []
        
        for dl in table_data:
            a_tag = dl.find('a')
            if not a_tag:
                continue
                
            href = a_tag['href']
            title = a_tag['title']
            #print(f"レース名: {title}")
            #print(f"URL: {href}")
            
            match = re.search(r"/race/(\d+)", href)
            if match:
                race_ids.append(match.group(1))
            else:
                print(f"マッチしないURLがありました: {href}")
        
        return race_ids

    def save_race_data(self, race_id: str, base_dir: Path, date_id: str) -> bool:
        """レース情報をCSVファイルに保存する"""
        # 日付ごとのディレクトリを作成
        output_dir = base_dir / date_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 既に保存済みの場合はスキップ
        csv_path = output_dir / f"{race_id}.csv"
        if csv_path.exists():
            return True
        
        url = self.RACE_URL.format(race_id=race_id)
        html = self._make_request(url)
        if not html:
            return False
        
        try:
            df = parse_race_results(html)
            if df is not None and not df.empty:
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                return True
            return False
        except Exception as e:
            print(f"\nデータ保存エラー ({race_id}): {e}")
            return False

def main():
    client = NetkeibaClient()
    base_dir = Path("./data/training")
    base_dir.mkdir(exist_ok=True)
    
    # 2023年のデータを取得する
    all_date_ids = []
    print("カレンダーからレース開催日を取得中...")
    for month in range(12, 13):
        date_ids = client.get_calendar_dates(2024, month)
        all_date_ids.extend(date_ids)
    
    total_dates = len(all_date_ids)
    print(f"合計{total_dates}日のレース日を取得しました")
    
    for date_index, date_id in enumerate(all_date_ids, 1):
        print(f"\n日程 {date_index}/{total_dates}: {date_id}")
        try:
            race_ids = client.get_race_ids_by_date(date_id)
            total_races = len(race_ids)
            print(f"- {total_races}レースを取得します")
            
            for race_index, race_id in enumerate(race_ids, 1):
                print(f"  レース {race_index}/{total_races}: {race_id}", end="")
                if client.save_race_data(race_id, base_dir, date_id):
                    print(" ✓")
                else:
                    print(" ×")
                
        except KeyboardInterrupt:
            print("\n\n処理を中断しました")
            return
        except Exception as e:
            print(f"\nエラーが発生しました: {e}")
            continue

if __name__ == "__main__":
    main()

