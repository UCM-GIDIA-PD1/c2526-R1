import time
from tqdm import tqdm
from get_video_info_api import get_info
from get_video_ids_bs4 import get_random_ids
import pandas as pd
from pandas import DataFrame
import json
import datetime

def collect_all_data(num_videos):
    print("PART 1.1 - GETTING RANDOM IDS FOR ADULTS VIDEOS")
    palabras, ids = get_random_ids(num_ids=num_videos, after_date=str(datetime.date.today()-datetime.timedelta(days=1)))
    print(list(zip(palabras,ids)))

    print("PART 1.2 - GETTING RANDOM IDS FOR KIDS VIDEOS")

    print("PART 2 - PROCESSING VIDEOS")
    df_videos = []

    for id in tqdm(ids): #tqdm
        try:
            df_videos.append(get_info(id))
            #print(df_videos)
            time.sleep(0.2) #to not get too many requests error
        except Exception as e: print("Ran into exception", e, "for video", id) #for some videos the downloader does not work for some reason

    df_data = pd.concat(df_videos, ignore_index=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    data_parquet = df_data.to_parquet(f"src/data/data_videos_{timestamp}.parquet", index=False)
    return df_data

if __name__ == '__main__':
    data = collect_all_data(20)#(1000)