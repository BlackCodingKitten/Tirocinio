import json
import pandas as pd


whisper_df = pd.read_parquet('data/metrics/pandas_parquet/whisper.parquet')
gpt4o_df = pd.read_parquet('data/metrics/pandas_parquet/gpt.parquet')
    
whisper = whisper_df.to_dict("index")
gpt4o = gpt4o_df.to_dict("index")

for i in whisper.values():
    print(f"{i}\n")

# def merge (d1, d2):
    