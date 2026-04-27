import pandas as pd
import requests
from datetime import date

response = requests.get("https://api.votehub.com/polls")
data = response.json()

df = pd.DataFrame(data)

df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")

mask = ((pd.Timestamp(date.today()) - df["end_date"]).dt.days < 45)
new = df[mask].copy()

print(new.head())
print(new.shape)
print(df.shape)
