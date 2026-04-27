import pandas as pd
import numpy as np
import requests
from datetime import date

response = requests.get("https://api.votehub.com/polls")
data = response.json()

df = pd.DataFrame(data)

df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")

mask = ((pd.Timestamp(date.today()) - df["end_date"]).dt.days < 45)
new = df[mask].copy()

#Trump Favorability aggregator

favorabilityMask = ((new["poll_type"] == "favorability") & (new['subject'] == "Donald Trump"))
favorabilityDF = new[favorabilityMask].copy()

favorabilityDF["approve_pct"] = favorabilityDF["answers"].apply(
    lambda x: next((item["pct"] for item in x if item["choice"] == "Favorable"), None)
)

favorabilityDF["MoE"] = 1.96 * (
    (favorabilityDF["approve_pct"] / 100) *
    (1 - favorabilityDF["approve_pct"] / 100) /
    favorabilityDF["sample_size"]
) ** 0.5 * 100

favorabilityDF["MoE"] = favorabilityDF["MoE"].round(3)

print(favorabilityDF[["approve_pct","MoE"]])
print(favorabilityDF.shape)
print(favorabilityDF.columns)
print(favorabilityDF["answers"].head())
