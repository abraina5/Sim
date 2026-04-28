import pandas as pd
import numpy as np
import requests
from datetime import date
from scipy.stats import norm

response = requests.get("https://api.votehub.com/polls")
data = response.json()

df = pd.DataFrame(data)

df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")

mask = ((pd.Timestamp(date.today()) - df["end_date"]).dt.days < 45)
new = df[mask].copy()

#Trump Approval aggregator

favorabilityMask = ((new["poll_type"] == "approval") & (new['subject'] == "Donald Trump"))
favorabilityDF = new[favorabilityMask].copy()

favorabilityDF["approve_pct"] = favorabilityDF["answers"].apply(
    lambda x: next((item["pct"] for item in x if item["choice"] == "Approve"), None)
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

# Older election day approval vs vote received (All from Gallup)

# Structure is approval : vote total

# LSRL is y = 0.39x + 0.3 where x is approval and y is vote total
# R = 0.765 (Moderate to Strong correlation)

electionDay = {
    0.75 : 0.574, #1956 Eisenhower
    0.59 : 0.4955, # 1960 Eiseinhower approval : Nixon votes
    0.7 : 0.611, # 1964 Johnson
    0.43 : 0.43, # 1968 Johnson approval : Humphrey votes
    0.62 : 0.607, # 1972 Nixon
    # No November data for Ford (1976)
    0.31 : 0.41, # 1980 Carter
    0.61 : 0.588, # 1984 Reagan
    0.57 : 0.534, # 1988 Reagan approval : HW Bush votes
    0.43 : 0.375, # 1992 HW Bush
    0.58 : 0.492, # 1996 Clinton
    0.63 : 0.484, # 2000 Clinton approval : Gore votes
    0.53 : 0.507, # 2004 Bush
    0.29 : 0.457, # 2008 Bush approval : McCain votes
    0.5 : 0.511, # 2012 Obama
    0.54 : 0.482, # 2016 Obama approval : Clinton votes
    0.43 : 0.468, # 2020 Trump
    0.37 : 0.483 # 2024 Biden approval : Harris votes
}
