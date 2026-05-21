import pandas as pd
import numpy as np
import requests
from datetime import date
import time
import random

response = requests.get("https://api.votehub.com/polls")
data = response.json()

df = pd.DataFrame(data)

df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")

mask = ((pd.Timestamp(date.today()) - df["end_date"]).dt.days < 60)
new = df[mask].copy()

#Trump Approval aggregator

favorabilityMask = ((new["poll_type"] == "approval") & (new['subject'] == "Donald Trump"))
favorabilityDF = new[favorabilityMask].copy()

favorabilityDF["approve_pct"] = favorabilityDF["answers"].apply(
    lambda x: next((item["pct"] for item in x if item["choice"] == "Approve"), None)
)

# Utilizing 0.5 to maximize MoE

favorabilityDF["MoE"] = 1.96 * ((0.5*0.5) / favorabilityDF["sample_size"]) ** 0.5 * 100

favorabilityDF["MoE"] = favorabilityDF["MoE"].round(3)

print(favorabilityDF[["approve_pct","MoE"]])
print(favorabilityDF.shape)
print(favorabilityDF.columns)
print(favorabilityDF["answers"].head())

# Polls weighting for aggregate based on days since

favorabilityDF["days_old"] = (pd.Timestamp.today() - favorabilityDF["end_date"]).dt.days
lmbda = 0.05
favorabilityDF["weight"] = np.exp(-lmbda * favorabilityDF["days_old"])
favorabilityDF["norm_weight"] = favorabilityDF["weight"] / favorabilityDF["weight"].sum()

# Calculate Weighted Metrics
weighted_approval = (favorabilityDF["approve_pct"] * favorabilityDF["norm_weight"]).sum()

# Aggregate MoE (Using combined sample size for overall uncertainty)
total_n = favorabilityDF["sample_size"].sum()

# We use p=0.5 for a conservative, model-based estimate
sampling_error = np.sqrt(0.25 / total_n) * 100

non_sampling_error = 2.0

sigma = np.sqrt(sampling_error**2 + non_sampling_error**2)
sigma = (sigma / 100).round(5)

estimated_votes_approval = (0.39*(weighted_approval / 100) + 0.3)

print(estimated_votes_approval)
print(sigma)

# Senate elections





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


# Sim model
tick_speed = 1
total_ticks = 0

opposite_total = 1 - (random.randint(90,300) * 0.0001) - estimated_votes_approval
print(opposite_total)

states = {"Alabama":{"pvi":15,"precincts_in":0},"Alaska":{"pvi":9,"precincts_in":0},"Arizona":{"pvi":3,"precincts_in":0},"Arkansas":{"pvi":16,"precincts_in":0},"California":{"pvi":-13,"precincts_in":0},"Colorado":{"pvi":-4,"precincts_in":0},"Connecticut":{"pvi":-6,"precincts_in":0},"Delaware":{"pvi":-7,"precincts_in":0},"Florida":{"pvi":3,"precincts_in":0},"Georgia":{"pvi":1,"precincts_in":0},"Hawaii":{"pvi":-16,"precincts_in":0},"Idaho":{"pvi":18,"precincts_in":0},"Illinois":{"pvi":-7,"precincts_in":0},"Indiana":{"pvi":11,"precincts_in":0},"Iowa":{"pvi":6,"precincts_in":0},"Kansas":{"pvi":11,"precincts_in":0},"Kentucky":{"pvi":16,"precincts_in":0},"Louisiana":{"pvi":12,"precincts_in":0},"Maine":{"pvi":-2,"precincts_in":0},"Maryland":{"pvi":-13,"precincts_in":0},"Massachusetts":{"pvi":-14,"precincts_in":0},"Michigan":{"pvi":-1,"precincts_in":0},"Minnesota":{"pvi":-1,"precincts_in":0},"Mississippi":{"pvi":10,"precincts_in":0},"Missouri":{"pvi":10,"precincts_in":0},"Montana":{"pvi":11,"precincts_in":0},"Nebraska":{"pvi":13,"precincts_in":0},"Nevada":{"pvi":-1,"precincts_in":0},"New Hampshire":{"pvi":-1,"precincts_in":0},"New Jersey":{"pvi":-6,"precincts_in":0},"New Mexico":{"pvi":-5,"precincts_in":0},"New York":{"pvi":-11,"precincts_in":0},"North Carolina":{"pvi":1,"precincts_in":0},"North Dakota":{"pvi":19,"precincts_in":0},"Ohio":{"pvi":6,"precincts_in":0},"Oklahoma":{"pvi":20,"precincts_in":0},"Oregon":{"pvi":-4,"precincts_in":0},"Pennsylvania":{"pvi":-1,"precincts_in":0},"Rhode Island":{"pvi":-6,"precincts_in":0},"South Carolina":{"pvi":7,"precincts_in":0},"South Dakota":{"pvi":16,"precincts_in":0},"Tennessee":{"pvi":14,"precincts_in":0},"Texas":{"pvi":5,"precincts_in":0},"Utah":{"pvi":14,"precincts_in":0},"Vermont":{"pvi":-15,"precincts_in":0},"Virginia":{"pvi":-3,"precincts_in":0},"Washington":{"pvi":-5,"precincts_in":0},"West Virginia":{"pvi":22,"precincts_in":0},"Wisconsin":{"pvi":-1,"precincts_in":0},"Wyoming":{"pvi":25,"precincts_in":0}}
votes = {"Alabama":{"rVote":0,"dVote":0},"Alaska":{"rVote":0,"dVote":0},"Arizona":{"rVote":0,"dVote":0},"Arkansas":{"rVote":0,"dVote":0},"California":{"rVote":0,"dVote":0},"Colorado":{"rVote":0,"dVote":0},"Connecticut":{"rVote":0,"dVote":0},"Delaware":{"rVote":0,"dVote":0},"Florida":{"rVote":0,"dVote":0},"Georgia":{"rVote":0,"dVote":0},"Hawaii":{"rVote":0,"dVote":0},"Idaho":{"rVote":0,"dVote":0},"Illinois":{"rVote":0,"dVote":0},"Indiana":{"rVote":0,"dVote":0},"Iowa":{"rVote":0,"dVote":0},"Kansas":{"rVote":0,"dVote":0},"Kentucky":{"rVote":0,"dVote":0},"Louisiana":{"rVote":0,"dVote":0},"Maine":{"rVote":0,"dVote":0},"Maryland":{"rVote":0,"dVote":0},"Massachusetts":{"rVote":0,"dVote":0},"Michigan":{"rVote":0,"dVote":0},"Minnesota":{"rVote":0,"dVote":0},"Mississippi":{"rVote":0,"dVote":0},"Missouri":{"rVote":0,"dVote":0},"Montana":{"rVote":0,"dVote":0},"Nebraska":{"rVote":0,"dVote":0},"Nevada":{"rVote":0,"dVote":0},"New Hampshire":{"rVote":0,"dVote":0},"New Jersey":{"rVote":0,"dVote":0},"New Mexico":{"rVote":0,"dVote":0},"New York":{"rVote":0,"dVote":0},"North Carolina":{"rVote":0,"dVote":0},"North Dakota":{"rVote":0,"dVote":0},"Ohio":{"rVote":0,"dVote":0},"Oklahoma":{"rVote":0,"dVote":0},"Oregon":{"rVote":0,"dVote":0},"Pennsylvania":{"rVote":0,"dVote":0},"Rhode Island":{"rVote":0,"dVote":0},"South Carolina":{"rVote":0,"dVote":0},"South Dakota":{"rVote":0,"dVote":0},"Tennessee":{"rVote":0,"dVote":0},"Texas":{"rVote":0,"dVote":0},"Utah":{"rVote":0,"dVote":0},"Vermont":{"rVote":0,"dVote":0},"Virginia":{"rVote":0,"dVote":0},"Washington":{"rVote":0,"dVote":0},"West Virginia":{"rVote":0,"dVote":0},"Wisconsin":{"rVote":0,"dVote":0},"Wyoming":{"rVote":0,"dVote":0}}
for state in states:
    states[state]["rVote"] = estimated_votes_approval
    states[state]["dVote"] = opposite_total
    states[state]["rVote"] += states[state]["pvi"] / 200
    states[state]["dVote"] -= states[state]["pvi"] / 200
    print(states[state])

while (total_ticks < 10) :
    total_ticks += 1
    time.sleep(1 / tick_speed)
    print(total_ticks)
