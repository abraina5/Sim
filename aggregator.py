import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime

response = requests.get("https://api.votehub.com/polls")
data = response.json()

df = pd.DataFrame(data)

print(df.columns)
