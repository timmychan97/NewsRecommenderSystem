import pandas as pd

df = pd.read_json('dataset/active1000/20170101', lines=True)
print(df.iloc[0])

#print(df.to_string()) 