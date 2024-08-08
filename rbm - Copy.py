import pandas as pd 

dfStone = pd.read_parquet('dados-marcelo\part-00782-f271eeb3-f9cd-4b31-8b36-059b245d4ed0-c000.snappy.parquet')
dfStone.to_csv('dfStone.csv')
print('Fim')