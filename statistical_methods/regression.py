import pandas as pd
import quandl
import math

df = quandl.get('WIKI/GOOGL')
#stock example
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100
df = df[['Adj. Close', 'HL_PCT', 'Adj. Volume']]

forecast_column = 'Adj. Close'
df.fillna(-9999, inplace=True)

#predicting 1% of a data frame
forecast_out = int(math.ceil(0.01*len(df)))
#column shifted up
df['label'] = df[forecast_column].shift(-forecast_out)

print(df.head())

