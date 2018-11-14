import pandas as pd
from random import sample, randrange

d = {
        'python' : pd.Series([randrange(1, 11) for _ in range(0, 50)], index=sample(range(100), 50)),
        'csharp' : pd.Series([randrange(1, 11) for _ in range(0, 50)], index=sample(range(100), 50)),
        'cpp' : pd.Series([randrange(1, 11) for _ in range(0, 50)], index=sample(range(100), 50)),
        'java' : pd.Series([randrange(1, 11) for _ in range(0, 50)], index=sample(range(100), 50)),
        'javascript' : pd.Series([randrange(1, 11) for _ in range(0, 50)], index=sample(range(100), 50)),
        'haskel' : pd.Series([randrange(1, 11) for _ in range(0, 50)], index=sample(range(100), 50)),
        'c' : pd.Series([randrange(1, 11) for _ in range(0, 50)], index=sample(range(100), 50)),
        'r' : pd.Series([randrange(1, 11) for _ in range(0, 50)], index=sample(range(100), 50)),
        'php' : pd.Series([randrange(1, 11) for _ in range(0, 50)], index=sample(range(100), 50)),
        'swift' : pd.Series([randrange(1, 11) for _ in range(0, 50)], index=sample(range(100), 50)),
        'assembly' : pd.Series([randrange(1, 11) for _ in range(0, 50)], index=sample(range(100), 50)),
        'go' : pd.Series([randrange(1, 11) for _ in range(0, 50)], index=sample(range(100), 50)),
        'ruby' : pd.Series([randrange(1, 11) for _ in range(0, 50)], index=sample(range(100), 50)),
        'perl' : pd.Series([randrange(1, 11) for _ in range(0, 50)], index=sample(range(100), 50))
         }

df = pd.DataFrame(d)
print(df.head())
print(df.shape)

df_columns = list(df.columns)

# most popular languages
most_popular_dict = { key: df[key].mean() for key in df_columns }
most_popular_dict = sorted(most_popular_dict.items(), key=lambda kv: kv[1])
most_popular_dict = most_popular_dict[::-1]
print(most_popular_dict)

# most popular language per user

