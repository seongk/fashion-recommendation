import pandas as pd

df = pd.read_csv("../data/vali_modified.csv")

groups = df.groupby('split')
groups.get_group('test').to_csv('../data/test_modified2.csv', index=False)
groups.get_group('val').to_csv('../data/vali_modified.csv', index=False)
groups.get_group('train').to_csv('../data/train_modified2.csv', index=False)
