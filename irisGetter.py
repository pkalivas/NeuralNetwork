import pandas as pd
import numpy as np


# Download the data
table = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
table.columns = ['one', 'two', 'three', 'four', 'flower']

# Normalize the data
for column in list(table.columns):
	if column != 'flower':
		table[column] = (table[column] - table[column].mean()) / table[column].std();

# Replace the flower names 
table['flower'] = table.flower.map({'Iris-setosa' : 0, 'Iris-versacolor' : 1, 'Iris-virginica' : 2})

# Shuffle the dataframe and make sure we dont lose data
table = table.sample(n = table.shape[0])
table.fillna(1, inplace = True)
table.reset_index(inplace = True, drop = True)

# Split into training and testing data
cutoff = int(table.shape[0] - (table.shape[0] * .2))
training = table[table.index < cutoff].copy()
testing = table[table.index >= cutoff].copy()

training_features = training.loc[:, training.columns != 'flower'].copy()
training_target = training.loc[:, training.columns == 'flower'].copy()
testing_features = testing.loc[:, testing.columns != 'flower'].copy()
testing_target = testing.loc[:, testing.columns == 'flower'].copy()

training_features.to_csv('irisdata/training_features.csv', index = False, header = False)
training_target.to_csv('irisdata/training_target.csv', index = False, header = False)
testing_features.to_csv('irisdata/testing_features.csv', index = False, header = False)
testing_target.to_csv('irisdata/testing_target.csv', index = False, header = False)

print('Done:\n', table.groupby(['flower']).agg('count'))
