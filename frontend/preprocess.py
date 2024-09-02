import pandas as pd

data = pd.read_csv('heart.csv')

data.columns = ['age', 'sex', 'chestpaintype', 'trestbps', 'cholesterol', 
                'fbs', 'restecg', 'thalach', 'exang', 'stdepression', 'stslope', 
                'num_major_vessels', 'thal', 'target']


X = data.drop('target', axis=1)
y = data['target']