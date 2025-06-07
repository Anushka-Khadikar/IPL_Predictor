import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pickle
import os

# Load synthetic dataset
df = pd.read_csv('synthetic_ipl_data.csv')

X = df.drop('winner', axis=1)
y = df['winner']

categorical_features = ['batting_team', 'bowling_team', 'venue']
numerical_features = ['target_score', 'current_score', 'overs_left', 'wickets_left']

preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(drop='first'), categorical_features),
    ('num', StandardScaler(), numerical_features)
])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

os.makedirs('model', exist_ok=True)
with open('model/pipe.pkl', 'wb') as f:
    pickle.dump(pipeline, f)