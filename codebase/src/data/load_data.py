import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path)
    X = df.drop('sentiment', axis=1)
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

if __name__=='__main__':
    X_train, X_test, y_train, y_test = \
        load_data('/home/tvnl/Desktop/Desktop Folders/python/ML/movie-review-analysis/codebase/data/processed/IMDB Dataset.csv')
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)