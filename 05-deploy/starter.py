import pickle
import argparse
import pandas as pd

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def predict(args):
    print(args)
    file_name = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{int(args.year):04d}-{int(args.month):02d}.parquet'
    #file_name = f'../../data/yellow_tripdata_{int(args.year):04d}-{int(args.month):02d}.parquet'
    print('reading:', file_name)
    df = read_data(file_name)
    print('data size:', df.shape)
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    return y_pred


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--year')
    parser.add_argument('--month')
    args = parser.parse_args()

    y_pred = predict(args)
    print(f'y_pred: {y_pred.mean():.2f} +- {y_pred.std():.2f}')


    



