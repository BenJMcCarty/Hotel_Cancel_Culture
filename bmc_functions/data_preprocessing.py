import pandas as pd

def split_dataset(df, percentage=10):
    """
    Split the dataset into training and holdout sets 
    based on a specified percentage of the total rows.
    """
    split_idx = int(len(df) * (1 - (percentage / 100)))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    holdout_df = df.iloc[split_idx:].reset_index(drop=True)
    return train_df, holdout_df

def save_dataset(df, path='./data', filename='dataset.parquet',
                 engine='pyarrow', compression='brotli'):
    """
    Save the dataset to a given path with a specified filename.
    """
    df.to_parquet(f'{path}/{filename}',
                  engine = engine,
                  compression = compression)

def split_and_save_dataset_by_percentage(df, percentage=10,
                                         save_path='./data',
                                         filename_prefix='H'):
    """
    Split the dataset into training and holdout sets 
    based on a specified percentage of the total rows
    and save them with appropriate filenames.
    """
    train_df, holdout_df = split_dataset(df, percentage)
    training_path = f'{save_path}/{filename_prefix}_Training.parquet'
    holdout_path = f'{save_path}/{filename_prefix}_Validation.parquet'
    save_dataset(train_df, training_path)
    save_dataset(holdout_df, holdout_path)

def create_arrival_date_columns(df, arrival_date_cols=['ArrivalDateYear',
                                                       'ArrivalDateMonth',
                                                       'ArrivalDateDayOfMonth']):
    """
    Create arrival date and departure date columns 
    based on the provided columns representing year, month, and day of month.
    """
    arrival_date_full_str = df[arrival_date_cols].astype(str).agg('-'.join, axis=1)
    arrival_date_dt = pd.to_datetime(arrival_date_full_str, yearfirst=True)
    departure_date = arrival_date_dt + pd.to_timedelta(df['StaysInWeekendNights'],
                                                       unit='D') + \
                     pd.to_timedelta(df['StaysInWeekNights'], unit='D')
    df['Arrival_Date'] = arrival_date_dt
    df['Departure_Date'] = departure_date
    df['Booking_Date'] = df['Arrival_Date'] - pd.to_timedelta(df['LeadTime'], unit='D')
    return df

def create_and_save_arrival_date_dataset(df, hotel_number, path='./data',
                                         percentage=10, filename_prefix='H'):
    """
    Create arrival date columns and split the dataset by percentage, then save them.
    """
    df = create_arrival_date_columns(df)
    split_and_save_dataset_by_percentage(df, percentage,
                                         save_path=path,
                                         filename_prefix=filename_prefix)

# Example usage:
# df = pd.read_parquet('./data/H1.parquet')
# create_and_save_arrival_date_dataset(df, hotel_number=1)
