## Custom-built tools for use with the repository's DuckDB database

from contextlib import contextmanager
import duckdb
import os
import pandas as pd
import uuid

# --- FUNCTIONS FOR PREPARING DATABASE --- #

def generate_uuid_list(df):
    """
    Generate a UUID for each row in the provided DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame for which to generate UUIDs.

    Returns:
    - list: A list of unique UUID strings, one for each row in the DataFrame.
    """
    return [str(uuid.uuid4()) for _ in df.index]
    

def extract_hotel_number(filepath):
    """
    Extract the hotel number from a filepath.

    Assumes the filename contains the hotel number prefixed by 'H',
    followed by a numeric identifier (e.g., 'H1').

    Parameters:
    - filepath (str): The filepath from which to extract the hotel number.

    Returns:
    - int: The extracted hotel number as an integer.
    """
    base_filename = os.path.basename(filepath).split('.')[0]
    hotel_number = base_filename[1:]
    return int(hotel_number)



def add_hotel_number_to_dataframe(input_filepath, output_filepath=None,
                                  col_hotelnum='HotelNumber', col_id='UUID',
                                  save_to_parquet=True, engine='pyarrow',
                                  compression='snappy'):
    """
    Add a hotel number and UUIDs to each row of a DataFrame loaded from a Parquet file.

    Optionally, save the modified DataFrame back to a Parquet file.

    Parameters:
    - input_filepath (str): Path to the input Parquet file.
    - output_filepath (str, optional): Path to save the modified DataFrame to a Parquet file.
    - col_hotelnum (str, optional): Name of the column for hotel numbers. Defaults to 'HotelNumber'.
    - col_id (str, optional): Name of the column for UUIDs. Defaults to 'UUID'.
    - save_to_parquet (bool, optional): Whether to save the modified DataFrame to a Parquet file. Defaults to True.
    - engine (str, optional): Parquet engine to use ('pyarrow' or 'fastparquet'). Defaults to 'pyarrow'.
    - compression (str, optional): Compression to use for saving to Parquet. Defaults to 'snappy'.

    Returns:
    - pandas.DataFrame: The modified DataFrame with added hotel numbers and UUIDs.
    """
    df = pd.read_parquet(input_filepath)
    
    df[col_hotelnum] = extract_hotel_number(input_filepath)
    df[col_id] = generate_uuid_list(df)
    
    if save_to_parquet and output_filepath:
        df.to_parquet(output_filepath, engine=engine, compression=compression)
    
    return df

# --- FUNCTIONS FOR INTERACTING WITH DATABASE --- #

@contextmanager
def duckdb_connection(database_path):
    """
    Context manager for managing DuckDB database connections.

    This context manager simplifies the process of connecting to and
    closing connections with a DuckDB database, ensuring that the database
    connection is properly closed after use, even if exceptions occur.

    Parameters:
    - database_path (str): The file path to the DuckDB database. If the file
      does not exist, DuckDB will create a new database at this location.

    Yields:
    - conn: A DuckDB connection object that can be used to execute SQL
      commands within a 'with' block.

    Example:
        with duckdb_connection('./data/my_database.duckdb') as conn:
            conn.execute("SELECT * FROM my_table")

    Note: Replace './data/my_database.duckdb' with the path to your actual
    DuckDB database file.
    """
    try:
        conn = duckdb.connect(database_path)
        yield conn
    finally:
        conn.close()


def alter_table(command, db_path):
    
    with duckdb_connection(db_path) as conn:
        conn.execute(command)
        print('Completed successfully.')


def get_col_dtypes(table_name, db_path):
    

    with duckdb_connection(db_path) as conn:
        column_info = conn.execute(f"PRAGMA table_info({table_name})").fetchall()

    # Print information about each column
    for column in column_info:
        column_name, column_dtype = column[1], column[2]
        print(f"Column Name: {column_name}, Data Type: {column_dtype}")
        
    return column_info
