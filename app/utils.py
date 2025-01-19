import pandas as pd
import numpy as np
# from sklearn.preprocessing import LabelEncoder

# Function to load raw data
def load_data(filepath):
    return pd.read_csv(filepath)


def preprocess_data(df) : 
    
    # -------- Handle missing value ------------>
    missing_values = df.isnull().sum()
    print(missing_values)
    # No missing values found thus no need to handle such cases. 


    # -------- Handle categorical data ------------------>
    # Categorical, impact, urgency, assignment_group
    # Apply one hot encoding
    df = pd.get_dummies(df, columns = ['category', 'impact', 'urgency', 'assignment_group'], drop_first = True)
    print(df.head())



    # Convert Date Columns
    df['opened_at'] = pd.to_datetime(df['opened_at'])
    df['resolved_at'] = pd.to_datetime(df['resolved_at'])
    df['closed_at'] = pd.to_datetime(df['closed_at'])



    # Handle boolean columns
    # Convert boolean columns to 0 and 1
    df['made_sla'] = df['made_sla'].astype(int)
    df['knowledge'] = df['knowledge'].astype(int)
    df['u_priority_confirmation'] = df['u_priority_confirmation'].astype(int)

    return df


# Function to save processed data
def save_processed_data(df, filepath='./data/processed_data.pkl'):
    """
    Saves the preprocessed data to a specified file in pickle format.

    Parameters:
        df (DataFrame): The preprocessed pandas DataFrame.
        filepath (str): The file path where the processed data will be saved.
    """
    df.to_pickle(filepath)
    print(f"Processed data saved to {filepath}")





# Text Data preprocessing
# Text preprocessing steps:

# Convert text to lowercase.
# Remove stopwords.
# Remove punctuation and special characters.
# Perform tokenization (splitting text into words).
# Stem or lemmatize the words (optional but can improve performance).