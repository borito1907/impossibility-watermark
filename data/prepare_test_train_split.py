import pandas as pd
from sklearn.model_selection import train_test_split

def split_csv(csv_file, test_size, output_train_file, output_test_file, random_state=None):
    # Load the CSV file into a DataFrame
    data = pd.read_csv(csv_file)
    
    # Perform the train-test split
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    
    # Save the split data to new CSV files
    train_data.to_csv(output_train_file, index=False)
    test_data.to_csv(output_test_file, index=False)
    
    print(f"Train data saved to {output_train_file}")
    print(f"Test data saved to {output_test_file}")

# Example usage:
csv_file = './data/IMP/all.csv'
test_size = 0.3  
output_train_file = './data/IMP/train.csv'
output_test_file = './data/IMP/test.csv'

split_csv(csv_file, test_size, output_train_file, output_test_file)
