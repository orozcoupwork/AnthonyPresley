import gzip
import os
import sys

def gz_to_csv(gz_filepath):
    """
    Convert a .gz file to .csv in the same directory
    """
    # Check if file exists
    if not os.path.exists(gz_filepath):
        print(f"Error: File '{gz_filepath}' not found")
        return
    
    # Check if it's a .gz file
    if not gz_filepath.endswith('.gz'):
        print("Error: File must have .gz extension")
        return
    
    # Create output filename by removing .gz extension
    # If the file is named 'data.csv.gz', this will create 'data.csv'
    csv_filepath = gz_filepath[:-3]
    
    # If the .gz file doesn't end with .csv.gz, add .csv extension
    if not csv_filepath.endswith('.csv'):
        csv_filepath = gz_filepath[:-3] + '.csv'
    
    try:
        # Read from .gz file and write to .csv file
        with gzip.open(gz_filepath, 'rb') as f_in:
            with open(csv_filepath, 'wb') as f_out:
                f_out.write(f_in.read())
        
        print(f"Successfully converted '{gz_filepath}' to '{csv_filepath}'")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")

# Main execution
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python gz_to_csv.py <path_to_gz_file>")
        print("Example: python gz_to_csv.py data.csv.gz")
    else:
        gz_to_csv(sys.argv[1])