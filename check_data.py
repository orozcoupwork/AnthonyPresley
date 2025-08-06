# check_data.py
import pandas as pd
import numpy as np

print("="*60)
print("DATA DIAGNOSTIC CHECK")
print("="*60)

# Load sales data
print("\nLoading sales data...")
sales_df = pd.read_csv('data/sales.csv')
print(f"Total sales records: {len(sales_df):,}")

# Check location IDs
print(f"\nUnique locations in sales data: {sales_df['location_id'].nunique()}")
locations = sorted(sales_df['location_id'].unique())
print(f"Location IDs: {locations}")

# Check if 145137 exists
if 145137 in sales_df['location_id'].values:
    print(f"\n✓ Location 145137 EXISTS in sales data")
    loc_data = sales_df[sales_df['location_id'] == 145137]
    print(f"  Records for 145137: {len(loc_data)}")
else:
    print(f"\n✗ Location 145137 NOT FOUND in sales data")

# Load locations.csv to cross-check
print("\n" + "-"*40)
print("Checking locations.csv...")
locations_df = pd.read_csv('data/locations.csv')
print(f"Total locations in locations.csv: {len(locations_df)}")
print("\nAll locations in locations.csv:")
for idx, row in locations_df.iterrows():
    print(f"  ID: {row['id']}, Corporation: {row['corporation_id']}, Postal: {row['postal_code']}")

# Check if 145137 is in locations.csv
if 145137 in locations_df['id'].values:
    print(f"\n✓ Location 145137 EXISTS in locations.csv")
else:
    print(f"\n✗ Location 145137 NOT FOUND in locations.csv")

# Show sample of sales data
print("\n" + "-"*40)
print("Sample of sales data (first 10 records):")
print(sales_df.head(10))

# Show sales breakdown by location
print("\n" + "-"*40)
print("Sales records by location:")
location_counts = sales_df['location_id'].value_counts().sort_index()
for loc_id, count in location_counts.items():
    print(f"  Location {loc_id}: {count:,} records")

# Check data types
print("\n" + "-"*40)
print("Data types in sales.csv:")
print(sales_df.dtypes)

# Check for any data issues
print("\n" + "-"*40)
print("Checking for data issues...")
print(f"NULL location_ids: {sales_df['location_id'].isna().sum()}")
print(f"NULL sales values (y): {sales_df['y'].isna().sum()}")
print(f"Min location_id: {sales_df['location_id'].min()}")
print(f"Max location_id: {sales_df['location_id'].max()}")