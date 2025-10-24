import pandas as pd

csv_file = "data.csv"

# Try different encodings
encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']

for encoding in encodings:
    try:
        df = pd.read_csv(csv_file, encoding=encoding)
        print(f"✓ Successfully loaded CSV with {encoding} encoding")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        break
    except Exception as e:
        print(f"✗ Failed with {encoding}: {str(e)[:50]}")
