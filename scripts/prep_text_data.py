# scripts/prep_text_data.py
import os
import json
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split

RAW_DIR = "data/raw/listings/metadata"
OUTPUT_SAMPLE = "data/processed/sample.csv"
OUTPUT_TRAIN = "data/processed/text_train.csv"
OUTPUT_VAL = "data/processed/text_val.csv"
OUTPUT_TEST = "data/processed/text_test.csv"

def extract_field(obj, field):
    """Helper to safely extract first 'value' from a field list."""
    if field in obj and isinstance(obj[field], list) and len(obj[field]) > 0:
        return obj[field][0].get("value")
    return None

def load_json_files():
    """Load raw SNAP JSON lines into DataFrame."""
    rows = []
    files = glob(os.path.join(RAW_DIR, "*.json"))
    print(f"Found {len(files)} JSON files in {RAW_DIR}")

    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    row = {
                        "item_id": obj.get("item_id"),
                        "brand": extract_field(obj, "brand"),
                        "item_name": extract_field(obj, "item_name"),
                        "color": extract_field(obj, "color"),
                        "product_type": extract_field(obj, "product_type"),
                        "model_number": extract_field(obj, "model_number"),
                        "model_year": extract_field(obj, "model_year"),
                        "country": obj.get("country"),
                        "marketplace": obj.get("marketplace"),
                        "domain_name": obj.get("domain_name"),
                    }
                    rows.append(row)
                except Exception as e:
                    print(f"⚠️ Skipping line in {file} due to error: {e}")
    return pd.DataFrame(rows)

def preprocess_and_split(df, top_k=100):
    """Clean, normalize, and create train/val/test splits."""
    # Drop empty rows
    df.dropna(how="all", inplace=True)

    # Rename + keep relevant cols
    df = df[['item_id', 'item_name', 'brand', 'color']].rename(columns={'item_name': 'title'})
    df['title'] = df['title'].fillna("").astype(str).str.strip().str.lower()
    df['brand'] = df['brand'].fillna("unknown").astype(str).str.strip().str.lower()

    # Filter out bad rows
    df = df[(df['title'] != "") & (df['brand'] != "unknown")]

    # Limit to top-k brands
    top_brands = df['brand'].value_counts().nlargest(top_k).index.tolist()
    df['brand'] = df['brand'].apply(lambda x: x if x in top_brands else "other")

    # Split
    train, temp = train_test_split(df, test_size=0.2, stratify=df['brand'], random_state=42)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp['brand'], random_state=42)

    return df, train, val, test

def main():
    df = load_json_files()
    os.makedirs("data/processed", exist_ok=True)

    # Save raw sample
    df.to_csv(OUTPUT_SAMPLE, index=False, encoding="utf-8")
    print(f"✅ Saved raw sample with {len(df)} rows to {OUTPUT_SAMPLE}")

    # Process + split
    df, train, val, test = preprocess_and_split(df)

    train.to_csv(OUTPUT_TRAIN, index=False)
    val.to_csv(OUTPUT_VAL, index=False)
    test.to_csv(OUTPUT_TEST, index=False)

    print("✅ Final splits saved:")
    print("   Train:", train.shape, " Val:", val.shape, " Test:", test.shape)

if __name__ == "__main__":
    main()
