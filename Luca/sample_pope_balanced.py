"""
Sample 100 balanced POPE examples: 50 "yes" (no hallucination) + 50 "no" (hallucination).
Output saved to POPE/balanced_100/balanced_sample.parquet
"""

import pandas as pd
import os

POPE_DATA_DIR = os.path.join(os.path.dirname(__file__), "POPE", "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "POPE", "balanced_100")
OUT_FILE = os.path.join(OUT_DIR, "balanced_sample.parquet")

SEED = 42
N_PER_CLASS = 50

def main():
    # Load all test parquet shards
    shards = sorted([
        os.path.join(POPE_DATA_DIR, f)
        for f in os.listdir(POPE_DATA_DIR)
        if f.endswith(".parquet")
    ])
    df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)

    print(f"Loaded {len(df)} total examples")
    print(f"Answer distribution:\n{df['answer'].value_counts()}")

    yes = df[df["answer"] == "yes"].sample(n=N_PER_CLASS, random_state=SEED)
    no  = df[df["answer"] == "no"].sample(n=N_PER_CLASS, random_state=SEED)

    balanced = pd.concat([yes, no], ignore_index=True).sample(frac=1, random_state=SEED)

    os.makedirs(OUT_DIR, exist_ok=True)
    balanced.to_parquet(OUT_FILE, index=False)

    print(f"\nSaved {len(balanced)} samples to {OUT_FILE}")
    print(f"Final distribution:\n{balanced['answer'].value_counts()}")

if __name__ == "__main__":
    main()
