import pandas as pd
from pathlib import Path

files = sorted(Path("data").glob("*.tsv"))
df = pd.concat([pd.read_csv(f, sep="\t") for f in files], ignore_index=True)


print(f"{len(df)} openings loaded from {len(files)} files")
print(df.head())

# How long are the move sequences?
df["num_moves"] = df["pgn"].str.count(r"\d+\.")
print(df["num_moves"].describe())

# Try a search by name
print(df[df["name"].str.contains("najdorf", case=False)][["eco", "name"]])