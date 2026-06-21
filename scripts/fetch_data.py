from datacentertracesdatasets import loadtraces
import pandas as pd
from pathlib import Path

# Pull machine_usage at a 300-second (5-min) timestep as a DataFrame.
df = loadtraces.get_trace(
    trace_name="alibaba2018",
    trace_type="machine_usage",
    stride_seconds=300,
    format="dataframe",
)

print("Full slice shape:", df.shape)
print(df.head())

# Save it to the gitignored data folder as Parquet (compact + fast to read).
out = Path("data/alibaba_machine_usage_300s.parquet")
out.parent.mkdir(exist_ok=True)
df.to_parquet(out)
print(f"Saved to {out}")