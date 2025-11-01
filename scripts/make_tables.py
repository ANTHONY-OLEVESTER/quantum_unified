import pandas as pd
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "paper" / "tables_auto.tex"

rows = []
if DATA.exists():
    for csv in DATA.glob("*.csv"):
        try:
            df = pd.read_csv(csv, nrows=1)
            cols = ", ".join(df.columns[:5])
            rows.append((csv.name, cols))
        except Exception:
            pass

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8") as f:
    f.write("\\section*{Data inventory}\n")
    f.write("\\begin{tabular}{ll}\n\\toprule\nFile & Columns (first few)\\\\\\midrule\n")
    for name, cols in sorted(rows):
        safe_cols = cols.replace("_", "\\_")
        f.write(f"{name} & {safe_cols}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
print(f"[tables] Wrote {OUT}")

