import re, sys, json, pathlib
p = pathlib.Path("data/phase6_summary.txt")
d = {"alpha_intercept": None, "varY_slope": None, "varY_ci": None}
if p.exists():
    txt = p.read_text()
    m = re.search(r"alpha intercept\s*˜\s*([+\-]?\d+\.\d+).*?\[([+\-]\d+\.\d+)\s*,\s*([+\-]\d+\.\d+)\]", txt, re.I|re.S)
    if m:
        d["alpha_intercept"] = float(m.group(1))
        d["alpha_ci"] = [float(m.group(2)), float(m.group(3))]
    m = re.search(r"Var\(Y\)\s*slope\s*˜\s*([+\-]?\d+\.\d+)\s*\[\s*([+\-]\d+\.\d+)\s*,\s*([+\-]\d+\.\d+)\s*\]", txt, re.I)
    if m:
        d["varY_slope"] = float(m.group(1))
        d["varY_ci"] = [float(m.group(2)), float(m.group(3))]
print(json.dumps(d, indent=2))
