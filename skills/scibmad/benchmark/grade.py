import json, sys, glob, os
ref = json.load(open(sys.argv[2])); tot = 0
for d in sorted(glob.glob(sys.argv[1] + "/ch*")):
    ch = os.path.basename(d)
    try: t = json.load(open(d + "/transcript.json")); cost = t.get("total_cost_usd", 0); sec = t.get("duration_ms", 0)/1000
    except Exception: cost = sec = 0
    tot += cost
    try: r = json.load(open(d + "/results.json"))
    except Exception as e: print(f"{ch}: no results.json (${cost:.2f}, {sec:.0f}s)"); continue
    rows = []
    for k, v in r.items():
        kk = k.replace("ring.context.", "context.")
        if kk in ref and isinstance(v, (int, float)):
            x, y = float(v), ref[kk]
            if kk[-2:] in ("x1", "x2", "y1", "y2"): diff = abs((x - y + 0.5) % 1 - 0.5)
            else: diff = abs(x - y) / max(abs(y), 1e-12)
            rows.append(f"{kk}:{diff:.0e}")
    print(f"{ch} ${cost:.2f} {sec:.0f}s | " + (" ".join(rows) if rows else "no comparable keys: " + ",".join(list(r)[:8])))
print(f"total ${tot:.2f}")
