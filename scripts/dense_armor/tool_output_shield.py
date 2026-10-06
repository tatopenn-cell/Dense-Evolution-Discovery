import json
import time
import urllib.request
import zipfile
from datetime import datetime, timezone

import numpy as np
from dense_armor import Armatura

URL = "https://raw.githubusercontent.com/OTRF/Security-Datasets/master/datasets/compound/apt29/day1/apt29_evals_day1_manual.zip"
ZIP_PATH = "data/apt29_day1_manual.zip"


def fetch_real_series():
    urllib.request.urlretrieve(URL, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        member = zf.namelist()[0]
        timestamps, event_ids = [], []
        with zf.open(member) as f:
            for raw_line in f:
                try:
                    obj = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                ts = obj.get("@timestamp")
                if ts is None:
                    continue
                try:
                    dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
                timestamps.append(dt.timestamp())
                event_ids.append(obj.get("EventID"))
    timestamps = np.array(timestamps)
    event_ids = np.array(event_ids, dtype=object)
    order = np.argsort(timestamps)
    timestamps, event_ids = timestamps[order], event_ids[order]
    bin_edges = np.linspace(timestamps.min(), timestamps.max(), 501)
    counts, _ = np.histogram(timestamps[event_ids == 10], bins=bin_edges)
    return counts.astype(np.float64)


real_series = fetch_real_series()
print(f"serie reale (EventID10, APT29 day1): {len(real_series)} punti")

corrupted = real_series.copy()
injected = {50: np.nan, 200: 999999999999999999.0, 350: -1.0}
for idx, val in injected.items():
    corrupted[idx] = val

t0 = time.perf_counter()
shield = Armatura(livello_ia=0.0)
pulito, K, anomalie = shield.analizza(corrupted)
elapsed_ms = (time.perf_counter() - t0) * 1000

print(f"\nscudo eseguito in {elapsed_ms:.1f} ms su {len(corrupted)} punti")
print(f"valori corrotti iniettati agli indici: {sorted(injected.keys())}")
print(f"anomalie rilevate dallo scudo: {anomalie[:20]}{'...' if len(anomalie) > 20 else ''}")
print(f"totale anomalie: {len(anomalie)}")

print("\ndettaglio sui 3 punti iniettati:")
for idx, val in injected.items():
    caught = idx in anomalie
    print(f"  indice {idx}: iniettato={val!r} rilevato={caught} valore_pulito={pulito[idx]:.2f}")

mean_raw_naive = np.nanmean(corrupted[np.isfinite(corrupted)])
mean_shielded = np.mean(pulito)
print(f"\nse un agente calcolasse una media grezza sul dato non filtrato (ignorando solo i NaN): {mean_raw_naive:.2f}")
print(f"media sul dato passato dallo scudo: {mean_shielded:.2f}")
print(f"media sulla serie originale pulita (verita' di riferimento): {real_series.mean():.2f}")
