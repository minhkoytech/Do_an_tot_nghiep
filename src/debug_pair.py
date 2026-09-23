"""
Script debug - chay TRUC TIEP tren may ban de kiem tra dung nguyen nhan
cua ket qua bat thuong voi cap original_9_2.png vs forgeries_7_5.png

CO THE DAT O DAU CUNG DUOC (thu muc goc project HOAC trong src/) -
script tu tim dung thu muc src/ va data/.
"""
import sys
from pathlib import Path
import importlib.util

SCRIPT_DIR = Path(__file__).resolve().parent

# Tu tim thu muc goc project: neu script dang nam trong src/, thu muc goc
# la cha cua src/. Neu script nam o thu muc goc (co san san folder src/
# ben trong), thu muc goc la chinh no.
if (SCRIPT_DIR / "src").exists():
    PROJECT_ROOT = SCRIPT_DIR
elif (SCRIPT_DIR.parent / "src").exists():
    PROJECT_ROOT = SCRIPT_DIR.parent
else:
    raise RuntimeError(f"Khong tim thay thu muc src/ gan {SCRIPT_DIR}. Kiem tra lai vi tri file nay.")

SRC_DIR = PROJECT_ROOT / "src"
print(f"Project root xac dinh: {PROJECT_ROOT}")
print(f"Src dir: {SRC_DIR}\n")


def load_module(filename, module_name):
    spec = importlib.util.spec_from_file_location(module_name, SRC_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preprocess_mod = load_module("01_preprocessing.py", "preprocess_mod")
features_mod = load_module("03_features.py", "features_mod")
siamese_mod = load_module("06_siamese_network.py", "siamese_mod")

ref_raw = str(PROJECT_ROOT / "data" / "raw" / "cedar" / "full_org" / "original_9_2.png")
query_raw = str(PROJECT_ROOT / "data" / "raw" / "cedar" / "full_forg" / "forgeries_7_5.png")

print(f"Reference goc: {ref_raw}")
print(f"Query goc: {query_raw}")

if not Path(ref_raw).exists():
    raise FileNotFoundError(f"Khong tim thay: {ref_raw}")
if not Path(query_raw).exists():
    raise FileNotFoundError(f"Khong tim thay: {query_raw}")

ref_processed = preprocess_mod.preprocess_image(ref_raw)
query_processed = preprocess_mod.preprocess_image(query_raw)

import cv2
ref_out = str(PROJECT_ROOT / "debug_ref_processed.png")
query_out = str(PROJECT_ROOT / "debug_query_processed.png")
cv2.imwrite(ref_out, ref_processed)
cv2.imwrite(query_out, query_processed)
print(f"\nDa luu: {ref_out}")
print(f"Da luu: {query_out}")
print("MO 2 FILE NAY XEM CO DUNG LA 2 CHU KY KHONG, HAY BI LOI (trong/giong nhau)")

ref_features = features_mod.extract_all_features(ref_out)
query_features = features_mod.extract_all_features(query_out)

print(f"\nSELECTED_FEATURES dang dung: {siamese_mod.SELECTED_FEATURES}")
print(f"\n{'Feature':<20} {'Ref':>12} {'Query':>12} {'Delta':>12}")
for f in siamese_mod.SELECTED_FEATURES:
    ref_v = ref_features[f]
    query_v = query_features[f]
    delta = abs(ref_v - query_v)
    print(f"{f:<20} {ref_v:>12.4f} {query_v:>12.4f} {delta:>12.4f}")