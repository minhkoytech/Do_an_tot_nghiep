"""
13_tables_section_4_6.py
-------------------------
In ra cac bang so lieu cho muc 4.6 cua bao cao (phan tich theo loai gia mao
va theo nguoi ky), doc truc tiep tu cac file ket qua cua lan chay gan nhat:

    results/tables/pairwise_baseline_per_pairtype.csv   (RF, SVM)
    results/tables/siamese_per_pairtype.csv
    results/tables/combined_per_pairtype.csv
    results/tables/pairwise_baseline_per_writer.csv
    results/tables/siamese_per_writer.csv
    results/tables/combined_per_writer.csv

Cach dung:
    cd src
    python 13_tables_section_4_6.py

Ket qua in ra man hinh dang bang, dong thoi luu vao
results/tables/section_4_6_tables.txt de copy vao bao cao.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
T = ROOT / "results" / "tables"

NAMES = {"rf": "Random Forest", "svm": "SVM", "siamese": "Siamese Network", "combined": "Mô hình kết hợp"}
ORDER = ["rf", "svm", "siamese", "combined"]


def load(files):
    frames = []
    for f in files:
        p = T / f
        if p.exists():
            frames.append(pd.read_csv(p))
        else:
            print(f"[THIEU] {p} - bo qua")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fmt(x, pct=False):
    if pd.isna(x):
        return "-"
    return f"{x*100:.1f}%".replace(".", ",") if pct else f"{x:.3f}".replace(".", ",")


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def main():
    out = []

    # ---------- Bang theo loai gia mao ----------
    pt = load(["pairwise_baseline_per_pairtype.csv", "siamese_per_pairtype.csv", "combined_per_pairtype.csv"])
    if len(pt):
        rows = []
        for m in ORDER:
            sub = pt[pt["model"] == m]
            if sub.empty:
                continue
            def get(ptype, col):
                s = sub[sub["pair_type"] == ptype]
                return s[col].iloc[0] if (len(s) and col in s) else float("nan")
            rows.append([NAMES[m],
                         fmt(get("skilled_forgery", "FAR")),
                         fmt(get("random_forgery", "FAR")),
                         fmt(get("genuine_genuine", "FRR"))])
        out.append("Bang: FAR theo loai gia mao va FRR tren tap kiem thu")
        out.append(table(["Mô hình", "FAR giả có kỹ năng", "FAR giả ngẫu nhiên", "FRR"], rows))

    # ---------- Thong ke theo nguoi ky ----------
    pw = load(["pairwise_baseline_per_writer.csv", "siamese_per_writer.csv", "combined_per_writer.csv"])
    if len(pw):
        rows = []
        for m in ORDER:
            sub = pw[pw["model"] == m]
            if sub.empty:
                continue
            for col, label in [("accuracy", "Accuracy"), ("FAR", "FAR"), ("FRR", "FRR")]:
                s = sub[col]
                rows.append([NAMES[m], label, fmt(s.mean()), fmt(s.std()), fmt(s.min()), fmt(s.max())])
        out.append("\nBang: Thong ke hieu nang theo tung nguoi ky (10 nguoi ky tap kiem thu)")
        out.append(table(["Mô hình", "Chỉ số", "Trung bình", "Độ lệch chuẩn", "Thấp nhất", "Cao nhất"], rows))

        # Chi tiet tung nguoi ky cua mo hinh de xuat
        best = "combined" if "combined" in pw["model"].values else "siamese"
        sub = pw[pw["model"] == best].sort_values("FAR", ascending=False)
        rows = [[str(r.writer_id), fmt(r.accuracy), fmt(r.FAR), fmt(r.FRR)] for r in sub.itertuples()]
        out.append(f"\nBang: Chi tiet tung nguoi ky - {NAMES[best]}, sap xep theo FAR giam dan")
        out.append(table(["Người ký", "Accuracy", "FAR", "FRR"], rows))

    text = "\n".join(out)
    print(text)
    dst = T / "section_4_6_tables.txt"
    dst.write_text(text, encoding="utf-8")
    print(f"\nDa luu: {dst}")


if __name__ == "__main__":
    main()