"""合并多个 replay pkl，输出到 replay_v2_argmax_merged.pkl。"""
import os
import sys
import pickle
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = Path(__file__).parent / "nn_checkpoints"


def main(files, out):
    all_records = []
    for f in files:
        p = HERE / f
        with open(p, "rb") as fh:
            recs = pickle.load(fh)
        print(f"[MERGE] {f}: {len(recs)} 局")
        all_records.extend(recs)
    out_path = HERE / out
    with open(out_path, "wb") as fh:
        pickle.dump(all_records, fh)
    print(f"[MERGE] 合计 {len(all_records)} 局 → {out_path}")


if __name__ == "__main__":
    main(["replay_v2_argmax.pkl", "replay_v2_argmax_p2.pkl"], "replay_v2_argmax_230.pkl")
