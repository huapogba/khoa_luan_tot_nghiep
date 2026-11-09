"""
Comparison table: CLIP / CLIP4Clip / X-CLIP, zero-shot and fine-tuned (results/*.json), vs Qwen3-VL
(evaluation_msvd_results.json from evaluation_msvd.py).

Example:
    python msvd_evaluate/compare.py
"""

from msvd_common import QWEN_RESULTS, RESULT_DIR, load_json, save_json, setup_stdout

SOURCES = {
    "CLIP zero-shot": RESULT_DIR / "clip_results.json",
    "CLIP4Clip (WebVid)": RESULT_DIR / "clip4clip_results.json",
    "X-CLIP (K400)": RESULT_DIR / "xclip_results.json",
    "CLIP4Clip FT-MSVD": RESULT_DIR / "clip4clip_ft_results.json",
    "X-CLIP FT-MSVD": RESULT_DIR / "xclip_ft_results.json",
    "LanguageBind zero-shot": RESULT_DIR / "languagebind_results.json",
    "LanguageBind FT-MSVD": RESULT_DIR / "languagebind_ft_results.json",
    "Qwen3-VL-Emb-2B": QWEN_RESULTS,
}


def main():
    setup_stdout()

    found = {}
    for name, path in SOURCES.items():
        if path.exists():
            found[name] = load_json(path)
        else:
            print(f"Missing (skipped): {path}")

    if not found:
        return

    counts = {d["evaluation"]["candidate_videos"] for d in found.values()}
    if len(counts) > 1:
        print(f"WARNING: candidate counts differ {sorted(counts)}, not directly comparable.")

    for task in ("t2v", "v2t"):
        print(f"\n[{task}]  MSVD test, all captions")
        print(f"{'Model':<24}{'N':>7}{'R@1':>8}{'R@5':>8}{'R@10':>8}{'MRR':>8}{'MdR':>6}{'MnR':>8}")
        for name, d in found.items():
            m = d["metrics"][task]
            print(
                f"{name:<24}{m['queries']:>7}{m['R@1'] * 100:>7.1f}%{m['R@5'] * 100:>7.1f}%"
                f"{m['R@10'] * 100:>7.1f}%{m['MRR']:>8.4f}{m['MdR']:>6.1f}{m['MeanRank']:>8.2f}"
            )

    save_json(
        {name: {"model": d["evaluation"]["model"],
                "candidate_videos": d["evaluation"]["candidate_videos"],
                "metrics": d["metrics"]} for name, d in found.items()},
        RESULT_DIR / "comparison.json",
    )
    print(f"\nSaved: {RESULT_DIR / 'comparison.json'}")


if __name__ == "__main__":
    main()
