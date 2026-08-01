import json, os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    results    = json.load(open(os.path.join(RESULTS_DIR, "eval_results.json")))
    ours       = [r for r in results if r["group"] == "ours"]
    variant_A  = next((r for r in ours if r["name"] == "ours_A"), None)
    if variant_A is None:
        raise SystemExit("Variant A not found in eval_results.json")
    a_integrity = variant_A["devanagari_integrity"]

    eligible, rejected = [], []
    for r in ours:
        mean_fert    = sum(r["flores_fertility"].values()) / 3
        passes_rt    = r["round_trip_fidelity"] >= 0.999
        di           = r["devanagari_integrity"]
        passes_di    = di is not None and di >= a_integrity
        r["_mean_flores_fertility"] = round(mean_fert, 4)
        if passes_rt and passes_di:
            eligible.append(r)
        else:
            reasons = []
            if not passes_rt: reasons.append(f"round_trip={r['round_trip_fidelity']:.4f} < 0.999")
            if not passes_di: reasons.append(f"dev_integrity={di} < variant_A={a_integrity}")
            rejected.append({"name": r["name"], "reasons": reasons})

    eligible.sort(key=lambda r: r["_mean_flores_fertility"])
    shortlist = eligible[:3]

    print("=== Eligibility gate (100% round-trip AND dev-integrity >= variant A) ===")
    for r in rejected:
        print(f"  REJECTED {r['name']}: {'; '.join(r['reasons'])}")

    print("\n=== Ranked by mean FLORES fertility (lower = better) ===")
    for r in eligible:
        marker = " <-- SHORTLISTED" if r in shortlist else ""
        ff = r["flores_fertility"]
        print(f"  {r['name']}: mean={r['_mean_flores_fertility']:.3f}"
              f"  (hi={ff['hi']:.2f} mr={ff['mr']:.2f} en={ff['en']:.2f}){marker}")

    out = {
        "shortlist": [r["name"] for r in shortlist],
        "shortlist_detail": shortlist,
        "rejected": rejected,
        "rule": "top-3 by mean FLORES fertility (hi,mr,en), gated on 100% round-trip "
                "and Devanagari integrity >= variant A",
    }
    out_path = os.path.join(RESULTS_DIR, "shortlist.json")
    json.dump(out, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"\nShortlist -> {[r['name'] for r in shortlist]}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
