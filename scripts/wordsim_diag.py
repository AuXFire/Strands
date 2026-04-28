"""Where is strand losing on WordSim/MEN/RG-65? Side-by-side per-pair diff vs GloVe-300."""

from pathlib import Path
import gensim.downloader as api
import numpy as np
from scipy.stats import spearmanr
from strands import compare
from tests.benchmarks._loaders import DATA_DIR, load_men3000, load_rg65, load_pairs_tsv


def cos(m, a, b):
    if a not in m or b not in m: return 0.0
    va, vb = m[a], m[b]
    d = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / d) if d > 0 else 0.0


def main():
    g300 = api.load("glove-wiki-gigaword-300")
    pairs_set = {
        "WordSim-353": load_pairs_tsv(DATA_DIR / "wordsim353.tsv"),
        "MEN-3000":    load_men3000(),
        "RG-65":       load_rg65(),
    }
    for name, pairs in pairs_set.items():
        gold = []
        strand_pred = []
        glove_pred = []
        diffs = []
        for a, b, g in pairs:
            s = compare(a, b, conceptnet_bridge=True).score
            gp = max(0.0, cos(g300, a, b))
            gold.append(g)
            strand_pred.append(s)
            glove_pred.append(gp)
            diffs.append((a, b, g, s, gp, abs(s - gp)))
        rho_s, _ = spearmanr(gold, strand_pred)
        rho_g, _ = spearmanr(gold, glove_pred)
        print(f"\n{name}: strand={rho_s:.4f} glove={rho_g:.4f} (gap {rho_s-rho_g:+.4f})")

        # Find pairs where strand and glove disagree most, AND strand is wrong
        # vs gold. Sort by gold descending — focus on high-relatedness pairs
        # where strand under-rates.
        diffs.sort(key=lambda x: x[2], reverse=True)
        print("  Top relatedness pairs where strand UNDERrates:")
        shown = 0
        for a, b, g, s, gp, _ in diffs[:50]:
            # Convert all to ranks first; just heuristic: strand should be
            # close to glove for high gold pairs.
            if s < gp - 0.20:
                print(f"    {a:14} {b:14} gold={g:5.2f}  strand={s:.3f}  glove={gp:.3f}")
                shown += 1
                if shown >= 10:
                    break


if __name__ == "__main__":
    main()
