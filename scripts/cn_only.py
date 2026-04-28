"""Pure ConceptNet/Numberbatch baseline (no strand) for sanity."""
import gensim.downloader as api
import numpy as np
from scipy.stats import spearmanr
from tests.benchmarks._loaders import DATA_DIR, load_men3000, load_rg65, load_simverb, load_pairs_tsv


def cos(m, a, b):
    ka, kb = f"/c/en/{a}", f"/c/en/{b}"
    if ka not in m or kb not in m: return 0.0
    va, vb = m[ka], m[kb]
    d = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / d) if d > 0 else 0.0


def main():
    cn = api.load("conceptnet-numberbatch-17-06-300")
    g300 = api.load("glove-wiki-gigaword-300")

    sets = {
        "SimLex-999":   load_pairs_tsv(DATA_DIR / "simlex999.txt"),
        "WordSim-353":  load_pairs_tsv(DATA_DIR / "wordsim353.tsv"),
        "MEN-3000":     load_men3000(),
        "RG-65":        load_rg65(),
        "SimVerb-3500": load_simverb(),
    }
    print(f"{'Dataset':<14} {'CN-only ρ':>12} {'GloVe-300 ρ':>14}")
    for name, pairs in sets.items():
        gold = [g for _, _, g in pairs]
        cn_pred = [cos(cn, a, b) for a, b, _ in pairs]

        def gcos(a, b):
            if a not in g300 or b not in g300: return 0.0
            va, vb = g300[a], g300[b]
            d = np.linalg.norm(va) * np.linalg.norm(vb)
            return float(np.dot(va, vb) / d) if d > 0 else 0.0
        g_pred = [gcos(a, b) for a, b, _ in pairs]
        rho_cn, _ = spearmanr(gold, cn_pred)
        rho_g, _ = spearmanr(gold, g_pred)
        print(f"{name:<14} {rho_cn:>12.4f} {rho_g:>14.4f}")


if __name__ == "__main__":
    main()
