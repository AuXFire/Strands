"""Pure CN/Numberbatch mean-of-vectors baseline for sentence STS."""
import re, gensim.downloader as api, numpy as np
from scipy.stats import spearmanr
from tests.benchmarks._loaders import DATA_DIR, load_sts, load_sick

_TOK = re.compile(r"[A-Za-z]+")

def main():
    cn = api.load("conceptnet-numberbatch-17-06-300")
    g = api.load("glove-wiki-gigaword-300")

    def cn_mean(s):
        toks = [t.lower() for t in _TOK.findall(s)]
        vs = [cn[f'/c/en/{t}'] for t in toks if f'/c/en/{t}' in cn]
        return np.mean(vs, axis=0) if vs else None

    def g_mean(s):
        toks = [t.lower() for t in _TOK.findall(s)]
        vs = [g[t] for t in toks if t in g]
        return np.mean(vs, axis=0) if vs else None

    def cos(a, b):
        if a is None or b is None: return 0.0
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / d) if d > 0 else 0.0

    sets = {
        "STS-2012-MSRpar":   (DATA_DIR / "sts2012_msrpar.tsv", load_sts),
        "STS-2014-headlines":(DATA_DIR / "sts2014_headlines.tsv", load_sts),
        "STS-2014-images":   (DATA_DIR / "sts2014_images.tsv", load_sts),
        "STS-2015-headlines":(DATA_DIR / "sts2015_headlines.tsv", load_sts),
        "SICK-test":         (DATA_DIR / "sick_test.txt", load_sick),
    }
    print(f"{'Dataset':<22} {'CN-mean ρ':>11} {'GloVe-mean ρ':>14}")
    for name, (path, loader) in sets.items():
        pairs = loader(path)
        gold = [g for _, _, g in pairs]
        cn_p = [cos(cn_mean(a), cn_mean(b)) for a, b, _ in pairs]
        g_p  = [cos(g_mean(a), g_mean(b)) for a, b, _ in pairs]
        rho_cn, _ = spearmanr(gold, cn_p)
        rho_g, _ = spearmanr(gold, g_p)
        print(f"{name:<22} {rho_cn:>11.4f} {rho_g:>14.4f}")

if __name__ == "__main__":
    main()
