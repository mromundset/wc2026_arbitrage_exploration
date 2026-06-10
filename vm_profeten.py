"""
vm_profeten.py  --  Exact-score picks for VG's "VM Profeten".
=============================================================
Reuses the calibrated bivariate-Poisson engine from wc2026_joint.py. For every group
match we fit (lam1, lam2, lam3) to Polymarket's 1X2 + Over/Under 2.5 and build the full
joint score distribution, then pick the scoreline that MAXIMISES EXPECTED POINTS under
VG VM Profeten's scoring:

    EXPECTED POINTS(i-j) = 2 * P(correct outcome) + 3 * P(exact score)

i.e. 2 pts for the right result (home/draw/away) and 3 pts for the exact score (a correct
exact score also nails the outcome, so it scores the full 5). This is NOT generally the
modal scoreline: the 2-pt outcome term carries far more probability than the 3-pt exact
term, so the EV pick is the most-likely OUTCOME's most-likely score -- which turns low-
scoring "draw" modes into the favourite's narrow win whenever a side is favoured.

Output is sorted chronologically (matchday 1 first). Run:
    python3 vm_profeten.py            # all 72 group games
    python3 vm_profeten.py --round1   # only the earliest matchday-1 games
"""
from __future__ import annotations
import sys, re
import numpy as np
try:                                       # keep output clean on Windows cp1252 consoles
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from wc2026_joint import (fetch_games_full, fetch_over25, fit_biv, _joint,
                          canon, T2G)

W_OUTCOME, W_EXACT = 2, 3                # VG VM Profeten: 2 pts outcome, 3 pts exact

def _norm_joint(l1, l2, l3):
    J = _joint(l1, l2, l3)
    return J / J.sum()                   # normalise (truncation-safe)

def _outcome_probs(J):
    return np.tril(J, -1).sum(), np.trace(J), np.triu(J, 1).sum()   # home, draw, away

def top_scores(J, n=3):
    """The n most-likely (home, away, prob) scorelines from the joint pmf."""
    flat = J.ravel(); ncols = J.shape[1]
    return [(*divmod(int(k), ncols), float(flat[k])) for k in np.argsort(-flat)[:n]]

def ev_pick(J):
    """Scoreline maximising expected points = 2*P(outcome) + 3*P(exact).
       Returns (i, j, ev, p_exact, p_outcome) for the best, scanning the full grid."""
    ph, pd, pa = _outcome_probs(J)
    K = J.shape[0]; best = None
    for i in range(K):
        for j in range(K):
            res = ph if i > j else (pa if i < j else pd)
            ev = W_OUTCOME * res + W_EXACT * J[i, j]
            if best is None or ev > best[2]:
                best = (i, j, ev, float(J[i, j]), float(res))
    return best

def _date(slug):
    m = re.search(r"(\d{4}-\d{2}-\d{2})$", slug)
    return m.group(1) if m else "????-??-??"

def main(round1_only=False):
    print("  fetching games + totals from Polymarket ...", flush=True)
    games = fetch_games_full()
    rows = []
    for gm in games:
        h, a = canon(gm["home"]), canon(gm["away"])
        if not (h in T2G and a in T2G and T2G[h] == T2G[a]):
            continue                      # group matches only
        po = fetch_over25(gm["slug"])
        l1, l2, l3 = fit_biv(gm["p_home"], gm["p_away"], po)
        J = _norm_joint(l1, l2, l3)
        rows.append(dict(date=_date(gm["slug"]), grp=T2G[h],
                         home=gm["home"], away=gm["away"],
                         ph=gm["p_home"], pd=gm["p_draw"], pa=gm["p_away"],
                         xgh=l1 + l3, xga=l2 + l3,
                         ev=ev_pick(J), modal=top_scores(J, 1)[0]))
    rows.sort(key=lambda r: (r["date"], r["grp"]))

    # completeness: each of the 12 groups should yield 6 games (72 total)
    from collections import Counter
    per_grp = Counter(r["grp"] for r in rows)
    short = {g: per_grp.get(g, 0) for g in "ABCDEFGHIJKL" if per_grp.get(g, 0) != 6}
    if short:
        print(f"  NOTE: incomplete (live feed) — groups not at 6 games: {short}")
        print("  (a transient missing price on Polymarket; re-run to pick them up)")

    # matchday-1 = each group's first two games (its two earliest fixtures)
    if round1_only:
        seen = {}; r1 = []
        for r in rows:
            seen.setdefault(r["grp"], 0)
            if seen[r["grp"]] < 2:        # first 2 chronological games per group = MD1
                r1.append(r); seen[r["grp"]] += 1
        rows = sorted(r1, key=lambda r: (r["date"], r["grp"]))

    print("\n" + "=" * 90)
    print("  VM PROFETEN -- EV-optimal picks (maximise 2*P(outcome) + 3*P(exact))")
    print("  PICK = expected-points-optimal score.  Pe = P(exact), Po = P(outcome), EV = exp. pts.")
    print("  *  marks where the EV pick differs from the single most-likely score (modal).")
    print("=" * 90)
    cur = None; tot_ev = 0.0
    for r in rows:
        if r["date"] != cur:
            cur = r["date"]; print(f"\n  -- {cur} " + "-" * 64)
        i, j, ev, pe, po = r["ev"]; tot_ev += ev
        mi, mj, _ = r["modal"]
        flag = "" if (i, j) == (mi, mj) else f" * (modal {mi}-{mj})"
        print(f"  {r['grp']}  {r['home']:>14} {i}-{j} {r['away']:<18}"
              f"  Pe {100*pe:4.1f}%  Po {100*po:4.1f}%  EV {ev:4.2f}"
              f"   1X2 {100*r['ph']:2.0f}/{100*r['pd']:2.0f}/{100*r['pa']:2.0f}{flag}")
    print("\n" + "=" * 90)
    print(f"  {len(rows)} matches.  Model's expected haul: {tot_ev:.1f} pts "
          f"(avg {tot_ev/len(rows):.2f}/match).  Max possible = {5*len(rows)}.")
    print("=" * 90)

if __name__ == "__main__":
    main(round1_only=("--round1" in sys.argv))
