"""
wc2026_model.py
================
A Monte Carlo model of the 2026 FIFA World Cup that turns team-strength ratings
into coherent probabilities for every derived market (group winner, top-2,
qualification incl. best-third, deep runs, title), then compares those model
probabilities against market prices and flags discrepancies that exceed the
model's own input-noise band.

Design follows the strategy discussion:
  - INPUT layer  : a single strength rating per team (calibrate these from
                   sharp match-odds markets -- see calibrate_rating_from_match()).
  - MATCH engine : bivariate Poisson scorelines (gives goals, needed for
                   goal-difference / goals-scored tiebreakers).
  - PROPAGATION  : simulate the full tournament structure many times.
  - COMPARISON   : devig market prices, compute edge = model - market.
  - DISCIPLINE   : a sensitivity pass perturbs ratings to estimate how much each
                   output probability moves under input uncertainty, so a flagged
                   "edge" must clear the noise it could just be made of.

IMPORTANT honesty notes:
  * GROUP-STAGE outputs (win group / top-2 / qualify) use the exact 2026 rules
    and are the trustworthy core -- this is where the edge thesis lives.
  * KNOCKOUT outputs (reach R16/QF/SF/final, win cup) use a bracket that is
    RANDOMISED each simulation rather than the official fixed R32 third-place
    assignment table (which depends on which 8 groups the thirds come from).
    That makes deep-run numbers an unbiased average over possible brackets,
    good for relative reads but NOT a substitute for the official map. Plug the
    official R32 table into build_knockout_bracket() to make these exact.
  * The ratings below are ILLUSTRATIVE placeholders. Replace them with values
    calibrated to devigged match odds before trusting any flagged edge.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict

# ----------------------------------------------------------------------------
# 1. TOURNAMENT DATA  (real 2026 draw; placeholder strength ratings)
# ----------------------------------------------------------------------------

GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Korea", "South Africa", "Czechia"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia-Herzegovina"],
    "C": ["Brazil", "Morocco", "Scotland", "Haiti"],
    "D": ["USA", "Paraguay", "Australia", "Turkiye"],
    "E": ["Germany", "Ecuador", "Ivory Coast", "Curacao"],
    "F": ["Netherlands", "Japan", "Tunisia", "Sweden"],
    "G": ["Belgium", "Iran", "Egypt", "New Zealand"],
    "H": ["Spain", "Uruguay", "Saudi Arabia", "Cape Verde"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Austria", "Algeria", "Jordan"],
    "K": ["Portugal", "Colombia", "Uzbekistan", "DR Congo"],
    "L": ["England", "Croatia", "Panama", "Ghana"],
}

# Elo-style ratings (~1500 weak .. ~2050 elite). PLACEHOLDERS -- replace with
# ratings fitted to devigged match-odds so the model inherits the market's
# (trusted) view of singular-match strength.
RATINGS: dict[str, float] = {
    "Mexico": 1840, "South Korea": 1790, "South Africa": 1660, "Czechia": 1820,
    "Canada": 1800, "Switzerland": 1860, "Qatar": 1690, "Bosnia-Herzegovina": 1760,
    "Brazil": 2030, "Morocco": 1900, "Scotland": 1780, "Haiti": 1560,
    "USA": 1830, "Paraguay": 1760, "Australia": 1770, "Turkiye": 1840,
    "Germany": 1980, "Ecuador": 1830, "Ivory Coast": 1820, "Curacao": 1560,
    "Netherlands": 1960, "Japan": 1860, "Tunisia": 1740, "Sweden": 1800,
    "Belgium": 1930, "Iran": 1780, "Egypt": 1800, "New Zealand": 1620,
    "Spain": 2050, "Uruguay": 1900, "Saudi Arabia": 1680, "Cape Verde": 1600,
    "France": 2010, "Senegal": 1870, "Norway": 1880, "Iraq": 1620,
    "Argentina": 2020, "Austria": 1820, "Algeria": 1790, "Jordan": 1620,
    "Portugal": 1970, "Colombia": 1890, "Uzbekistan": 1700, "DR Congo": 1750,
    "England": 1990, "Croatia": 1880, "Panama": 1700, "Ghana": 1760,
}

# ----------------------------------------------------------------------------
# 2. CONFIG
# ----------------------------------------------------------------------------

@dataclass
class Config:
    base_goals: float = 1.30      # expected goals per side in an even match
    k: float = 0.40               # strength -> goals sensitivity
    scale: float = 200.0          # rating points per unit supremacy
    corr: float = 0.10            # bivariate-Poisson shared component (draw inflation)
    n_sims: int = 10_000          # raise for tighter estimates
    h2h_before_overall_gd: bool = True  # FIFA 2026 order is debated in public
    seed: int | None = 12345

CFG = Config()
_RNG = np.random.default_rng(CFG.seed)

# ----------------------------------------------------------------------------
# 3. MATCH ENGINE  (bivariate Poisson)
# ----------------------------------------------------------------------------

def expected_goals(r_a: float, r_b: float, cfg: Config = CFG) -> tuple[float, float]:
    """Map two ratings to expected goals for each side."""
    d = (r_a - r_b) / cfg.scale
    lam_a = cfg.base_goals * np.exp(cfg.k * d)
    lam_b = cfg.base_goals * np.exp(-cfg.k * d)
    return lam_a, lam_b

def play_match(r_a: float, r_b: float, rng, cfg: Config = CFG) -> tuple[int, int]:
    """One scoreline via bivariate Poisson: ga = U + W, gb = V + W."""
    lam_a, lam_b = expected_goals(r_a, r_b, cfg)
    c = min(cfg.corr, lam_a - 0.05, lam_b - 0.05)
    c = max(c, 0.0)
    w = rng.poisson(c)
    ga = rng.poisson(lam_a - c) + w
    gb = rng.poisson(lam_b - c) + w
    return int(ga), int(gb)

def play_knockout(r_a: float, r_b: float, rng, cfg: Config = CFG) -> int:
    """Return 0 if team A advances, 1 if team B (ties -> coin-flip for ET/pens)."""
    ga, gb = play_match(r_a, r_b, rng, cfg)
    if ga > gb:
        return 0
    if gb > ga:
        return 1
    return int(rng.random() < 0.5)

# ----------------------------------------------------------------------------
# 4. GROUP SIMULATION + TIEBREAKERS
# ----------------------------------------------------------------------------

@dataclass
class TeamRow:
    name: str
    pts: int = 0
    gf: int = 0
    ga: int = 0
    # head-to-head accumulators are computed on demand
    @property
    def gd(self) -> int:
        return self.gf - self.ga

# round-robin fixture index pairs for a 4-team group
_RR_PAIRS = [(0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)]

def simulate_group(teams: list[str], ratings: dict[str, float], rng,
                   cfg: Config = CFG):
    """Simulate a 4-team round robin; return ordered list of (name, row) plus
    the raw results so tiebreakers can use head-to-head."""
    rows = {t: TeamRow(t) for t in teams}
    results = {}  # (i,j) -> (gi, gj)
    for i, j in _RR_PAIRS:
        ti, tj = teams[i], teams[j]
        gi, gj = play_match(ratings[ti], ratings[tj], rng, cfg)
        results[(i, j)] = (gi, gj)
        rows[ti].gf += gi; rows[ti].ga += gj
        rows[tj].gf += gj; rows[tj].ga += gi
        if gi > gj:
            rows[ti].pts += 3
        elif gj > gi:
            rows[tj].pts += 3
        else:
            rows[ti].pts += 1; rows[tj].pts += 1

    order = _rank_group(teams, rows, results, rng, cfg)
    return order, rows

def _mini_table(idxs, teams, results):
    """Head-to-head sub-table among the given team indices."""
    pts = defaultdict(int); gf = defaultdict(int); ga = defaultdict(int)
    s = set(idxs)
    for (i, j), (gi, gj) in results.items():
        if i in s and j in s:
            gf[i] += gi; ga[i] += gj; gf[j] += gj; ga[j] += gi
            if gi > gj: pts[i] += 3
            elif gj > gi: pts[j] += 3
            else: pts[i] += 1; pts[j] += 1
    return {i: (pts[i], gf[i] - ga[i], gf[i]) for i in idxs}

def _rank_group(teams, rows, results, rng, cfg: Config):
    """Return team names best->worst applying FIFA 2026 tiebreakers."""
    idx = list(range(len(teams)))

    def overall_key(i):
        r = rows[teams[i]]
        return (r.pts, r.gd, r.gf)

    # group by points first
    idx.sort(key=lambda i: rows[teams[i]].pts, reverse=True)
    final = []
    p = 0
    while p < len(idx):
        q = p
        while q + 1 < len(idx) and rows[teams[idx[q + 1]]].pts == rows[teams[idx[p]]].pts:
            q += 1
        tied = idx[p:q + 1]
        if len(tied) == 1:
            final.append(tied[0])
        else:
            final.extend(_break_tie(tied, teams, rows, results, rng, cfg))
        p = q + 1
    return [teams[i] for i in final]

def _break_tie(tied, teams, rows, results, rng, cfg: Config):
    """Resolve a set of teams level on points."""
    def h2h_key(i, mini):
        return mini[i]  # (pts, gd, gf) among tied
    def overall_key(i):
        r = rows[teams[i]]
        return (r.gd, r.gf)

    if cfg.h2h_before_overall_gd:
        mini = _mini_table(tied, teams, results)
        keyfn = lambda i: (h2h_key(i, mini), overall_key(i), rng.random())
    else:
        keyfn = lambda i: (overall_key(i), rng.random())
    # rng.random() is the terminal fallback standing in for fair-play / FIFA
    # ranking / drawing of lots (not modelled here).
    return sorted(tied, key=keyfn, reverse=True)

# ----------------------------------------------------------------------------
# 5. THIRD-PLACE SELECTION (8 best across the 12 groups)
# ----------------------------------------------------------------------------

def select_best_thirds(thirds, rng):
    """thirds: list of (name, pts, gd, gf). Return set of 8 qualifying names."""
    ranked = sorted(thirds, key=lambda x: (x[1], x[2], x[3], rng.random()),
                    reverse=True)
    return {name for name, *_ in ranked[:8]}

# ----------------------------------------------------------------------------
# 6. KNOCKOUT  (randomised bracket -- see honesty note at top)
# ----------------------------------------------------------------------------

def build_knockout_bracket(qualifiers, rng):
    """qualifiers: list of 32 team names. Returns them shuffled into bracket
    order. REPLACE with the official FIFA R32 third-place assignment table for
    exact deep-run probabilities."""
    q = qualifiers[:]
    rng.shuffle(q)
    return q

def run_knockout(qualifiers, ratings, rng, cfg: Config):
    """Single-elimination from 32. Returns dict round_reached per team where
    1=R32 entrant, 2=R16, 3=QF, 4=SF, 5=Final, 6=Champion."""
    bracket = build_knockout_bracket(qualifiers, rng)
    reached = {t: 1 for t in bracket}
    round_no = 1
    while len(bracket) > 1:
        round_no += 1
        nxt = []
        for k in range(0, len(bracket), 2):
            a, b = bracket[k], bracket[k + 1]
            w = a if play_knockout(ratings[a], ratings[b], rng, cfg) == 0 else b
            reached[w] = round_no
            nxt.append(w)
        bracket = nxt
    return reached  # winner has round_no == 6

# ----------------------------------------------------------------------------
# 7. MONTE CARLO DRIVER
# ----------------------------------------------------------------------------

MARKETS = ["win_group", "top2", "qualify", "reach_r16", "reach_qf",
           "reach_sf", "reach_final", "win_cup"]

def simulate_tournament(ratings, cfg: Config = CFG, rng=None):
    """Run cfg.n_sims tournaments; return {team: {market: prob}}."""
    if rng is None:
        rng = np.random.default_rng(cfg.seed)
    teams = [t for g in GROUPS.values() for t in g]
    counts = {t: dict.fromkeys(MARKETS, 0) for t in teams}

    for _ in range(cfg.n_sims):
        qualifiers = []
        third_candidates = []
        for gid, gteams in GROUPS.items():
            order, rows = simulate_group(gteams, ratings, rng, cfg)
            counts[order[0]]["win_group"] += 1
            counts[order[0]]["top2"] += 1
            counts[order[1]]["top2"] += 1
            qualifiers.extend([order[0], order[1]])
            r3 = rows[order[2]]
            third_candidates.append((order[2], r3.pts, r3.gd, r3.gf))

        best_thirds = select_best_thirds(third_candidates, rng)
        for name in best_thirds:
            qualifiers.append(name)
        for t in qualifiers:
            counts[t]["qualify"] += 1

        reached = run_knockout(qualifiers, ratings, rng, cfg)
        for t, r in reached.items():
            if r >= 2: counts[t]["reach_r16"] += 1
            if r >= 3: counts[t]["reach_qf"] += 1
            if r >= 4: counts[t]["reach_sf"] += 1
            if r >= 5: counts[t]["reach_final"] += 1
            if r >= 6: counts[t]["win_cup"] += 1

    n = cfg.n_sims
    return {t: {m: counts[t][m] / n for m in MARKETS} for t in teams}

# ----------------------------------------------------------------------------
# 8. DEVIG UTILITIES
# ----------------------------------------------------------------------------

def devig_multiplicative(prices: dict[str, float]) -> dict[str, float]:
    s = sum(prices.values())
    return {k: v / s for k, v in prices.items()}

def devig_power(prices: dict[str, float], tol=1e-9) -> dict[str, float]:
    """Solve for exponent n so sum(p_i**n) = 1 (favourite-longshot aware)."""
    p = np.array(list(prices.values()), float)
    lo, hi = 0.5, 3.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if (p ** mid).sum() > 1:
            lo = mid
        else:
            hi = mid
        if abs((p ** mid).sum() - 1) < tol:
            break
    n = (lo + hi) / 2
    out = p ** n
    out /= out.sum()
    return dict(zip(prices.keys(), out))

# ----------------------------------------------------------------------------
# 9. SENSITIVITY  -> noise band on each output probability
# ----------------------------------------------------------------------------

def sensitivity_bands(base_ratings, market_for, cfg: Config = CFG,
                      rating_sd=25.0, n_scenarios=12):
    """Perturb every rating by Gaussian noise (sd in rating points) and re-run
    a smaller sim; return per-(team, market) std of the output probability.
    This is the noise an apparent edge must clear to be taken seriously."""
    small = Config(**{**cfg.__dict__})
    small.n_sims = max(2000, cfg.n_sims // 5)
    rng = np.random.default_rng(999)
    stacks = defaultdict(lambda: defaultdict(list))
    for s in range(n_scenarios):
        perturbed = {t: r + rng.normal(0, rating_sd) for t, r in base_ratings.items()}
        probs = simulate_tournament(perturbed, small, rng)
        for t in probs:
            for m in MARKETS:
                stacks[t][m].append(probs[t][m])
    return {t: {m: float(np.std(stacks[t][m])) for m in MARKETS} for t in stacks}

# ----------------------------------------------------------------------------
# 10. COMPARISON / FLAGGING
# ----------------------------------------------------------------------------

def compare_market(model_probs, market_prices, market_key, bands=None,
                   edge_threshold=0.03, devig=devig_multiplicative):
    """market_prices: {group_or_set_id: {team: raw_implied_price}}.
    Returns list of rows with model, fair (devigged), edge, noise, flag."""
    rows = []
    for set_id, prices in market_prices.items():
        fair = devig(prices)
        for team, fair_p in fair.items():
            model_p = model_probs[team][market_key]
            edge = model_p - fair_p
            noise = bands[team][market_key] if bands else 0.0
            flag = (abs(edge) > edge_threshold) and (abs(edge) > 2 * noise)
            rows.append(dict(set=set_id, team=team, market_fair=fair_p,
                             model=model_p, edge=edge, noise=noise, flag=flag))
    rows.sort(key=lambda r: abs(r["edge"]), reverse=True)
    return rows

# ----------------------------------------------------------------------------
# 11. DEMO / REPORT
# ----------------------------------------------------------------------------

# Sample MARKET group-winner prices (raw implied probs incl. overround),
# converted from real bookmaker American odds for three groups. These are the
# "market" the model is benchmarked against.
MARKET_GROUP_WINNER = {
    "H": {"Spain": 0.818, "Uruguay": 0.213, "Saudi Arabia": 0.0526, "Cape Verde": 0.0244},
    "I": {"France": 0.697, "Norway": 0.267, "Senegal": 0.1176, "Iraq": 0.0196},
    "J": {"Argentina": 0.773, "Austria": 0.182, "Algeria": 0.125, "Jordan": 0.0244},
}

def _pct(x): return f"{100 * x:5.1f}%"

def main():
    print("=" * 74)
    print("  2026 WORLD CUP MODEL  -- %d simulations" % CFG.n_sims)
    print("=" * 74)

    probs = simulate_tournament(RATINGS, CFG)

    # ---- group-stage report (the trustworthy core) ----
    print("\nGROUP-STAGE PROBABILITIES (exact rules)")
    print("-" * 74)
    for gid, teams in GROUPS.items():
        ranked = sorted(teams, key=lambda t: probs[t]["qualify"], reverse=True)
        print(f"\nGroup {gid}")
        print(f"  {'Team':<20}{'Win grp':>9}{'Top 2':>9}{'Qualify':>10}")
        for t in ranked:
            print(f"  {t:<20}{_pct(probs[t]['win_group']):>9}"
                  f"{_pct(probs[t]['top2']):>9}{_pct(probs[t]['qualify']):>10}")

    # ---- title odds (approx bracket) ----
    print("\n" + "-" * 74)
    print("TITLE / DEEP-RUN (knockout bracket randomised -- relative read only)")
    print("-" * 74)
    top = sorted(probs, key=lambda t: probs[t]["win_cup"], reverse=True)[:10]
    print(f"  {'Team':<20}{'Reach SF':>10}{'Final':>9}{'Win cup':>9}")
    for t in top:
        print(f"  {t:<20}{_pct(probs[t]['reach_sf']):>10}"
              f"{_pct(probs[t]['reach_final']):>9}{_pct(probs[t]['win_cup']):>9}")

    # ---- sensitivity bands ----
    print("\n" + "-" * 74)
    print("Estimating input-noise bands (rating sd = 25 pts) ...")
    bands = sensitivity_bands(RATINGS, "win_group", CFG)

    # ---- comparison vs market group-winner prices ----
    print("\n" + "-" * 74)
    print("MARKET COMPARISON -- group winner (devig = multiplicative)")
    print("model - fair > 3pp AND > 2x noise  => FLAG")
    print("-" * 74)
    rows = compare_market(probs, MARKET_GROUP_WINNER, "win_group", bands)
    print(f"  {'Grp':<4}{'Team':<18}{'Fair':>8}{'Model':>8}{'Edge':>8}{'Noise':>8}  Flag")
    for r in rows:
        flag = "<<< FLAG" if r["flag"] else ""
        print(f"  {r['set']:<4}{r['team']:<18}{_pct(r['market_fair']):>8}"
              f"{_pct(r['model']):>8}{r['edge']*100:+7.1f}p{r['noise']*100:6.1f}p  {flag}")

    print("\nNote: with placeholder ratings, headline group-winner edges should be")
    print("small -- the market is sharp there. Real signal hunting belongs in the")
    print("third-place-qualification and exact-order corners, and requires ratings")
    print("calibrated to devigged match odds. Swap RATINGS, then re-run.")
    print("=" * 74)

if __name__ == "__main__":
    main()
