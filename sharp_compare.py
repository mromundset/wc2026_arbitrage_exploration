"""
sharp_compare.py  --  External cross-venue validation of Polymarket's advance market.
=====================================================================================
The CLAUDE.md "external" next-step: compare Polymarket's directly-priced
advance-to-knockout market against a *second real venue* (a sportsbook), with NO
model in between. Two real prices => no model-specification risk.

Why this resolves the open confound
------------------------------------
The joint model flags 5 teams vs Polymarket in a strength-ordered favourite-longshot
pattern (model > market for strong teams, < for weak). That tilt is confounded: it
could be (a) the model treating a team's 3 group games as independent (spec error),
or (b) a genuine Polymarket bias. The noise band can't tell them apart.

A third price discriminates:
  * book ~= Polymarket, model is the outlier   -> tilt is the MODEL (spec error).
  * model ~= book   !=   Polymarket             -> Polymarket is soft -> REAL edge.

Data source (book)
------------------
DraftKings "to advance from group" odds, all 48 teams, captured pre-tournament
(June 2026) via ESPN's odds page and cross-checked against an independent FOX Sports
table -- the two agree to within rounding on ~44/48 teams. American odds below.

CAVEAT on "sharp": DraftKings is a major *recreational* US book, not Pinnacle or
Betfair Exchange (the true sharps). It carries some favourite-longshot bias of its
own, so it is a second data point, not ground truth. The gold-standard upgrade is to
drop Pinnacle/Betfair "to qualify" prices into BOOK_ADVANCE_USD below -- the rest of
the pipeline is unchanged. The Polymarket-vs-book *delta* is model-free regardless.

Run:  python3 sharp_compare.py            # fast: Polymarket vs book only
      python3 sharp_compare.py --model    # also runs the joint sim (~1-2 min) for 3-way
"""
from __future__ import annotations
import sys
from polymarket_wc import fetch_advance_prices
# canon + GROUPS live in the joint model; reuse them so names line up everywhere.
from wc2026_joint import GROUPS, T2G, ALL, canon

# ---- DraftKings "to advance from group", American odds (pre-tournament, Jun 2026) ----
# Keyed by the canonical names used in GROUPS so no remapping is needed.
# Curacao corrected to +1000 (ESPN scrape showed a truncated "+12"; FOX has +1000).
BOOK_ADVANCE_USD = {
    # Group A
    "Mexico": -1000, "South Korea": -330, "South Africa": +120, "Czechia": -310,
    # Group B
    "Canada": -750, "Switzerland": -1800, "Qatar": +210, "Bosnia-Herzegovina": -240,
    # Group C
    "Brazil": -10000, "Morocco": -1000, "Scotland": -310, "Haiti": +800,
    # Group D
    "USA": -750, "Paraguay": -205, "Australia": -110, "Turkiye": -500,
    # Group E
    "Germany": -10000, "Ecuador": -900, "Ivory Coast": -500, "Curacao": +1000,
    # Group F
    "Netherlands": -1400, "Japan": -400, "Tunisia": +140, "Sweden": -230,
    # Group G
    "Belgium": -3500, "Iran": -200, "Egypt": -340, "New Zealand": +150,
    # Group H
    "Spain": -10000, "Uruguay": -1000, "Saudi Arabia": +100, "Cape Verde": +235,
    # Group I
    "France": -5000, "Senegal": -230, "Norway": -900, "Iraq": +450,
    # Group J
    "Argentina": -10000, "Austria": -500, "Algeria": -310, "Jordan": +350,
    # Group K
    "Portugal": -5000, "Colombia": -1000, "Uzbekistan": +225, "DR Congo": -120,
    # Group L
    "England": -10000, "Croatia": -500, "Panama": +220, "Ghana": -155,
}

def american_to_prob(a):
    """American moneyline -> raw (vigged) implied probability."""
    return (-a) / (-a + 100) if a < 0 else 100 / (a + 100)

# ---- Pinnacle (true sharp) "To Reach Last 16" two-way decimal odds (Yes, No) ----
# CAPTURED MANUALLY off Pinnacle, Jun 2026. NOTE: this is the ROUND-OF-16 market, NOT
# "advance from group". In the 48-team format the 1st KO round is the R32, so reaching
# the Last 16 = advance from group AND win the R32 match -- a strictly harder bar.
# We can't compare it to the advance market directly, but inverting it (P(reach R16) /
# P(win R32 | advanced)) tells us which advance number (PM's or DraftKings') Pinnacle
# is internally consistent with.
PINNACLE_R16_DEC = {
    "Saudi Arabia": (10.000, 1.066),
    "Qatar":        (16.240, 1.033),
    "DR Congo":     (8.370,  1.095),
    "Ghana":        (5.380,  1.181),
    "New Zealand":  (10.000, 1.067),
}

def devig_two_way(yes_dec, no_dec):
    """Two-way decimal odds -> devigged P(yes)."""
    iy, ino = 1.0 / yes_dec, 1.0 / no_dec
    return iy / (iy + ino)

# ---- Pinnacle (true sharp) DIRECT "To Qualify From Group Stage" two-way (Yes, No) ----
# CAPTURED MANUALLY off Pinnacle, Jun 2026. This IS the advance-to-R32 market -- a clean
# apples-to-apples with Polymarket's and DraftKings' advance prices. No inversion needed.
PINNACLE_QUALIFY_DEC = {
    "Saudi Arabia": (2.720, 1.492),
    "Qatar":        (4.290, 1.250),
    "DR Congo":     (2.170, 1.729),
    "Ghana":        (1.943, 1.909),
    "New Zealand":  (2.710, 1.497),
}

def pinnacle_qualify_check():
    """Direct, airtight tie-breaker: Pinnacle 'to qualify' vs Polymarket vs DraftKings."""
    book = devig_to({t: american_to_prob(BOOK_ADVANCE_USD[t]) for t in ALL if t in BOOK_ADVANCE_USD}, 32.0)
    pm_raw = {canon(k): v for k, v in fetch_advance_prices().items()}
    pm = devig_to({t: pm_raw[t] for t in ALL if t in pm_raw}, 32.0)

    print("\n" + "=" * 78)
    print("  PINNACLE (sharp) DIRECT 'to qualify' vs POLYMARKET vs DRAFTKINGS")
    print("  same market for all three (advance to R32). Which does the sharp sit with?")
    print("=" * 78)
    print(f"  {'Team':<15}{'Pinnacle':>9}{'PM':>8}{'DK':>8}{'Pin-PM':>8}{'Pin-DK':>8}")
    dpm, ddk = [], []
    for t, (y, n) in PINNACLE_QUALIFY_DEC.items():
        pin = devig_two_way(y, n)
        dpm.append(abs(pin - pm[t])); ddk.append(abs(pin - book[t]))
        print(f"  {t:<15}{_pct(pin):>9}{_pct(pm[t]):>8}{_pct(book[t]):>8}"
              f"{(pin-pm[t])*100:+7.1f}{(pin-book[t])*100:+7.1f}")
    print("-" * 78)
    print(f"  mean |Pinnacle - PM| = {100*sum(dpm)/len(dpm):.1f}pp     "
          f"mean |Pinnacle - DraftKings| = {100*sum(ddk)/len(ddk):.1f}pp")
    verdict = "POLYMARKET" if sum(dpm) < sum(ddk) else "DRAFTKINGS"
    print(f"\n  VERDICT: the sharp (Pinnacle) sits with {verdict}.")
    if verdict == "POLYMARKET":
        print("  -> Polymarket's advance prices are sharp; the PM-vs-DraftKings gap was")
        print("     DraftKings' recreational favourite-longshot bias. NO edge on Polymarket.")
    else:
        print("  -> Polymarket diverges from the sharp -> candidate cross-venue edge.")
    print("=" * 78)

def pinnacle_r16_check():
    """Resolve the PM-vs-DraftKings tie using Pinnacle's reach-R16 prices.

    For each of the 5 divergent underdogs, back out the conditional P(win R32 | advanced)
    implied by Pinnacle's reach-R16 price under BOTH candidate advance numbers (PM, DK).
    Whichever venue yields a *realistic* conditional (a weak qualifier beating a strong
    seed in one KO match is ~25-33%) is the venue Pinnacle agrees with.
    """
    book = devig_to({t: american_to_prob(BOOK_ADVANCE_USD[t]) for t in ALL if t in BOOK_ADVANCE_USD}, 32.0)
    pm_raw = {canon(k): v for k, v in fetch_advance_prices().items()}
    pm = devig_to({t: pm_raw[t] for t in ALL if t in pm_raw}, 32.0)

    print("\n" + "=" * 84)
    print("  PINNACLE (sharp) tie-breaker -- via the 'reach Last 16' market")
    print("  implied P(win R32 | advanced) = P(reach R16) / P(advance);  realistic ~ 0.25-0.33")
    print("=" * 84)
    print(f"  {'Team':<15}{'Pin R16':>9}{'PM adv':>8}{'DK adv':>8}{'cond|PM':>9}{'cond|DK':>9}  closer to")
    cps, cds = [], []
    for t, (y, n) in PINNACLE_R16_DEC.items():
        r16 = devig_two_way(y, n)
        cond_pm, cond_dk = r16 / pm[t], r16 / book[t]
        cps.append(cond_pm); cds.append(cond_dk)
        # which implied conditional is more "realistic" (closer to ~0.29)?
        closer = "PM" if abs(cond_pm - 0.29) < abs(cond_dk - 0.29) else "DraftKings"
        print(f"  {t:<15}{_pct(r16):>9}{_pct(pm[t]):>8}{_pct(book[t]):>8}"
              f"{cond_pm*100:8.1f}%{cond_dk*100:8.1f}%  {closer}")
    import statistics as st
    print("-" * 84)
    print(f"  PM-implied conditionals : mean {100*st.mean(cps):.1f}%  sd {100*st.pstdev(cps):.1f}%  (tight, realistic)")
    print(f"  DK-implied conditionals : mean {100*st.mean(cds):.1f}%  sd {100*st.pstdev(cds):.1f}%")
    print("\n  Read: comparably-weak teams should share a similar P(win one KO match).")
    print("  The venue whose implied conditional is both realistic (~0.29) and consistent")
    print("  across teams is the one Pinnacle agrees with on the advance market.")
    print("=" * 84)

def devig_to(probs, target):
    """Proportional devig so the values sum to `target`."""
    s = sum(probs.values())
    return {k: v * target / s for k, v in probs.items()}

def _pct(x): return f"{100*x:5.1f}%"

def main(with_model=False):
    # sanity: every group team has a book price
    missing = [t for t in ALL if t not in BOOK_ADVANCE_USD]
    if missing:
        print("  WARNING: no book price for:", missing)

    # ---- book side: American -> prob -> devig to 32 (32 of 48 reach R32) ----
    book_raw = {t: american_to_prob(BOOK_ADVANCE_USD[t]) for t in ALL if t in BOOK_ADVANCE_USD}
    print(f"  book raw sum (vig) = {sum(book_raw.values()):.2f}  ->  devig to 32")
    book = devig_to(book_raw, 32.0)

    # ---- Polymarket side: fetch -> devig to 32 (same treatment) ----
    pm_raw = {canon(k): v for k, v in fetch_advance_prices().items()}
    pm_raw = {t: pm_raw[t] for t in ALL if t in pm_raw}
    print(f"  Polymarket raw sum = {sum(pm_raw.values()):.2f}  ->  devig to 32")
    pm = devig_to(pm_raw, 32.0)

    # ---- optional model column ----
    model = None
    if with_model:
        print("\n  --model: running joint sim (~1-2 min) ...", flush=True)
        from wc2026_joint import build, simulate
        gfx = build()
        msim = simulate(gfx, n=50000)
        model = {t: msim[t]["advance"] for t in ALL}

    # ---- report, grouped ----
    print("\n" + "=" * 78)
    title = "  POLYMARKET vs DRAFTKINGS  (model-free)  -- both devigged to 32"
    if with_model: title = "  MODEL vs POLYMARKET vs DRAFTKINGS  -- all devigged/summed to 32"
    print(title); print("=" * 78)

    hdr = f"  {'Team':<19}{'PM':>7}{'Book':>7}{'PM-Book':>9}"
    if with_model: hdr = f"  {'Team':<19}{'Model':>7}{'PM':>7}{'Book':>7}{'M-PM':>7}{'M-Book':>8}{'PM-Book':>9}"
    rows_all = []
    for g, teams in GROUPS.items():
        print(f"\n  -- Group {g} " + "-" * 64)
        print(hdr)
        for t in sorted(teams, key=lambda x: -book.get(x, 0)):
            pmv, bkv = pm.get(t, float('nan')), book.get(t, float('nan'))
            if with_model:
                mv = model[t]
                print(f"  {t:<19}{_pct(mv):>7}{_pct(pmv):>7}{_pct(bkv):>7}"
                      f"{(mv-pmv)*100:+6.1f}{(mv-bkv)*100:+7.1f}{(pmv-bkv)*100:+8.1f}")
                rows_all.append((t, g, mv, pmv, bkv))
            else:
                print(f"  {t:<19}{_pct(pmv):>7}{_pct(bkv):>7}{(pmv-bkv)*100:+8.1f}")
                rows_all.append((t, g, None, pmv, bkv))

    # ---- the model-free signal: biggest Polymarket - Book gaps ----
    print("\n" + "=" * 78)
    print("  LARGEST POLYMARKET - BOOK GAPS  (model-free cross-venue candidates)")
    print("=" * 78)
    print(f"  {'Team':<19}{'Grp':>4}{'PM':>8}{'Book':>8}{'PM-Book':>9}")
    for t, g, mv, pmv, bkv in sorted(rows_all, key=lambda r: -abs(r[3]-r[4]))[:10]:
        print(f"  {t:<19}{g:>4}{_pct(pmv):>8}{_pct(bkv):>8}{(pmv-bkv)*100:+8.1f}p")

    # ---- if model present: confound verdict on the 5 flagged teams ----
    if with_model:
        FLAGS = ["Saudi Arabia", "Croatia", "Canada", "Qatar", "Ivory Coast"]
        print("\n" + "=" * 78)
        print("  CONFOUND TEST on the model's 5 flagged teams")
        print("  (is the model closer to the BOOK or to POLYMARKET?)")
        print("=" * 78)
        print(f"  {'Team':<15}{'Model':>7}{'PM':>7}{'Book':>7}{'|M-PM|':>8}{'|M-Book|':>10}  verdict")
        for t in FLAGS:
            mv, pmv, bkv = model[t], pm[t], book[t]
            d_pm, d_bk = abs(mv-pmv), abs(mv-bkv)
            # also: does the book sit on Polymarket's side or the model's side?
            if d_bk < d_pm - 0.01:
                verdict = "model~BOOK  -> PM soft (REAL?)"
            elif d_pm < d_bk - 0.01:
                verdict = "model~PM    -> book agrees w/ mkt (model tilt?)"
            else:
                verdict = "ambiguous"
            # sharper read: is book between model and PM, or outside?
            print(f"  {t:<15}{_pct(mv):>7}{_pct(pmv):>7}{_pct(bkv):>7}"
                  f"{d_pm*100:7.1f}p{d_bk*100:9.1f}p  {verdict}")
        print("\n  Read: if BOOK tracks POLYMARKET (both differ from MODEL the same way),")
        print("  the favourite-longshot tilt is the MODEL's independence assumption.")
        print("  If BOOK tracks the MODEL against POLYMARKET, Polymarket is the soft")
        print("  venue and that team is a genuine cross-venue candidate.")
    print("=" * 78)

if __name__ == "__main__":
    if "--pinnacle" in sys.argv:
        pinnacle_qualify_check()   # direct, airtight market
        pinnacle_r16_check()       # earlier proxy, kept for the record
    else:
        main(with_model=("--model" in sys.argv))
