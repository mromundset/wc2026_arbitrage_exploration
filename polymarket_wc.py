"""
polymarket_wc.py  (v2 -- corrected: WC games DO exist)
======================================================
Single-source pipeline on Polymarket for the 2026 World Cup.

Correct tags discovered:
  - Match (game) events : tag 'fifa-world-cup' (102232) + 'games' (100639),
                          slug pattern  fifwc-<home>-<away>-YYYY-MM-DD
                          Each game = 3 binary markets (home win / draw / away win).
  - Group-winner events : slug  world-cup-group-<x>-winner   (tag 'world-cup' 519)
  - Advance-to-knockout : slug  world-cup-team-to-advance-to-knockout-stages
  - Outright winner     : slug  world-cup-winner

Pipeline:
  match 1X2  --devig-->  per-match P(home/draw/away)
            --fit-->     independent-Poisson goal rates (lam_home, lam_away)
            --simulate-->group standings w/ tiebreakers
            --aggregate->model P(win group), P(top 2)
  then juxtapose vs Polymarket's own group-winner and advance markets.
"""
from __future__ import annotations
import json, urllib.request, re, math
import numpy as np
from collections import defaultdict

GAMMA = "https://gamma-api.polymarket.com"
FIFA_WC_TAG = 102232   # 'fifa-world-cup'

def _get(url, timeout=45):
    req = urllib.request.Request(url, headers={"Accept":"application/json","User-Agent":"wc/2.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def _slug(s):
    d = _get(f"{GAMMA}/events/slug/{s}")
    return d[0] if isinstance(d, list) else d

def _yes(m):
    pr = m.get("outcomePrices")
    if not pr: return None
    try: return float(json.loads(pr)[0] if isinstance(pr,str) else pr[0])
    except: return None

def devig(prices, target=1.0):
    s = sum(v for v in prices.values() if v is not None)
    return {k:(v*target/s if v is not None else None) for k,v in prices.items()} if s else prices

# ---- fetchers ---------------------------------------------------------------
def fetch_games():
    """Return list of {home, away, date, p_home, p_draw, p_away} (devigged 1X2)."""
    games, seen = [], set()
    for off in range(0, 1600, 100):
        b = _get(f"{GAMMA}/events?tag_id={FIFA_WC_TAG}&limit=100&offset={off}&closed=false")
        if not b: break
        for e in b:
            sl = e.get("slug","")
            if not sl.startswith("fifwc-") or sl in seen: continue
            seen.add(sl)
            title = e.get("title","")
            parts = re.split(r"\s+vs\.?\s+", title)
            if len(parts)!=2: continue
            home, away = parts[0].strip(), parts[1].strip()
            ph=pa=pd=None
            for m in e.get("markets",[]):
                q=m.get("question",""); y=_yes(m)
                if y is None: continue
                mm=re.match(r"Will (.+?) win on", q)
                if mm:
                    if mm.group(1).strip()==home: ph=y
                    elif mm.group(1).strip()==away: pa=y
                elif "end in a draw" in q: pd=y
            if None in (ph,pd,pa): continue
            dv = devig({"h":ph,"d":pd,"a":pa})
            md = re.search(r"(\d{4}-\d{2}-\d{2})$", sl)
            games.append(dict(home=home, away=away, date=md.group(1) if md else "",
                              p_home=dv["h"], p_draw=dv["d"], p_away=dv["a"]))
    return games

def fetch_group_winner(letter):
    ev=_slug(f"world-cup-group-{letter.lower()}-winner")
    out={}
    for m in ev.get("markets",[]):
        y=_yes(m); mm=re.match(r"Will (.+?) win Group", m.get("question",""))
        if mm and y is not None: out[mm.group(1).strip()]=y
    return out

def fetch_advance_prices():
    ev=_slug("world-cup-team-to-advance-to-knockout-stages")
    out={}
    for m in ev.get("markets",[]):
        y=_yes(m); mm=re.match(r"Will (.+?) advance", m.get("question",""))
        if mm and y is not None: out[mm.group(1).strip()]=y
    return out

# ---- fit goal rates from 1X2 (independent Poisson) --------------------------
def _wdl(l1, l2, kmax=12):
    k=np.arange(kmax)
    p1=np.exp(-l1)*l1**k/np.array([math.factorial(i) for i in k])
    p2=np.exp(-l2)*l2**k/np.array([math.factorial(i) for i in k])
    M=np.outer(p1,p2)
    home=np.tril(M,-1).sum(); draw=np.trace(M); away=np.triu(M,1).sum()
    return home, draw, away

def fit_lambdas(ph, pd, pa):
    """Grid+refine for (lam_home, lam_away) reproducing the devigged 1X2."""
    best=None
    grid=np.arange(0.15,3.6,0.05)
    for l1 in grid:
        for l2 in grid:
            h,d,a=_wdl(l1,l2)
            err=(h-ph)**2+(a-pa)**2
            if best is None or err<best[0]: best=(err,l1,l2)
    # local refine
    _,l1,l2=best
    for l1r in np.arange(l1-0.05,l1+0.05,0.01):
        for l2r in np.arange(l2-0.05,l2+0.05,0.01):
            h,d,a=_wdl(l1r,l2r)
            err=(h-ph)**2+(a-pa)**2
            if err<best[0]: best=(err,l1r,l2r)
    return best[1], best[2]

# ---- simulate a group from per-match goal rates -----------------------------
def simulate_group_from_market(teams, fixtures, n=20000, seed=7):
    """fixtures: list of (home, away, lam_home, lam_away)."""
    rng=np.random.default_rng(seed)
    pre={(h,a):(rng.poisson(lh,n), rng.poisson(la,n)) for (h,a,lh,la) in fixtures}
    win=dict.fromkeys(teams,0); top2=dict.fromkeys(teams,0)
    for s in range(n):
        pts=dict.fromkeys(teams,0); gf=dict.fromkeys(teams,0); ga=dict.fromkeys(teams,0)
        res={}
        for (h,a) in pre:
            gh=pre[(h,a)][0][s]; gax=pre[(h,a)][1][s]
            res[(h,a)]=(gh,gax)
            gf[h]+=gh; ga[h]+=gax; gf[a]+=gax; ga[a]+=gh
            if gh>gax: pts[h]+=3
            elif gax>gh: pts[a]+=3
            else: pts[h]+=1; pts[a]+=1
        # rank: points -> H2H(among tied: pts,gd,gf) -> overall GD -> GF -> random
        order=sorted(teams, key=lambda t: pts[t], reverse=True)
        # resolve ties
        final=[]; i=0
        while i<len(order):
            j=i
            while j+1<len(order) and pts[order[j+1]]==pts[order[i]]: j+=1
            grp=order[i:j+1]
            if len(grp)>1:
                hp=defaultdict(int); hgf=defaultdict(int); hga=defaultdict(int); S=set(grp)
                for (h,a),(gh,gax) in res.items():
                    if h in S and a in S:
                        hgf[h]+=gh; hga[h]+=gax; hgf[a]+=gax; hga[a]+=gh
                        if gh>gax: hp[h]+=3
                        elif gax>gh: hp[a]+=3
                        else: hp[h]+=1; hp[a]+=1
                grp=sorted(grp, key=lambda t:(hp[t], hgf[t]-hga[t], hgf[t],
                                              gf[t]-ga[t], gf[t], rng.random()), reverse=True)
            final.extend(grp); i=j+1
        win[final[0]]+=1; top2[final[0]]+=1; top2[final[1]]+=1
    return {t:dict(win_group=win[t]/n, top2=top2[t]/n) for t in teams}

# ---- demo: Group I end to end ----------------------------------------------
def _pct(x): return f"{100*x:5.1f}%"

def main():
    GI=["France","Senegal","Norway","Iraq"]
    print("="*70); print("  GROUP I  -- propagate Polymarket match 1X2 -> compare to its")
    print("            own group-winner & advance markets"); print("="*70)

    games=fetch_games()
    gi=[g for g in games if g["home"] in GI and g["away"] in GI]
    print(f"\nFetched {len(games)} WC group games; {len(gi)} are Group I fixtures.")
    fixtures=[]
    print("\nMarket-implied match probabilities -> fitted goal rates:")
    for g in gi:
        lh,la=fit_lambdas(g["p_home"],g["p_draw"],g["p_away"])
        fixtures.append((g["home"],g["away"],lh,la))
        print(f"  {g['home']:>9} v {g['away']:<9} "
              f"P {g['p_home']:.2f}/{g['p_draw']:.2f}/{g['p_away']:.2f}  "
              f"-> lam {lh:.2f}/{la:.2f}")

    model=simulate_group_from_market(GI, fixtures, n=20000)

    gw=devig(fetch_group_winner("I"))           # market group winner (devigged)
    adv=fetch_advance_prices()                   # market advance (already ~prob)

    print("\n--- WIN GROUP: model (from match odds) vs market group-winner ---")
    print(f"  {'Team':<9}{'model':>8}{'market':>9}{'edge':>8}")
    for t in sorted(GI, key=lambda x:-model[x]['win_group']):
        e=model[t]['win_group']-gw.get(t,float('nan'))
        print(f"  {t:<9}{_pct(model[t]['win_group']):>8}{_pct(gw.get(t,0)):>9}{e*100:+7.1f}p")

    print("\n--- ADVANCE: model P(top2) vs market P(advance, incl. 3rd-place path) ---")
    print(f"  {'Team':<9}{'model top2':>11}{'mkt advance':>12}{'edge':>8}")
    for t in sorted(GI, key=lambda x:-model[x]['top2']):
        e=model[t]['top2']-adv.get(t,float('nan'))
        print(f"  {t:<9}{_pct(model[t]['top2']):>11}{_pct(adv.get(t,0)):>12}{e*100:+7.1f}p")
    print("\nNote: model 'top2' excludes the best-third path, so it should sit a")
    print("touch BELOW market 'advance' for mid teams -- that gap is the third-place")
    print("qualification value, which needs all 12 groups simulated jointly.")
    print("="*70)

if __name__=="__main__":
    main()
