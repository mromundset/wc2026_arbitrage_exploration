"""
wc2026_joint.py  (v2)  --  Joint 12-group WC model, Polymarket-sourced,
                           bivariate-Poisson match engine calibrated to 1X2 + totals.
=====================================================================================
Per match we fit a BIVARIATE Poisson (lam1, lam2, lam3-shared) to THREE market
targets: P(home win), P(away win) from the moneyline, and P(over 2.5) from the
totals (O/U) market in the '-more-markets' sibling event. The shared term lets
the model match the market's goal total -- which drives goal-difference and
goals-scored, hence the third-place cutoff -- instead of being forced by the 1X2
alone. The advance market is devigged to sum to 32 before comparison.

Needs polymarket_wc.py (low-level fetchers), numpy, scipy, internet.

Honest limits unchanged: still assumes a team's three matches are INDEPENDENT
(no tournament-form correlation) and ignores dead-rubber rotation -- the two
effects most likely behind any residual favourite tilt. Tiebreak = GD-first.
"""
from __future__ import annotations
import numpy as np, math, json, re
from scipy.optimize import least_squares
from polymarket_wc import _get, _slug, devig, fetch_advance_prices, FIFA_WC_TAG

GROUPS = {
 "A":["Mexico","South Korea","South Africa","Czechia"],
 "B":["Canada","Switzerland","Qatar","Bosnia-Herzegovina"],
 "C":["Brazil","Morocco","Scotland","Haiti"],
 "D":["USA","Paraguay","Australia","Turkiye"],
 "E":["Germany","Ecuador","Ivory Coast","Curacao"],
 "F":["Netherlands","Japan","Tunisia","Sweden"],
 "G":["Belgium","Iran","Egypt","New Zealand"],
 "H":["Spain","Uruguay","Saudi Arabia","Cape Verde"],
 "I":["France","Senegal","Norway","Iraq"],
 "J":["Argentina","Austria","Algeria","Jordan"],
 "K":["Portugal","Colombia","Uzbekistan","DR Congo"],
 "L":["England","Croatia","Panama","Ghana"],
}
NAME_MAP = {"Korea Republic":"South Korea","United States":"USA","Türkiye":"Turkiye",
 "Côte d'Ivoire":"Ivory Coast","Curaçao":"Curacao","Cabo Verde":"Cape Verde",
 "IR Iran":"Iran","Bosnia and Herzegovina":"Bosnia-Herzegovina","Congo DR":"DR Congo"}
def canon(n): return NAME_MAP.get(n.strip(), n.strip())
ALL=set(t for g in GROUPS.values() for t in g); T2G={t:g for g,ts in GROUPS.items() for t in ts}

# ---------- fetch games (with slug) + totals ----------
def fetch_games_full():
    out, seen = [], set()
    for off in range(0,1600,100):
        b=_get(f"https://gamma-api.polymarket.com/events?tag_id={FIFA_WC_TAG}&limit=100&offset={off}&closed=false")
        if not b: break
        for e in b:
            sl=e.get("slug","")
            if not sl.startswith("fifwc-") or sl.endswith("-more-markets") or sl in seen: continue
            seen.add(sl)
            parts=re.split(r"\s+vs\.?\s+", e.get("title",""))
            if len(parts)!=2: continue
            h,a=parts[0].strip(),parts[1].strip(); ph=pd=pa=None
            for m in e.get("markets",[]):
                pr=m.get("outcomePrices"); 
                if not pr: continue
                y=float(json.loads(pr)[0]); q=m.get("question","")
                mm=re.match(r"Will (.+?) win on",q)
                if mm:                                   # canon both sides: Polymarket
                    nm=canon(mm.group(1).strip())        # spells some names differently
                    if nm==canon(h): ph=y                # in the question vs the title
                    elif nm==canon(a): pa=y              # (e.g. Bosnia and/-Herzegovina)
                elif "end in a draw" in q: pd=y
            if None in (ph,pd,pa): continue
            dv=devig({"h":ph,"d":pd,"a":pa})
            out.append(dict(slug=sl,home=h,away=a,p_home=dv["h"],p_draw=dv["d"],p_away=dv["a"]))
    return out

def fetch_over25(slug):
    try: ev=_slug(slug+"-more-markets")
    except Exception: return None
    for m in ev.get("markets",[]):
        if "O/U 2.5" in m.get("question",""):
            pr=m.get("outcomePrices")
            if pr:
                o,u=[float(x) for x in json.loads(pr)[:2]]
                return o/(o+u)
    return None

# ---------- bivariate Poisson ----------
K=13; _F=np.array([math.factorial(i) for i in range(K)],float)
def _pois(l): k=np.arange(K); return np.exp(-l)*l**k/_F
def _joint(l1,l2,l3):
    p1,p2,p3=_pois(l1),_pois(l2),_pois(l3); J=np.zeros((K,K))
    for k in range(K):
        if k>0 and p3[k]<1e-10: break
        a=np.zeros(K); a[k:]=p1[:K-k]; b=np.zeros(K); b[k:]=p2[:K-k]
        J+=p3[k]*np.outer(a,b)
    return J
_TOT=np.add.outer(np.arange(K),np.arange(K))
def _probs(l1,l2,l3):
    J=_joint(l1,l2,l3)
    return np.tril(J,-1).sum(), np.triu(J,1).sum(), J[_TOT>=3].sum()
def fit_biv(ph,pa,po):
    if po is None:  # no totals -> 2-param independent fit (lam3=0)
        def r(x): a,b,_=_probs(x[0],x[1],0.0); return [a-ph,b-pa]
        s=least_squares(r,[1.2,1.0],bounds=([0.05,0.05],[5,5])); return s.x[0],s.x[1],0.0
    def r(x): a,b,c=_probs(*x); return [a-ph,b-pa,c-po]
    s=least_squares(r,[1.2,1.0,0.1],bounds=([0.05,0.05,0.0],[5,5,1.5]))
    return tuple(s.x)

def build():
    games=fetch_games_full()
    print(f"  fetched {len(games)} games; pulling totals ...", flush=True)
    gfx={g:[] for g in GROUPS}; n_tot=0
    for gm in games:
        h,a=canon(gm["home"]),canon(gm["away"])
        if h in T2G and a in T2G and T2G[h]==T2G[a]:
            po=fetch_over25(gm["slug"]); n_tot+= po is not None
            l1,l2,l3=fit_biv(gm["p_home"],gm["p_away"],po)
            gfx[T2G[h]].append((h,a,l1,l2,l3))
    print(f"  totals found for {n_tot}/{len(games)} games", flush=True)
    return gfx

# ---------- vectorised joint sim (bivariate goals) ----------
def simulate(gfx, n=50000, rng=None):
    if rng is None: rng=np.random.default_rng(2026)
    win=dict.fromkeys(ALL,0.0); top2=dict.fromkeys(ALL,0.0)
    thirdq=dict.fromkeys(ALL,0.0); adv=dict.fromkeys(ALL,0.0)
    third_comp=[]; meta=[]
    for g,teams in GROUPS.items():
        idx={t:i for i,t in enumerate(teams)}
        pts=np.zeros((4,n)); gf=np.zeros((4,n)); ga=np.zeros((4,n))
        for (h,a,l1,l2,l3) in gfx[g]:
            w=rng.poisson(l3,n); gh=rng.poisson(l1,n)+w; gx=rng.poisson(l2,n)+w
            hi,ai=idx[h],idx[a]
            gf[hi]+=gh; ga[hi]+=gx; gf[ai]+=gx; ga[ai]+=gh
            pts[hi]+=3*(gh>gx)+(gh==gx); pts[ai]+=3*(gx>gh)+(gh==gx)
        comp=pts*1e6+((gf-ga)+100)*1e3+gf+rng.random((4,n))
        order=np.argsort(-comp,axis=0); w_i,r_i,t_i=order[0],order[1],order[2]
        for i,t in enumerate(teams):
            win[t]+=(w_i==i).mean(); top2[t]+=((w_i==i)|(r_i==i)).mean()
        tp=np.take_along_axis(pts,t_i[None],0)[0]
        tgd=np.take_along_axis(gf-ga,t_i[None],0)[0]
        tgf=np.take_along_axis(gf,t_i[None],0)[0]
        third_comp.append(tp*1e6+(tgd+100)*1e3+tgf+rng.random(n))
        meta.append((teams,t_i,w_i,r_i))
    C=np.vstack(third_comp); cutoff=-np.sort(-C,axis=0)[7]
    for gi,(teams,t_i,w_i,r_i) in enumerate(meta):
        q=C[gi]>=cutoff
        for i,t in enumerate(teams):
            is3=(t_i==i)
            thirdq[t]+=(is3&q).mean()
            adv[t]+=((w_i==i)|(r_i==i)|(is3&q)).mean()
    return {t:dict(win_group=win[t],top2=top2[t],third_q=thirdq[t],advance=adv[t]) for t in ALL}

def noise_bands(gfx,n=15000,scen=15,sd=0.07):
    rng=np.random.default_rng(404); st={t:[] for t in ALL}
    for _ in range(scen):
        p={g:[(h,a,max(.05,l1+rng.normal(0,sd)),max(.05,l2+rng.normal(0,sd)),max(0,l3+rng.normal(0,sd/2)))
              for (h,a,l1,l2,l3) in fx] for g,fx in gfx.items()}
        m=simulate(p,n,rng)
        for t in ALL: st[t].append(m[t]["advance"])
    return {t:float(np.std(st[t])) for t in ALL}

def _pct(x): return f"{100*x:5.1f}%"
def main(n_main=50000):
    print("="*80); print("  JOINT MODEL v2 -- bivariate Poisson fit to 1X2 + TOTALS"); print("="*80)
    gfx=build()
    bad=[g for g in GROUPS if len(gfx[g])!=6]; print("  incomplete groups:", bad or "none")
    model=simulate(gfx,n=n_main)
    print(f"  sum model P(advance) = {sum(model[t]['advance'] for t in ALL):.2f} (=32)")
    mkt={canon(k):v for k,v in fetch_advance_prices().items()}
    s=sum(mkt.get(t,0) for t in ALL); mkt={t:mkt.get(t,0)*32/s for t in ALL}  # devig to 32
    print(f"  market devigged to sum 32 (raw was {s:.2f})")
    print("  noise bands ...", flush=True); bands=noise_bands(gfx)
    rows=[]
    for t in ALL:
        e=model[t]["advance"]-mkt[t]; nz=bands[t]
        rows.append((t,T2G[t],model[t]["top2"],model[t]["third_q"],model[t]["advance"],
                     mkt[t],e,nz, abs(e)>0.04 and abs(e)>2*nz))
    rows.sort(key=lambda r:-abs(r[6]))
    print("\n  model advance = top2 + 3rd-qualify | market devigged to 32 | FLAG=|edge|>4pp & >2xnoise")
    print("-"*80)
    print(f"  {'Team':<17}{'Grp':>3}{'top2':>7}{'3rdQ':>7}{'mAdv':>8}{'mkt':>8}{'edge':>8}{'noise':>7}")
    for t,g,t2,t3,ad,mk,e,nz,fl in rows:
        print(f"  {t:<17}{g:>3}{_pct(t2):>7}{_pct(t3):>7}{_pct(ad):>8}{_pct(mk):>8}{e*100:+7.1f}p{nz*100:6.1f}p"
              + ("  <<FLAG" if fl else ""))
    print("-"*80)
    print(f"  flags: {sum(1 for r in rows if r[8])}  (1X2-only v1 produced 7)")
    print("="*80)
if __name__=="__main__": main()
