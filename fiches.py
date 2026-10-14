# -*- coding: utf-8 -*-
"""Teste les 12 fiches de strategies (regles fixees d'avance, aucun reglage) sur les bougies M5 de Neon,
regroupees en 15 min, 30 min, 1H et 4H. Decision a la cloture, entree a l'ouverture suivante, spread inclus.
Garde-fous : REGLAGE 2020-2022 / VALID 2023-2024 / FINAL 2025-2026 cache, test INVERSE,
controle direction au hasard (memes moments, memes stop et objectif), critere dur ecrit dans le code.
Resultats en R : 1R = distance du stop loss + spread."""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import psycopg2
import requests
from numpy.lib.stride_tricks import sliding_window_view as swv

DB = os.environ.get("DATABASE_URL", "")
if not DB:
    sys.exit("ERREUR : DATABASE_URL manquant (Secrets GitHub).")
SHOW_FINAL = os.environ.get("SHOW_FINAL", "no").strip().lower() in ("yes", "oui", "1", "true")
SPREAD_FACTOR = 0.0 if os.environ.get("SPREAD", "normal").strip().lower() in ("zero", "0") else 1.0
SPREADS = {"XAU/USD": 0.55, "BTC/USD": 30.0, "GBP/JPY": 0.03}
NAMES = {"XAU/USD": "GOLD", "BTC/USD": "BTCUSD", "GBP/JPY": "GBPJPY"}
MINS = {"15min": 15, "30min": 30, "1h": 60, "4h": 240}

# critere dur, fixe avant de voir les resultats
MIN_R, MIN_PF, MIN_YEARS, MIN_VALID_TRADES = 0.10, 1.15, 3, 100


def telegram(text):
    t, c = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not t or not c:
        return
    try:
        requests.post("https://api.telegram.org/bot%s/sendMessage" % t,
                      json={"chat_id": c, "text": text[:4000]}, timeout=30)
    except Exception as e:
        print("Telegram :", e)


def load(conn, sym):
    cur = conn.cursor()
    cur.execute("select ts, o, h, l, c from candles where symbol=%s order by ts", (sym,))
    rows = cur.fetchall()
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c"])
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def resample(df, rule):
    x = df.set_index("ts")
    return x.resample(rule, label="left", closed="left").agg(
        {"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()


# ---------------------------------------------------------------- indicateurs
def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().values


def sma(x, n):
    return pd.Series(x).rolling(n).mean().values


def rma(x, n):
    return pd.Series(x).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean().values


def wma(x, n):
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        w = np.arange(1, n + 1, dtype=float)
        out[n - 1:] = swv(x, n) @ (w / w.sum())
    return out


def rsi(x, n):
    d = np.diff(x, prepend=x[0])
    ag, al = rma(np.maximum(d, 0), n), rma(np.maximum(-d, 0), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 100 - 100 / (1 + ag / al)
    return np.where(al == 0, 100.0, r)


def cci(tp, n):
    out = np.full(len(tp), np.nan)
    if len(tp) >= n:
        w = swv(tp, n)
        md = np.abs(w - w.mean(axis=1, keepdims=True)).mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[n - 1:] = (tp[n - 1:] - w.mean(axis=1)) / (0.015 * md)
    return out


def dmi(h, l, tr, n):
    up = np.diff(h, prepend=h[0])
    dn = -np.diff(l, prepend=l[0])
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    trn = rma(tr, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100 * rma(pdm, n) / trn, 100 * rma(mdm, n) / trn


def sar(h, l, c, af0, step, mx):
    n = len(h)
    bullv = np.zeros(n, dtype=bool)
    if n < 3:
        return bullv
    bull = c[1] >= c[0]
    ep = h[1] if bull else l[1]
    s = l[0] if bull else h[0]
    af = af0
    bullv[1] = bull
    for i in range(2, n):
        s = s + af * (ep - s)
        if bull:
            s = min(s, l[i - 1], l[i - 2])
            if l[i] < s:
                bull, s, ep, af = False, ep, l[i], af0
            elif h[i] > ep:
                ep, af = h[i], min(af + step, mx)
        else:
            s = max(s, h[i - 1], h[i - 2])
            if h[i] > s:
                bull, s, ep, af = True, ep, h[i], af0
            elif l[i] < ep:
                ep, af = l[i], min(af + step, mx)
        bullv[i] = bull
    return bullv


def pivots(h, l, k):
    """Dernier sommet / creux confirme (k bougies de chaque cote), connu seulement k bougies apres."""
    ph = h == pd.Series(h).rolling(2 * k + 1, center=True).max().values
    pl = l == pd.Series(l).rolling(2 * k + 1, center=True).min().values
    hi = pd.Series(np.where(ph, h, np.nan)).shift(k).ffill().values
    lo = pd.Series(np.where(pl, l, np.nan)).shift(k).ffill().values
    return hi, lo


def lag(x, k):
    return np.r_[np.full(k, np.nan), x[:-k]]


def cross_up(a, b):
    b = np.full(len(a), b, dtype=float) if np.isscalar(b) else b
    with np.errstate(invalid="ignore"):
        return (a > b) & (lag(a, 1) <= lag(b, 1))


def cross_dn(a, b):
    b = np.full(len(a), b, dtype=float) if np.isscalar(b) else b
    with np.errstate(invalid="ignore"):
        return (a < b) & (lag(a, 1) >= lag(b, 1))


def first(b):
    return b & ~np.r_[False, b[:-1]]


def since(b):
    """Nombre de bougies depuis le dernier True (tres grand si jamais)."""
    i = np.arange(len(b))
    last = np.maximum.accumulate(np.where(b, i, -10 ** 9))
    return i - last


def mk(up, dn):
    d = np.zeros(len(up), dtype=int)
    d[up] = 1
    d[dn] = -1
    return d


def prepare(df, rule):
    b = resample(df, rule)
    o, h, l, c = (b[x].values for x in ("o", "h", "l", "c"))
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    tr[0] = h[0] - l[0]
    return SimpleNamespace(n=len(c), o=o, h=h, l=l, c=c, tr=tr, atr=rma(tr, 14), idx=b.index.values,
                           ts=b.index, mins=MINS[rule], m5=df)


# ---------------------------------------------------------------- les 12 fiches (regles fixees d'avance)
# chaque fiche renvoie : d (direction), slb (distance du stop), rr (objectif en multiple du risque) ou tpb
# (distance de l'objectif) ou rien, ex = (sortie des achats, sortie des ventes), hold = bougies maximum
def sg(d, slb, hold, rr=None, tpb=None, ex=None):
    return SimpleNamespace(d=d, slb=slb, rr=rr, tpb=tpb, ex=ex, hold=hold)


def f1_bb_macd_ma(H):
    mid, ma = sma(H.c, 20), rma(H.c, 2)
    macd = ema(H.c, 11) - ema(H.c, 27)
    d = mk(cross_up(ma, mid) & (macd < 0), cross_dn(ma, mid) & (macd > 0))
    return sg(d, 2.0 * H.atr, 48, tpb=1.6 * H.atr)


def f2_ema200_ao(H):
    e200, mp = ema(H.c, 200), (H.h + H.l) / 2
    ao = sma(mp, 5) - sma(mp, 34)
    with np.errstate(invalid="ignore"):
        buy = (H.c > e200) & (ao > 0) & (ao > lag(ao, 1))
        sell = (H.c < e200) & (ao < 0) & (ao < lag(ao, 1))
        hi, lo = pivots(H.h, H.l, 5)
        slL, slS = H.c - (lo - 0.1 * H.atr), (hi + 0.1 * H.atr) - H.c
        d = mk(first(buy) & (slL > 0), first(sell) & (slS > 0))
    return sg(d, np.where(d == 1, slL, slS), 48, rr=1.0)


def f3_right_moment(H):
    c = H.c
    basis = sma(c, 20)
    dev = 2 * pd.Series(c).rolling(20).std(ddof=0).values
    r = rsi(c, 5)
    hh, ll = pd.Series(H.h).rolling(25).max().values, pd.Series(H.l).rolling(25).min().values
    with np.errstate(divide="ignore", invalid="ignore"):
        wr = 100 * (c - hh) / (hh - ll)
        up = cross_up(r, 30) & (since(cross_up(wr, -80)) <= 2) & (since(H.l <= basis - dev) <= 2) & (basis > c)
        dn = cross_dn(r, 70) & (since(cross_dn(wr, -20)) <= 2) & (since(H.h >= basis + dev) <= 2) & (basis < c)
    return sg(mk(up, dn), 1.5 * H.atr, 24, tpb=np.abs(basis - c))


def f4_alligator(H):
    mp = (H.h + H.l) / 2
    jaw, teeth, lips = lag(rma(mp, 13), 8), lag(rma(mp, 8), 5), lag(rma(mp, 5), 3)
    s144 = sma(H.c, 144)
    with np.errstate(invalid="ignore"):
        buy = (H.c > s144) & (lips > teeth) & (teeth > jaw)
        sell = (H.c < s144) & (lips < teeth) & (teeth < jaw)
        dL, dS = H.c - (s144 - 0.1 * H.atr), (s144 + 0.1 * H.atr) - H.c
        up = first(buy) & (dL >= 0.5 * H.atr) & (dL <= 4 * H.atr)
        dn = first(sell) & (dS >= 0.5 * H.atr) & (dS <= 4 * H.atr)
    d = mk(up, dn)
    return sg(d, np.where(d == 1, dL, dS), 500, ex=(cross_dn(lips, teeth), cross_up(lips, teeth)))


def f5_bb_ema50(H):
    c = H.c
    basis = sma(c, 20)
    dev = 3 * pd.Series(c).rolling(20).std(ddof=0).values
    e50 = ema(c, 50)
    with np.errstate(invalid="ignore"):
        tL = lag(H.l, 1) <= lag(basis, 1) + 0.1 * lag(H.atr, 1)
        tS = lag(H.h, 1) >= lag(basis, 1) - 0.1 * lag(H.atr, 1)
        slL, slS = c - (basis - 0.1 * H.atr), (basis + 0.1 * H.atr) - c
        up = (c > e50) & tL & (H.h > lag(H.h, 1)) & (slL >= 0.5 * H.atr) & (slL <= 4 * H.atr) & (basis + dev > c)
        dn = (c < e50) & tS & (H.l < lag(H.l, 1)) & (slS >= 0.5 * H.atr) & (slS <= 4 * H.atr) & (basis - dev < c)
    d = mk(up, dn)
    return sg(d, np.where(d == 1, slL, slS), 40, tpb=np.where(d == 1, basis + dev - c, c - (basis - dev)))


def f6_three_sma(H):
    m13, m26, m100 = sma(H.c, 13), sma(H.c, 26), sma(H.c, 100)
    with np.errstate(invalid="ignore"):
        d = mk(first((m13 > m26) & (m26 > m100)), first((m13 < m26) & (m26 < m100)))
    return sg(d, 3.0 * H.atr, 500, ex=(cross_dn(m13, m26), cross_up(m13, m26)))


def f7_follow_trend(H):
    e4, e10 = ema(H.c, 4), ema(H.c, 10)
    macd = ema(H.c, 5) - ema(H.c, 10)
    pdi, mdi = dmi(H.h, H.l, H.tr, 28)
    with np.errstate(invalid="ignore"):
        d = mk(cross_up(e4, e10) & (pdi > mdi) & (macd > 0), cross_dn(e4, e10) & (mdi > pdi) & (macd < 0))
    return sg(d, 1.0 * H.atr, 40, tpb=3.0 * H.atr)


def f8_intelligent(H):
    c = H.c
    src = (H.h + H.l + c) / 3
    ma8, ma18 = rma(src, 8), rma(src, 18)
    macd = ema(c, 8) - ema(c, 21)
    hh, ll = pd.Series(H.h).rolling(12).max().values, pd.Series(H.l).rolling(12).min().values
    with np.errstate(divide="ignore", invalid="ignore"):
        k = sma(100 * (c - ll) / (hh - ll), 5)
    dd = sma(k, 12)
    cu, cd = cross_up(k, 50) | cross_up(dd, 50), cross_dn(k, 50) | cross_dn(dd, 50)
    bull = sar(H.h, H.l, c, 0.026, 0.026, 0.5)
    pb = np.r_[False, bull[:-1]]
    with np.errstate(invalid="ignore"):
        d = mk(cu & (ma8 > ma18) & (macd > 0), cd & (ma8 < ma18) & (macd < 0))
    return sg(d, 3.0 * H.atr, 500, ex=(pb & ~bull, ~pb & bull))


def f9_ema100_two_tf(H):
    e5, e100 = ema(H.c, 5), ema(H.c, 100)
    h1 = resample(H.m5, "1h")
    c1 = h1["c"].values
    up1 = pd.Series(np.where(np.arange(len(c1)) >= 100, (ema(c1, 5) > ema(c1, 100)).astype(float), np.nan),
                    index=h1.index)
    dec = H.ts + pd.Timedelta(minutes=H.mins)             # moment de la decision (cloture de la bougie)
    f = up1.reindex(dec.floor("1h") - pd.Timedelta(hours=1)).values   # derniere bougie 1H deja fermee
    sess = (dec.hour.values >= 7) & (dec.hour.values < 20)
    with np.errstate(invalid="ignore"):
        d = mk(cross_up(e5, e100) & (f == 1) & sess, cross_dn(e5, e100) & (f == 0) & sess)
    return sg(d, 1.0 * H.atr, 48, tpb=3.0 * H.atr)


def f10_range(H):
    c = H.c
    e16, e30, w5, w12, r = ema(c, 16), ema(c, 30), wma(c, 5), wma(c, 12), rsi(c, 19)
    rh, rl = np.maximum(e16, e30), np.minimum(e16, e30)
    with np.errstate(invalid="ignore"):
        d = mk(first((w5 > rh) & (w12 > rh) & (r > 50)), first((w5 < rl) & (w12 < rl) & (r < 50)))
        ex = (w5 < rh, w5 > rl)
    return sg(d, 3.0 * H.atr, 500, ex=ex)


def f11_cci_ema(H):
    e8, e28, cc = ema(H.c, 8), ema(H.c, 28), cci((H.h + H.l + H.c) / 3, 30)
    with np.errstate(invalid="ignore"):
        d = mk(cross_up(e8, e28) & (cc > 0), cross_dn(e8, e28) & (cc < 0))
    return sg(d, 2.0 * H.atr, 48, tpb=5.0 * H.atr)


def f12_macd_trend(H):
    w85, w75, w5 = wma(H.l, 85), wma(H.l, 75), wma(H.l, 5)
    macd = ema(H.c, 15) - ema(H.c, 26)
    rh, rl = np.maximum(w85, w75), np.minimum(w85, w75)
    with np.errstate(invalid="ignore"):
        d = mk(first((w5 > rh) & (macd > 0)), first((w5 < rl) & (macd < 0)))
    return sg(d, 1.0 * H.atr, 48, tpb=2.0 * H.atr)


FICHES = [  # (nom, fonction, unites de temps)
    ("F1_BB_MACD_MA", f1_bb_macd_ma, ["15min"]),
    ("F2_EMA200_AO", f2_ema200_ao, ["1h"]),
    ("F3_RIGHT_MOMENT", f3_right_moment, ["15min", "1h"]),
    ("F4_ALLIGATOR", f4_alligator, ["15min", "1h"]),
    ("F5_BB_EMA50", f5_bb_ema50, ["1h", "4h"]),
    ("F6_TROIS_SMA", f6_three_sma, ["1h", "4h"]),
    ("F7_FOLLOW_TREND", f7_follow_trend, ["4h"]),
    ("F8_INTELLIGENT", f8_intelligent, ["1h"]),
    ("F9_100EMA_2TF", f9_ema100_two_tf, ["15min"]),
    ("F10_RANGE", f10_range, ["15min", "30min", "1h"]),
    ("F11_CCI_EMA", f11_cci_ema, ["30min", "1h"]),
    ("F12_MACD_TREND", f12_macd_trend, ["30min", "1h"]),
]


# ---------------------------------------------------------------- simulateur (ecarts de prix inclus)
def run_trades(H, s, dirs, spr):
    o, h, l, c, n = H.o, H.h, H.l, H.c, H.n
    out, free = [], 0
    for i in np.where(s.d != 0)[0]:
        if i < free or i + s.hold + 1 >= n:
            continue
        sb = s.slb[i]
        if np.isnan(sb) or sb <= 0:
            continue
        dr = int(dirs[i])
        fill = o[i + 1] + dr * spr / 2
        sld = sb + spr
        td = None
        if s.tpb is not None:
            td = s.tpb[i]
            if np.isnan(td) or td <= 0:
                continue
        elif s.rr is not None:
            td = s.rr * sld
        sl = fill - dr * sld
        tp = None if td is None else fill + dr * td
        res = None
        for j in range(i + 1, i + 1 + s.hold):
            if dr == 1:
                bo, blo, bhi = o[j] - spr / 2, l[j] - spr / 2, h[j] - spr / 2
                if j > i + 1 and (bo <= sl or (tp is not None and bo >= tp)):
                    res = ((bo - fill) / sld, j)
                elif blo <= sl:
                    res = (-1.0, j)
                elif tp is not None and bhi >= tp:
                    res = (td / sld, j)
            else:
                ao, ahi, alo = o[j] + spr / 2, h[j] + spr / 2, l[j] + spr / 2
                if j > i + 1 and (ao >= sl or (tp is not None and ao <= tp)):
                    res = ((fill - ao) / sld, j)
                elif ahi >= sl:
                    res = (-1.0, j)
                elif tp is not None and alo <= tp:
                    res = (td / sld, j)
            if res is None and s.ex is not None and j + 1 < n:
                if (s.ex[0][j] if s.d[i] == 1 else s.ex[1][j]):      # sortie sur signal, a l'ouverture suivante
                    res = (dr * ((o[j + 1] - dr * spr / 2) - fill) / sld, j + 1)
            if res is not None:
                break
        if res is None:
            res = (dr * ((c[i + s.hold] - dr * spr / 2) - fill) / sld, i + s.hold)
        out.append((H.idx[i], res[0]))
        free = res[1]
    return out


def pstats(r):
    if len(r) == 0:
        return 0, float("nan"), float("nan")
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    return len(r), r.mean(), (gp / gl if gl > 0 else float("inf"))


def summarize(trades):
    if not trades:
        return None
    ts = np.array([t for t, _ in trades])
    R = np.array([r for _, r in trades])
    yr = ts.astype("datetime64[Y]").astype(int) + 1970
    if not SHOW_FINAL:
        keep = yr <= 2024
        R, yr = R[keep], yr[keep]
    if len(R) == 0:
        return None
    years = [(y, R[yr == y].mean()) for y in np.unique(yr) if (yr == y).sum() >= 10]
    return {"n": len(R), "train": pstats(R[yr <= 2022]), "valid": pstats(R[(yr >= 2023) & (yr <= 2024)]),
            "final": pstats(R[yr >= 2025]), "pos": sum(1 for _, m in years if m > 0), "ny": len(years),
            "worst": min([m for _, m in years]) if years else float("nan")}


def fmt(v):
    return "%+.2f" % v if v == v else "  n/a"


def main():
    conn = psycopg2.connect(DB)
    rng = np.random.default_rng(1)
    lines, control, cands = [], [], []
    ntests = 0
    for sym in ("XAU/USD", "BTC/USD", "GBP/JPY"):
        df = load(conn, sym)
        if df is None or len(df) < 5000:
            lines.append("%s : pas assez de bougies." % NAMES[sym])
            continue
        name, spr = NAMES[sym], SPREADS[sym] * SPREAD_FACTOR
        for rule in ("15min", "30min", "1h", "4h"):
            todo = [(nm, fn) for nm, fn, rules in FICHES if rule in rules]
            if not todo:
                continue
            H = prepare(df, rule)
            print("Calcul", name, rule, "-", H.n, "bougies", flush=True)
            for nm, fn in todo:
                s = fn(H)
                rd = np.where(s.d != 0, rng.choice([-1, 1], size=H.n), 0)
                if rule == "1h" and nm == "F12_MACD_TREND":    # controle du simulateur : sans frais, au hasard
                    z = [r for _, r in run_trades(H, s, rd, 0.0)]
                    control.append("%s : direction au hasard, spread 0 -> %s R (%d trades)" % (
                        name, fmt(np.mean(z) if z else float("nan")), len(z)))
                real = summarize(run_trades(H, s, s.d, spr))
                inv = summarize(run_trades(H, s, -s.d, spr))
                alea = summarize(run_trades(H, s, rd, spr))
                ntests += 1
                lab = "%-15s %-5s %-6s" % (nm, rule, name)
                if real is None:
                    lines.append(lab + " aucun trade")
                    continue
                tr, va, pf = real["train"][1], real["valid"][1], real["valid"][2]
                txt = "%s n=%5d | REGLAGE %s | VALID %s PF %.2f | ans>0 %d/%d | INV %s | ALEA %s" % (
                    lab, real["n"], fmt(tr), fmt(va), pf if pf == pf else 0.0, real["pos"], real["ny"],
                    fmt(inv["valid"][1]) if inv else "n/a", fmt(alea["valid"][1]) if alea else "n/a")
                if SHOW_FINAL:
                    txt += " | FINAL %s" % fmt(real["final"][1])
                lines.append(txt)
                al = alea["valid"][1] if alea and alea["valid"][1] == alea["valid"][1] else -9
                if (tr == tr and va == va and tr > MIN_R and va > MIN_R and pf > MIN_PF and real["pos"] >= MIN_YEARS
                        and real["valid"][0] >= MIN_VALID_TRADES and va > al):
                    cands.append((va, nm, rule, name, tr, real["pos"], real["ny"], real["valid"][0]))

    head = ("TEST DES 12 FICHES (R moyen apres spread, 1R = risque du trade)\n"
            "Periodes : REGLAGE 2020-2022, VALID 2023-2024%s\n" % (", FINAL 2025-2026 (affiche)" if SHOW_FINAL
                                                                  else " (2025-2026 cache : test final)"))
    if SPREAD_FACTOR == 0:
        head = "*** TEST SANS FRAIS (spread = 0) ***\n" + head
    crit = ("CRITERE DUR : REGLAGE et VALID > %+.2f R, PF VALID > %.2f, au moins %d annees positives, "
            "au moins %d trades en VALID, VALID meilleur que ALEA.\n" % (MIN_R, MIN_PF, MIN_YEARS, MIN_VALID_TRADES))
    summary = head + crit + "\nCONTROLE (doit etre proche de 0) :\n" + "\n".join(control) + "\n\n"
    if cands:
        cands.sort(reverse=True)
        summary += "CANDIDATES (%d tests au total) :\n" % ntests
        for va, nm, rule, name, tr, p, ny, nv in cands[:8]:
            summary += "- %s %s %s : REGLAGE %s, VALID %s (%d trades), annees positives %d/%d\n" % (
                nm, rule, name, fmt(tr), fmt(va), nv, p, ny)
        summary += ("Avec %d tests, quelques candidates peuvent sortir par hasard : le test FINAL decide, "
                    "une seule fois, pour UNE candidate.\n" % ntests)
    else:
        summary += "AUCUNE fiche ne passe le critere dur (%d tests).\n" % ntests
    guide = ("\nComment lire : INV = memes moments, direction opposee. ALEA = memes moments et memes stop/objectif, "
             "direction au hasard, avec frais : c'est le niveau a battre. ans>0 = annees positives. "
             "Resultats theoriques, sans glissement ni frais de nuit.")
    full = summary + "\nDETAIL :\n" + "\n".join(lines) + "\n" + guide
    print(full)
    sf = os.environ.get("GITHUB_STEP_SUMMARY")
    if sf:
        with open(sf, "a", encoding="utf-8") as f:
            f.write("```\n" + full + "\n```\n")
    telegram(summary + guide)


if __name__ == "__main__":
    main()
