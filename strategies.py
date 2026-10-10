# -*- coding: utf-8 -*-
"""Teste plusieurs strategies sur les bougies M5 stockees dans Neon.
Garde-fous : peu de strategies fixees d'avance, periodes REGLAGE / VALIDATION / FINAL,
resultats par annee, test INVERSE, controle du simulateur avec un spread a 0.
Resultats en R : 1R = le risque prevu du trade (distance du stop loss, spread inclus)."""
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
# SPREAD=zero : meme test mais sans frais, pour mesurer l'avantage brut du signal
SPREAD_FACTOR = 0.0 if os.environ.get("SPREAD", "normal").strip().lower() in ("zero", "0") else 1.0

SPREADS = {"XAU/USD": 0.55, "BTC/USD": 30.0, "GBP/JPY": 0.03}
NAMES = {"XAU/USD": "GOLD", "BTC/USD": "BTCUSD", "GBP/JPY": "GBPJPY"}


def telegram(text):
    t, c = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not t or not c:
        return
    try:
        requests.post("https://api.telegram.org/bot%s/sendMessage" % t,
                      json={"chat_id": c, "text": text[:4000]}, timeout=30)
    except Exception as e:
        print("Telegram :", e)


# ---------------------------------------------------------------- donnees et indicateurs
def load(conn, sym):
    cur = conn.cursor()
    cur.execute("select ts, o, h, l, c from candles where symbol=%s order by ts", (sym,))
    rows = cur.fetchall()
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c"])
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def ewm(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().values


def wilder(x, n=14):
    G = np.full(len(x), np.nan)
    L = np.full(len(x), np.nan)
    if len(x) <= n:
        return G, L
    d = np.diff(x, prepend=x[0])
    g, l = np.maximum(d, 0), np.maximum(-d, 0)
    ag, al = g[1:n + 1].mean(), l[1:n + 1].mean()
    G[n], L[n] = ag, al
    for i in range(n + 1, len(x)):
        ag = (ag * (n - 1) + g[i]) / n
        al = (al * (n - 1) + l[i]) / n
        G[i], L[i] = ag, al
    return G, L


def rsi_from(g, l):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 100 - 100 / (1 + g / l)
    return np.where(l == 0, 100.0, r)


def state(e20, e50, c):
    return np.where((e20 > e50) & (c > e50), 1, np.where((e20 < e50) & (c < e50), -1, 0))


def htf(df, c, tf):
    ts = df["ts"].values.astype("datetime64[s]").astype(np.int64)
    grp = ts // (tf * 60)
    change = np.r_[True, grp[1:] != grp[:-1]]
    bi = np.cumsum(change) - 1
    last_idx = np.r_[np.where(change)[0][1:] - 1, len(c) - 1]
    bc = c[last_idx]
    E20, E50 = ewm(bc, 20), ewm(bc, 50)
    G, L = wilder(bc)
    prev = bi - 1
    ok = prev >= 60
    p = np.where(ok, prev, 0)
    k20, k50 = 2 / 21, 2 / 51
    e20p = E20[p] * (1 - k20) + c * k20
    e50p = E50[p] * (1 - k50) + c * k50
    st = state(e20p, e50p, c)
    d = c - bc[p]
    g = (G[p] * 13 + np.maximum(d, 0)) / 14
    l = (L[p] * 13 + np.maximum(-d, 0)) / 14
    r = rsi_from(g, l)
    return np.where(ok, st, 0), np.where(ok, r, np.nan), ok


def features(df):
    n = len(df)
    o, h, l, c = (df[x].values for x in ("o", "h", "l", "c"))
    s5 = state(ewm(c, 20), ewm(c, 50), c)
    r5 = rsi_from(*wilder(c))
    s30, r30, ok30 = htf(df, c, 30)
    s60, r60, ok60 = htf(df, c, 60)
    valid = (np.arange(n) >= 60) & ok30 & ok60
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = pd.Series(tr).rolling(14).mean().values
    dec = df["ts"] + pd.Timedelta(minutes=5)
    dow = dec.dt.dayofweek.values
    mins = (dec.dt.hour * 60 + dec.dt.minute).values
    sess = (dow < 5) & (((mins >= 420) & (mins < 660)) | ((mins >= 780) & (mins < 960)))
    return SimpleNamespace(n=n, o=o, h=h, l=l, c=c, idx=df["ts"].values, s5=s5, r5=r5, s30=s30, r30=r30,
                           s60=s60, r60=r60, valid=valid, atr=atr, sess=sess)


# ---------------------------------------------------------------- strategies (fixees d'avance)
def sig_trend(F):
    up = (F.s5 == 1) & (F.s30 == 1) & (F.s60 == 1) & F.valid
    dn = (F.s5 == -1) & (F.s30 == -1) & (F.s60 == -1) & F.valid
    late_up = (F.r30 > 70) | (F.r60 > 70)
    late_dn = (F.r30 < 30) | (F.r60 < 30)
    d = np.zeros(F.n, dtype=int)
    d[up & (F.r5 < 70) & ~late_up] = 1
    d[dn & (F.r5 > 30) & ~late_dn] = -1
    return d


def sig_meanrev(F):
    d = np.zeros(F.n, dtype=int)
    ok = np.arange(F.n) >= 60
    d[ok & (F.r5 < 20)] = 1
    d[ok & (F.r5 > 80)] = -1
    return d


def sig_break(F, df):
    day = df["ts"].dt.floor("D")
    g = df.groupby(day).agg(hi=("h", "max"), lo=("l", "min"))
    prevhi = day.map(g["hi"].shift(1)).values
    prevlo = day.map(g["lo"].shift(1)).values
    c = F.c
    pc = np.r_[c[0], c[:-1]]
    up = (c > prevhi) & (pc <= prevhi) & F.sess
    dn = (c < prevlo) & (pc >= prevlo) & F.sess
    d = np.zeros(F.n, dtype=int)
    dayv = day.values
    for direction, mask in ((1, up), (-1, dn)):
        idx = np.where(mask)[0]
        if len(idx):
            s = pd.Series(idx, index=dayv[idx])
            d[s[~s.index.duplicated(keep="first")].values] = direction  # une seule fois par jour
    return d


def fwd(x, w, fn):
    out = np.full(len(x), np.nan)
    if len(x) > w:
        out[:len(x) - w] = fn(swv(x[1:], w), axis=1)
    return out


def sig_ob(F):
    """Order Block simplifie, sans regarder le futur : la zone n'est connue qu'apres l'impulsion,
    le signal arrive au premier retour dans la zone avec rejet, dans le sens de la tendance H1."""
    IMP, EXP = 2.0, 48
    o, h, l, c, atr, n = F.o, F.h, F.l, F.c, F.atr, F.n
    nxt_max, nxt_min = fwd(h, 3, np.max), fwd(l, 3, np.min)
    prev_max = pd.Series(h).rolling(13).max().values
    prev_min = pd.Series(l).rolling(13).min().values
    c3 = np.r_[c[3:], np.full(3, np.nan)]
    ok = (np.arange(n) >= 60) & ~np.isnan(atr)
    body = np.abs(o - c)
    sized = (body >= 0.2 * atr) & (body <= 2.5 * atr)
    with np.errstate(invalid="ignore"):
        bull = ok & sized & (c < o) & ((nxt_max - c) >= IMP * atr) & (nxt_max > prev_max) & (c3 > c)
        bear = ok & sized & (c > o) & ((c - nxt_min) >= IMP * atr) & (nxt_min < prev_min) & (c3 < c)
    d = np.zeros(n, dtype=int)
    sl = np.full(n, np.nan)
    for k in np.where(bull)[0]:                     # OB haussier : zone = corps de la bougie baissiere
        a, b = k + 4, min(k + 4 + EXP, n)
        if a >= b:
            continue
        ztop, zbot = o[k], c[k]
        touch = np.where(l[a:b] <= ztop)[0]
        if len(touch) == 0:
            continue
        m = a + touch[0]                            # premier retour seulement
        if c[m] < ztop or F.s60[m] != 1 or np.isnan(atr[m]):
            continue
        d[m] = 1
        sl[m] = c[m] - zbot + 0.5 * atr[m]
    for k in np.where(bear)[0]:                     # OB baissier : zone = corps de la bougie haussiere
        a, b = k + 4, min(k + 4 + EXP, n)
        if a >= b:
            continue
        zbot, ztop = o[k], c[k]
        touch = np.where(h[a:b] >= zbot)[0]
        if len(touch) == 0:
            continue
        m = a + touch[0]
        if c[m] > zbot or F.s60[m] != -1 or np.isnan(atr[m]):
            continue
        d[m] = -1
        sl[m] = ztop - c[m] + 0.5 * atr[m]
    return d, sl


# ---------------------------------------------------------------- simulateur
def run_trades(F, d, sl_base, rr, hold, spr, flip=False):
    o, h, l, c, idx, n = F.o, F.h, F.l, F.c, F.idx, F.n
    step = np.timedelta64(5 * hold, "m")
    out = []
    free = 0
    for i in np.where(d != 0)[0]:
        if i < free or i + hold >= n or not F.sess[i]:
            continue
        sb = sl_base[i]
        if np.isnan(sb) or sb <= 0 or idx[i + hold] - idx[i] != step:
            continue
        dr = -d[i] if flip else d[i]
        fill = o[i + 1] + dr * spr / 2
        sld = sb + spr
        sl, tp = fill - dr * sld, fill + dr * rr * sld
        res = None
        for j in range(i + 1, i + 1 + hold):
            if dr == 1:
                hit_sl, hit_tp = (l[j] - spr / 2) <= sl, (h[j] - spr / 2) >= tp
            else:
                hit_sl, hit_tp = (h[j] + spr / 2) >= sl, (l[j] + spr / 2) <= tp
            if hit_sl:
                res = (-1.0, j)
                break
            if hit_tp:
                res = (rr, j)
                break
        if res is None:
            ex = c[i + hold] - dr * spr / 2
            res = (dr * (ex - fill) / sld, i + hold)
        out.append((idx[i], res[0]))
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
    out = {"n": len(R)}
    out["train"] = pstats(R[yr <= 2022])
    out["valid"] = pstats(R[(yr >= 2023) & (yr <= 2024)])
    out["final"] = pstats(R[yr >= 2025])
    years = [(y, R[yr == y].mean()) for y in np.unique(yr) if (yr == y).sum() >= 30]
    out["pos"] = sum(1 for _, m in years if m > 0)
    out["ny"] = len(years)
    out["worst"] = min([m for _, m in years]) if years else float("nan")
    return out


def fmt(v):
    return "%+.2f" % v if v == v else "  n/a"


STRATS = [
    ("TENDANCE_1H", 12, 2.0, 2.0),    # (nom, bougies M5 de maintien, K x ATR, RR)
    ("TENDANCE_4H", 48, 3.0, 2.0),
    ("RETOUR_MOYENNE", 12, 2.0, 1.0),
    ("CASSURE_VEILLE", 36, 2.0, 2.0),
    ("ORDER_BLOCK", 48, None, 2.0),
]

conn = psycopg2.connect(DB)
lines, control, cands = [], [], []
for sym in ("XAU/USD", "BTC/USD", "GBP/JPY"):
    df = load(conn, sym)
    if df is None or len(df) < 1000:
        lines.append("%s : pas assez de bougies." % NAMES[sym])
        continue
    F = features(df)
    spr = SPREADS[sym] * SPREAD_FACTOR
    name = NAMES[sym]
    print("Calcul de", name, flush=True)

    # controle : regles de depart (10 min) avec spread 0 -> doit etre proche de 0
    dt = sig_trend(F)
    base = 1.5 * F.atr
    a = [r for _, r in run_trades(F, dt, base, 1.5, 2, 0.0)]
    b = [r for _, r in run_trades(F, dt, base, 1.5, 2, 0.0, flip=True)]
    control.append("%s : spread 0 -> signal %s R, inverse %s R" % (
        name, fmt(np.mean(a) if a else float("nan")), fmt(np.mean(b) if b else float("nan"))))

    sigs = {"TENDANCE_1H": (dt, 2.0 * F.atr), "TENDANCE_4H": (dt, 3.0 * F.atr),
            "RETOUR_MOYENNE": (sig_meanrev(F), 2.0 * F.atr), "CASSURE_VEILLE": (sig_break(F, df), 2.0 * F.atr)}
    sigs["ORDER_BLOCK"] = sig_ob(F)
    for sname, hold, k, rr in STRATS:
        d, slb = sigs[sname]
        real = summarize(run_trades(F, d, slb, rr, hold, spr))
        inv = summarize(run_trades(F, d, slb, rr, hold, spr, flip=True))
        if real is None:
            lines.append("%-15s %-7s aucun trade" % (sname, name))
            continue
        txt = "%-15s %-7s n=%5d | REGLAGE %s | VALID %s PF %.2f | ans>0 %d/%d | pire %s | INV VALID %s" % (
            sname, name, real["n"], fmt(real["train"][1]), fmt(real["valid"][1]),
            real["valid"][2] if real["valid"][2] == real["valid"][2] else 0.0,
            real["pos"], real["ny"], fmt(real["worst"]), fmt(inv["valid"][1]) if inv else "n/a")
        if SHOW_FINAL:
            txt += " | FINAL %s" % fmt(real["final"][1])
        lines.append(txt)
        tr, va = real["train"][1], real["valid"][1]
        if tr == tr and va == va and tr > 0 and va > 0 and real["valid"][0] >= 300:
            cands.append((va, sname, name, tr, real["pos"], real["ny"]))

head = ("TEST DES STRATEGIES (R moyen apres spread, 1R = risque du trade)\n"
        "Periodes : REGLAGE 2020-2022, VALID 2023-2024%s\n\n" % (", FINAL 2025-2026 (affiche)" if SHOW_FINAL else
                                                                  " (2025-2026 cache : sert au test final)"))
if SPREAD_FACTOR == 0:
    head = "*** TEST SANS FRAIS (spread = 0) : mesure l'avantage brut du signal ***\n" + head
text = head + "CONTROLE DU SIMULATEUR (doit etre proche de 0) :\n" + "\n".join(control) + "\n\n" + "\n".join(lines) + "\n\n"
if cands:
    cands.sort(reverse=True)
    text += "CANDIDATES (positives sur REGLAGE et VALID) :\n"
    for va, s, a, tr, p, ny in cands[:5]:
        text += "- %s sur %s : REGLAGE %s, VALID %s, annees positives %d/%d\n" % (s, a, fmt(tr), fmt(va), p, ny)
else:
    text += "AUCUNE strategie positive a la fois sur REGLAGE et VALID.\n"
text += ("\nComment lire : R moyen positif = gagnant apres spread. INV = meme moments, direction opposee : "
         "si l'inverse fait aussi bien, la strategie n'a pas de valeur. ans>0 = annees gagnantes sur le total. "
         "Les trades qui traversent une pause de marche sont ignores. Resultats theoriques, sans glissement.")
print(text)
sf = os.environ.get("GITHUB_STEP_SUMMARY")
if sf:
    with open(sf, "a", encoding="utf-8") as f:
        f.write("```\n" + text + "\n```\n")
telegram(text)
    
