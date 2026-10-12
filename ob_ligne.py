# -*- coding: utf-8 -*-
"""Traduction de la strategie TradingView 'OB + Ligne Tendance' (Supertrend lisse + Order Blocks, sortie 2R)
testee sur les bougies de Neon (M5 regroupees en 1h et 4h), avec spread, periodes et test inverse.
Parametres du script d'origine, SANS aucun reglage : Supertrend 3.0 / 10, WMA 20, EMA 10, impulsion 1.5 ATR,
stop sur l'order block, objectif 2R, un seul trade a la fois. Resultats en R (1R = distance du stop)."""
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


# ---------------------------------------------------------------- indicateurs (comme Pine Script)
def rma(x, n):
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    out[n - 1] = np.mean(x[:n])
    a = 1.0 / n
    for i in range(n, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def wma(x, n):
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    w = np.arange(1, n + 1, dtype=float)
    w /= w.sum()
    out[n - 1:] = swv(x, n) @ w
    return out


def supertrend(h, l, c, tr, factor, period):
    n = len(c)
    atr = rma(tr, period)
    src = (h + l) / 2
    up_raw, lo_raw = src + factor * atr, src - factor * atr
    st = np.full(n, np.nan)
    fub = np.full(n, np.nan)
    flb = np.full(n, np.nan)
    for i in range(1, n):
        if np.isnan(atr[i]):
            continue
        prev_l = flb[i - 1] if not np.isnan(flb[i - 1]) else 0.0
        prev_u = fub[i - 1] if not np.isnan(fub[i - 1]) else 0.0
        lb = lo_raw[i] if (lo_raw[i] > prev_l or c[i - 1] < prev_l) else prev_l
        ub = up_raw[i] if (up_raw[i] < prev_u or c[i - 1] > prev_u) else prev_u
        flb[i], fub[i] = lb, ub
        if np.isnan(atr[i - 1]):
            d = 1
        elif st[i - 1] == prev_u:
            d = -1 if c[i] > ub else 1
        else:
            d = 1 if c[i] < lb else -1
        st[i] = lb if d == -1 else ub
    return st


def prepare(df, rule):
    b = resample(df, rule)
    o, h, l, c = (b[x].values for x in ("o", "h", "l", "c"))
    n = len(c)
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    tr[0] = h[0] - l[0]
    atr = rma(tr, 14)
    st = supertrend(h, l, c, tr, 3.0, 10)
    tl = pd.Series(wma(st, 20)).ewm(span=10, adjust=False).mean().values
    trend = np.zeros(n)
    t = 0
    for i in range(1, n):
        if not (np.isnan(tl[i]) or np.isnan(tl[i - 1])):
            if tl[i] > tl[i - 1]:
                t = 1
            elif tl[i] < tl[i - 1]:
                t = -1
        trend[i] = t
    pcx, pox = np.r_[np.nan, c[:-1]], np.r_[np.nan, o[:-1]]
    with np.errstate(invalid="ignore"):
        up, dn = (trend == 1) & (c > tl), (trend == -1) & (c < tl)
        bull = ((c - o) > 1.5 * atr) & (pcx < pox)
        bear = ((o - c) > 1.5 * atr) & (pcx > pox)
    return SimpleNamespace(n=n, o=o, h=h, l=l, c=c, atr=atr, up=up, dn=dn, bull=bull, bear=bear, idx=b.index.values)


# ---------------------------------------------------------------- simulation (une barre apres l'autre)
def sim(P, spr, mode="real", seed=1):
    """mode : real = strategie telle quelle ; flip = direction inversee ; rand = direction au hasard.
    En flip et rand, le stop et l'objectif reprennent les memes distances, de facon symetrique."""
    o, h, l, c, atr, n = P.o, P.h, P.l, P.c, P.atr, P.n
    rng = np.random.default_rng(seed)
    bull_top = bull_bot = bear_top = bear_bot = np.nan
    pos, pending, out = None, None, []
    for j in range(1, n):
        if pending is not None:                                  # entree a l'ouverture de cette bougie
            dr, sl_abs, tp_abs, sd = pending
            pending = None
            fill = o[j] + dr * spr / 2
            if mode == "real":
                sl, tp = sl_abs, tp_abs
            else:
                sl, tp = fill - dr * sd, fill + dr * 2 * sd
            sld = abs(fill - sl)
            if sld > 0 and ((dr == 1 and fill > sl) or (dr == -1 and fill < sl)):
                pos = {"dr": dr, "fill": fill, "sl": sl, "tp": tp, "sld": sld, "i0": j}
        if pos is not None:                                      # sortie sur cette bougie ?
            dr, fill, sl, tp, sld = pos["dr"], pos["fill"], pos["sl"], pos["tp"], pos["sld"]
            px = None
            if dr == 1:
                bo, bl, bh = o[j] - spr / 2, l[j] - spr / 2, h[j] - spr / 2
                if j > pos["i0"] and bo <= sl:
                    px = bo
                elif j > pos["i0"] and bo >= tp:
                    px = bo
                elif bl <= sl:
                    px = sl
                elif bh >= tp:
                    px = tp
            else:
                ao, ah, al = o[j] + spr / 2, h[j] + spr / 2, l[j] + spr / 2
                if j > pos["i0"] and ao >= sl:
                    px = ao
                elif j > pos["i0"] and ao <= tp:
                    px = ao
                elif ah >= sl:
                    px = sl
                elif al <= tp:
                    px = tp
            if px is not None:
                out.append((P.idx[pos["i0"]], dr * (px - fill) / sld, pos["i0"]))
                pos = None
        # mise a jour des order blocks puis signaux, a la cloture de la bougie j
        if P.bull[j]:
            bull_top, bull_bot = h[j - 1], l[j - 1]
        if P.bear[j]:
            bear_top, bear_bot = h[j - 1], l[j - 1]
        flat = pos is None and pending is None
        long_c = (P.up[j] and flat and l[j] <= bull_top and c[j] > bull_top and c[j] > o[j]
                  and (c[j] - bull_bot) > 0.3 * atr[j])
        short_c = (P.dn[j] and flat and h[j] >= bear_bot and c[j] < bear_bot and c[j] < o[j]
                   and (bear_top - c[j]) > 0.3 * atr[j])
        if long_c:
            sd = c[j] - bull_bot
            d0 = 1
            sl_abs, tp_abs = bull_bot, c[j] + 2 * sd
        elif short_c:
            sd = bear_top - c[j]
            d0 = -1
            sl_abs, tp_abs = bear_top, c[j] - 2 * sd
        if long_c or short_c:
            dr = d0 if mode == "real" else (-d0 if mode == "flip" else int(rng.choice([-1, 1])))
            pending = (dr, sl_abs, tp_abs, sd)
        used_bull = long_c or (c[j] < bull_bot)
        used_bear = short_c or (c[j] > bear_top)
        if used_bull:
            bull_top = bull_bot = np.nan
        if used_bear:
            bear_top = bear_bot = np.nan
    return out


# ---------------------------------------------------------------- resultats
def pstats(r):
    if len(r) == 0:
        return 0, float("nan"), float("nan")
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    return len(r), r.mean(), (gp / gl if gl > 0 else float("inf"))


def summarize(trades):
    if not trades:
        return None
    ts = np.array([t for t, _, _ in trades])
    R = np.array([r for _, r, _ in trades])
    yr = ts.astype("datetime64[Y]").astype(int) + 1970
    if not SHOW_FINAL:
        keep = yr <= 2024
        R, yr = R[keep], yr[keep]
    if len(R) == 0:
        return None
    out = {"n": len(R), "train": pstats(R[yr <= 2022]), "valid": pstats(R[(yr >= 2023) & (yr <= 2024)]),
           "final": pstats(R[yr >= 2025])}
    years = [(y, R[yr == y].mean()) for y in np.unique(yr) if (yr == y).sum() >= 8]
    out["pos"], out["ny"] = sum(1 for _, m in years if m > 0), len(years)
    out["worst"] = min([m for _, m in years]) if years else float("nan")
    return out


def fmt(v):
    return "%+.2f" % v if v == v else "  n/a"


conn = psycopg2.connect(DB)
lines, control, cands, fidel = [], [], [], []
for sym in ("BTC/USD", "XAU/USD", "GBP/JPY"):
    df = load(conn, sym)
    if df is None or len(df) < 5000:
        lines.append("%s : pas assez de bougies." % NAMES[sym])
        continue
    name = NAMES[sym]
    spr_base = SPREADS[sym] * SPREAD_FACTOR
    for rule, label in (("1h", "1H"), ("4h", "4H")):
        P = prepare(df, rule)
        # le spread de reference est le meme que pour les autres tests (en prix) ; il pese moins sur 1H et 4H
        real = sim(P, spr_base, "real")
        inv = sim(P, spr_base, "flip")
        s_real, s_inv = summarize(real), summarize(inv)
        print("Calcul", name, label, "-", P.n, "bougies,", len(real), "trades", flush=True)
        if label == "1H":
            rnd = [r for _, r, _ in sim(P, 0.0, "rand")]
            control.append("%s : direction au hasard, spread 0 -> %s R (%d trades)" % (
                name, fmt(np.mean(rnd) if rnd else float("nan")), len(rnd)))
            if sym == "BTC/USD":                       # fidelite avec TradingView (5000 dernieres bougies 1H)
                last = [r for _, r, i0 in sim(P, 0.0, "real") if i0 >= P.n - 5000]
                fidel.append("BTCUSD 1H, 5000 dernieres bougies, sans frais : %d trades, %d gagnants "
                             "(TradingView : 46 trades, 22 gagnants, autres prix et autres frais)"
                             % (len(last), sum(1 for r in last if r > 0)))
        if s_real is None:
            lines.append("%-7s %s aucun trade" % (name, label))
            continue
        pf = s_real["valid"][2]
        txt = "%-7s %s n=%4d | REGLAGE %s | VALID %s PF %.2f | ans>0 %d/%d | pire %s | INV VALID %s" % (
            name, label, s_real["n"], fmt(s_real["train"][1]), fmt(s_real["valid"][1]), pf if pf == pf else 0.0,
            s_real["pos"], s_real["ny"], fmt(s_real["worst"]), fmt(s_inv["valid"][1]) if s_inv else "n/a")
        if SHOW_FINAL:
            txt += " | FINAL %s" % fmt(s_real["final"][1])
        lines.append(txt)
        tr, va = s_real["train"][1], s_real["valid"][1]
        if tr == tr and va == va and tr > 0 and va > 0 and s_real["valid"][0] >= 30:
            cands.append((va, name, label, tr, s_real["pos"], s_real["ny"], s_real["valid"][0]))

head = ("TEST 'OB + LIGNE TENDANCE' (traduction TradingView, bougies 1h et 4h, R moyen apres spread)\n"
        "Periodes : REGLAGE 2020-2022, VALID 2023-2024%s\n\n" % (", FINAL 2025-2026 (affiche)" if SHOW_FINAL else
                                                                  " (2025-2026 cache : test final)"))
if SPREAD_FACTOR == 0:
    head = "*** TEST SANS FRAIS (spread = 0) ***\n" + head
text = head + "CONTROLE (doit etre proche de 0) :\n" + "\n".join(control) + "\n\n"
text += "FIDELITE :\n" + "\n".join(fidel) + "\n\n" + "\n".join(lines) + "\n\n"
if cands:
    cands.sort(reverse=True)
    text += "CANDIDATES (positives sur REGLAGE et VALID, au moins 30 trades en VALID) :\n"
    for va, a, lab, tr, p, ny, nv in cands[:6]:
        text += "- %s %s : REGLAGE %s, VALID %s (%d trades), annees positives %d/%d\n" % (a, lab, fmt(tr), fmt(va), nv, p, ny)
else:
    text += "AUCUNE combinaison positive a la fois sur REGLAGE et VALID.\n"
text += ("\nComment lire : peu de trades par ligne, donc chiffres peu fiables. INV = strategie inversee. "
         "Fidelite : si les nombres de trades sont tres differents de TradingView, la copie n'est pas exacte. "
         "Spread en prix, identique aux autres tests, sans glissement ni frais de nuit. Resultats theoriques.")
print(text)
sf = os.environ.get("GITHUB_STEP_SUMMARY")
if sf:
    with open(sf, "a", encoding="utf-8") as f:
        f.write("```\n" + text + "\n```\n")
telegram(text)
  
