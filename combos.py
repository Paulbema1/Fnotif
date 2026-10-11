# -*- coding: utf-8 -*-
"""Teste 5 combinaisons 'contexte + entree + sortie' sur des trades de quelques jours.
Bougies H4 et journalieres construites depuis les bougies M5 de Neon (aucune donnee du futur :
le contexte journalier utilise toujours la veille terminee).
Garde-fous : combinaisons fixees d'avance, periodes REGLAGE / VALIDATION / FINAL cachee, test INVERSE,
controle avec directions aleatoires. Resultats en R : 1R = risque prevu (stop loss + spread)."""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import psycopg2
import requests

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


def prepare(df):
    h4, d1 = resample(df, "4h"), resample(df, "1D")
    c1 = d1["c"]
    e50, e200 = c1.ewm(span=50, adjust=False).mean(), c1.ewm(span=200, adjust=False).mean()
    trend = pd.Series(np.where(e50 > e200, 1.0, np.where(e50 < e200, -1.0, 0.0)), index=d1.index)
    trend.iloc[:200] = 0.0
    mom = c1 / c1.shift(60) - 1
    dkey = h4.index.floor("D")
    ctx = trend.shift(1).reindex(dkey).values      # tendance journaliere de la veille terminee
    mo = mom.shift(1).reindex(dkey).values         # momentum 60 jours de la veille terminee
    o, h, l, c = (h4[x].values for x in ("o", "h", "l", "c"))
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return SimpleNamespace(
        n=len(c), o=o, h=h, l=l, c=c, idx=h4.index.values, ctx=np.nan_to_num(ctx), mo=mo,
        atr=pd.Series(tr).rolling(14).mean().values, atr100=pd.Series(tr).rolling(100).mean().values,
        ema20=pd.Series(c).ewm(span=20, adjust=False).mean().values,
        hour=h4.index.hour.values, dow=h4.index.dayofweek.values)


# ---------------------------------------------------------------- combinaisons (fixees d'avance)
def s_repli(H, need_mom=False):
    """Tendance journaliere + repli sur la EMA 20 (H4) puis rebond."""
    up = (H.ctx == 1) & (H.l <= H.ema20) & (H.c > H.ema20)
    dn = (H.ctx == -1) & (H.h >= H.ema20) & (H.c < H.ema20)
    if need_mom:
        up &= (H.mo > 0)
        dn &= (H.mo < 0)
    d = np.zeros(H.n, dtype=int)
    d[up], d[dn] = 1, -1
    return d


def s_mom(H):
    """Momentum 60 jours + tendance journaliere, decision chaque lundi."""
    first = (H.dow == 0) & (H.hour == 0)
    d = np.zeros(H.n, dtype=int)
    d[first & (H.mo > 0) & (H.ctx == 1)] = 1
    d[first & (H.mo < 0) & (H.ctx == -1)] = -1
    return d


def s_break(H, calm=False, trend=False):
    """Cassure du plus haut / plus bas des 20 bougies H4, avec filtre calme et/ou tendance."""
    pm = pd.Series(H.h).rolling(20).max().shift(1).values
    pn = pd.Series(H.l).rolling(20).min().shift(1).values
    up, dn = H.c > pm, H.c < pn
    if calm:
        ratio = H.atr / H.atr100
        cp = np.r_[np.nan, ratio[:-1]] < 0.8       # volatilite calme juste avant la cassure
        up &= cp
        dn &= cp
    if trend:
        up &= (H.ctx == 1)
        dn &= (H.ctx == -1)
    d = np.zeros(H.n, dtype=int)
    d[up], d[dn] = 1, -1
    return d


# (nom, fonction, K x ATR pour le stop, RR, bougies H4 maximum de maintien)
STRATS = [
    ("REPLI_TENDANCE", lambda H: s_repli(H), 2.0, 2.0, 30),
    ("MOMENTUM_HEBDO", lambda H: s_mom(H), 4.0, 50.0, 30),     # RR 50 = pas de take profit
    ("CASSURE_CALME", lambda H: s_break(H, calm=True), 2.0, 2.0, 30),
    ("CASSURE_TENDANCE", lambda H: s_break(H, trend=True), 2.0, 2.0, 30),
    ("REPLI_MOMENTUM", lambda H: s_repli(H, need_mom=True), 2.0, 2.0, 30),
]


# ---------------------------------------------------------------- simulateur (ecarts de prix inclus)
def run_trades(H, d, slb, rr, hold, spr, flip=False):
    o, h, l, c, n = H.o, H.h, H.l, H.c, H.n
    out = []
    free = 0
    for i in np.where(d != 0)[0]:
        if i < free or i + hold >= n:
            continue
        sb = slb[i]
        if np.isnan(sb) or sb <= 0:
            continue
        dr = -d[i] if flip else d[i]
        fill = o[i + 1] + dr * spr / 2
        sld = sb + spr
        sl, tp = fill - dr * sld, fill + dr * rr * sld
        res = None
        for j in range(i + 1, i + 1 + hold):
            if dr == 1:
                bo, blo, bhi = o[j] - spr / 2, l[j] - spr / 2, h[j] - spr / 2
                if j > i + 1 and (bo <= sl or bo >= tp):       # ecart (gap) a l'ouverture
                    res = ((bo - fill) / sld, j)
                    break
                if blo <= sl:
                    res = (-1.0, j)
                    break
                if bhi >= tp:
                    res = (rr, j)
                    break
            else:
                ao, ahi, alo = o[j] + spr / 2, h[j] + spr / 2, l[j] + spr / 2
                if j > i + 1 and (ao >= sl or ao <= tp):
                    res = ((fill - ao) / sld, j)
                    break
                if ahi >= sl:
                    res = (-1.0, j)
                    break
                if alo <= tp:
                    res = (rr, j)
                    break
        if res is None:
            ex = c[i + hold] - dr * spr / 2
            res = (dr * (ex - fill) / sld, i + hold)
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
    out = {"n": len(R), "train": pstats(R[yr <= 2022]), "valid": pstats(R[(yr >= 2023) & (yr <= 2024)]),
           "final": pstats(R[yr >= 2025])}
    years = [(y, R[yr == y].mean()) for y in np.unique(yr) if (yr == y).sum() >= 10]
    out["pos"], out["ny"] = sum(1 for _, m in years if m > 0), len(years)
    out["worst"] = min([m for _, m in years]) if years else float("nan")
    return out


def fmt(v):
    return "%+.2f" % v if v == v else "  n/a"


conn = psycopg2.connect(DB)
lines, control, cands = [], [], []
for sym in ("XAU/USD", "BTC/USD", "GBP/JPY"):
    df = load(conn, sym)
    if df is None or len(df) < 5000:
        lines.append("%s : pas assez de bougies." % NAMES[sym])
        continue
    H = prepare(df)
    spr = SPREADS[sym] * SPREAD_FACTOR
    name = NAMES[sym]
    print("Calcul de", name, "-", H.n, "bougies H4", flush=True)

    # controle : memes moments, direction au hasard, sans frais -> doit etre proche de 0
    d0 = s_repli(H)
    rd = np.where(d0 != 0, np.random.default_rng(1).choice([-1, 1], size=H.n), 0)
    ctrl = [r for _, r in run_trades(H, rd, 2.0 * H.atr, 2.0, 30, 0.0)]
    control.append("%s : direction au hasard, spread 0 -> %s R (%d trades)" % (
        name, fmt(np.mean(ctrl) if ctrl else float("nan")), len(ctrl)))

    for sname, fn, k, rr, hold in STRATS:
        d, slb = fn(H), k * H.atr
        real = summarize(run_trades(H, d, slb, rr, hold, spr))
        inv = summarize(run_trades(H, d, slb, rr, hold, spr, flip=True))
        if real is None:
            lines.append("%-17s %-7s aucun trade" % (sname, name))
            continue
        pf = real["valid"][2]
        txt = "%-17s %-7s n=%4d | REGLAGE %s | VALID %s PF %.2f | ans>0 %d/%d | pire %s | INV VALID %s" % (
            sname, name, real["n"], fmt(real["train"][1]), fmt(real["valid"][1]), pf if pf == pf else 0.0,
            real["pos"], real["ny"], fmt(real["worst"]), fmt(inv["valid"][1]) if inv else "n/a")
        if SHOW_FINAL:
            txt += " | FINAL %s" % fmt(real["final"][1])
        lines.append(txt)
        tr, va = real["train"][1], real["valid"][1]
        if tr == tr and va == va and tr > 0 and va > 0 and real["valid"][0] >= 30:
            cands.append((va, sname, name, tr, real["pos"], real["ny"], real["valid"][0]))

head = ("TEST DE COMBINAISONS (trades de quelques jours, bougies H4, R moyen apres spread)\n"
        "Periodes : REGLAGE 2020-2022, VALID 2023-2024%s\n\n" % (", FINAL 2025-2026 (affiche)" if SHOW_FINAL else
                                                                  " (2025-2026 cache : test final)"))
if SPREAD_FACTOR == 0:
    head = "*** TEST SANS FRAIS (spread = 0) ***\n" + head
text = head + "CONTROLE (doit etre proche de 0) :\n" + "\n".join(control) + "\n\n" + "\n".join(lines) + "\n\n"
if cands:
    cands.sort(reverse=True)
    text += "CANDIDATES (positives sur REGLAGE et VALID, au moins 30 trades en VALID) :\n"
    for va, s, a, tr, p, ny, nv in cands[:6]:
        text += "- %s sur %s : REGLAGE %s, VALID %s (%d trades), annees positives %d/%d\n" % (s, a, fmt(tr), fmt(va), nv, p, ny)
else:
    text += "AUCUNE combinaison positive a la fois sur REGLAGE et VALID.\n"
text += ("\nComment lire : peu de trades par combinaison, donc les chiffres sont moins fiables que pour les trades courts. "
         "INV = memes moments, direction opposee. ans>0 = annees gagnantes. "
         "Les frais de nuit (swap) ne sont PAS comptes : ils peuvent couter ou rapporter selon la direction. "
         "Ecarts d'ouverture (gaps) inclus. Resultats theoriques.")
print(text)
sf = os.environ.get("GITHUB_STEP_SUMMARY")
if sf:
    with open(sf, "a", encoding="utf-8") as f:
        f.write("```\n" + text + "\n```\n")
telegram(text)
  
