# -*- coding: utf-8 -*-
"""Backtest des regles du tableau (entree M5, confirmation M30 et H1) sur les bougies stockees dans Neon.
Resultats en R : 1R = le risque prevu sur le trade (distance du stop loss). Spread inclus."""
import os
import sys

import numpy as np
import pandas as pd
import psycopg2
import requests

DB = os.environ.get("DATABASE_URL", "")
if not DB:
    sys.exit("ERREUR : DATABASE_URL manquant (Secrets GitHub).")


def envf(name, default):
    return float(os.environ.get(name, default))


K = envf("ATR_K", 1.5)            # stop loss = K x ATR(M5) + spread
RR = envf("RR", 1.5)              # take profit = RR x stop loss
HOLD = int(envf("MAX_HOLD", 2))   # sortie apres 2 bougies M5 (10 min)
SLIP = envf("SLIP", 0.0)          # glissement en prix, par trade
TRAIN_FRAC = 0.6                  # 60 % du temps pour regler, 40 % pour valider
SPREADS = {"XAU/USD": envf("SPREAD_GOLD", 0.55), "BTC/USD": envf("SPREAD_BTC", 30),
           "GBP/JPY": envf("SPREAD_GBPJPY", 0.03)}
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
    """Etat et RSI d'une periode plus longue, vus a chaque cloture M5 (bougie en cours incluse, sans regarder le futur)."""
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


def build_signals(df):
    n = len(df)
    c, h, l = df["c"].values, df["h"].values, df["l"].values
    s5 = state(ewm(c, 20), ewm(c, 50), c)
    r5 = rsi_from(*wilder(c))
    s30, r30, ok30 = htf(df, c, 30)
    s60, r60, ok60 = htf(df, c, 60)
    valid = (np.arange(n) >= 60) & ok30 & ok60
    up = (s5 == 1) & (s30 == 1) & (s60 == 1) & valid
    dn = (s5 == -1) & (s30 == -1) & (s60 == -1) & valid
    late_up = (r30 > 70) | (r60 > 70)
    late_dn = (r30 < 30) | (r60 < 30)
    d = np.zeros(n, dtype=int)
    d[up & (r5 < 70) & ~late_up] = 1
    d[dn & (r5 > 30) & ~late_dn] = -1
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = pd.Series(tr).rolling(14).mean().values
    dec = df["ts"] + pd.Timedelta(minutes=5)  # moment de la decision = cloture de la bougie
    dow = dec.dt.dayofweek.values
    mins = (dec.dt.hour * 60 + dec.dt.minute).values
    sess = (dow < 5) & (((mins >= 420) & (mins < 660)) | ((mins >= 780) & (mins < 960)))
    return d, atr, sess


def run_trades(df, d, atr, sess, spr, sess_only, flip=False):
    o, h, l, c = (df[x].values for x in ("o", "h", "l", "c"))
    idx = df["ts"].values
    n = len(df)
    step = np.timedelta64(5 * HOLD, "m")
    out = []
    free = 0
    for i in np.where(d != 0)[0]:
        if i < free or i + HOLD >= n:
            continue
        if sess_only and not sess[i]:
            continue
        if np.isnan(atr[i]) or idx[i + HOLD] - idx[i] != step:
            continue
        dr = -d[i] if flip else d[i]
        fill = o[i + 1] + dr * spr / 2          # achat au Ask, vente au Bid
        sld = K * atr[i] + spr
        tpd = RR * sld
        sl, tp = fill - dr * sld, fill + dr * tpd
        res = None
        for j in range(i + 1, i + 1 + HOLD):
            if dr == 1:
                hit_sl, hit_tp = (l[j] - spr / 2) <= sl, (h[j] - spr / 2) >= tp
            else:
                hit_sl, hit_tp = (h[j] + spr / 2) >= sl, (l[j] + spr / 2) <= tp
            if hit_sl:                           # si SL et TP dans la meme bougie : SL d'abord
                res = (-1.0 - SLIP / sld, j)
                break
            if hit_tp:
                res = (RR - SLIP / sld, j)
                break
        if res is None:
            ex = c[i + HOLD] - dr * spr / 2
            res = ((dr * (ex - fill) - SLIP) / sld, i + HOLD)
        out.append((idx[i], res[0]))
        free = res[1]
    return out


def metrics(rs):
    if not rs:
        return None
    r = np.array(rs)
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    pf = gp / gl if gl > 0 else float("inf")
    cum = np.cumsum(r)
    dd = (np.maximum.accumulate(cum) - cum).max()
    return len(r), (r > 0).mean() * 100, r.mean(), pf, dd


def line(label, m):
    if m is None:
        return "  %-16s aucun trade" % label
    n, w, a, pf, dd = m
    return "  %-16s %6d trades | réussite %4.1f %% | R moyen %+.3f | PF %.2f | pire série %.1fR" % (label, n, w, a, pf, dd)


def split(trades, cutoff):
    train = [r for t, r in trades if t < cutoff]
    test = [r for t, r in trades if t >= cutoff]
    return [r for _, r in trades], train, test


conn = psycopg2.connect(DB)
report = []
for sym in ("XAU/USD", "BTC/USD", "GBP/JPY"):
    df = load(conn, sym)
    if df is None or len(df) < 500:
        report.append("%s : pas assez de bougies. Lance d'abord le téléchargement." % NAMES[sym])
        continue
    spr = SPREADS[sym]
    d, atr, sess = build_signals(df)
    t0, t1 = df["ts"].iloc[0], df["ts"].iloc[-1]
    cutoff = t0 + (t1 - t0) * TRAIN_FRAC
    report.append("%s : %d bougies M5, du %s au %s (spread %.3f)" % (NAMES[sym], len(df), t0.date(), t1.date(), spr))
    report.append("Séances 08-12 et 14-17 (Kinshasa), lundi au vendredi :")
    real = split(run_trades(df, d, atr, sess, spr, True), cutoff)
    report += [line("TOTAL", metrics(real[0])), line("REGLAGE (60 %)", metrics(real[1])),
               line("VALIDATION (40 %)", metrics(real[2]))]
    inv = split(run_trades(df, d, atr, sess, spr, True, flip=True), cutoff)
    report.append(line("INVERSE total", metrics(inv[0])))
    report.append(line("INVERSE valid.", metrics(inv[2])))
    allh = split(run_trades(df, d, atr, sess, spr, False), cutoff)
    report.append("Toutes les heures :")
    report += [line("TOTAL", metrics(allh[0])), line("VALIDATION (40 %)", metrics(allh[2]))]
    report.append("")

guide = ("Comment lire : R moyen au-dessus de 0 = gagnant après spread. PF au-dessus de 1 = gagnant. "
         "Regarde surtout la VALIDATION, qui n'a pas servi à régler les règles. "
         "INVERSE = mêmes moments, direction opposée : si l'inverse fait aussi bien, le signal n'a pas de valeur. "
         "Résultats théoriques : décisions à la cloture M5, entrée à l'ouverture suivante, pas de glissement (SLIP=0), "
         "spreads supposés constants.")
text = "BACKTEST (stop loss %.1f x ATR M5 + spread, take profit %.1f x stop, sortie après %d min)\n\n" % (K, RR, 5 * HOLD)
text += "\n".join(report) + "\n" + guide
print(text)
summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
if summary_file:
    with open(summary_file, "a", encoding="utf-8") as f:
        f.write("```\n" + text + "\n```\n")
telegram(text)
    
