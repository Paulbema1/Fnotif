# -*- coding: utf-8 -*-
"""LOT 1 : regles d'entree de 4 EA du depot 'metatrader5' (geraked), SANS grille, SANS martingale.
Meme moteur, memes frais, meme critere dur et meme periode finale cachee que fiches.py (il doit etre dans le depot).
Chaque EA est teste sur l'unite de temps indiquee dans sa description (2MOMEMA : 1h, choisi avant de voir les resultats).
Stop et objectif : ceux de l'EA (stop sur Bollinger, sur les plus bas/hauts recents ou sur les moyennes mobiles).
Les points du depot sont remplaces par des distances relatives a la volatilite (ATR) : 'SLDev' = 0,2 ATR."""
import os
import sys

import numpy as np
import pandas as pd
import psycopg2
from types import SimpleNamespace

from fiches import (DB, MIN_PF, MIN_R, MIN_VALID_TRADES, MIN_YEARS, NAMES, SPREAD_FACTOR, SPREADS, SHOW_FINAL, ema,
                    fmt, lag, load, resample, rma, rsi, run_trades, sg, sma, summarize, telegram)

TESTS_BEFORE = 60      # les 12 fiches deja testees


def prepare(df, rule):
    b = resample(df, rule)
    o, h, l, c = (b[x].values for x in ("o", "h", "l", "c"))
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    tr[0] = h[0] - l[0]
    return SimpleNamespace(n=len(c), o=o, h=h, l=l, c=c, tr=tr, atr=rma(tr, 14), idx=b.index.values, ts=b.index,
                           mins=int(pd.Timedelta(rule).total_seconds() / 60), m5=df)


def roll(x, n, fn):
    s = pd.Series(x).rolling(n)
    return getattr(s, fn)().values


def mk(buy, sell):
    d = np.zeros(len(buy), dtype=int)
    d[np.nan_to_num(buy).astype(bool)] = 1
    d[np.nan_to_num(sell).astype(bool)] = -1
    return d


def pos(slb):
    """Distance de stop invalide (negative) -> pas de trade."""
    return np.where(slb > 0, slb, np.nan)


# ---------------------------------------------------------------- les 4 EA (decision a la cloture de la bougie i = 'shift 1')
def ea_bbrsi(H):
    """Bollinger(500, 2) + RSI(7), retour dans les bandes. Stop : bande - 0,9 x (milieu - bande). Objectif : 1 x le stop."""
    c = H.c
    mid = sma(c, 500)
    sd = pd.Series(c).rolling(500).std(ddof=0).values
    up, lo = mid + 2 * sd, mid - 2 * sd
    r = rsi(c, 7)
    with np.errstate(invalid="ignore"):
        buy = (lag(r, 1) < 30) & (lag(c, 1) < lag(lo, 1)) & (r > 30) & (c > lo) & (r < 50) & (c < mid)
        sell = (lag(r, 1) > 70) & (lag(c, 1) > lag(up, 1)) & (r < 70) & (c < up) & (r > 50) & (c > mid)
    d = mk(buy, sell)
    slb = np.where(d == 1, c - (lo - 0.9 * (mid - lo)), (up + 0.9 * (up - mid)) - c)
    return sg(d, pos(slb), hold=576, rr=1.0)


def ea_2macdsto(H):
    """Deux MACD (13/21 et 34/144) + Stochastique (7, 3, 3) sur les plus bas/hauts. Stop : extreme des 10 dernieres bougies."""
    c, h, l = H.c, H.h, H.l
    m1, m2 = ema(c, 13) - ema(c, 21), ema(c, 34) - ema(c, 144)
    ll, hh = roll(l, 7, "min"), roll(h, 7, "max")
    num, den = roll(c - ll, 3, "sum"), roll(hh - ll, 3, "sum")
    with np.errstate(invalid="ignore", divide="ignore"):
        k = 100 * num / den
    dd = sma(k, 3)
    with np.errstate(invalid="ignore"):
        buy = (lag(m2, 1) > 0) & (lag(m1, 1) < 0) & (lag(k, 1) < 20) & (lag(k, 1) <= lag(dd, 1)) & (k > dd)
        sell = (lag(m2, 1) < 0) & (lag(m1, 1) > 0) & (lag(k, 1) > 80) & (lag(k, 1) >= lag(dd, 1)) & (k < dd)
    d = mk(buy, sell)
    dev = 0.2 * H.atr
    slb = np.where(d == 1, c - (roll(l, 10, "min") - dev), (roll(h, 10, "max") + dev) - c)
    return sg(d, pos(slb), hold=40, rr=1.0)


def ea_3maf(H):
    """Trois EMA (60, 350, 600) alignees + retour sur la EMA 60 + fractale de Williams. Stop sur EMA 350 ou 600. Objectif 1,5 x."""
    c, h, l, o = H.c, H.h, H.l, H.o
    m1, m2, m3 = ema(c, 60), ema(c, 350), ema(c, 600)
    onext = np.r_[o[1:], np.nan]                    # ouverture de la bougie suivante = 'bougie 0' au moment de la decision
    with np.errstate(invalid="ignore"):
        low_fr = (lag(l, 1) < lag(l, 3)) & (lag(l, 1) < lag(l, 2)) & (lag(l, 1) < l) & (lag(l, 1) < onext)
        high_fr = (lag(h, 1) > lag(h, 3)) & (lag(h, 1) > lag(h, 2)) & (lag(h, 1) > h) & (lag(h, 1) > onext)
        buy = (m1 > m2) & (m2 > m3) & (l > m3) & (l < m1) & low_fr
        sell = (m3 > m2) & (m2 > m1) & (h < m3) & (h > m1) & high_fr
    d = mk(buy, sell)
    sl_b = np.where(l > m2, m2, m3)
    sl_s = np.where(h < m2, m2, m3)
    slb = np.where(d == 1, c - sl_b, sl_s - c)
    slb = np.where(slb >= 0.3 * H.atr, slb, np.nan)  # stop minimum (l'EA exige 100 points)
    return sg(d, pos(slb), hold=192, rr=1.5)


def ea_2momema(H):
    """Deux momentums (10 et 34) + deux EMA (10 et 34) : croisement du momentum rapide avec 100, tendance lente alignee 9 bougies."""
    c, h, l = H.c, H.h, H.l
    fm, sm = 100 * c / lag(c, 10), 100 * c / lag(c, 34)
    fe, se = ema(c, 10), ema(c, 34)
    with np.errstate(invalid="ignore"):
        ok_up = pd.Series(((sm > 100) & (fe > se)).astype(float)).rolling(9).min().values == 1
        ok_dn = pd.Series(((sm < 100) & (fe < se)).astype(float)).rolling(9).min().values == 1
        buy = (fm > 100) & (lag(fm, 1) <= 100) & ok_up
        sell = (fm < 100) & (lag(fm, 1) >= 100) & ok_dn
    d = mk(buy, sell)
    dev = 0.2 * H.atr
    slb = np.where(d == 1, c - (roll(l, 7, "min") - dev), (roll(h, 7, "max") + dev) - c)
    return sg(d, pos(slb), hold=120, rr=1.0)


LOT = [  # (nom, fonction, unite de temps)
    ("EA_BBRSI", ea_bbrsi, "5min"),
    ("EA_2MACDSTO", ea_2macdsto, "3h"),
    ("EA_3MAF", ea_3maf, "15min"),
    ("EA_2MOMEMA", ea_2momema, "1h"),
]


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
        for nm, fn, rule in LOT:
            H = prepare(df, rule)
            print("Calcul", nm, rule, name, "-", H.n, "bougies", flush=True)
            s = fn(H)
            rd = np.where(s.d != 0, rng.choice([-1, 1], size=H.n), 0)
            if nm == "EA_2MOMEMA":                      # controle du simulateur : sans frais, au hasard
                z = [r for _, r in run_trades(H, s, rd, 0.0)]
                control.append("%s : direction au hasard, spread 0 -> %s R (%d trades)" % (
                    name, fmt(np.mean(z) if z else float("nan")), len(z)))
            real = summarize(run_trades(H, s, s.d, spr))
            inv = summarize(run_trades(H, s, -s.d, spr))
            alea = summarize(run_trades(H, s, rd, spr))
            ntests += 1
            lab = "%-12s %-5s %-6s" % (nm, rule, name)
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

    head = ("LOT 1 : 4 EA du depot metatrader5, sans grille (R moyen apres spread, 1R = risque du trade)\n"
            "Periodes : REGLAGE 2020-2022, VALID 2023-2024%s\n" % (", FINAL 2025-2026 (affiche)" if SHOW_FINAL
                                                                  else " (2025-2026 cache : test final)"))
    if SPREAD_FACTOR == 0:
        head = "*** TEST SANS FRAIS (spread = 0) ***\n" + head
    crit = ("CRITERE DUR : REGLAGE et VALID > %+.2f R, PF VALID > %.2f, au moins %d annees positives, "
            "au moins %d trades en VALID, VALID meilleur que ALEA.\n" % (MIN_R, MIN_PF, MIN_YEARS, MIN_VALID_TRADES))
    summary = head + crit + "\nCONTROLE (doit etre proche de 0) :\n" + "\n".join(control) + "\n\n"
    total = TESTS_BEFORE + ntests
    if cands:
        cands.sort(reverse=True)
        summary += "CANDIDATES (%d tests dans ce lot, %d au total depuis le debut) :\n" % (ntests, total)
        for va, nm, rule, name, tr, p, ny, nv in cands[:8]:
            summary += "- %s %s %s : REGLAGE %s, VALID %s (%d trades), annees positives %d/%d\n" % (
                nm, rule, name, fmt(tr), fmt(va), nv, p, ny)
        summary += "Avec %d tests au total, une candidate peut sortir par hasard : test FINAL une seule fois.\n" % total
    else:
        summary += "AUCUN EA ne passe le critere dur (%d tests dans ce lot, %d au total).\n" % (ntests, total)
    guide = ("\nComment lire : INV = memes moments, direction opposee. ALEA = memes moments et memes stop/objectif, direction "
             "au hasard, avec frais : c'est le niveau a battre. Grille et martingale de l'EA d'origine NON utilisees. "
             "Trades fermes de force apres 2 a 5 jours si ni stop ni objectif. Resultats theoriques, sans swap ni glissement.")
    full = summary + "\nDETAIL :\n" + "\n".join(lines) + "\n" + guide
    print(full)
    sf = os.environ.get("GITHUB_STEP_SUMMARY")
    if sf:
        with open(sf, "a", encoding="utf-8") as f:
            f.write("```\n" + full + "\n```\n")
    telegram(summary + guide)


if __name__ == "__main__":
    main()
  
