# -*- coding: utf-8 -*-
"""LOT 2 : les 5 EA du depot 'XAUUSD_EA' (EA_0 a EA_4), sur l'OR (GOLD) seulement, base M5.
Meme moteur, memes frais, meme critere dur et meme periode finale cachee que fiches.py (il doit etre dans le depot).

REGLES FIXEES D'AVANCE (avant de voir un seul resultat) :
- Entrees : celles du code .mq5 de chaque EA (decision a la cloture de la bougie M5, entree a l'ouverture suivante).
- Tendance M15 et H1 (EMA 50 > EMA 200) : on n'utilise que des bougies M15/H1 deja FERMEES au moment de la decision.
- Stop / objectif : les 'points' du depot sont remplaces par des multiples de l'ATR(14, M5), car le prix de l'or a
  beaucoup change depuis 2020. Hypothese : 1 ATR M5 ~ 3 $ (100 points = 1 $ -> 0,33 ATR).
    EA_0 : stop 4 ATR (1200 pts), objectif 1,5 x le stop (1800 pts), sortie max 60 min
    EA_1 : stop 4 ATR (1200 pts), objectif 1,25 x le stop (1500 pts), sortie max 60 min
    EA_2 : stop 1,7 ATR (500 pts), objectif 3,6 x le stop (1800 pts), sortie max 60 min
    EA_3 : stop 4 ATR (1200 pts), objectif 1,5 x le stop (1800 pts), sortie max 60 min
    EA_4 : stop 0,33 ATR (100 pts), objectif 18 x le stop (1800 pts), sortie apres 1 bougie (5 min)
- Filtre de corps de bougie : 30 points (0,30 $) = 0,1 ATR ; 300 points (3 $) = 1 ATR.
- SIMPLIFICATIONS (le depot fait autrement) : un seul trade a la fois (le depot en autorise 2 a 10) ; pas de sortie
  anticipee 'si gain apres N minutes' (EA_0, EA_1) ; pas d'arret apres N stops par session/jour (EA_2, EA_3, EA_4) ;
  pas de filtre de spread (le spread de 0,55 est de toute facon deduit de chaque trade).
- EA_2 (acheter si la bougie monte, vendre si elle baisse) est le CONTROLE DE BRUIT : il doit etre proche de 0 sans frais.
- Les resultats du depot (7 semaines, reglages choisis sur ces memes semaines) ne sont PAS repris.
"""
import os

import numpy as np
import pandas as pd
import psycopg2
from types import SimpleNamespace

from fiches import (DB, MIN_PF, MIN_R, MIN_VALID_TRADES, MIN_YEARS, NAMES, SPREAD_FACTOR, SPREADS, SHOW_FINAL, ema,
                    fmt, lag, load, resample, rma, run_trades, sg, summarize, telegram)

TESTS_BEFORE = 72      # tests deja faits (fiches + lot 1)
SYM = "XAU/USD"


def prepare(df, rule):
    b = resample(df, rule)
    o, h, l, c = (b[x].values for x in ("o", "h", "l", "c"))
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    tr[0] = h[0] - l[0]
    return SimpleNamespace(n=len(c), o=o, h=h, l=l, c=c, tr=tr, atr=rma(tr, 14), idx=b.index.values, ts=b.index,
                           mins=int(pd.Timedelta(rule).total_seconds() / 60), m5=df)


def mk(buy, sell):
    d = np.zeros(len(buy), dtype=int)
    d[np.nan_to_num(buy).astype(bool)] = 1
    d[np.nan_to_num(sell).astype(bool)] = -1
    return d


def htf_trend(H, rule, minutes):
    """Tendance (EMA50 > EMA200 : +1, inverse : -1) de la derniere bougie M15/H1 deja FERMEE a la cloture de la
    bougie M5 : debut de la bougie haute <= debut M5 + 5 min - duree. Aucun regard vers le futur."""
    b = resample(H.m5, rule)
    c = b["c"].values
    tr = np.sign(ema(c, 50) - ema(c, 200))
    starts = b.index.values
    key = (H.ts + pd.Timedelta(minutes=5 - minutes)).values
    k = np.searchsorted(starts, key, side="right") - 1
    return np.where(k >= 0, tr[np.clip(k, 0, None)], 0.0)


def aligned(H):
    t15 = htf_trend(H, "15min", 15)
    t1h = htf_trend(H, "1h", 60)
    return np.where(t15 == t1h, t15, 0.0)


# ---------------------------------------------------------------- les 5 EA (decision a la cloture de la bougie i = 'shift 1')
def ea_0(H):
    """Tendance M15+H1 alignee, retour sur EMA20 (M5) : cloture precedente sous l'EMA20, cloture actuelle au-dessus."""
    c = H.c
    e = ema(c, 20)
    t = aligned(H)
    with np.errstate(invalid="ignore"):
        buy = (t == 1) & (lag(c, 1) < lag(e, 1)) & (c > e)
        sell = (t == -1) & (lag(c, 1) > lag(e, 1)) & (c < e)
    return sg(mk(buy, sell), 4.0 * H.atr, 12, rr=1.5)


def ea_1(H):
    """Comme EA_0 + cassure du plus haut/bas precedent + corps de bougie >= 0,1 ATR."""
    c, h, l, o = H.c, H.h, H.l, H.o
    e = ema(c, 20)
    t = aligned(H)
    body = np.abs(c - o) >= 0.1 * H.atr
    with np.errstate(invalid="ignore"):
        buy = (t == 1) & (lag(l, 1) <= lag(e, 1)) & (h > lag(h, 1)) & body
        sell = (t == -1) & (lag(h, 1) >= lag(e, 1)) & (l < lag(l, 1)) & body
    return sg(mk(buy, sell), 4.0 * H.atr, 12, rr=1.25)


def ea_2(H):
    """CONTROLE DE BRUIT : achat si la cloture monte, vente si elle baisse. Aucun filtre."""
    c = H.c
    with np.errstate(invalid="ignore"):
        buy = c > lag(c, 1)
        sell = c < lag(c, 1)
    return sg(mk(buy, sell), 1.7 * H.atr, 12, rr=3.6)


def ea_3(H):
    """Tendance M15+H1 alignee, corps de la bougie precedente >= 0,1 ATR, precedente cloturee de l'autre cote de l'EMA20
    puis la bougie actuelle touche l'EMA20."""
    c, h, l, o = H.c, H.h, H.l, H.o
    e = ema(c, 20)
    t = aligned(H)
    with np.errstate(invalid="ignore"):
        body = np.abs(lag(c, 1) - lag(o, 1)) >= 0.1 * lag(H.atr, 1)
        buy = (t == 1) & body & (lag(c, 1) < lag(e, 1)) & (h > lag(e, 1))
        sell = (t == -1) & body & (lag(c, 1) > lag(e, 1)) & (l < lag(e, 1))
    return sg(mk(buy, sell), 4.0 * H.atr, 12, rr=1.5)


def ea_4(H):
    """EMA200 (M5) + gros corps de bougie (>= 1 ATR) + continuation. Sortie apres 1 bougie (5 min)."""
    c, o = H.c, H.o
    e200 = ema(c, 200)
    body = np.abs(c - o) >= 1.0 * H.atr
    with np.errstate(invalid="ignore"):
        buy = body & (c > e200) & (c > lag(c, 1))
        sell = body & (c < e200) & (c < lag(c, 1))
    return sg(mk(buy, sell), 0.33 * H.atr, 1, rr=18.0)


LOT = [  # (nom, fonction)
    ("EA_0", ea_0),
    ("EA_1", ea_1),
    ("EA_2", ea_2),
    ("EA_3", ea_3),
    ("EA_4", ea_4),
]


def main():
    conn = psycopg2.connect(DB)
    rng = np.random.default_rng(1)
    lines, control, cands = [], [], []
    ntests = 0
    df = load(conn, SYM)
    if df is None or len(df) < 5000:
        print("Pas assez de bougies.")
        telegram("LOT 2 : pas assez de bougies GOLD dans la base.")
        return
    name, spr = NAMES[SYM], SPREADS[SYM] * SPREAD_FACTOR
    H = prepare(df, "5min")
    for nm, fn in LOT:
        print("Calcul", nm, "5min", name, "-", H.n, "bougies", flush=True)
        s = fn(H)
        rd = np.where(s.d != 0, rng.choice([-1, 1], size=H.n), 0)
        if nm == "EA_2":                                # controle du simulateur : sans frais, au hasard
            z = [r for _, r in run_trades(H, s, rd, 0.0)]
            control.append("%s : direction au hasard, spread 0 -> %s R (%d trades)" % (
                name, fmt(np.mean(z) if z else float("nan")), len(z)))
        real = summarize(run_trades(H, s, s.d, spr))
        inv = summarize(run_trades(H, s, -s.d, spr))
        alea = summarize(run_trades(H, s, rd, spr))
        ntests += 1
        lab = "%-6s %-5s %-6s" % (nm, "5min", name)
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
            cands.append((va, nm, "5min", name, tr, real["pos"], real["ny"], real["valid"][0]))

    head = ("LOT 2 : 5 EA du depot XAUUSD_EA sur l'OR, sans martingale (R moyen apres spread, 1R = risque du trade)\n"
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
             "au hasard, avec frais : c'est le niveau a battre. EA_2 = controle de bruit. Stops en multiples d'ATR "
             "(1 ATR ~ 3 $ en 2026), un seul trade a la fois, sans sortie anticipee ni arret par session. "
             "Resultats theoriques, sans swap ni glissement.")
    full = summary + "\nDETAIL :\n" + "\n".join(lines) + "\n" + guide
    print(full)
    sf = os.environ.get("GITHUB_STEP_SUMMARY")
    if sf:
        with open(sf, "a", encoding="utf-8") as f:
            f.write("```\n" + full + "\n```\n")
    telegram(summary + guide)


if __name__ == "__main__":
    main()
