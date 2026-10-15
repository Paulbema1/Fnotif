# -*- coding: utf-8 -*-
"""LOT 3 : strategie ICT du depot 'ICT-Trading-Strategy' (sweep de liquidite -> MSS -> trou de prix FVG), M15,
sur GOLD, BTCUSD et GBPJPY. Meme moteur, memes frais, meme critere dur et meme periode finale cachee que fiches.py.

POURQUOI UN PORTAGE 'HONNETE' : le test du depot (+513 418 %) entre a un prix PASSE (le bas/haut du FVG) sans verifier
que le prix y revient, et tire l'objectif au hasard entre 3 et 5 R. Ici on corrige ces deux points.

REGLES FIXEES D'AVANCE (avant de voir un seul resultat) :
- Bougies M15. Decision a la cloture de la bougie i, entree a l'OUVERTURE de la bougie suivante (marche), spread deduit.
- Seance : 7h a 21h UTC (hypothese : les donnees de la base sont en UTC). Detection et entrees seulement dans la seance.
- Sweep : la bougie i depasse le plus haut (ou le plus bas) des 20 bougies precedentes, puis cloture en retrait d'au moins
  0,1 ATR par rapport a son extreme. Un sweep reste valable 30 bougies.
- MSS : apres un sweep haussier (bas balaye), une bougie suivante depasse le plus haut de la bougie du sweep ; baissier : inverse.
- FVG : trou de prix de la bougie i (bas de i > haut de i-2, ou l'inverse) d'au moins 0,2 ATR, dans le sens attendu.
- Filtre de tendance de l'original conserve (8 dernieres bougies contre 8 bougies d'il y a 25 a 32, 'neutre' autorise).
- Stop : extreme du sweep -/+ 0,1 ATR ; si le stop est plus petit que 0,3 ATR, pas de trade.
- Objectif FIXE : 3 x le risque. Sortie forcee apres 96 bougies (24 h). Un seul trade a la fois.
- 3 tests dans ce lot (GOLD, BTCUSD, GBPJPY, une seule configuration chacun).
"""
import os

import numpy as np
import pandas as pd
import psycopg2

from fiches import (DB, MIN_PF, MIN_R, MIN_VALID_TRADES, MIN_YEARS, NAMES, SPREAD_FACTOR, SPREADS, SHOW_FINAL,
                    fmt, lag, load, prepare, run_trades, sg, summarize, telegram)

TESTS_BEFORE = 77      # tests deja faits (fiches + lots 1 et 2)
RULE = "15min"
HOLD = 96
RR = 3.0


def ict_signals(H):
    n, h, l, c, atr = H.n, H.h, H.l, H.c, H.atr
    hs, ls = pd.Series(h), pd.Series(l)
    pmax = hs.shift(1).rolling(20).max().values
    pmin = ls.shift(1).rolling(20).min().values
    rh, rl = hs.shift(1).rolling(8).max().values, ls.shift(1).rolling(8).min().values
    oh, ol = hs.shift(25).rolling(8).max().values, ls.shift(25).rolling(8).min().values
    with np.errstate(invalid="ignore"):
        bear_sw = (h > pmax) & (c < h - 0.1 * atr)
        bull_sw = (l < pmin) & (c > l + 0.1 * atr) & ~bear_sw
        trend = np.where((rh > oh) & (rl > ol), 1, np.where((rh < oh) & (rl < ol), -1, 0))
        fvg_up = (l > lag(h, 2)) & ((l - lag(h, 2)) >= 0.2 * atr)
        fvg_dn = (h < lag(l, 2)) & ((lag(l, 2) - h) >= 0.2 * atr)
    hours = H.ts.hour.values
    sess = (hours >= 7) & (hours < 21)

    d = np.zeros(n, dtype=int)
    slb = np.full(n, np.nan)
    sweeps = []                                    # [indice, sens (+1 = bas balaye, -1 = haut balaye), prix, tendance]
    for i in range(50, n):
        sweeps = [s for s in sweeps if i - s[0] <= 30]
        if not sess[i] or np.isnan(atr[i]):
            continue
        if bear_sw[i]:
            sweeps.append([i, -1, h[i], trend[i]])
        elif bull_sw[i]:
            sweeps.append([i, 1, l[i], trend[i]])
        for s in sweeps:
            si, sd, sp, st = s
            if si >= i:
                continue
            if trend[i] != 0 and st != trend[i]:
                continue
            if sd == 1:
                if not (h[si + 1:i + 1].max() > h[si] and fvg_up[i]):
                    continue
                d[i] = 1
                slb[i] = c[i] - (sp - 0.1 * atr[i])
            else:
                if not (l[si + 1:i + 1].min() < l[si] and fvg_dn[i]):
                    continue
                d[i] = -1
                slb[i] = (sp + 0.1 * atr[i]) - c[i]
            sweeps.remove(s)
            break
    slb = np.where(slb >= 0.3 * atr, slb, np.nan)
    return sg(d, slb, HOLD, rr=RR)


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
        H = prepare(df, RULE)
        print("Calcul ICT", RULE, name, "-", H.n, "bougies", flush=True)
        s = ict_signals(H)
        rd = np.where(s.d != 0, rng.choice([-1, 1], size=H.n), 0)
        z = [r for _, r in run_trades(H, s, rd, 0.0)]
        control.append("%s : direction au hasard, spread 0 -> %s R (%d trades)" % (
            name, fmt(np.mean(z) if z else float("nan")), len(z)))
        real = summarize(run_trades(H, s, s.d, spr))
        inv = summarize(run_trades(H, s, -s.d, spr))
        alea = summarize(run_trades(H, s, rd, spr))
        ntests += 1
        lab = "%-4s %-5s %-6s" % ("ICT", RULE, name)
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
            cands.append((va, "ICT", RULE, name, tr, real["pos"], real["ny"], real["valid"][0]))

    head = ("LOT 3 : strategie ICT (sweep + MSS + FVG), entree au marche, objectif fixe 3 R (R moyen apres spread)\n"
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
        summary += "AUCUNE configuration ne passe le critere dur (%d tests dans ce lot, %d au total).\n" % (ntests, total)
    guide = ("\nComment lire : INV = memes moments, direction opposee. ALEA = memes moments et memes stop/objectif, direction "
             "au hasard, avec frais : c'est le niveau a battre. Entree au marche (le depot entrait a un prix passe), "
             "objectif fixe (le depot le tirait au hasard), seuils en fractions d'ATR. "
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
                        
