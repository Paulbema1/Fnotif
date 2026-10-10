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

