# -*- coding: utf-8 -*-
"""Telecharge les bougies M5 (XAU/USD, BTC/USD, GBP/JPY) de Twelve Data vers Neon.
Reprend automatiquement la ou il s'est arrete. Relance-le le lendemain si le quota est atteint."""
import os
import sys
import time

import psycopg2
import requests
from psycopg2.extras import execute_values

KEY = os.environ.get("TWELVE_KEY", "")
DB = os.environ.get("DATABASE_URL", "")
if not KEY or not DB:
    sys.exit("ERREUR : TWELVE_KEY ou DATABASE_URL manquant (Secrets GitHub).")

SYMBOLS = ["XAU/USD", "BTC/USD", "GBP/JPY"]
MAX_REQ = int(os.environ.get("MAX_REQ", "450"))  # laisse de la marge pour ton robot Telegram
PAUSE = 8.0  # 8 requetes par minute au maximum sur le plan gratuit
URL = "https://api.twelvedata.com/time_series"
req_count = 0


def telegram(text):
    t, c = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not t or not c:
        return
    try:
        requests.post("https://api.telegram.org/bot%s/sendMessage" % t,
                      json={"chat_id": c, "text": text[:4000]}, timeout=30)
    except Exception as e:
        print("Telegram :", e)


conn = psycopg2.connect(DB)
conn.autocommit = True
cur = conn.cursor()
cur.execute("""create table if not exists candles (
    symbol text not null,
