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
    ts timestamp not null,
    o double precision, h double precision, l double precision, c double precision,
    primary key (symbol, ts))""")


def fetch(symbol, end_date=None):
    """Renvoie une liste de bougies, [] s'il n'y a plus de donnees, None si le quota du jour est atteint."""
    global req_count
    params = {"symbol": symbol, "interval": "5min", "outputsize": 5000,
              "timezone": "UTC", "apikey": KEY}
    if end_date:
        params["end_date"] = end_date
    for _ in range(5):
        if req_count >= MAX_REQ:
            return None
        req_count += 1
        try:
            j = requests.get(URL, params=params, timeout=90).json()
        except Exception as e:
            print("Erreur réseau :", e)
            time.sleep(20)
            continue
        time.sleep(PAUSE)
        if j.get("status") == "ok":
            return j.get("values") or []
        code = j.get("code")
        if code == 429:
            print("Limite de requêtes atteinte, pause de 65 s")
            time.sleep(65)
            continue
        if code == 404:
            return []
        raise RuntimeError(str(j))
    raise RuntimeError("Trop de tentatives pour " + symbol)


def store(symbol, values):
    rows = [(symbol, v["datetime"], float(v["open"]), float(v["high"]),
             float(v["low"]), float(v["close"])) for v in values]
    execute_values(cur, "insert into candles (symbol, ts, o, h, l, c) values %s "
                        "on conflict do nothing", rows, page_size=2000)


def count(symbol):
    cur.execute("select count(*) from candles where symbol=%s", (symbol,))
    return cur.fetchone()[0]


def process(sym):
    """Renvoie 'fini' ou 'quota'."""
    vals = fetch(sym)  # bougies les plus recentes
    if vals is None:
        return "quota"
    if vals:
        store(sym, vals)
    while True:
        cur.execute("select min(ts) from candles where symbol=%s", (sym,))
        oldest = cur.fetchone()[0]
        if oldest is None:
            print("Aucune donnée pour", sym)
            return "fini"
        before = count(sym)
        vals = fetch(sym, oldest.strftime("%Y-%m-%d %H:%M:%S"))
        if vals is None:
            return "quota"
        if not vals:
            print("Début de l'historique atteint pour", sym)
            return "fini"
        store(sym, vals)
        added = count(sym) - before
        print("%s : +%d bougies, plus ancienne : %s, requêtes utilisées : %d/%d"
              % (sym, added, vals[-1]["datetime"], req_count, MAX_REQ))
        if added == 0:
            print("Début de l'historique atteint pour", sym)
            return "fini"


status = {}
for s in SYMBOLS:
    print("==", s, "==")
    status[s] = process(s)
    if status[s] == "quota":
        print("Quota du jour atteint : relance demain pour continuer.")
        break

lines = []
for s in SYMBOLS:
    cur.execute("select count(*), min(ts), max(ts) from candles where symbol=%s", (s,))
    n, a, b = cur.fetchone()
    etat = status.get(s, "pas commencé")
    lines.append("%s : %d bougies, de %s à %s (%s)" % (s, n, a, b, etat))
summary = "Téléchargement terminé\n" + "\n".join(lines) + "\nRequêtes utilisées : %d" % req_count
print(summary)
telegram(summary)
