"""Pull monthly S&P 500 realized-volatility dataset (Li et al. 2505.13933v2).

Sources (all keyless, free):
  - Yahoo Finance chart API : ^GSPC daily closes -> monthly log RV
  - Ken French data library : Mkt-RF, SMB, HML, RF, ST_Rev (monthly)
  - Shiller ie_data.xls     : S&P P/D/E, CPI -> DP, EP, INF
  - FRED fredgraph.csv      : INDPRO -> IP growth, TB3MS -> TB, BAA-AAA -> DEF

Output: data/rv_monthly.csv  (Feb 1950 - Dec 2017, 815 rows, << 2 MB Qollab cap)

Usage: .venv/bin/python scripts/pull_data.py
"""

from __future__ import annotations

import io
import pathlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import xlrd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
OUT = DATA_DIR / "rv_monthly.csv"

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) qollab-hackathon-research"}
START, END = "1950-01", "2017-12"  # 816 monthly rows like the paper's Data.CSV
# (paper text says Feb 1950; their repo data starts Jan 1950. OOS eval unaffected.)


def _get(url: str, **kw) -> requests.Response:
    r = requests.get(url, headers=UA, timeout=60, **kw)
    r.raise_for_status()
    return r


# ---------------------------------------------------------------- Yahoo RV
def pull_rv() -> pd.DataFrame:
    p1 = int(datetime(1950, 1, 1, tzinfo=timezone.utc).timestamp())
    p2 = int(datetime(2018, 2, 1, tzinfo=timezone.utc).timestamp())
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC"
        f"?period1={p1}&period2={p2}&interval=1d&events=div"
    )
    j = _get(url).json()["chart"]["result"][0]
    ts = pd.to_datetime(j["timestamp"], unit="s", utc=True).tz_convert(None)
    ts = ts.floor("D")  # Yahoo stamps carry intraday times; floor to calendar day
    close = pd.Series(j["indicators"]["quote"][0]["close"], index=ts, dtype=float)
    daily = close.dropna().groupby(level=0).last().sort_index().asfreq("B")
    daily = daily.ffill(limit=5)  # bridge suspensions only; months still need real data
    r = np.log(daily / daily.shift(1)).dropna()
    m = r.index.to_period("M")
    rv = np.sqrt(r.groupby(m).apply(lambda x: (x**2).sum()))
    df = pd.DataFrame({"RV": rv, "RV_log": np.log(rv)})
    df.index = df.index.to_timestamp("M")
    df["RVq"] = df["RV"].rolling(3).mean()  # quarterly avg, cf. HARX Eq.(3)
    df["RVa"] = df["RV"].rolling(12).mean()  # annual avg
    return df


# ------------------------------------------------------- French factors
def _french_csv(url: str) -> pd.DataFrame:
    txt = _get(url).text
    # data block starts at first YYYYMM line, ends at blank/alpha line
    lines = txt.splitlines()
    start = next(i for i, l in enumerate(lines) if l[:6].isdigit() and "," in l)
    rows = []
    for line in lines[start:]:
        if not line.strip() or not line[:6].strip().isdigit():
            break
        rows.append(line)
    return pd.read_csv(io.StringIO("\n".join(rows)), header=None)


def pull_french() -> pd.DataFrame:
    base = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
    f = _french_csv(f"{base}/F-F_Research_Data_Factors.CSV")
    f.columns = ["ym", "MKT", "SMB", "HML", "RF"]
    s = _french_csv(f"{base}/F-F_ST_Reversal_Factor.CSV")
    s.columns = ["ym", "STR"]
    df = f.merge(s, on="ym", how="outer")
    df["date"] = pd.to_datetime(df["ym"].astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    df = df.set_index("date").drop(columns="ym").replace(-99.99, np.nan).replace(-999, np.nan)
    return df.astype(float)


# ------------------------------------------------------- Shiller DP/EP/INF
def pull_shiller() -> pd.DataFrame:
    url = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
    content = _get(url).content
    wb = xlrd.open_workbook(file_contents=content)
    sh = wb.sheet_by_name("Data")

    def val(r, c):
        try:
            v = sh.cell_value(r, c)
            return float(v) if v != "" else np.nan
        except (IndexError, ValueError):
            return np.nan

    rows = []
    for r in range(8, sh.nrows):
        d = val(r, 0)
        if np.isnan(d):
            continue
        y, m = int(d), int(round((d - int(d)) * 100))
        if not 1 <= m <= 12:
            continue
        rows.append((datetime(y, m, 1), val(r, 1), val(r, 2), val(r, 3), val(r, 4)))
    df = pd.DataFrame(rows, columns=["date", "P", "D", "E", "CPI"])
    df = df.set_index("date").sort_index()
    df["DP"] = df["D"] / df["P"]
    df["EP"] = df["E"] / df["P"]
    df["INF"] = 100 * (df["CPI"] / df["CPI"].shift(1) - 1)  # monthly CPI inflation, %
    df.index = df.index + pd.offsets.MonthEnd(0)
    return df[["DP", "EP", "INF"]]


# ------------------------------------------------------- FRED IP/TB/DEF
def _fred(series: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd=1950-01-01"
    df = pd.read_csv(io.StringIO(_get(url).text), parse_dates=["observation_date"])
    s = df.set_index("observation_date")[series].replace(".", np.nan).astype(float)
    return s.resample("ME").last()


def pull_fred() -> pd.DataFrame:
    ip = _fred("INDPRO")
    tb = _fred("TB3MS")
    baa = _fred("BAA")
    aaa = _fred("AAA")
    return pd.DataFrame(
        {
            "IP": 100 * np.log(ip / ip.shift(1)),  # industrial production growth, %
            "TB": tb,  # 3-month T-bill rate, %
            "DEF": baa - aaa,  # default spread, pp
        }
    )


# ---------------------------------------------------------------- main
def main() -> None:
    print("pulling Yahoo RV ...")
    rv = pull_rv()
    print("pulling French factors ...")
    fr = pull_french()
    print("pulling Shiller DP/EP/INF ...")
    sh = pull_shiller()
    print("pulling FRED IP/TB/DEF ...")
    fred = pull_fred()

    df = rv.join(fr, how="left").join(sh, how="left").join(fred, how="left")
    # TB3MS starts 1982 on FRED -> backfill early window with French 1-mo T-bill return
    df["TB"] = df["TB"].fillna(df["RF"])

    window = df.loc[START:END].copy()
    cols = ["RV_log", "RV", "RVq", "RVa", "MKT", "SMB", "HML", "STR",
            "DP", "EP", "INF", "IP", "TB", "DEF", "RF"]
    window = window[cols]
    window.to_csv(OUT, index_label="date")

    print(f"\nwrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB, "
          f"Qollab file cap 2048 KB)")
    print(f"rows: {len(window)} (paper: 815), range: "
          f"{window.index.min().date()} -> {window.index.max().date()}")
    print("\nNaNs per column in window:")
    print(window.isna().sum())
    print("\ndescribe:")
    print(window.describe().T[["mean", "std", "min", "max"]].round(4))


if __name__ == "__main__":
    main()
