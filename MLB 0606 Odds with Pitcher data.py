import pandas as pd
import numpy as np
import requests
import re
from datetime import date, timedelta
from pybaseball import playerid_lookup, statcast_pitcher, statcast_batter
from bs4 import BeautifulSoup

# ============================================================
# MLB BETTING MODEL v6 — SAMPLE-WEIGHTED LINEUP VULNERABILITY
#   v2 FIXES (carried forward):
#     FIX 1: Avg_IP_Start now counts actual out-events
#     FIX 2: grade_bullpen() closer fallback when no CL on roster
#     FIX 3: safe_playerid_lookup() handles suffixes/hyphens/accents
#   v3 NEW (carried forward):
#     NEW 1: pitcher_profile() adds K%_vL and K%_vR splits
#     NEW 2: build_matchup_cards() stores lineup handedness counts
#     NEW 3: _platoon_weighted_k() computes handedness-weighted K%
#     NEW 4: attach_betting_signals() uses platoon K% for k-props
#     NEW 5: export_full_cards() includes platoon K% columns
#   v4 NEW (carried forward):
#     NEW 6: log_bets_to_results()   — appends PENDING bets after each run
#     NEW 7: fetch_and_log_results() — auto-grades ML/F5/Total/NRFI next day
#     NEW 8: print_results_summary() — win rate by market + confidence tier
#   v5 NEW (carried forward):
#     NEW 9:  IP gate (4.5 inn minimum) on ML / F5 / Total / NRFI signals
#     NEW 10: LOW confidence suppression — filtered from log + flagged in print
#   v5 FIXES (carried forward):
#     FIX 4: results dedup key = date+matchup+bet_type (lean excluded)
#     FIX 5: K_PROP dedup keyed per pitcher so both SPs in game log
#     FIX 6: _load_results_log() normalises date format on read
#   v6 NEW:
#     NEW 11: batter_profile() adds xwoba_recent (last 21 days),
#             pa_vL and pa_vR sample counts
#     NEW 12: lineup_vulnerability_score() uses Bayesian sample-weighted
#             blend of platoon split vs overall xwOBA/K%/HH% — thin splits
#             shrink toward overall rather than defaulting to 50.0 neutral
#     NEW 13: per-batter split_confidence flag (HIGH/MED/LOW) shown in
#             breakdown so you can see which slot scores are reliable
#   v6 GRADING FIXES:
#     FIX G1: _fetch_final_score() — requires 9+ innings, guards in-progress
#     FIX G2: _fetch_f5_score() — pads missing 0-run innings from MLB API
#     FIX G3: fetch_and_log_results() — casts game_pk int to strip pandas .0
#     FIX G4: TOTAL grading warns + skips if no line in Notes instead of
#             silently defaulting to wrong 8.5 threshold
# ============================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", None)

# Use Sydney time to determine "today" — script runs on UTC servers but
# we want the date as seen from Sydney (AEST UTC+10 / AEDT UTC+11).
# This ensures the correct MLB slate is fetched when running at 7pm Sydney.
try:
    from zoneinfo import ZoneInfo as _ZoneInfo
    import datetime as _datetime
    _sydney_now = _datetime.datetime.now(_ZoneInfo("Australia/Sydney"))
    today_dt  = _sydney_now.date()
except Exception:
    today_dt  = date.today()
today     = today_dt.strftime("%Y-%m-%d")
print(f"📅  Script date (Sydney time): {today}")
start_date = f"{today_dt.year}-01-01"
end_date   = today

# Out events used to count outs recorded per start
OUT_EVENTS = {
    "field_out", "strikeout", "grounded_into_double_play",
    "double_play", "triple_play", "strikeout_double_play",
    "fielders_choice_out", "force_out", "sac_fly", "sac_bunt",
    "caught_stealing_2b", "caught_stealing_3b", "caught_stealing_home",
    "pickoff_caught_stealing_2b", "pickoff_caught_stealing_3b",
    "pickoff_caught_stealing_home", "other_out",
}

# ============================================================
# HELPERS
# ============================================================

def split_name(full_name):
    parts = full_name.strip().split()
    return parts[0], " ".join(parts[1:])

def safe_playerid_lookup(full_name):
    """Try multiple name splits to handle suffixes, hyphens, accents."""
    parts = full_name.strip().split()
    if len(parts) < 2:
        return pd.DataFrame()
    attempts = []
    attempts.append((parts[0], " ".join(parts[1:])))
    if len(parts) > 2:
        attempts.append((parts[0], parts[-1]))
    suffixes = {"jr", "sr", "ii", "iii", "iv"}
    if parts[-1].lower().rstrip(".") in suffixes and len(parts) > 2:
        attempts.append((parts[0], " ".join(parts[1:-1])))
        attempts.append((parts[0], parts[-2]))
    for first, last in attempts:
        try:
            result = playerid_lookup(last, first)
            if not result.empty:
                return result
        except Exception:
            continue
    return pd.DataFrame()

# ============================================================
# PHASE 1 — PITCHER PROFILING
# ============================================================

def pitcher_profile(full_name):
    try:
        first, last = split_name(full_name)
        pid = playerid_lookup(last, first)
        if pid.empty:
            pid = safe_playerid_lookup(full_name)
        if pid.empty:
            return None
        mlbam_id = pid.iloc[0]["key_mlbam"]
        df = statcast_pitcher(start_date, end_date, mlbam_id)
        if df.empty:
            return None

        hand = "UNK"
        if "p_throws" in df.columns:
            hands = df["p_throws"].dropna().unique()
            if len(hands) > 0:
                hand = hands[0]

        batted        = df[df["events"].notnull()].copy()
        batters_faced = len(batted)
        sample_flag   = "LOW SAMPLE" if batters_faced < 50 else "OK"

        xwoba    = batted["estimated_woba_using_speedangle"].mean()
        xba      = batted["estimated_ba_using_speedangle"].mean()
        xslg     = batted["estimated_slg_using_speedangle"].mean()
        la       = batted["launch_angle"].mean()
        ev       = batted["launch_speed"].mean()
        hard_hit = (batted["launch_speed"] >= 95).mean() * 100

        strikeouts = len(df[df["events"] == "strikeout"])
        walks      = len(df[df["events"] == "walk"])
        k_pct      = (strikeouts / batters_faced) * 100 if batters_faced > 0 else 0
        bb_pct     = (walks / batters_faced) * 100 if batters_faced > 0 else 0

        def platoon_k(stand_val):
            sub = batted[batted["stand"] == stand_val] if "stand" in batted.columns else pd.DataFrame()
            if len(sub) < 15:
                return None
            return round((sub["events"] == "strikeout").mean() * 100, 1)

        k_pct_vl = platoon_k("L")
        k_pct_vr = platoon_k("R")

        first_tto_batters = []
        game_pitch_counts = []
        game_outs         = []

        if "game_pk" in df.columns and "batter" in df.columns:
            for game_id in df["game_pk"].dropna().unique():
                game_df = df[df["game_pk"] == game_id].copy()

                seen = set()
                first_trip_rows = []
                for _, row in game_df.iterrows():
                    batter = row.get("batter")
                    if batter not in seen:
                        seen.add(batter)
                        first_trip_rows.append(row)
                    if len(seen) >= 9:
                        break
                if first_trip_rows:
                    first_tto_batters.extend(first_trip_rows)

                game_pitch_counts.append(len(game_df))

                if "events" in game_df.columns:
                    outs = int(game_df["events"].isin(OUT_EVENTS).sum())
                    if outs > 0:
                        game_outs.append(outs)

        tto_df = pd.DataFrame(first_tto_batters)
        if not tto_df.empty:
            tto_batted = tto_df[tto_df["events"].notnull()].copy()
            tto_xwoba  = tto_batted["estimated_woba_using_speedangle"].mean()
            tto_k      = (tto_df["events"] == "strikeout").sum() / max(len(tto_batted), 1) * 100
            tto_bb     = (tto_df["events"] == "walk").sum() / max(len(tto_batted), 1) * 100
        else:
            tto_xwoba = tto_k = tto_bb = np.nan

        avg_pitches_start = np.mean(game_pitch_counts) if game_pitch_counts else np.nan
        avg_ip_start = np.mean(game_outs) / 3 if game_outs else np.nan

        return {
            "Pitcher": full_name, "Hand": hand,
            "xwOBA":   round(xwoba, 3) if pd.notnull(xwoba) else None,
            "xBA":     round(xba,   3) if pd.notnull(xba)   else None,
            "xSLG":    round(xslg,  3) if pd.notnull(xslg)  else None,
            "LA":      round(la,    1) if pd.notnull(la)     else None,
            "EV":      round(ev,    1) if pd.notnull(ev)     else None,
            "HardHit%": round(hard_hit, 1),
            "K%":      round(k_pct,  1), "BB%": round(bb_pct, 1),
            "K-BB%":   round(k_pct - bb_pct, 1), "Sample": sample_flag,
            "TTO_xwOBA": round(tto_xwoba, 3) if pd.notnull(tto_xwoba) else None,
            "TTO_K%":    round(tto_k, 1)     if pd.notnull(tto_k)     else None,
            "TTO_BB%":   round(tto_bb, 1)    if pd.notnull(tto_bb)    else None,
            "Avg_Pitches_Start": round(avg_pitches_start, 1) if pd.notnull(avg_pitches_start) else None,
            "Avg_IP_Start":      round(avg_ip_start, 2)      if pd.notnull(avg_ip_start)      else None,
            "K%_vL": k_pct_vl,
            "K%_vR": k_pct_vr,
        }
    except Exception as e:
        print(f"Error {full_name}: {e}")
        return None

# ============================================================
# PHASE 2A — LINEUP SCRAPER (RotoWire + MLB API)
# ============================================================

ROTOWIRE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

ROTOWIRE_TEAM_MAP = {
    "ARI":"Arizona Diamondbacks","ATL":"Atlanta Braves","BAL":"Baltimore Orioles",
    "BOS":"Boston Red Sox","CHC":"Chicago Cubs","CWS":"Chicago White Sox",
    "CIN":"Cincinnati Reds","CLE":"Cleveland Guardians","COL":"Colorado Rockies",
    "DET":"Detroit Tigers","HOU":"Houston Astros","KC":"Kansas City Royals",
    "LAA":"Los Angeles Angels","LAD":"Los Angeles Dodgers","MIA":"Miami Marlins",
    "MIL":"Milwaukee Brewers","MIN":"Minnesota Twins","NYM":"New York Mets",
    "NYY":"New York Yankees","OAK":"Oakland Athletics","PHI":"Philadelphia Phillies",
    "PIT":"Pittsburgh Pirates","SD":"San Diego Padres","SF":"San Francisco Giants",
    "SEA":"Seattle Mariners","STL":"St. Louis Cardinals","TB":"Tampa Bay Rays",
    "TEX":"Texas Rangers","TOR":"Toronto Blue Jays","WSH":"Washington Nationals",
}
FULL_TO_ABBR = {v: k for k, v in ROTOWIRE_TEAM_MAP.items()}

_roster_cache = {}

def get_team_roster(team_id):
    if team_id in _roster_cache:
        return _roster_cache[team_id]
    url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType=active&season={today_dt.year}"
    try:
        data   = requests.get(url, timeout=10).json()
        roster = [{"full_name": p["person"]["fullName"],
                   "person_id": p["person"]["id"],
                   "position":  p.get("position", {}).get("abbreviation", "")}
                  for p in data.get("roster", [])]
        _roster_cache[team_id] = roster
        return roster
    except:
        return []

def resolve_short_name(short_name, team_id):
    roster      = get_team_roster(team_id)
    short_clean = short_name.strip()
    if not roster:
        return short_name, None
    for p in roster:
        if p["full_name"].lower() == short_clean.lower():
            return p["full_name"], p["person_id"]
    if " " not in short_clean:
        matches = [p for p in roster if p["full_name"].lower().endswith(" " + short_clean.lower())]
        if len(matches) == 1:
            return matches[0]["full_name"], matches[0]["person_id"]
    m = re.match(r'^([A-Z])\.\s+(.+)$', short_clean)
    if m:
        initial  = m.group(1).upper()
        lastname = m.group(2).strip().lower()
        matches  = [p for p in roster
                    if p["full_name"].lower().split()[-1] == lastname
                    and p["full_name"][0].upper() == initial]
        if matches:
            return matches[0]["full_name"], matches[0]["person_id"]
    parts = short_clean.split()
    if parts:
        last    = parts[-1].lower()
        matches = [p for p in roster if p["full_name"].lower().split()[-1].startswith(last[:4])]
        if len(matches) == 1:
            return matches[0]["full_name"], matches[0]["person_id"]
    return short_name, None

def scrape_rotowire_lineups():
    lineups = {}
    try:
        resp = requests.get("https://www.rotowire.com/baseball/daily-lineups.php",
                            headers=ROTOWIRE_HEADERS, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        print(f"  ⚠  RotoWire fetch failed: {e}")
        return {}

    soup = BeautifulSoup(resp.text, "html.parser")
    containers = (soup.find_all("div", class_=re.compile(r"lineup__box")) or
                  soup.find_all("div", class_=re.compile(r"lineups-box")) or
                  soup.find_all("div", class_=re.compile(r"lineup-card")))
    print(f"  RotoWire: {len(containers)} game containers found")

    for container in containers:
        away_abbr, home_abbr = _rw_get_abbrs(container)
        if not away_abbr or not home_abbr:
            continue
        away_players, home_players = _rw_get_players(container)
        if away_players:
            lineups[away_abbr] = away_players
        if home_players:
            lineups[home_abbr] = home_players

    print(f"  RotoWire: lineups for {len(lineups)} teams: {sorted(lineups.keys())}")
    return lineups

def _rw_get_abbrs(container):
    for cls_pat in [r"lineup__team", r"team-abbrev", r"lineup__abbr", r"lineup-team"]:
        tags  = container.find_all(class_=re.compile(cls_pat))
        found = []
        for tag in tags:
            text  = tag.get_text(strip=True).upper()
            match = re.search(r'\b([A-Z]{2,3})\b', text)
            if match and match.group(1) in ROTOWIRE_TEAM_MAP:
                found.append(match.group(1))
        if len(found) >= 2:
            return found[0], found[1]
    links = container.find_all("a", href=re.compile(r'/baseball/team'))
    found = []
    for link in links:
        m = re.search(r'/baseball/team[s]?/([a-z]{2,3})', link.get("href",""), re.I)
        if m:
            cand = m.group(1).upper()
            if cand in ROTOWIRE_TEAM_MAP and cand not in found:
                found.append(cand)
        if len(found) >= 2:
            return found[0], found[1]
    all_strings = container.find_all(
        string=re.compile(r'^(ARI|ATL|BAL|BOS|CHC|CWS|CIN|CLE|COL|DET|HOU|KC|LAA|LAD|MIA|MIL|MIN|NYM|NYY|OAK|PHI|PIT|SD|SF|SEA|STL|TB|TEX|TOR|WSH)$'))
    found = []
    for s in all_strings:
        cand = s.strip()
        if cand in ROTOWIRE_TEAM_MAP and cand not in found:
            found.append(cand)
        if len(found) >= 2:
            return found[0], found[1]
    return None, None

def _rw_get_players(container):
    all_lists = (container.find_all("ul", class_=re.compile(r"lineup__list")) or
                 container.find_all("ol", class_=re.compile(r"lineup__list")) or
                 container.find_all("ul", class_=re.compile(r"lineup-list")) or
                 container.find_all("div", class_=re.compile(r"lineup__players")))
    valid = []
    for lst in all_lists:
        items = (lst.find_all("li", class_=re.compile(r"lineup__player")) or
                 lst.find_all("li") or
                 lst.find_all("div", class_=re.compile(r"player")))
        if 5 <= len(items) <= 11:
            valid.append(items)
    if len(valid) < 2:
        return [], []
    return _rw_parse_list(valid[0]), _rw_parse_list(valid[1])

def _rw_parse_list(items):
    batters = []
    for idx, item in enumerate(items[:9], 1):
        name = _rw_extract_name(item)
        if name and len(name) > 2:
            batters.append({"name": name, "order": idx, "source": "RotoWire"})
    return batters

def _rw_extract_name(item):
    a = item.find("a")
    if a:
        name = a.get_text(strip=True)
        name = re.sub(r'\s+[A-Z]{1,2}$', '', name).strip()
        name = re.sub(r'^\d+[\.\s]+', '', name).strip()
        if name:
            return name
    span = item.find("span", class_=re.compile(r"name|player-name"))
    if span:
        return span.get_text(strip=True)
    text  = re.sub(r'^\d+[\.\s]+', '', item.get_text(strip=True)).strip()
    parts = text.split()
    return " ".join(parts[:2]) if len(parts) >= 2 else None

def get_confirmed_lineup(game_pk):
    try:
        data  = requests.get(f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore", timeout=10).json()
        teams = data.get("teams", {})
        def extract(players_dict):
            batters = []
            for _, val in players_dict.items():
                order = val.get("battingOrder")
                if order:
                    try:
                        oi = int(str(order).strip())
                        if oi % 100 == 0 and oi <= 900:
                            batters.append({"name": val["person"]["fullName"],
                                            "batter_id": val["person"]["id"],
                                            "order": oi // 100, "source": "MLB-Confirmed"})
                    except:
                        continue
            batters.sort(key=lambda x: x["order"])
            return batters
        home = extract(teams.get("home", {}).get("players", {}))
        away = extract(teams.get("away", {}).get("players", {}))
        return away, home
    except:
        return [], []

def _remove_pitcher_from_lineup(lineup, pitcher_name):
    if not lineup or not pitcher_name or pitcher_name == "TBD":
        return lineup
    pitcher_last  = pitcher_name.strip().split()[-1].lower()
    pitcher_first = pitcher_name.strip().split()[0][0].upper()
    filtered = []
    for b in lineup:
        bname = b.get("name", "").strip()
        if bname.lower() == pitcher_name.lower():
            continue
        if bname.lower().split()[-1] == pitcher_last:
            continue
        m = re.match(r'^([A-Z])\.\s+(.+)$', bname)
        if m and m.group(1).upper() == pitcher_first and m.group(2).strip().lower() == pitcher_last:
            continue
        filtered.append(b)
    for i, b in enumerate(filtered):
        b["order"] = i + 1
    return filtered

def _resolve_lineup_names(lineup, team_id):
    resolved = []
    for b in lineup:
        short = b.get("name", "")
        full_name, person_id = resolve_short_name(short, team_id)
        resolved.append({**b, "name": full_name, "batter_id": person_id, "name_raw": short})
    return resolved

def _get_team_id_map():
    try:
        data = requests.get("https://statsapi.mlb.com/api/v1/teams?sportId=1", timeout=10).json()
        return {t["name"]: t["id"] for t in data.get("teams", [])}
    except:
        return {}

def get_todays_lineups_with_probables():
    print("  📋 Scraping RotoWire probable lineups...")
    rw_cache    = scrape_rotowire_lineups()
    team_id_map = _get_team_id_map()

    try:
        sched = requests.get(
            f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}&hydrate=probablePitcher,team",
            timeout=10).json()
    except Exception as e:
        print(f"  ⚠  Schedule fetch failed: {e}")
        return []

    dates = sched.get("dates", [])
    if not dates:
        print("  No games scheduled today.")
        return []

    matchups = []
    for g in dates[0].get("games", []):
        game_pk      = g.get("game_pk")
        home_team    = g["teams"]["home"]["team"]["name"]
        away_team    = g["teams"]["away"]["team"]["name"]
        hp_info      = g["teams"]["home"].get("probablePitcher")
        ap_info      = g["teams"]["away"].get("probablePitcher")
        home_pitcher = hp_info["fullName"] if hp_info else "TBD"
        away_pitcher = ap_info["fullName"] if ap_info else "TBD"

        away_confirmed, home_confirmed = get_confirmed_lineup(game_pk)

        home_abbr  = FULL_TO_ABBR.get(home_team, "")
        away_abbr  = FULL_TO_ABBR.get(away_team, "")
        rw_home    = rw_cache.get(home_abbr, [])
        rw_away    = rw_cache.get(away_abbr, [])

        final_home  = home_confirmed if home_confirmed else rw_home
        final_away  = away_confirmed if away_confirmed else rw_away
        source_home = "MLB-Confirmed" if home_confirmed else ("RotoWire" if rw_home else "UNCONFIRMED")
        source_away = "MLB-Confirmed" if away_confirmed else ("RotoWire" if rw_away else "UNCONFIRMED")

        final_home = _remove_pitcher_from_lineup(final_home, home_pitcher)
        final_away = _remove_pitcher_from_lineup(final_away, away_pitcher)

        home_team_id = team_id_map.get(home_team)
        away_team_id = team_id_map.get(away_team)
        if source_home == "RotoWire" and home_team_id:
            final_home = _resolve_lineup_names(final_home, home_team_id)
        if source_away == "RotoWire" and away_team_id:
            final_away = _resolve_lineup_names(final_away, away_team_id)

        def src_label(src, n):
            if src == "MLB-Confirmed": return f"✅ CONFIRMED ({n} batters)"
            if src == "RotoWire":      return f"📋 ROTOWIRE ({n} batters)"
            return                            f"⚠  UNCONFIRMED"

        print(f"    {away_team[:20]:<20} @ {home_team[:20]:<20}")
        print(f"      Away: {src_label(source_away, len(final_away))}  Home: {src_label(source_home, len(final_home))}")

        matchups.append({
            "game_pk": game_pk, "home_team": home_team, "away_team": away_team,
            "home_pitcher": home_pitcher, "away_pitcher": away_pitcher,
            "home_lineup": final_home, "away_lineup": final_away,
            "lineup_source_home": source_home, "lineup_source_away": source_away,
        })
    return matchups

# ============================================================
# PHASE 2B — BATTER PROFILING + MATCHUP CARDS
# ============================================================

def batter_profile(batter_name, batter_mlbam_id=None):
    try:
        if batter_mlbam_id:
            mlbam_id = batter_mlbam_id
        else:
            pid = safe_playerid_lookup(batter_name)
            if pid.empty:
                return None
            mlbam_id = int(pid.iloc[0]["key_mlbam"])

        df = statcast_batter(start_date, end_date, mlbam_id)
        if df.empty:
            return None

        batted   = df[df["events"].notnull()].copy()
        pa_total = len(batted)

        xwoba_all  = batted["estimated_woba_using_speedangle"].mean()
        xba_all    = batted["estimated_ba_using_speedangle"].mean()
        k_pct_all  = (batted["events"] == "strikeout").mean() * 100
        bb_pct_all = (batted["events"] == "walk").mean() * 100
        hard_hit   = (batted["launch_speed"] >= 95).mean() * 100

        if "zone" in df.columns and "description" in df.columns:
            outside   = df[df["zone"] > 9].copy()
            chase_pct = outside["description"].isin(
                ["swinging_strike","foul","hit_into_play","swinging_strike_blocked","foul_tip"]
            ).mean() * 100 if len(outside) > 0 else np.nan
        else:
            chase_pct = np.nan

        if "description" in df.columns:
            swings    = df[df["description"].isin(["swinging_strike","foul","hit_into_play","swinging_strike_blocked","foul_tip"])]
            whiffs    = df[df["description"].isin(["swinging_strike","swinging_strike_blocked"])]
            whiff_pct = len(whiffs) / max(len(swings), 1) * 100
        else:
            whiff_pct = np.nan

        def split_stats(hand):
            sub = batted[batted["p_throws"] == hand].copy() if "p_throws" in batted.columns else pd.DataFrame()
            n   = len(sub)
            if sub.empty or n < 5:
                return None, None, None, None, n
            xw = sub["estimated_woba_using_speedangle"].mean()
            k  = (sub["events"] == "strikeout").mean() * 100
            bb = (sub["events"] == "walk").mean() * 100
            hh = (sub["launch_speed"] >= 95).mean() * 100
            return (round(xw,3) if pd.notnull(xw) else None,
                    round(k,1)  if pd.notnull(k)  else None,
                    round(bb,1) if pd.notnull(bb) else None,
                    round(hh,1) if pd.notnull(hh) else None,
                    n)

        xwoba_vl, k_vl, bb_vl, hh_vl, pa_vl = split_stats("L")
        xwoba_vr, k_vr, bb_vr, hh_vr, pa_vr = split_stats("R")

        xwoba_recent = None
        recent_pa    = 0
        if "game_date" in df.columns:
            cutoff_recent = (today_dt - timedelta(days=21)).strftime("%Y-%m-%d")
            recent_batted = batted[batted["game_date"] >= cutoff_recent].copy()
            recent_pa     = len(recent_batted)
            if recent_pa >= 5:
                xw_r = recent_batted["estimated_woba_using_speedangle"].mean()
                xwoba_recent = round(xw_r, 3) if pd.notnull(xw_r) else None

        return {
            "name": batter_name, "batter_id": mlbam_id,
            "sample": "LOW" if pa_total < 30 else "OK", "pa": pa_total,
            "xwOBA": round(xwoba_all,3) if pd.notnull(xwoba_all) else None,
            "xBA":   round(xba_all,3)   if pd.notnull(xba_all)   else None,
            "K%":    round(k_pct_all,1) if pd.notnull(k_pct_all) else None,
            "BB%":   round(bb_pct_all,1)if pd.notnull(bb_pct_all)else None,
            "HardHit%": round(hard_hit,1) if pd.notnull(hard_hit) else None,
            "Chase%":   round(chase_pct,1) if pd.notnull(chase_pct) else None,
            "Whiff%":   round(whiff_pct,1) if pd.notnull(whiff_pct) else None,
            "xwOBA_vL": xwoba_vl, "K%_vL": k_vl, "BB%_vL": bb_vl, "HardHit%_vL": hh_vl,
            "xwOBA_vR": xwoba_vr, "K%_vR": k_vr, "BB%_vR": bb_vr, "HardHit%_vR": hh_vr,
            "pa_vL": pa_vl, "pa_vR": pa_vr,
            "xwOBA_recent": xwoba_recent, "recent_pa": recent_pa,
        }
    except Exception as e:
        print(f"    Batter error ({batter_name}): {e}")
        return None

def lineup_vulnerability_score(lineup, pitcher_hand):
    if not lineup:
        return None, {}

    slot_weights   = {1:1.3,2:1.3,3:1.4,4:1.4,5:1.2,6:1.1,7:0.9,8:0.8,9:0.6}
    scores         = []
    slot_breakdown = []

    PRIOR_PA     = 30
    RECENT_BLEND = 0.20

    def _blend(split_val, split_pa, overall_val, recent_val=None):
        if overall_val is None:
            return split_val, "LOW"
        if split_val is None or split_pa is None or split_pa == 0:
            base = overall_val
            if recent_val is not None:
                base = (1 - RECENT_BLEND) * base + RECENT_BLEND * recent_val
            return round(base, 3), "LOW"
        w_split   = split_pa / (split_pa + PRIOR_PA)
        w_overall = 1 - w_split
        blended   = w_split * split_val + w_overall * overall_val
        if recent_val is not None:
            blended = (1 - RECENT_BLEND) * blended + RECENT_BLEND * recent_val
        if split_pa >= 60:   conf = "HIGH"
        elif split_pa >= 25: conf = "MED"
        else:                conf = "LOW"
        return round(blended, 3), conf

    def _blend_pct(split_val, split_pa, overall_val):
        if overall_val is None:
            return split_val, "LOW"
        if split_val is None or split_pa is None or split_pa == 0:
            return overall_val, "LOW"
        w_split = split_pa / (split_pa + PRIOR_PA)
        blended = w_split * split_val + (1 - w_split) * overall_val
        conf    = "HIGH" if split_pa >= 60 else "MED" if split_pa >= 25 else "LOW"
        return round(blended, 1), conf

    for i, batter in enumerate(lineup[:9]):
        slot    = i + 1
        weight  = slot_weights.get(slot, 1.0)
        profile = batter.get("profile")

        if not profile:
            scores.append((50.0, weight))
            slot_breakdown.append({
                "slot": slot, "name": batter.get("name", "Unknown"),
                "score": 50.0, "xwOBA": None, "K%": None, "HH%": None,
                "note": "no profile", "split_conf": "—",
            })
            continue

        overall_xwoba  = profile.get("xwOBA")
        overall_k      = profile.get("K%")
        overall_hh     = profile.get("HardHit%")
        recent_xwoba   = profile.get("xwOBA_recent")

        if pitcher_hand == "L":
            split_xwoba = profile.get("xwOBA_vL")
            split_k     = profile.get("K%_vL")
            split_hh    = profile.get("HardHit%_vL")
            split_pa    = profile.get("pa_vL") or 0
        else:
            split_xwoba = profile.get("xwOBA_vR")
            split_k     = profile.get("K%_vR")
            split_hh    = profile.get("HardHit%_vR")
            split_pa    = profile.get("pa_vR") or 0

        xwoba, xwoba_conf = _blend(split_xwoba, split_pa, overall_xwoba, recent_xwoba)
        k_pct, _          = _blend_pct(split_k,  split_pa, overall_k)
        hh,    _          = _blend_pct(split_hh, split_pa, overall_hh)

        xwoba_score   = max(0, min(40, (xwoba - 0.200) / 0.200 * 40)) if xwoba else 0
        k_score       = max(0, min(30, (35 - k_pct) / 25 * 30))       if k_pct  else 0
        hh_score      = max(0, min(20, (hh - 25) / 25 * 20))          if hh     else 0
        chase = profile.get("Chase%")
        whiff = profile.get("Whiff%")
        contact_score = max(0, min(10, (45-chase)/35*5+(30-whiff)/20*5)) if (chase and whiff) else 10
        batter_score  = max(0, min(100, xwoba_score + k_score + hh_score + contact_score))

        scores.append((batter_score, weight))

        recent_note = f" r{profile.get('recent_pa',0)}PA" if recent_xwoba else ""
        split_note  = f"{split_pa}PA vs {'L' if pitcher_hand=='L' else 'R'}{recent_note}"
        slot_breakdown.append({
            "slot":       slot,
            "name":       batter.get("name", "?"),
            "score":      round(batter_score, 1),
            "xwOBA":      xwoba,
            "K%":         k_pct,
            "HH%":        hh,
            "note":       split_note,
            "split_conf": xwoba_conf,
        })

    if not scores:
        return None, {}

    total_weight = sum(w for _, w in scores)
    weighted_sum = sum(s * w for s, w in scores)
    final_score  = round(weighted_sum / total_weight, 1)

    low_conf_slots = sum(1 for b in slot_breakdown if b.get("split_conf") == "LOW")
    grade = ("DANGEROUS ⚡" if final_score >= 70 else "ABOVE AVG" if final_score >= 55
             else "AVERAGE" if final_score >= 40 else "WEAK" if final_score >= 25
             else "VERY WEAK 🎯")
    if low_conf_slots >= 5:
        grade += " ⚠low splits"

    return final_score, {"grade": grade, "breakdown": slot_breakdown}


# ============================================================
# PLATOON-WEIGHTED K% HELPER
# ============================================================

def _platoon_weighted_k(overall_k, k_vl, k_vr, lhh_count, rhh_count):
    total = lhh_count + rhh_count
    if total == 0 or (k_vl is None and k_vr is None):
        return overall_k, "overall K% (no split data)"
    effective_vl = k_vl if k_vl is not None else overall_k
    effective_vr = k_vr if k_vr is not None else overall_k
    weighted = (rhh_count * effective_vr + lhh_count * effective_vl) / total
    note = (f"platoon-weighted: {lhh_count}×L({effective_vl}%) + "
            f"{rhh_count}×R({effective_vr}%) = {weighted:.1f}%")
    return round(weighted, 1), note


# ============================================================
# BETTING SIGNALS
# ============================================================

def attach_betting_signals(card):
    # TBD/unknown SP detection — suppress F5 and NRFI when either SP has no data
    hp_is_tbd = (card.get("home_pitcher", "").strip().upper() == "TBD" or
                 (card.get("home_pitcher_xwOBA") is None and
                  card.get("home_pitcher_K%") is None and
                  card.get("home_pitcher_AvgIP") is None))
    ap_is_tbd = (card.get("away_pitcher", "").strip().upper() == "TBD" or
                 (card.get("away_pitcher_xwOBA") is None and
                  card.get("away_pitcher_K%") is None and
                  card.get("away_pitcher_AvgIP") is None))

    hp_xwoba = card.get("home_pitcher_xwOBA") or 0.320
    ap_xwoba = card.get("away_pitcher_xwOBA") or 0.320
    hp_ip    = card.get("home_pitcher_AvgIP") or 5.0
    ap_ip    = card.get("away_pitcher_AvgIP") or 5.0
    hp_tto   = card.get("home_pitcher_TTO_xwOBA") or 0.310
    ap_tto   = card.get("away_pitcher_TTO_xwOBA") or 0.310
    away_vuln = card.get("away_lineup_vuln_score") or 50
    home_vuln = card.get("home_lineup_vuln_score") or 50

    hp_k_raw = card.get("home_pitcher_K%") or 20.0
    ap_k_raw = card.get("away_pitcher_K%") or 20.0

    hp_k, hp_k_note = _platoon_weighted_k(
        overall_k = hp_k_raw,
        k_vl      = card.get("home_pitcher_K%_vL"),
        k_vr      = card.get("home_pitcher_K%_vR"),
        lhh_count = card.get("away_lineup_lhh", 0),
        rhh_count = card.get("away_lineup_rhh", 0),
    )
    ap_k, ap_k_note = _platoon_weighted_k(
        overall_k = ap_k_raw,
        k_vl      = card.get("away_pitcher_K%_vL"),
        k_vr      = card.get("away_pitcher_K%_vR"),
        lhh_count = card.get("home_lineup_lhh", 0),
        rhh_count = card.get("home_lineup_rhh", 0),
    )
    card["home_pitcher_K%_platoon"]      = hp_k
    card["home_pitcher_K%_platoon_note"] = hp_k_note
    card["away_pitcher_K%_platoon"]      = ap_k
    card["away_pitcher_K%_platoon_note"] = ap_k_note

    IP_MIN = 4.5

    def _ip_gate_both(signal_name):
        short = []
        if hp_ip < IP_MIN:
            short.append(f"{card['home_pitcher']} ({hp_ip} IP avg)")
        if ap_ip < IP_MIN:
            short.append(f"{card['away_pitcher']} ({ap_ip} IP avg)")
        if short:
            return True, f"IP gate: {' + '.join(short)} below {IP_MIN} inn minimum"
        return False, ""

    def _ip_gate_single(pitcher_name, ip_val):
        if ip_val < IP_MIN:
            return True, f"IP gate: {pitcher_name} ({ip_val} IP avg) below {IP_MIN} inn minimum"
        return False, ""

    signals = {}

    # ── ML signal ────────────────────────────────────────────────────────
    ml_suppressed, ml_ip_note = _ip_gate_both("ML")
    if ml_suppressed:
        signals["ML_lean"] = "SUPPRESSED"
        signals["ML_conf"] = "N/A"
        signals["ML_ip_note"] = ml_ip_note
    else:
        home_edge = (away_vuln - 50) * 0.4 + (0.320 - ap_xwoba) * 200
        away_edge = (home_vuln - 50) * 0.4 + (0.320 - hp_xwoba) * 200
        if abs(home_edge - away_edge) < 5:
            signals["ML_lean"] = "NEUTRAL"; signals["ML_conf"] = "LOW"
        elif home_edge > away_edge:
            signals["ML_lean"] = f"HOME ({card['home_team']})"
            signals["ML_conf"] = "MED" if home_edge - away_edge < 15 else "HIGH"
        else:
            signals["ML_lean"] = f"AWAY ({card['away_team']})"
            signals["ML_conf"] = "MED" if away_edge - home_edge < 15 else "HIGH"

        xwoba_diff = hp_xwoba - ap_xwoba
        ml_lean    = signals["ML_lean"]
        if xwoba_diff >= 0.030 and "HOME" in ml_lean:
            signals["ML_sanity_note"] = (
                f"⚠ ML leans HOME but home SP xwOBA {hp_xwoba} "
                f"is {xwoba_diff:.3f} worse than away SP {ap_xwoba} — "
                f"driven by lineup differential"
            )
        elif xwoba_diff <= -0.030 and "AWAY" in ml_lean:
            signals["ML_sanity_note"] = (
                f"⚠ ML leans AWAY but away SP xwOBA {ap_xwoba} "
                f"is {abs(xwoba_diff):.3f} worse than home SP {hp_xwoba} — "
                f"driven by lineup differential"
            )

    # ── Total signal ──────────────────────────────────────────────────────
    total_suppressed, total_ip_note = _ip_gate_both("Total")
    if total_suppressed:
        signals["total_lean"] = "SUPPRESSED"
        signals["total_conf"] = "N/A"
        signals["total_ip_note"] = total_ip_note
    else:
        avg_px = (hp_xwoba + ap_xwoba) / 2
        avg_lv = (away_vuln + home_vuln) / 2
        if avg_px >= 0.340 and avg_lv >= 60:   signals["total_lean"]="OVER";    signals["total_conf"]="HIGH"
        elif avg_px >= 0.330 or avg_lv >= 55:  signals["total_lean"]="OVER";    signals["total_conf"]="MED"
        elif avg_px <= 0.290 and avg_lv <= 40: signals["total_lean"]="UNDER";   signals["total_conf"]="HIGH"
        elif avg_px <= 0.300 or avg_lv <= 45:  signals["total_lean"]="UNDER";   signals["total_conf"]="MED"
        else:                                  signals["total_lean"]="NEUTRAL";  signals["total_conf"]="LOW"

    # ── NRFI signal ───────────────────────────────────────────────────────
    nrfi_suppressed, nrfi_ip_note = _ip_gate_both("NRFI")
    if not nrfi_suppressed and (hp_is_tbd or ap_is_tbd):
        nrfi_suppressed = True
        tbd_name = card["home_pitcher"] if hp_is_tbd else card["away_pitcher"]
        nrfi_ip_note = f"TBD gate: {tbd_name} has no Statcast data — NRFI needs both SPs profiled"
    if nrfi_suppressed:
        signals["nrfi_lean"] = "SUPPRESSED"
        signals["nrfi_conf"] = "N/A"
        signals["nrfi_ip_note"] = nrfi_ip_note
    else:
        nrfi_score = 0
        if hp_tto <= 0.280: nrfi_score += 2
        elif hp_tto <= 0.300: nrfi_score += 1
        elif hp_tto <= 0.310: nrfi_score += 1   # loosened: catches more games
        if ap_tto <= 0.280: nrfi_score += 2
        elif ap_tto <= 0.300: nrfi_score += 1
        elif ap_tto <= 0.310: nrfi_score += 1   # loosened: catches more games
        if nrfi_score >= 4:   signals["nrfi_lean"]="STRONG NRFI"; signals["nrfi_conf"]="HIGH"
        elif nrfi_score >= 2: signals["nrfi_lean"]="LEAN NRFI";   signals["nrfi_conf"]="MED"
        elif nrfi_score >= 1: signals["nrfi_lean"]="LEAN NRFI";   signals["nrfi_conf"]="MED"  # loosened: 1 point now qualifies
        else:                 signals["nrfi_lean"]="NEUTRAL / AVOID"; signals["nrfi_conf"]="LOW"

    # ── F5 signal ─────────────────────────────────────────────────────────
    hp_f5_supp, hp_f5_note = _ip_gate_single(card["home_pitcher"], hp_ip)
    ap_f5_supp, ap_f5_note = _ip_gate_single(card["away_pitcher"], ap_ip)
    # TBD gate: suppress F5 entirely when either SP is unknown
    if hp_is_tbd:
        hp_f5_supp = True
        hp_f5_note = f"TBD gate: {card['home_pitcher']} has no Statcast data — F5 needs both SPs profiled"
    if ap_is_tbd:
        ap_f5_supp = True
        ap_f5_note = f"TBD gate: {card['away_pitcher']} has no Statcast data — F5 needs both SPs profiled"

    if hp_f5_supp and ap_f5_supp:
        signals["f5_lean"] = "SUPPRESSED"
        signals["f5_conf"] = "N/A"
        signals["f5_ip_note"] = f"{hp_f5_note} | {ap_f5_note}"
    elif hp_f5_supp and not ap_f5_supp:
        if ap_xwoba <= 0.300:
            signals["f5_lean"] = f"AWAY F5 ({card['away_team']})"
            signals["f5_conf"] = "MED"
            signals["f5_ip_note"] = f"⚠ {hp_f5_note} — F5 based on away SP only"
        else:
            signals["f5_lean"] = "SUPPRESSED"
            signals["f5_conf"] = "N/A"
            signals["f5_ip_note"] = f"{hp_f5_note} — away SP not strong enough to lean alone"
    elif ap_f5_supp and not hp_f5_supp:
        if hp_xwoba <= 0.300:
            signals["f5_lean"] = f"HOME F5 ({card['home_team']})"
            signals["f5_conf"] = "MED"
            signals["f5_ip_note"] = f"⚠ {ap_f5_note} — F5 based on home SP only"
        else:
            signals["f5_lean"] = "SUPPRESSED"
            signals["f5_conf"] = "N/A"
            signals["f5_ip_note"] = f"{ap_f5_note} — home SP not strong enough to lean alone"
    else:
        f5_home_edge = (away_vuln - 50) * 0.2 + (0.320 - ap_xwoba) * 250
        f5_away_edge = (home_vuln - 50) * 0.2 + (0.320 - hp_xwoba) * 250

        if hp_xwoba <= 0.295 and ap_xwoba <= 0.295:
            signals["f5_lean"] = "UNDER"; signals["f5_conf"] = "HIGH"
        elif hp_xwoba <= 0.305 and ap_xwoba <= 0.305:
            signals["f5_lean"] = "UNDER"; signals["f5_conf"] = "MED"
        elif abs(f5_home_edge - f5_away_edge) < 4:
            signals["f5_lean"] = "NEUTRAL"; signals["f5_conf"] = "LOW"
        elif f5_home_edge > f5_away_edge:
            diff = f5_home_edge - f5_away_edge
            signals["f5_lean"] = f"HOME F5 ({card['home_team']})"
            signals["f5_conf"] = "HIGH" if diff >= 18 else "MED"
        else:
            diff = f5_away_edge - f5_home_edge
            signals["f5_lean"] = f"AWAY F5 ({card['away_team']})"
            signals["f5_conf"] = "HIGH" if diff >= 18 else "MED"

        ml_lean = signals.get("ML_lean", "NEUTRAL")
        f5_lean = signals.get("f5_lean", "NEUTRAL")
        if ml_lean not in ("NEUTRAL", "SUPPRESSED") and f5_lean not in ("NEUTRAL", "UNDER", "SUPPRESSED"):
            ml_home = "HOME" in ml_lean
            f5_home = "HOME" in f5_lean
            if ml_home != f5_home:
                signals["f5_lean"] = "NEUTRAL"
                signals["f5_conf"] = "LOW"
                signals["f5_conflict_note"] = "F5 conflicts with ML lean — neutralised"

    # ── K-prop targets ────────────────────────────────────────────────────
    k_props = []
    if hp_k >= 23.0 and hp_ip >= IP_MIN:
        k_props.append(f"{card['home_pitcher']} (platoon K% {hp_k}% — {hp_k_note})")
    if ap_k >= 23.0 and ap_ip >= IP_MIN:
        k_props.append(f"{card['away_pitcher']} (platoon K% {ap_k}% — {ap_k_note})")
    signals["k_prop_targets"] = k_props if k_props else ["None"]

    # ── Fade notes ────────────────────────────────────────────────────────
    fade_notes = []
    if hp_xwoba and hp_xwoba >= 0.340 and away_vuln >= 60:
        fade_notes.append(f"FADE {card['home_team']} — weak home SP ({hp_xwoba} xwOBA) vs dangerous away lineup (score {away_vuln})")
    if ap_xwoba and ap_xwoba >= 0.340 and home_vuln >= 60:
        fade_notes.append(f"FADE {card['away_team']} — weak away SP ({ap_xwoba} xwOBA) vs dangerous home lineup (score {home_vuln})")
    signals["fade_notes"] = fade_notes if fade_notes else ["No strong fade signal"]

    # ── Fade vs F5 conflict guard ─────────────────────────────────────────
    f5_lean_now = signals.get("f5_lean", "NEUTRAL")
    if f5_lean_now not in ("NEUTRAL", "SUPPRESSED", "UNDER") and fade_notes:
        for fn in fade_notes:
            faded_team = fn.split("FADE ")[1].split(" —")[0].strip()
            f5_is_away = "AWAY" in f5_lean_now
            f5_team = card["away_team"] if f5_is_away else card["home_team"]
            if faded_team == f5_team:
                signals["f5_lean"] = "SUPPRESSED"
                signals["f5_conf"] = "N/A"
                signals["f5_ip_note"] = (signals.get("f5_ip_note") or "") + \
                    f" | F5 suppressed — conflicts with FADE signal on {faded_team}"
                break

    # ── Fade vs ML conflict guard ─────────────────────────────────────────
    ml_lean_now = signals.get("ML_lean", "NEUTRAL")
    if ml_lean_now not in ("NEUTRAL", "SUPPRESSED") and fade_notes:
        for fn in fade_notes:
            faded_team = fn.split("FADE ")[1].split(" —")[0].strip()
            ml_is_away = "AWAY" in ml_lean_now
            ml_team = card["away_team"] if ml_is_away else card["home_team"]
            if faded_team == ml_team:
                signals["ML_lean"] = "SUPPRESSED"
                signals["ML_conf"] = "N/A"
                signals["ML_ip_note"] = (signals.get("ML_ip_note") or "") + \
                    f" | ML suppressed — conflicts with FADE signal on {faded_team}"
                break

    # ── TEAM_TOTAL signal (fade pitcher → opposition team total OVER) ─────────
    # Fires when either SP qualifies as a fade target (xwOBA ≥ 0.330, HH% ≥ 28)
    fade_threshold_xwoba = 0.330
    fade_threshold_hh    = 28.0

    hp_hh = card.get("home_pitcher_HardHit%") or 0
    ap_hh = card.get("away_pitcher_HardHit%") or 0

    team_total_signals = []

    # Home SP is a fade target → AWAY team total OVER
    if (hp_xwoba >= fade_threshold_xwoba and hp_hh >= fade_threshold_hh
            and hp_ip >= 3.5 and not hp_is_tbd):
        team_total_signals.append({
            "lean": f"AWAY TEAM OVER (opp: {card['home_pitcher']} xwOBA {hp_xwoba:.3f} HH% {hp_hh}%)",
            "conf": "HIGH" if hp_xwoba >= 0.370 else "MED",
            "side": "AWAY",
        })

    # Away SP is a fade target → HOME team total OVER
    if (ap_xwoba >= fade_threshold_xwoba and ap_hh >= fade_threshold_hh
            and ap_ip >= 3.5 and not ap_is_tbd):
        team_total_signals.append({
            "lean": f"HOME TEAM OVER (opp: {card['away_pitcher']} xwOBA {ap_xwoba:.3f} HH% {ap_hh}%)",
            "conf": "HIGH" if ap_xwoba >= 0.370 else "MED",
            "side": "HOME",
        })

    signals["team_total_signals"] = team_total_signals

    # ── BB_PROP signals ────────────────────────────────────────────────────────
    # OVER walks: fade targets with BB% ≥ 7.0 and IP ≥ 3.5
    # UNDER walks: K-BB% pitchers with BB% ≤ 7.0, K% ≥ 25, K-BB% ≥ 14
    bb_prop_signals = []

    for side, pitcher_name, bb_pct, k_pct, kbb_pct, sp_ip, sp_xwoba, is_tbd in [
        ("home", card.get("home_pitcher",""), card.get("home_pitcher_BB%") or 0,
         card.get("home_pitcher_K%") or 0, 0, hp_ip, hp_xwoba, hp_is_tbd),
        ("away", card.get("away_pitcher",""), card.get("away_pitcher_BB%") or 0,
         card.get("away_pitcher_K%") or 0, 0, ap_ip, ap_xwoba, ap_is_tbd),
    ]:
        if is_tbd or not pitcher_name or sp_ip < 3.5:
            continue
        kbb = k_pct - bb_pct

        # OVER walks: high BB%, hittable pitcher (fade target profile)
        if bb_pct >= 7.0 and sp_xwoba >= 0.330 and sp_ip >= 3.5:
            bb_prop_signals.append({
                "lean": f"{pitcher_name} WALKS OVER ({bb_pct}% BB, {sp_ip:.1f}ip)",
                "conf": "HIGH" if bb_pct >= 10 else "MED",
                "side": side,
                "direction": "OVER",
                "pitcher": pitcher_name,
            })

        # UNDER walks: elite control pitcher (K-BB% target profile)
        if bb_pct <= 7.0 and k_pct >= 25.0 and kbb >= 14.0 and sp_ip >= 4.5:
            bb_prop_signals.append({
                "lean": f"{pitcher_name} WALKS UNDER ({bb_pct}% BB, K-BB% {kbb:.1f})",
                "conf": "HIGH" if bb_pct <= 5.0 else "MED",
                "side": side,
                "direction": "UNDER",
                "pitcher": pitcher_name,
            })

    signals["bb_prop_signals"] = bb_prop_signals

    card["signals"] = signals
    return card


# ============================================================
# COMPOSITE PLAY QUALITY SCORE  (mirrors dashboard Strict Mode)
# ============================================================

def score_play_quality(card, bet_type, lean=None):
    """
    Returns a 0-100 composite quality score for a bet type.
    Requires convergence of multiple signals to score high.
    Mirrors the dashboard's scorePlayQuality() JS function so the
    Python log and dashboard Strict Mode agree on quality.

    bet_type: one of "ML", "TOTAL", "F5", "F5_TOTAL", "NRFI", "K_PROP"
    lean:     the signal lean string (e.g. "HOME (Yankees)", "OVER",
              the K-Prop target string). Required for ML/F5/TOTAL/K_PROP.
    """
    s         = card.get("signals", {})
    hp_xwoba  = card.get("home_pitcher_xwOBA") or 0.320
    ap_xwoba  = card.get("away_pitcher_xwOBA") or 0.320
    hp_ip     = card.get("home_pitcher_AvgIP") or 5.0
    ap_ip     = card.get("away_pitcher_AvgIP") or 5.0
    away_vuln = card.get("away_lineup_vuln_score") or 50
    home_vuln = card.get("home_lineup_vuln_score") or 50
    hbp_score = card.get("home_bullpen", {}).get("bullpen_score") or 50
    abp_score = card.get("away_bullpen", {}).get("bullpen_score") or 50
    env_adj   = card.get("environment", {}).get("env_total_adj", 0)
    score     = 0

    if bet_type == "ML":
        ml_lean   = lean or s.get("ML_lean", "")
        lean_home = "HOME" in ml_lean
        sp_xwoba  = ap_xwoba if lean_home else hp_xwoba   # SP facing the team we're backing
        bp_score  = hbp_score if lean_home else abp_score
        vuln      = home_vuln if lean_home else away_vuln
        if sp_xwoba <= 0.295: score += 30
        elif sp_xwoba <= 0.310: score += 15
        if bp_score >= 65: score += 25
        elif bp_score >= 55: score += 12
        if vuln >= 60: score += 20
        elif vuln >= 52: score += 10
        if hp_ip >= 5.5 and ap_ip >= 5.5: score += 15
        if s.get("ML_conf") == "HIGH": score += 10
        opp_xwoba = hp_xwoba if lean_home else ap_xwoba
        if opp_xwoba <= 0.300: score -= 20

    elif bet_type == "TOTAL":
        total_lean = lean or s.get("total_lean", "")
        avg_xwoba  = (hp_xwoba + ap_xwoba) / 2
        avg_vuln   = (away_vuln + home_vuln) / 2
        vr         = s.get("total_value_rating", "") or ""
        if total_lean == "OVER":
            if avg_xwoba >= 0.340: score += 30
            elif avg_xwoba >= 0.330: score += 15
            if avg_vuln >= 60: score += 20
            elif avg_vuln >= 55: score += 10
            if env_adj >= 2.0: score += 20
            elif env_adj >= 1.0: score += 10
            if "HIGH VALUE" in vr: score += 20
            elif "FAIR" in vr: score += 10
        elif total_lean == "UNDER":
            if avg_xwoba <= 0.290: score += 30
            elif avg_xwoba <= 0.300: score += 15
            if avg_vuln <= 40: score += 20
            elif avg_vuln <= 45: score += 10
            if env_adj <= -2.0: score += 20
            elif env_adj <= -1.0: score += 10
            if "HIGH VALUE" in vr: score += 20
            elif "FAIR" in vr: score += 10
        if s.get("total_conf") == "HIGH": score += 10

    elif bet_type in ("F5", "F5_TOTAL"):
        f5_lean   = lean or s.get("f5_lean", "")
        lean_home = "HOME" in f5_lean
        sp_xwoba  = ap_xwoba if lean_home else hp_xwoba
        sp_ip     = ap_ip   if lean_home else hp_ip
        if sp_xwoba <= 0.290: score += 35
        elif sp_xwoba <= 0.305: score += 18
        if sp_ip >= 6.0: score += 20
        elif sp_ip >= 5.5: score += 10
        if s.get("f5_conf") == "HIGH": score += 25
        vuln = home_vuln if lean_home else away_vuln
        if vuln >= 58: score += 15
        elif vuln >= 50: score += 7
        if f5_lean == "UNDER" and (hp_xwoba <= 0.295 and ap_xwoba <= 0.295): score += 10

    elif bet_type == "NRFI":
        hp_tto = card.get("home_pitcher_TTO_xwOBA") or 0.310
        ap_tto = card.get("away_pitcher_TTO_xwOBA") or 0.310
        if hp_tto <= 0.270 and ap_tto <= 0.270: score += 50
        elif hp_tto <= 0.280 and ap_tto <= 0.280: score += 35
        elif hp_tto <= 0.290 and ap_tto <= 0.290: score += 20   # loosened: was 15
        elif hp_tto <= 0.300 and ap_tto <= 0.300: score += 12   # new tier
        elif hp_tto <= 0.310 or ap_tto <= 0.310: score += 8     # new tier — one good SP enough
        if s.get("nrfi_conf") == "HIGH": score += 20
        if s.get("nrfi_conf") == "MED":  score += 10            # new: MED also gets a bump
        if env_adj <= -1.0: score += 15
        elif env_adj >= 2.0: score -= 20

    elif bet_type == "K_PROP":
        kp_text = lean or ""
        m = re.search(r'platoon K%\s*([\d.]+)%', kp_text)
        kp_k = float(m.group(1)) if m else (
            card.get("home_pitcher_K%_platoon") or card.get("away_pitcher_K%_platoon") or 0
        )
        # Determine which pitcher this K_PROP is for (home or away)
        kp_is_home = card.get("home_pitcher", "") in kp_text
        kp_xwoba = (card.get("home_pitcher_xwOBA") if kp_is_home else card.get("away_pitcher_xwOBA")) or 0.320
        kp_ip    = (hp_ip if kp_is_home else ap_ip)

        # K% tiers
        if kp_k >= 30:   score += 50
        elif kp_k >= 27: score += 30
        elif kp_k >= 25: score += 20
        elif kp_k >= 23: score += 10

        # IP tiers — based on this pitcher's actual IP, not max
        if kp_ip >= 6.0:   score += 25
        elif kp_ip >= 5.5: score += 15
        elif kp_ip >= 5.0: score += 10
        elif kp_ip >= 4.5: score += 5

        # xwOBA bonus/penalty — K_PROP needs a dominant pitcher
        if kp_xwoba <= 0.280:   score += 15
        elif kp_xwoba <= 0.295: score += 8
        elif kp_xwoba >= 0.320: score -= 10  # penalty: pitcher is too hittable
        elif kp_xwoba >= 0.310: score -= 5

        # Environment penalty — suppress K_PROP in hitter-friendly conditions
        if env_adj >= 3.0:  score -= 20  # e.g. Oracle Park blowing out
        elif env_adj >= 1.5: score -= 10

        # Suppress K_PROP when TOTAL signal is OVER (contradictory)
        total_lean = s.get("total_lean", "NEUTRAL")
        if "OVER" in str(total_lean): score -= 20

    elif bet_type == "TEAM_TOTAL":
        # Score a team-total OVER bet driven by a fade-worthy opposing pitcher
        # lean = "AWAY TEAM OVER (opp: PitcherName)" or "HOME TEAM OVER (opp: PitcherName)"
        lean_str   = lean or ""
        is_away    = "AWAY" in lean_str
        opp_xwoba  = hp_xwoba if is_away else ap_xwoba   # pitcher facing the team we back
        opp_ip     = hp_ip    if is_away else ap_ip
        opp_hh     = (card.get("home_pitcher_HardHit%") if is_away
                      else card.get("away_pitcher_HardHit%")) or 28
        our_vuln   = away_vuln if is_away else home_vuln

        # Fade pitcher quality — higher xwOBA = more confidence in team total OVER
        if opp_xwoba >= 0.380:   score += 40
        elif opp_xwoba >= 0.360: score += 30
        elif opp_xwoba >= 0.340: score += 20
        elif opp_xwoba >= 0.320: score += 10

        # Short IP = fewer outs before bullpen, more run risk
        if opp_ip <= 3.0:   score += 25
        elif opp_ip <= 4.0: score += 15
        elif opp_ip <= 4.5: score += 8

        # Hard-hit % against the pitcher — higher = more contact quality
        if opp_hh >= 35: score += 15
        elif opp_hh >= 30: score += 8

        # Batting team vulnerability (higher = team hits well)
        if our_vuln >= 60: score += 15
        elif our_vuln >= 52: score += 8

        # Environment boost
        if env_adj >= 2.0: score += 10
        elif env_adj >= 1.0: score += 5

    elif bet_type == "BB_PROP":
        # Score a walks OVER (fade pitcher) or UNDER (K-BB% pitcher)
        lean_str   = lean or ""
        is_over    = "OVER" in lean_str
        # Extract pitcher name from lean string format "PitcherName WALKS OVER/UNDER X.X"
        # Use the relevant pitcher's BB% data
        if is_over:
            # Fade target — high BB% pitcher, backing walks OVER
            pitcher_bb = (card.get("home_pitcher_BB%") if "home" in lean_str.lower()
                          else card.get("away_pitcher_BB%")) or 0
            if pitcher_bb >= 11: score += 50
            elif pitcher_bb >= 9: score += 35
            elif pitcher_bb >= 7: score += 20
            # Short IP reduces walk opportunities — penalise
            relevant_ip = hp_ip if "home" in lean_str.lower() else ap_ip
            if relevant_ip <= 3.5: score -= 20
            elif relevant_ip <= 4.0: score -= 10
            # xwOBA context — fade pitchers tend to be hittable, not necessarily walk-prone
            relevant_xwoba = (hp_xwoba if "home" in lean_str.lower() else ap_xwoba) or 0.320
            if relevant_xwoba >= 0.360: score += 10
        else:
            # K-BB% target — low BB% pitcher, backing walks UNDER
            pitcher_bb = (card.get("home_pitcher_BB%") if "home" in lean_str.lower()
                          else card.get("away_pitcher_BB%")) or 99
            if pitcher_bb <= 5:   score += 50
            elif pitcher_bb <= 6: score += 35
            elif pitcher_bb <= 7: score += 25
            elif pitcher_bb <= 8: score += 15
            # Long IP helps — more innings means more chances to maintain control
            relevant_ip = hp_ip if "home" in lean_str.lower() else ap_ip
            if relevant_ip >= 5.5: score += 15
            elif relevant_ip >= 5.0: score += 8

    return max(0, min(100, score))


# Minimum composite scores to qualify as a logged bet
QUALITY_THRESHOLDS = {
    "ML":         58,   # +3 — tighten ML signal
    "TOTAL":      68,   # +3 — TOTAL has gone cold, raise bar
    "F5":         56,   # +3 — reduce marginal F5 triggers
    "F5_TOTAL":   50,   # +5 — was too loose, correlated with TOTAL losses
    "NRFI":       45,   # +5 — NRFI sample too small to be loose
    "K_PROP":     40,   # +10 — consistently underperforming, biggest tighten
    "TEAM_TOTAL": 40,   # +5 — new market, be conservative
    "BB_PROP":    35,   # +5 — raise slightly, too many marginal fires
}

# ── Kelly staking constants ───────────────────────────────────────────────────
KELLY_FRACTION   = 0.25   # Quarter-Kelly (conservative, standard for sports betting)
MAX_KELLY_UNITS  = 5.0    # Hard per-bet ceiling in units
DAILY_STAKE_CAP  = 20.0   # Max total units across all bets in one day before scaling
MAX_DAILY_PLAYS  = 10     # Hard cap on number of bets logged per day — keep top N by quality
MAX_BETS_PER_GAME = 2    # Hard cap on bets per matchup — prevents correlated stacking
UNIT_SIZE_AUD    = 20.0   # $20 per unit
FLAT_STAKE_UNITS = 1.0    # Fallback when no odds available (NRFI, K_PROP)


# ============================================================
# LINEUP HANDEDNESS + BUILD MATCHUP CARDS
# ============================================================

def _count_lineup_handedness(lineup_with_profiles):
    lhh = rhh = 0
    for b in lineup_with_profiles:
        profile = b.get("profile")
        if not profile:
            continue
        has_vl = profile.get("xwOBA_vL") is not None
        has_vr = profile.get("xwOBA_vR") is not None
        if has_vl and not has_vr:
            lhh += 1
        elif has_vr and not has_vl:
            rhh += 1
        elif has_vl and has_vr:
            rhh += 1
    return lhh, rhh


def build_matchup_cards(pitcher_df, matchups):
    pitcher_lookup = {}
    if pitcher_df is not None and not pitcher_df.empty:
        for _, row in pitcher_df.iterrows():
            pitcher_lookup[row["Pitcher"]] = row.to_dict()

    game_cards = []
    for game in matchups:
        print(f"\n{'='*60}\n  {game['away_team']} @ {game['home_team']}")
        print(f"  Away SP: {game['away_pitcher']}  |  Home SP: {game['home_pitcher']}\n{'='*60}")

        card = {"game_pk": game["game_pk"],
                "matchup": f"{game['away_team']} @ {game['home_team']}",
                "home_team": game["home_team"], "away_team": game["away_team"],
                "home_pitcher": game["home_pitcher"], "away_pitcher": game["away_pitcher"]}

        hp_stats = pitcher_lookup.get(game["home_pitcher"])
        card["home_pitcher_xwOBA"]     = hp_stats["xwOBA"]        if hp_stats else None
        card["home_pitcher_K%"]        = hp_stats["K%"]           if hp_stats else None
        card["home_pitcher_K%_vL"]     = hp_stats.get("K%_vL")    if hp_stats else None
        card["home_pitcher_K%_vR"]     = hp_stats.get("K%_vR")    if hp_stats else None
        card["home_pitcher_BB%"]       = hp_stats["BB%"]          if hp_stats else None
        card["home_pitcher_HH%"]       = hp_stats["HardHit%"]     if hp_stats else None
        card["home_pitcher_hand"]      = hp_stats["Hand"]         if hp_stats else "R"
        card["home_pitcher_TTO_xwOBA"] = hp_stats["TTO_xwOBA"]   if hp_stats else None
        card["home_pitcher_AvgIP"]     = hp_stats["Avg_IP_Start"] if hp_stats else None

        ap_stats = pitcher_lookup.get(game["away_pitcher"])
        card["away_pitcher_xwOBA"]     = ap_stats["xwOBA"]        if ap_stats else None
        card["away_pitcher_K%"]        = ap_stats["K%"]           if ap_stats else None
        card["away_pitcher_K%_vL"]     = ap_stats.get("K%_vL")    if ap_stats else None
        card["away_pitcher_K%_vR"]     = ap_stats.get("K%_vR")    if ap_stats else None
        card["away_pitcher_BB%"]       = ap_stats["BB%"]          if ap_stats else None
        card["away_pitcher_HH%"]       = ap_stats["HardHit%"]     if ap_stats else None
        card["away_pitcher_hand"]      = ap_stats["Hand"]         if ap_stats else "R"
        card["away_pitcher_TTO_xwOBA"] = ap_stats["TTO_xwOBA"]   if ap_stats else None
        card["away_pitcher_AvgIP"]     = ap_stats["Avg_IP_Start"] if ap_stats else None

        src_away = game.get("lineup_source_away", "?")
        src_home = game.get("lineup_source_home", "?")
        print(f"  Profiling away lineup ({len(game['away_lineup'])} batters — {src_away})...")
        away_lineup_with_profiles = []
        for b in game["away_lineup"]:
            print(f"    > {b['name']}")
            profile = batter_profile(b["name"], b.get("batter_id"))
            away_lineup_with_profiles.append({**b, "profile": profile})

        print(f"  Profiling home lineup ({len(game['home_lineup'])} batters — {src_home})...")
        home_lineup_with_profiles = []
        for b in game["home_lineup"]:
            print(f"    > {b['name']}")
            profile = batter_profile(b["name"], b.get("batter_id"))
            home_lineup_with_profiles.append({**b, "profile": profile})

        away_lhh, away_rhh = _count_lineup_handedness(away_lineup_with_profiles)
        home_lhh, home_rhh = _count_lineup_handedness(home_lineup_with_profiles)
        card["away_lineup_lhh"] = away_lhh
        card["away_lineup_rhh"] = away_rhh
        card["home_lineup_lhh"] = home_lhh
        card["home_lineup_rhh"] = home_rhh
        print(f"  Away lineup handedness: {away_lhh} LHH / {away_rhh} RHH")
        print(f"  Home lineup handedness: {home_lhh} LHH / {home_rhh} RHH")

        away_vuln_score, away_vuln_detail = lineup_vulnerability_score(away_lineup_with_profiles, card["home_pitcher_hand"])
        home_vuln_score, home_vuln_detail = lineup_vulnerability_score(home_lineup_with_profiles, card["away_pitcher_hand"])

        card["away_lineup_vuln_score"] = away_vuln_score
        card["away_lineup_grade"]      = away_vuln_detail.get("grade")
        card["away_lineup_breakdown"]  = away_vuln_detail.get("breakdown", [])
        card["home_lineup_vuln_score"] = home_vuln_score
        card["home_lineup_grade"]      = home_vuln_detail.get("grade")
        card["home_lineup_breakdown"]  = home_vuln_detail.get("breakdown", [])
        card = attach_betting_signals(card)
        game_cards.append(card)
    return game_cards

def print_game_card(card):
    s = card.get("signals", {})

    def _fmt(label, lean, conf, ip_note_key=None):
        ip_note = s.get(ip_note_key, "") if ip_note_key else ""
        if lean == "SUPPRESSED":
            return f"  {label:<18} {'— SUPPRESSED —':<30} [{conf}]  ← {ip_note}"
        if conf == "LOW":
            return f"  {label:<18} {lean:<30} [LOW — not logged]"
        line = f"  {label:<18} {lean:<30} [{conf}]"
        if ip_note:
            line += f"  ← {ip_note}"
        return line

    print(f"\n{'█'*62}\n  GAME:  {card['matchup']}\n{'─'*62}")
    print(f"  AWAY SP : {card['away_pitcher']:<25} xwOBA {card.get('away_pitcher_xwOBA','?')}  K% {card.get('away_pitcher_K%','?')}  AvgIP {card.get('away_pitcher_AvgIP','?')}  HH% {card.get('away_pitcher_HH%','?')}")
    print(f"  HOME SP : {card['home_pitcher']:<25} xwOBA {card.get('home_pitcher_xwOBA','?')}  K% {card.get('home_pitcher_K%','?')}  AvgIP {card.get('home_pitcher_AvgIP','?')}  HH% {card.get('home_pitcher_HH%','?')}")
    if card.get("away_pitcher_K%_platoon") is not None:
        print(f"           Away platoon K% (vs home lineup): {card['away_pitcher_K%_platoon']}%  ← {card.get('away_pitcher_K%_platoon_note','')}")
    if card.get("home_pitcher_K%_platoon") is not None:
        print(f"           Home platoon K% (vs away lineup): {card['home_pitcher_K%_platoon']}%  ← {card.get('home_pitcher_K%_platoon_note','')}")
    print(f"{'─'*62}")
    print(f"  AWAY LINEUP vs {card['home_pitcher_hand']}HP  →  Vuln Score: {card.get('away_lineup_vuln_score','?')}  {card.get('away_lineup_grade','')}")
    print(f"  HOME LINEUP vs {card['away_pitcher_hand']}HP  →  Vuln Score: {card.get('home_lineup_vuln_score','?')}  {card.get('home_lineup_grade','')}")
    print(f"{'─'*62}")
    print(_fmt("ML LEAN",    s.get("ML_lean","?"),    s.get("ML_conf","?"),    "ML_ip_note"))
    if s.get("ML_sanity_note"):
        print(f"  {'':18} {s['ML_sanity_note']}")
    ml_h = card.get("odds_ml_home"); ml_a = card.get("odds_ml_away")
    if ml_h or ml_a:
        hi = card.get("odds_ml_home_implied","?"); ai = card.get("odds_ml_away_implied","?")
        print(f"  {'ML ODDS':<18} Home {ml_h} ({hi}%)  Away {ml_a} ({ai}%)"
              + (f"  ← {s['ml_value_note']}" if s.get("ml_value_note") else ""))
    print(_fmt("TOTAL LEAN", s.get("total_lean","?"), s.get("total_conf","?"), "total_ip_note"))
    tl = card.get("odds_total_line")
    if tl:
        op = card.get("odds_total_over_price","?"); up = card.get("odds_total_under_price","?")
        vr = s.get("total_value_rating",""); vn = s.get("total_value_note","")
        print(f"  {'TOTAL LINE':<18} {tl} (Over {op} / Under {up})  {vr}")
        if vn: print(f"  {'':18} {vn}")
    print(_fmt("F5 LEAN",    s.get("f5_lean","?"),    s.get("f5_conf","?"),    "f5_ip_note"))
    f5l = card.get("odds_f5_total_line")
    if f5l:
        f5h = card.get("odds_f5_ml_home","?"); f5a = card.get("odds_f5_ml_away","?")
        f5op = card.get("odds_f5_over_price","?"); f5up = card.get("odds_f5_under_price","?")
        print(f"  {'F5 ODDS':<18} Home ML {f5h}  Away ML {f5a}  Total {f5l} (O{f5op}/U{f5up})")
    print(_fmt("NRFI",       s.get("nrfi_lean","?"),  s.get("nrfi_conf","?"),  "nrfi_ip_note"))
    print(f"  {'K-PROP TARGETS':<18} {', '.join(s.get('k_prop_targets',['None']))}")
    tt_sigs = s.get('team_total_signals', [])
    tt_display = ' | '.join(f"{t['lean']} [{t['conf']}]" for t in tt_sigs) if tt_sigs else 'None'
    print(f"  {'TEAM TOTAL':<18} {tt_display}")
    bb_sigs = s.get('bb_prop_signals', [])
    bb_display = ' | '.join(f"{b['lean']} [{b['conf']}]" for b in bb_sigs) if bb_sigs else 'None'
    print(f"  {'BB PROP':<18} {bb_display}")
    print(f"{'─'*62}")
    for fn in s.get("fade_notes", []):
        print(f"  ⚠  {fn}")
    print(f"\n  AWAY LINEUP BREAKDOWN (vs {card['home_pitcher_hand']}HP):")
    for b in card.get("away_lineup_breakdown", []):
        conf_tag = f"[{b.get('split_conf','?')}]" if b.get('split_conf') else ""
        print(f"    {b['slot']}. {b['name']:<22} xwOBA {str(b.get('xwOBA','?')):<6} K% {str(b.get('K%','?')):<5} HH% {str(b.get('HH%','?')):<5} Score {b.get('score','?'):<5} {conf_tag:<6} {b.get('note','')}")
    print(f"\n  HOME LINEUP BREAKDOWN (vs {card['away_pitcher_hand']}HP):")
    for b in card.get("home_lineup_breakdown", []):
        conf_tag = f"[{b.get('split_conf','?')}]" if b.get('split_conf') else ""
        print(f"    {b['slot']}. {b['name']:<22} xwOBA {str(b.get('xwOBA','?')):<6} K% {str(b.get('K%','?')):<5} HH% {str(b.get('HH%','?')):<5} Score {b.get('score','?'):<5} {conf_tag:<6} {b.get('note','')}")
    print(f"{'█'*62}\n")

# ============================================================
# PHASE 3 — PARK FACTORS + WEATHER
# ============================================================

PARK_DATA = {
    "Arizona Diamondbacks":  {"park":"Chase Field","factor":1.03,"roof":"retractable","lat":33.4453,"lon":-112.0667,"orientation":"NE","elevation_ft":1082},
    "Atlanta Braves":        {"park":"Truist Park","factor":1.01,"roof":"open","lat":33.8908,"lon":-84.4678,"orientation":"NE","elevation_ft":1027},
    "Baltimore Orioles":     {"park":"Oriole Park at Camden Yards","factor":1.00,"roof":"open","lat":39.2838,"lon":-76.6218,"orientation":"NE","elevation_ft":43},
    "Boston Red Sox":        {"park":"Fenway Park","factor":1.07,"roof":"open","lat":42.3467,"lon":-71.0972,"orientation":"NE","elevation_ft":21},
    "Chicago Cubs":          {"park":"Wrigley Field","factor":1.05,"roof":"open","lat":41.9484,"lon":-87.6553,"orientation":"NE","elevation_ft":595},
    "Chicago White Sox":     {"park":"Guaranteed Rate Field","factor":1.02,"roof":"open","lat":41.8300,"lon":-87.6338,"orientation":"NW","elevation_ft":595},
    "Cincinnati Reds":       {"park":"Great American Ball Park","factor":1.07,"roof":"open","lat":39.0979,"lon":-84.5082,"orientation":"NW","elevation_ft":482},
    "Cleveland Guardians":   {"park":"Progressive Field","factor":0.97,"roof":"open","lat":41.4962,"lon":-81.6852,"orientation":"NW","elevation_ft":653},
    "Colorado Rockies":      {"park":"Coors Field","factor":1.28,"roof":"open","lat":39.7559,"lon":-104.9942,"orientation":"NE","elevation_ft":5200},
    "Detroit Tigers":        {"park":"Comerica Park","factor":0.94,"roof":"open","lat":42.3390,"lon":-83.0485,"orientation":"NE","elevation_ft":583},
    "Houston Astros":        {"park":"Minute Maid Park","factor":0.99,"roof":"retractable","lat":29.7573,"lon":-95.3555,"orientation":"NW","elevation_ft":43},
    "Kansas City Royals":    {"park":"Kauffman Stadium","factor":0.96,"roof":"open","lat":39.0517,"lon":-94.4803,"orientation":"NE","elevation_ft":1014},
    "Los Angeles Angels":    {"park":"Angel Stadium","factor":0.97,"roof":"open","lat":33.8003,"lon":-117.8827,"orientation":"NE","elevation_ft":160},
    "Los Angeles Dodgers":   {"park":"Dodger Stadium","factor":0.96,"roof":"open","lat":34.0739,"lon":-118.2400,"orientation":"NW","elevation_ft":512},
    "Miami Marlins":         {"park":"loanDepot park","factor":0.92,"roof":"retractable","lat":25.7781,"lon":-80.2197,"orientation":"NE","elevation_ft":6},
    "Milwaukee Brewers":     {"park":"American Family Field","factor":1.01,"roof":"retractable","lat":43.0280,"lon":-87.9712,"orientation":"NE","elevation_ft":635},
    "Minnesota Twins":       {"park":"Target Field","factor":0.97,"roof":"open","lat":44.9817,"lon":-93.2781,"orientation":"NE","elevation_ft":830},
    "New York Mets":         {"park":"Citi Field","factor":0.95,"roof":"open","lat":40.7571,"lon":-73.8458,"orientation":"NE","elevation_ft":23},
    "New York Yankees":      {"park":"Yankee Stadium","factor":1.06,"roof":"open","lat":40.8296,"lon":-73.9262,"orientation":"NE","elevation_ft":55},
    "Oakland Athletics":     {"park":"Sutter Health Park","factor":1.00,"roof":"open","lat":38.5804,"lon":-121.5080,"orientation":"NE","elevation_ft":25},
    "Philadelphia Phillies": {"park":"Citizens Bank Park","factor":1.06,"roof":"open","lat":39.9061,"lon":-75.1665,"orientation":"NE","elevation_ft":20},
    "Pittsburgh Pirates":    {"park":"PNC Park","factor":0.96,"roof":"open","lat":40.4469,"lon":-80.0057,"orientation":"NE","elevation_ft":730},
    "San Diego Padres":      {"park":"Petco Park","factor":0.89,"roof":"open","lat":32.7076,"lon":-117.1570,"orientation":"NW","elevation_ft":62},
    "San Francisco Giants":  {"park":"Oracle Park","factor":0.93,"roof":"open","lat":37.7786,"lon":-122.3893,"orientation":"NW","elevation_ft":0},
    "Seattle Mariners":      {"park":"T-Mobile Park","factor":0.93,"roof":"retractable","lat":47.5914,"lon":-122.3325,"orientation":"NE","elevation_ft":17},
    "St. Louis Cardinals":   {"park":"Busch Stadium","factor":0.98,"roof":"open","lat":38.6226,"lon":-90.1928,"orientation":"NE","elevation_ft":466},
    "Tampa Bay Rays":        {"park":"Tropicana Field","factor":0.96,"roof":"dome","lat":27.7682,"lon":-82.6534,"orientation":"NE","elevation_ft":15},
    "Texas Rangers":         {"park":"Globe Life Field","factor":1.03,"roof":"retractable","lat":32.7473,"lon":-97.0822,"orientation":"NE","elevation_ft":551},
    "Toronto Blue Jays":     {"park":"Rogers Centre","factor":1.01,"roof":"retractable","lat":43.6414,"lon":-79.3894,"orientation":"NE","elevation_ft":250},
    "Washington Nationals":  {"park":"Nationals Park","factor":1.02,"roof":"open","lat":38.8730,"lon":-77.0074,"orientation":"NE","elevation_ft":25},
}

TEAM_ALIASES = {
    "D-backs":"Arizona Diamondbacks","Diamondbacks":"Arizona Diamondbacks",
    "Braves":"Atlanta Braves","Orioles":"Baltimore Orioles","Red Sox":"Boston Red Sox",
    "Cubs":"Chicago Cubs","White Sox":"Chicago White Sox","Reds":"Cincinnati Reds",
    "Guardians":"Cleveland Guardians","Rockies":"Colorado Rockies","Tigers":"Detroit Tigers",
    "Astros":"Houston Astros","Royals":"Kansas City Royals","Angels":"Los Angeles Angels",
    "Dodgers":"Los Angeles Dodgers","Marlins":"Miami Marlins","Brewers":"Milwaukee Brewers",
    "Twins":"Minnesota Twins","Mets":"New York Mets","Yankees":"New York Yankees",
    "Athletics":"Oakland Athletics","Phillies":"Philadelphia Phillies","Pirates":"Pittsburgh Pirates",
    "Padres":"San Diego Padres","Giants":"San Francisco Giants","Mariners":"Seattle Mariners",
    "Cardinals":"St. Louis Cardinals","Rays":"Tampa Bay Rays","Rangers":"Texas Rangers",
    "Blue Jays":"Toronto Blue Jays","Nationals":"Washington Nationals",
}

COMPASS_TO_DEG = {"N":0,"NNE":22.5,"NE":45,"ENE":67.5,"E":90,"ESE":112.5,"SE":135,"SSE":157.5,
                  "S":180,"SSW":202.5,"SW":225,"WSW":247.5,"W":270,"WNW":292.5,"NW":315,"NNW":337.5}

def resolve_team(name):
    if name in PARK_DATA:
        return name
    for alias, canonical in TEAM_ALIASES.items():
        if alias.lower() in name.lower():
            return canonical
    return None

def degrees_to_compass(deg):
    if deg is None: return "UNK"
    dirs = ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]
    return dirs[round(deg / (360 / len(dirs))) % len(dirs)]

def wx_code_to_label(code):
    if code is None: return "Unknown"
    if code == 0:   return "Clear"
    if code <= 3:   return "Partly Cloudy"
    if code <= 19:  return "Drizzle"
    if code <= 29:  return "Rain Showers"
    if code <= 69:  return "Rain"
    if code <= 84:  return "Rain Showers"
    if code <= 94:  return "Thunderstorms"
    return "Severe Weather"

def get_weather(lat, lon, park_name="Stadium"):
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           "&hourly=temperature_2m,precipitation_probability,windspeed_10m,winddirection_10m,weathercode"
           "&temperature_unit=fahrenheit&windspeed_unit=mph&forecast_days=1&timezone=auto")
    try:
        data   = requests.get(url, timeout=10).json()
        hourly = data.get("hourly", {})
        temps  = hourly.get("temperature_2m", [])
        precip = hourly.get("precipitation_probability", [])
        wspd   = hourly.get("windspeed_10m", [])
        wdir   = hourly.get("winddirection_10m", [])
        wxc    = hourly.get("weathercode", [])
        idx    = min(19, len(temps) - 1) if temps else 0
        wx = {"park": park_name,
              "temp_f":       round(temps[idx], 1)  if temps  else None,
              "precip_pct":   precip[idx]            if precip else None,
              "wind_mph":     round(wspd[idx], 1)    if wspd   else None,
              "wind_dir_deg": wdir[idx]              if wdir   else None,
              "wx_code":      wxc[idx]               if wxc    else None}
        wx["wind_dir_label"] = degrees_to_compass(wx["wind_dir_deg"])
        wx["condition"]      = wx_code_to_label(wx["wx_code"])
        return wx
    except Exception as e:
        print(f"  Weather fetch failed ({park_name}): {e}")
        return {"park":park_name,"temp_f":None,"precip_pct":None,"wind_mph":None,
                "wind_dir_deg":None,"wind_dir_label":"UNK","condition":"Unknown"}

def wind_impact(wind_mph, wind_dir_label, park_orientation):
    if wind_mph is None or wind_mph < 3:
        return 0.0, "Calm — negligible wind effect"
    wind_deg  = COMPASS_TO_DEG.get(wind_dir_label, 0)
    park_deg  = COMPASS_TO_DEG.get(park_orientation, 45)
    diff      = abs(wind_deg - park_deg) % 360
    if diff > 180: diff = 360 - diff
    alignment = np.cos(np.radians(diff))
    score     = round(alignment * (wind_mph / 10) * 5, 2)
    if score >= 3:   label = f"Wind OUT ({wind_mph}mph {wind_dir_label}) — significant run boost ⬆"
    elif score >= 1: label = f"Wind OUT ({wind_mph}mph {wind_dir_label}) — mild run boost"
    elif score <= -3:label = f"Wind IN ({wind_mph}mph {wind_dir_label}) — significant run suppressor ⬇"
    elif score <= -1:label = f"Wind IN ({wind_mph}mph {wind_dir_label}) — mild run suppressor"
    else:            label = f"Crosswind ({wind_mph}mph {wind_dir_label}) — minimal effect"
    return score, label

def temp_adjustment(temp_f):
    if temp_f is None: return 0.0, "Temp unknown"
    if temp_f < 40:    return -1.0, f"{temp_f}°F — very cold ❄"
    if temp_f < 50:    return -0.6, f"{temp_f}°F — cold"
    if temp_f < 60:    return -0.3, f"{temp_f}°F — cool"
    if temp_f <= 80:   return  0.0, f"{temp_f}°F — neutral"
    if temp_f <= 90:   return  0.2, f"{temp_f}°F — warm"
    return 0.4, f"{temp_f}°F — hot ☀"

def rain_risk(precip_pct, wx_condition):
    severe = ["Thunderstorms","Severe Weather","Rain","Rain Showers"]
    if wx_condition in severe and precip_pct and precip_pct >= 60:
        return "HIGH ⛈", f"{precip_pct}% precip — consider avoiding"
    elif precip_pct and precip_pct >= 40:
        return "MEDIUM 🌧", f"{precip_pct}% precip — monitor"
    return "LOW ✅", f"{precip_pct or 0}% precip — game likely on"

def get_park_weather_profile(home_team):
    canonical = resolve_team(home_team)
    if not canonical:
        return {"team":home_team,"park":"Unknown","park_factor":1.00,"roof":"open",
                "weather":None,"wind_impact_score":0.0,"wind_label":"Unknown",
                "temp_adj":0.0,"temp_label":"Unknown","rain_risk":"UNKNOWN",
                "rain_note":"","env_total_adj":0.0,"env_summary":"No park data"}
    park       = PARK_DATA[canonical]
    is_covered = park["roof"] in ("dome","retractable")
    if is_covered:
        weather    = {"park":park["park"],"temp_f":72,"precip_pct":0,"wind_mph":0,
                      "wind_dir_label":"N/A","condition":"Indoor / Controlled"}
        wind_score, wind_label = 0.0, f"Roof ({park['roof']}) — weather neutral"
        temp_adj,   temp_label = 0.0, "Climate controlled"
        rain_level, rain_note  = "LOW ✅", "Covered stadium"
    else:
        weather    = get_weather(park["lat"], park["lon"], park["park"])
        wind_score, wind_label = wind_impact(weather.get("wind_mph"), weather.get("wind_dir_label"), park["orientation"])
        temp_adj,   temp_label = temp_adjustment(weather.get("temp_f"))
        rain_level, rain_note  = rain_risk(weather.get("precip_pct"), weather.get("condition",""))

    park_dev     = (park["factor"] - 1.00) * 10
    env_total_adj = round(park_dev + wind_score + temp_adj, 2)
    parts = []
    if abs(park_dev) >= 0.5:
        parts.append(f"{park['park']} ({'hitter' if park_dev>0 else 'pitcher'}-friendly, PF {park['factor']})")
    if abs(wind_score) >= 1: parts.append(wind_label)
    if abs(temp_adj) >= 0.3: parts.append(temp_label)
    return {"team":canonical,"park":park["park"],"park_factor":park["factor"],"roof":park["roof"],
            "weather":weather,"wind_impact_score":wind_score,"wind_label":wind_label,
            "temp_adj":temp_adj,"temp_label":temp_label,"rain_risk":rain_level,"rain_note":rain_note,
            "env_total_adj":env_total_adj,"env_summary":" | ".join(parts) if parts else "Neutral environment"}

def recalibrate_total_signal(card, env):
    signals      = card.get("signals", {})
    adj          = env.get("env_total_adj", 0)
    rain_level   = env.get("rain_risk", "LOW ✅")
    current_lean = signals.get("total_lean", "NEUTRAL")
    current_conf = signals.get("total_conf", "LOW")
    note = ""
    if "HIGH" in rain_level:
        signals["total_lean"]="AVOID — RAIN RISK"; signals["total_conf"]="N/A"
        note = f"⛈ Rain risk — {env.get('rain_note')}"
    elif adj >= 2.5:
        if current_lean == "UNDER": signals["total_lean"]="NEUTRAL"; signals["total_conf"]="LOW"; note=f"OVER environment offsets UNDER lean"
        else: signals["total_lean"]="OVER"; signals["total_conf"]="HIGH" if current_conf in ("MED","HIGH") else "MED"; note=f"Environment strongly favours OVER (+{adj})"
    elif adj >= 1.0:
        if current_lean=="OVER": signals["total_conf"]="HIGH" if current_conf=="MED" else current_conf; note=f"Environment supports OVER (+{adj})"
        elif current_lean=="UNDER": signals["total_conf"]="LOW"; note=f"Environment partially offsets UNDER lean"
    elif adj <= -2.5:
        if current_lean=="OVER": signals["total_lean"]="NEUTRAL"; signals["total_conf"]="LOW"; note=f"UNDER environment offsets OVER lean"
        else: signals["total_lean"]="UNDER"; signals["total_conf"]="HIGH" if current_conf in ("MED","HIGH") else "MED"; note=f"Environment strongly favours UNDER ({adj})"
    elif adj <= -1.0:
        if current_lean=="UNDER": signals["total_conf"]="HIGH" if current_conf=="MED" else current_conf; note=f"Environment supports UNDER ({adj})"
        elif current_lean=="OVER": signals["total_conf"]="LOW"; note=f"Environment partially offsets OVER lean"
    if note: signals["total_env_note"] = note
    signals["env_total_adj"] = adj
    signals["env_summary"]   = env.get("env_summary","")
    card["signals"] = signals
    return card

def recalibrate_nrfi_signal(card, env):
    signals      = card.get("signals", {})
    adj          = env.get("env_total_adj", 0)
    current_nrfi = signals.get("nrfi_lean", "NEUTRAL / AVOID")
    current_conf = signals.get("nrfi_conf", "LOW")
    if adj >= 2.0 and "NRFI" in current_nrfi:
        if current_conf=="HIGH": signals["nrfi_conf"]="MED"; signals["nrfi_env_note"]=f"Hitter-friendly park reduces NRFI confidence"
        elif current_conf=="MED": signals["nrfi_lean"]="NEUTRAL / AVOID"; signals["nrfi_conf"]="LOW"; signals["nrfi_env_note"]="Hitter-friendly environment — NRFI undermined"
    elif adj <= -1.5 and "NRFI" in current_nrfi:
        if current_conf=="MED": signals["nrfi_conf"]="HIGH"; signals["nrfi_env_note"]="Pitcher-friendly environment boosts NRFI confidence"
    card["signals"] = signals
    return card

def apply_park_weather(game_cards):
    enriched = []
    for card in game_cards:
        home_team = card.get("home_team","")
        print(f"  ☁  Fetching park/weather: {home_team}")
        env              = get_park_weather_profile(home_team)
        card["environment"] = env
        card = recalibrate_total_signal(card, env)
        card = recalibrate_nrfi_signal(card, env)
        enriched.append(card)
    return enriched

def print_environment_card(card):
    env = card.get("environment", {})
    wx  = env.get("weather", {}) or {}
    s   = card.get("signals", {})
    print(f"\n  ── ENVIRONMENT: {env.get('park','Unknown')} ──")
    print(f"     Park Factor : {env.get('park_factor','?')}  ({env.get('roof','open').upper()})")
    if wx.get("condition") != "Indoor / Controlled":
        print(f"     Temperature : {wx.get('temp_f','?')}°F")
        print(f"     Wind        : {wx.get('wind_mph','?')} mph {wx.get('wind_dir_label','')} — {env.get('wind_label','')}")
        print(f"     Conditions  : {wx.get('condition','?')}  ({wx.get('precip_pct','?')}% precip)")
        print(f"     Rain Risk   : {env.get('rain_risk','?')} — {env.get('rain_note','')}")
    else:
        print(f"     Conditions  : Climate controlled (indoor)")
    print(f"     Env Adj     : {env.get('env_total_adj',0):+.1f} runs vs neutral")
    print(f"  ── ADJUSTED SIGNALS ──")
    print(f"     Total Lean  : {s.get('total_lean','?')} [{s.get('total_conf','?')}]" + (f"  ← {s.get('total_env_note','')}" if s.get('total_env_note') else ""))
    print(f"     NRFI        : {s.get('nrfi_lean','?')} [{s.get('nrfi_conf','?')}]" + (f"  ← {s.get('nrfi_env_note','')}" if s.get('nrfi_env_note') else ""))

# ============================================================
# PHASE 4 — BULLPEN ANALYSIS
# ============================================================

FATIGUE_DAYS = 3

def get_bullpen_roster(team_id):
    url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType=active&season={today_dt.year}"
    try:
        data = requests.get(url, timeout=10).json()
        # Include all pitcher position codes - MLB API uses P, RP, CL, CP, TWP
        all_pitchers = [{"name": p["person"]["fullName"], "person_id": p["person"]["id"],
                 "position": p.get("position",{}).get("abbreviation","")}
                for p in data.get("roster",[])
                if p.get("position",{}).get("abbreviation","") in ("RP","CL","CP","P","TWP","SP")]
        # Filter out known starters by checking if they appear in the pitcher targets
        # Fallback: just use everyone with P-type position
        relievers = [p for p in all_pitchers if p["position"] in ("RP","CL","CP","TWP")]
        if not relievers:
            # No explicit relievers found — use all pitchers (broad fallback)
            relievers = all_pitchers
        return relievers
    except Exception as e:
        print(f"    Roster fetch error (team {team_id}): {e}")
        return []

def get_team_id_map():
    try:
        data = requests.get("https://statsapi.mlb.com/api/v1/teams?sportId=1", timeout=10).json()
        return {t["name"]: t["id"] for t in data.get("teams",[])}
    except Exception as e:
        print(f"    Team ID map fetch error: {e}")
        return {}

def get_recent_usage(person_id, days=FATIGUE_DAYS):
    since = (today_dt - timedelta(days=days)).strftime("%Y-%m-%d")
    url   = (f"https://statsapi.mlb.com/api/v1/people/{person_id}/stats"
             f"?stats=gameLog&group=pitching&season={today_dt.year}&startDate={since}&endDate={today}")
    try:
        data   = requests.get(url, timeout=10).json()
        splits = data.get("stats",[{}])[0].get("splits",[])
        pitches = sum(int(g.get("stat",{}).get("numberOfPitches",0) or 0) for g in splits)
        return {"appearances_last_3d": len(splits), "pitches_last_3d": pitches}
    except:
        return {"appearances_last_3d": 0, "pitches_last_3d": 0}

def reliever_statcast_profile(name, mlbam_id):
    try:
        df = statcast_pitcher(start_date, end_date, int(mlbam_id))
        if df is None or df.empty: return None
        batted = df[df["events"].notnull()].copy()
        pa     = len(batted)
        if pa < 10: return None
        xwoba    = batted["estimated_woba_using_speedangle"].mean()
        k_pct    = (batted["events"]=="strikeout").mean()*100
        bb_pct   = (batted["events"]=="walk").mean()*100
        hard_hit = (batted["launch_speed"]>=95).mean()*100
        hrs      = (batted["events"]=="home_run").sum()
        hr_per9  = (hrs / max(pa,1)) * 27
        if "description" in df.columns:
            swings    = df[df["description"].isin(["swinging_strike","foul","hit_into_play","swinging_strike_blocked","foul_tip"])]
            whiffs    = df[df["description"].isin(["swinging_strike","swinging_strike_blocked"])]
            whiff_pct = len(whiffs)/max(len(swings),1)*100
        else:
            whiff_pct = np.nan
        return {"name":name,"pa":pa,
                "xwOBA":   round(xwoba,3)    if pd.notnull(xwoba)    else None,
                "K%":      round(k_pct,1)    if pd.notnull(k_pct)    else None,
                "BB%":     round(bb_pct,1)   if pd.notnull(bb_pct)   else None,
                "HardHit%":round(hard_hit,1) if pd.notnull(hard_hit) else None,
                "HR_per9": round(hr_per9,2)  if pd.notnull(hr_per9)  else None,
                "Whiff%":  round(whiff_pct,1)if pd.notnull(whiff_pct)else None,
                "G": df["game_pk"].nunique() if "game_pk" in df.columns else 1}
    except Exception as e:
        print(f"      Reliever profile error ({name}): {e}")
        return None

def grade_bullpen(relievers_with_profiles):
    scored = []
    tired_arms = []
    elite_arms = []
    closer = None
    best_closer_score = -1

    for r in relievers_with_profiles:
        profile  = r.get("profile")
        usage    = r.get("usage", {})
        name     = r.get("name", "?")
        pos      = r.get("position", "RP")
        p3d = usage.get("pitches_last_3d", 0)
        a3d = usage.get("appearances_last_3d", 0)
        is_tired = (a3d >= 2) or (p3d >= 50)
        if is_tired:
            tired_arms.append({"name": name, "appearances_3d": a3d, "pitches_3d": p3d})
        if not profile:
            scored.append(50.0)
            continue
        xwoba = profile.get("xwOBA") or 0.320
        k     = profile.get("K%")    or 20.0
        bb    = profile.get("BB%")   or 9.0
        hh    = profile.get("HardHit%") or 35.0
        arm = (
            max(0, min(40, (0.400 - xwoba) / 0.200 * 40)) +
            max(0, min(25, (k - 10) / 25 * 25)) +
            max(0, min(20, (15 - bb) / 10 * 20)) +
            max(0, min(15, (50 - hh) / 25 * 15))
        )
        arm = max(0, min(100, arm))
        if is_tired:
            arm = max(0, arm - 15)
        scored.append(arm)
        if arm >= 70 and not is_tired:
            elite_arms.append({"name": name, "score": round(arm, 1), "xwOBA": xwoba})
        is_closer_pos = pos in ("CL", "CP")
        if is_closer_pos and arm > best_closer_score:
            best_closer_score = arm
            closer = {"name": name, "xwOBA": xwoba, "K%": k, "score": round(arm, 1)}

    if closer is None and scored:
        best_idx = None; best_sc = -1
        for i, r in enumerate(relievers_with_profiles):
            sc = scored[i]
            tired = (r.get("usage", {}).get("appearances_last_3d", 0) >= 2 or
                     r.get("usage", {}).get("pitches_last_3d", 0) >= 50)
            if sc > best_sc and not tired and r.get("profile"):
                best_sc = sc; best_idx = i
        if best_idx is not None:
            r = relievers_with_profiles[best_idx]
            p = r.get("profile", {})
            closer = {"name": r.get("name","?"), "xwOBA": p.get("xwOBA",0.320),
                      "K%": p.get("K%",20.0), "score": round(scored[best_idx],1)}

    if not scored:
        return 50.0, "AVERAGE", None, [], [], []

    bs = round(np.mean(scored), 1)
    grade = ("ELITE 🔒" if bs>=72 else "STRONG" if bs>=60 else
             "AVERAGE" if bs>=48 else "WEAK" if bs>=35 else "VULNERABLE 💣")
    breakdown = [{"name":r.get("name"),"score":s,
                  "xwOBA":r.get("profile",{}).get("xwOBA") if r.get("profile") else None,
                  "tired":r.get("usage",{}).get("appearances_last_3d",0)>=2}
                 for r,s in zip(relievers_with_profiles,scored)]
    return bs, grade, closer, tired_arms, elite_arms, breakdown

def _empty_bullpen(team_name):
    return {"team":team_name,"bullpen_score":50.0,"grade":"UNKNOWN","closer":None,
            "tired_arms":[],"elite_arms":[],"n_relievers":0,"breakdown":[]}

def profile_team_bullpen(team_name, team_id):
    print(f"    Profiling bullpen: {team_name}")
    relievers = get_bullpen_roster(team_id)
    if not relievers:
        return _empty_bullpen(team_name)
    enriched = []
    for r in relievers:
        name  = r["name"]
        usage = get_recent_usage(r["person_id"])
        try:
            pid_df  = safe_playerid_lookup(name)
            if pid_df.empty:
                print(f"      ⚠ No MLBAM ID: {name}")
                profile = None
            else:
                profile = reliever_statcast_profile(name, int(pid_df.iloc[0]["key_mlbam"]))
        except Exception as e:
            print(f"      Lookup error ({name}): {e}")
            profile = None
        enriched.append({**r, "profile": profile, "usage": usage})
    score, grade, closer, tired, elite, breakdown = grade_bullpen(enriched)
    return {"team":team_name,"bullpen_score":score,"grade":grade,"closer":closer,
            "tired_arms":tired,"elite_arms":elite,"n_relievers":len(relievers),"breakdown":breakdown}

def calibrate_f5_vs_fullgame(card):
    signals=card.get("signals",{}); hbp=card.get("home_bullpen",{}); abp=card.get("away_bullpen",{})
    hbps=hbp.get("bullpen_score",50); abps=abp.get("bullpen_score",50)
    hbpg=hbp.get("grade","AVERAGE"); abpg=abp.get("grade","AVERAGE")
    hspx=card.get("home_pitcher_xwOBA") or 0.320; aspx=card.get("away_pitcher_xwOBA") or 0.320
    hspi=card.get("home_pitcher_AvgIP") or 5.0; aspi=card.get("away_pitcher_AvgIP") or 5.0
    htired=hbp.get("tired_arms",[]); atired=abp.get("tired_arms",[])
    notes=[]; f5_upgrade=False

    if hspx<=0.300 and hspi>=5.0:
        if hbps>=60: notes.append(f"✅ {card['home_pitcher']} (quality SP) + strong {card['home_team']} bullpen ({hbpg}) — full game ML holds")
        else: notes.append(f"⚠  {card['home_pitcher']} quality stops at F5 — {card['home_team']} bullpen is {hbpg}"); f5_upgrade=True
    elif hspx>=0.340:
        if hbps>=65: notes.append(f"ℹ  Weak {card['home_pitcher']} but {card['home_team']} bullpen ({hbpg}) may limit damage")
        else: notes.append(f"💣 {card['home_pitcher']} struggles + weak {card['home_team']} bullpen ({hbpg}) — high run risk all game")
    if aspx<=0.300 and aspi>=5.0:
        if abps>=60: notes.append(f"✅ {card['away_pitcher']} (quality SP) + strong {card['away_team']} bullpen ({abpg}) — full game ML holds")
        else: notes.append(f"⚠  {card['away_pitcher']} quality stops at F5 — {card['away_team']} bullpen is {abpg}"); f5_upgrade=True
    elif aspx>=0.340:
        if abps>=65: notes.append(f"ℹ  Weak {card['away_pitcher']} but {card['away_team']} bullpen ({abpg}) may limit damage")
        else: notes.append(f"💣 {card['away_pitcher']} struggles + weak {card['away_team']} bullpen ({abpg}) — high run risk all game")
    if len(htired)>=2: notes.append(f"😴 {card['home_team']} bullpen fatigue: {', '.join([a['name'] for a in htired[:3]])} used heavily last 3 days")
    if len(atired)>=2: notes.append(f"😴 {card['away_team']} bullpen fatigue: {', '.join([a['name'] for a in atired[:3]])} used heavily last 3 days")
    for bp_label, bp, tired in [("Home", hbp, htired), ("Away", abp, atired)]:
        c = bp.get("closer")
        if c:
            tired_names=[a["name"] for a in tired]; status="⚠ TIRED" if c["name"] in tired_names else "✅ AVAILABLE"
            notes.append(f"{bp_label} closer: {c['name']} (xwOBA {c['xwOBA']}, K% {c['K%']}) {status}")

    cf5  = signals.get("f5_lean","NEUTRAL")
    cf5c = signals.get("f5_conf","LOW")
    if f5_upgrade and cf5 != "NEUTRAL" and not signals.get("f5_conflict_note"):
        signals["f5_conf"] = "HIGH" if cf5c=="MED" else cf5c
        signals["f5_bp_note"] = "F5 preferred over full game — starter quality doesn't carry through bullpen"
    elif len(htired)>=2 or len(atired)>=2:
        if signals.get("total_lean","NEUTRAL") in ("NEUTRAL","OVER"):
            signals["total_lean"] = "OVER"
            signals["total_conf"] = "MED" if signals.get("total_conf")=="LOW" else signals.get("total_conf")
            signals["total_bp_note"] = "Tired bullpens increase late-game run risk"
    signals["bullpen_notes"] = notes
    card["signals"] = signals
    return card

# ============================================================
# PHASE 5 — ODDS API INTEGRATION
# Sign up free at: https://the-odds-api.com
# Free tier: 500 requests/month (~2/day used by this model)
# ============================================================
ODDS_API_KEY     = "a613bfad0e47fafb085b774c856f6ca6"   # ← paste your key
ODDS_API_BASE    = "https://api.the-odds-api.com/v4"
ODDS_BOOKMAKERS  = "draftkings,fanduel,betmgm,caesars"

def _odds_team_match(odds_name, card_name):
    o = odds_name.lower().strip()
    c = card_name.lower().strip()
    if o == c: return True
    o_words = o.split(); c_words = c.split()
    if o_words[-1] == c_words[-1]: return True
    if len(o_words) >= 2 and " ".join(o_words[-2:]) == " ".join(c_words[-2:]): return True
    if o in c or c in o: return True
    return False

def _get_best_price(outcomes, name_check_fn):
    for outcome in outcomes:
        if name_check_fn(outcome.get("name", "")):
            return outcome.get("price")
    return None

def _get_best_line(outcomes, side="Over"):
    for outcome in outcomes:
        if outcome.get("name", "").lower() == side.lower():
            return outcome.get("point")
    return None

def fetch_odds_api(market_keys="h2h,totals", label="full game"):
    if not ODDS_API_KEY or ODDS_API_KEY == "YOUR_API_KEY_HERE":
        print(f"  ⚠  Odds API key not set — skipping {label} odds")
        return []
    url = (f"{ODDS_API_BASE}/sports/baseball_mlb/odds"
           f"?regions=us&markets={market_keys}"
           f"&oddsFormat=american&apiKey={ODDS_API_KEY}"
           f"&bookmakers={ODDS_BOOKMAKERS}")
    try:
        resp = requests.get(url, timeout=15)
        remaining = resp.headers.get("x-requests-remaining", "?")
        used      = resp.headers.get("x-requests-used", "?")
        print(f"  Odds API ({label}): {used} credits used, {remaining} remaining")
        if resp.status_code != 200:
            print(f"  ⚠  Odds API error {resp.status_code}: {resp.text[:200]}")
            return []
        return resp.json()
    except Exception as e:
        print(f"  ⚠  Odds API fetch failed ({label}): {e}")
        return []

def _parse_odds_for_game(odds_game, home_team, away_team):
    result = {"ml_home": None, "ml_away": None,
              "total_line": None, "total_over_price": None,
              "total_under_price": None, "bookmaker": None}
    priority = [b.strip() for b in ODDS_BOOKMAKERS.split(",")]
    bookmakers = odds_game.get("bookmakers", [])
    def bk_priority(bk):
        try: return priority.index(bk["key"])
        except ValueError: return 999
    for bk in sorted(bookmakers, key=bk_priority):
        for market in bk.get("markets", []):
            key      = market.get("key", "")
            outcomes = market.get("outcomes", [])
            if key == "h2h":
                ml_h = _get_best_price(outcomes, lambda n: _odds_team_match(n, home_team))
                ml_a = _get_best_price(outcomes, lambda n: _odds_team_match(n, away_team))
                if ml_h and ml_a:
                    result["ml_home"] = ml_h; result["ml_away"] = ml_a
                    result["bookmaker"] = bk.get("title", bk["key"])
            if key == "totals":
                line  = _get_best_line(outcomes, "Over")
                o_prc = _get_best_price(outcomes, lambda n: n.lower()=="over")
                u_prc = _get_best_price(outcomes, lambda n: n.lower()=="under")
                if line:
                    result["total_line"] = line
                    result["total_over_price"] = o_prc
                    result["total_under_price"] = u_prc
        if result["ml_home"] and result["total_line"]:
            break
    return result

def _parse_f5_odds_for_game(odds_game, home_team, away_team):
    result = {"f5_ml_home": None, "f5_ml_away": None,
              "f5_total_line": None, "f5_total_over_price": None,
              "f5_total_under_price": None}
    for bk in odds_game.get("bookmakers", []):
        for market in bk.get("markets", []):
            key      = market.get("key", "")
            outcomes = market.get("outcomes", [])
            if key == "h2h_h1":
                ml_h = _get_best_price(outcomes, lambda n: _odds_team_match(n, home_team))
                ml_a = _get_best_price(outcomes, lambda n: _odds_team_match(n, away_team))
                if ml_h and ml_a:
                    result["f5_ml_home"] = ml_h; result["f5_ml_away"] = ml_a
            if key == "totals_h1":
                line  = _get_best_line(outcomes, "Over")
                o_prc = _get_best_price(outcomes, lambda n: n.lower()=="over")
                u_prc = _get_best_price(outcomes, lambda n: n.lower()=="under")
                if line:
                    result["f5_total_line"] = line
                    result["f5_total_over_price"] = o_prc
                    result["f5_total_under_price"] = u_prc
        if result["f5_ml_home"] and result["f5_total_line"]:
            break
    return result

def _american_to_implied(american_odds):
    if american_odds is None: return None
    try:
        o = float(american_odds)
        if o > 0: return round(100 / (o + 100) * 100, 1)
        else:     return round(-o / (-o + 100) * 100, 1)
    except: return None

def _value_rating(model_lean, total_line, env_adj, avg_xwoba):
    if total_line is None: return None, ""
    base_runs   = 8.8
    xwoba_adj   = (avg_xwoba - 0.320) / 0.010 * 0.4
    expected    = round(base_runs + xwoba_adj + (env_adj or 0), 1)
    gap         = round(expected - total_line, 1)
    if model_lean == "OVER":
        if gap >= 1.5:   return "HIGH VALUE ⬆",  f"Model expects ~{expected} runs vs line {total_line} (+{gap})"
        elif gap >= 0.5: return "FAIR VALUE",     f"Model expects ~{expected} runs vs line {total_line} (+{gap})"
        elif gap >= -0.5:return "MARGINAL",       f"Model expects ~{expected} runs, line {total_line} close ({gap})"
        else:            return "POOR VALUE ⚠",   f"Model expects ~{expected} runs but line is {total_line} ({gap})"
    elif model_lean == "UNDER":
        if gap <= -1.5:  return "HIGH VALUE ⬇",  f"Model expects ~{expected} runs vs line {total_line} ({gap})"
        elif gap <= -0.5:return "FAIR VALUE",     f"Model expects ~{expected} runs vs line {total_line} ({gap})"
        elif gap >= -0.5:return "MARGINAL",       f"Model expects ~{expected} runs, line {total_line} close ({gap})"
        else:            return "POOR VALUE ⚠",   f"Model expects ~{expected} runs but line is {total_line} ({gap})"
    return None, ""

def apply_odds_api(game_cards):
    print("\n💰  Fetching live odds from The Odds API...")
    full_odds = fetch_odds_api("h2h,totals",        "full game")
    f5_odds   = fetch_odds_api("h2h_h1,totals_h1", "F5")

    def match_card_to_odds(odds_list, home_team, away_team):
        for og in odds_list:
            oh = og.get("home_team", ""); oa = og.get("away_team", "")
            if _odds_team_match(oh, home_team) and _odds_team_match(oa, away_team):
                return og
        return None

    for card in game_cards:
        home_team = card.get("home_team", ""); away_team = card.get("away_team", "")
        signals   = card.get("signals", {})

        og_full = match_card_to_odds(full_odds, home_team, away_team)
        og_f5   = match_card_to_odds(f5_odds,  home_team, away_team)

        parsed    = _parse_odds_for_game(og_full, home_team, away_team) if og_full else \
                    {"ml_home": None, "ml_away": None, "total_line": None,
                     "total_over_price": None, "total_under_price": None, "bookmaker": None}
        parsed_f5 = _parse_f5_odds_for_game(og_f5, home_team, away_team) if og_f5 else \
                    {"f5_ml_home": None, "f5_ml_away": None,
                     "f5_total_line": None, "f5_total_over_price": None, "f5_total_under_price": None}

        card["odds_ml_home"]           = parsed["ml_home"]
        card["odds_ml_away"]           = parsed["ml_away"]
        card["odds_total_line"]        = parsed["total_line"]
        card["odds_total_over_price"]  = parsed["total_over_price"]
        card["odds_total_under_price"] = parsed["total_under_price"]
        card["odds_bookmaker"]         = parsed["bookmaker"]
        card["odds_f5_ml_home"]        = parsed_f5["f5_ml_home"]
        card["odds_f5_ml_away"]        = parsed_f5["f5_ml_away"]
        card["odds_f5_total_line"]     = parsed_f5["f5_total_line"]
        card["odds_f5_over_price"]     = parsed_f5["f5_total_over_price"]
        card["odds_f5_under_price"]    = parsed_f5["f5_total_under_price"]
        card["odds_ml_home_implied"]   = _american_to_implied(parsed["ml_home"])
        card["odds_ml_away_implied"]   = _american_to_implied(parsed["ml_away"])

        total_lean = signals.get("total_lean", "NEUTRAL")
        avg_xwoba  = ((card.get("home_pitcher_xwOBA") or 0.320) +
                      (card.get("away_pitcher_xwOBA") or 0.320)) / 2
        env_adj    = card.get("environment", {}).get("env_total_adj", 0)
        v_label, v_note = _value_rating(total_lean, parsed["total_line"], env_adj, avg_xwoba)
        signals["total_value_rating"] = v_label
        signals["total_value_note"]   = v_note

        if v_label and "POOR" in v_label:
            if signals.get("total_conf") == "HIGH":
                signals["total_conf"] = "MED"
                signals["total_env_note"] = (signals.get("total_env_note","") +
                    " | Confidence reduced — poor value vs book line").strip(" |")
            elif signals.get("total_conf") == "MED":
                signals["total_conf"] = "LOW"

        ml_lean = signals.get("ML_lean", "NEUTRAL")
        ml_value_note = ""
        if ml_lean not in ("NEUTRAL", "SUPPRESSED", "N/A") and parsed["ml_home"]:
            lean_is_home = "HOME" in ml_lean
            ml_price = parsed["ml_home"] if lean_is_home else parsed["ml_away"]
            if ml_price and ml_price > 0:
                ml_value_note = f"Model backs underdog at +{ml_price}"
            elif ml_price and ml_price < -200:
                ml_value_note = f"Heavy favourite at {ml_price} — low value"
        signals["ml_value_note"] = ml_value_note
        card["signals"] = signals

        line_str = f"Total {parsed['total_line']}" if parsed["total_line"] else "no line"
        ml_str   = (f"Home {parsed['ml_home']}/Away {parsed['ml_away']}"
                    if parsed["ml_home"] else "no ML odds")
        f5_str   = f"F5 total {parsed_f5['f5_total_line']}" if parsed_f5["f5_total_line"] else "no F5 line"
        print(f"    {away_team[:20]:<20} @ {home_team[:20]:<20} | {ml_str} | {line_str} | {f5_str}"
              + (f" | {v_label}" if v_label else ""))

    return game_cards

def apply_final_conflict_resolution(game_cards):
    for card in game_cards:
        signals    = card.get("signals", {})
        fade_notes = signals.get("fade_notes", [])
        faded_teams = set()
        for fn in fade_notes:
            if "FADE " in fn and " —" in fn:
                try:
                    team = fn.split("FADE ")[1].split(" —")[0].strip()
                    faded_teams.add(team)
                except:
                    pass
        if not faded_teams:
            continue
        ml_lean = signals.get("ML_lean", "NEUTRAL")
        if ml_lean not in ("NEUTRAL", "SUPPRESSED", "N/A"):
            ml_team = card["away_team"] if "AWAY" in ml_lean else card["home_team"]
            if ml_team in faded_teams:
                signals["ML_lean"] = "SUPPRESSED"; signals["ML_conf"] = "N/A"
                signals["ML_ip_note"] = (signals.get("ML_ip_note") or "") + \
                    f" | ML suppressed — contradicts FADE on {ml_team}"
        f5_lean = signals.get("f5_lean", "NEUTRAL")
        if f5_lean not in ("NEUTRAL", "SUPPRESSED", "UNDER", "N/A"):
            f5_team = card["away_team"] if "AWAY" in f5_lean else card["home_team"]
            if f5_team in faded_teams:
                signals["f5_lean"] = "SUPPRESSED"; signals["f5_conf"] = "N/A"
                signals["f5_ip_note"] = (signals.get("f5_ip_note") or "") + \
                    f" | F5 suppressed — contradicts FADE on {f5_team}"
        card["signals"] = signals
    return game_cards

def apply_bullpen_analysis(game_cards, team_id_map):
    enriched = []
    for card in game_cards:
        home_team = card.get("home_team",""); away_team = card.get("away_team","")
        home_id   = team_id_map.get(home_team) or team_id_map.get(resolve_team(home_team))
        away_id   = team_id_map.get(away_team) or team_id_map.get(resolve_team(away_team))
        card["home_bullpen"] = profile_team_bullpen(home_team, home_id) if home_id else _empty_bullpen(home_team)
        card["away_bullpen"] = profile_team_bullpen(away_team, away_id) if away_id else _empty_bullpen(away_team)
        card = calibrate_f5_vs_fullgame(card)
        enriched.append(card)
    return enriched

def print_bullpen_card(card):
    hbp=card.get("home_bullpen",{}); abp=card.get("away_bullpen",{}); s=card.get("signals",{})
    print(f"\n  ── BULLPENS ──")
    for label, bp in [("AWAY",abp),("HOME",hbp)]:
        c=bp.get("closer"); cs=f"  Closer: {c['name']} (xwOBA {c['xwOBA']})" if c else ""
        ts="  Tired: "+", ".join([f"{a['name']}({a['appearances_3d']}G/{a['pitches_3d']}P)" for a in bp.get("tired_arms",[])]) if bp.get("tired_arms") else ""
        es="  Elite: "+", ".join([f"{a['name']}({a['score']})" for a in bp.get("elite_arms",[])[:3]]) if bp.get("elite_arms") else ""
        print(f"     {label} {bp.get('team','?'):25} Score: {bp.get('bullpen_score','?'):<5} {bp.get('grade','?'):<15}{cs}")
        if ts: print(f"       {ts}")
        if es: print(f"       {es}")
    print(f"\n  ── F5 vs FULL GAME ──")
    print(f"     F5 Lean   : {s.get('f5_lean','?')} [{s.get('f5_conf','?')}]" + (f"  ← {s.get('f5_bp_note','')}" if s.get('f5_bp_note') else ""))
    print(f"     Full Game : {s.get('total_lean','?')} [{s.get('total_conf','?')}]" + (f"  ← {s.get('total_bp_note','')}" if s.get('total_bp_note') else ""))
    notes=s.get("bullpen_notes",[])
    if notes:
        print(f"\n  ── BULLPEN NOTES ──")
        for n in notes: print(f"     {n}")

# ============================================================
# EXPORT
# ============================================================

def export_full_cards(game_cards, filename="daily_matchup_full.csv"):
    rows=[]
    for c in game_cards:
        s=c.get("signals",{}); env=c.get("environment",{}); wx=env.get("weather",{}) or {}
        hbp=c.get("home_bullpen",{}); abp=c.get("away_bullpen",{})
        rows.append({
            "Date":today,"Matchup":c.get("matchup"),"Home Team":c.get("home_team"),"Away Team":c.get("away_team"),
            "Home SP":c.get("home_pitcher"),"Away SP":c.get("away_pitcher"),
            "Home SP xwOBA":c.get("home_pitcher_xwOBA"),"Away SP xwOBA":c.get("away_pitcher_xwOBA"),
            "Home SP K%":c.get("home_pitcher_K%"),"Away SP K%":c.get("away_pitcher_K%"),
            "Home SP K% vL":c.get("home_pitcher_K%_vL"),"Home SP K% vR":c.get("home_pitcher_K%_vR"),
            "Away SP K% vL":c.get("away_pitcher_K%_vL"),"Away SP K% vR":c.get("away_pitcher_K%_vR"),
            "Home SP K% Platoon":c.get("home_pitcher_K%_platoon"),
            "Away SP K% Platoon":c.get("away_pitcher_K%_platoon"),
            "Away Lineup LHH":c.get("away_lineup_lhh"),"Away Lineup RHH":c.get("away_lineup_rhh"),
            "Home Lineup LHH":c.get("home_lineup_lhh"),"Home Lineup RHH":c.get("home_lineup_rhh"),
            "Home SP AvgIP":c.get("home_pitcher_AvgIP"),"Away SP AvgIP":c.get("away_pitcher_AvgIP"),
            "Away Lineup Score":c.get("away_lineup_vuln_score"),"Home Lineup Score":c.get("home_lineup_vuln_score"),
            "Park":env.get("park"),"Park Factor":env.get("park_factor"),"Roof":env.get("roof"),
            "Temp F":wx.get("temp_f"),"Wind MPH":wx.get("wind_mph"),"Wind Dir":wx.get("wind_dir_label"),
            "Env Adj":env.get("env_total_adj"),"Rain Risk":env.get("rain_risk"),
            "Home BP Score":hbp.get("bullpen_score"),"Home BP Grade":hbp.get("grade"),
            "Home BP Tired Arms":len(hbp.get("tired_arms",[])),"Home Closer":hbp.get("closer",{}).get("name") if hbp.get("closer") else None,
            "Away BP Score":abp.get("bullpen_score"),"Away BP Grade":abp.get("grade"),
            "Away BP Tired Arms":len(abp.get("tired_arms",[])),"Away Closer":abp.get("closer",{}).get("name") if abp.get("closer") else None,
            "ML Lean":s.get("ML_lean"),"ML Conf":s.get("ML_conf"),
            "ML IP Note":s.get("ML_ip_note",""),
            "ML Odds Home":c.get("odds_ml_home"),"ML Odds Away":c.get("odds_ml_away"),
            "ML Home Implied":c.get("odds_ml_home_implied"),
            "ML Away Implied":c.get("odds_ml_away_implied"),
            "ML Value Note":s.get("ml_value_note",""),
            "Total Lean":s.get("total_lean"),"Total Conf":s.get("total_conf"),
            "Total IP Note":s.get("total_ip_note",""),
            "Total Line":c.get("odds_total_line"),
            "Total Over Price":c.get("odds_total_over_price"),
            "Total Under Price":c.get("odds_total_under_price"),
            "Total Value Rating":s.get("total_value_rating",""),
            "Total Value Note":s.get("total_value_note",""),
            "F5 Lean":s.get("f5_lean"),"F5 Conf":s.get("f5_conf"),
            "F5 IP Note":s.get("f5_ip_note",""),
            "F5 ML Home":c.get("odds_f5_ml_home"),"F5 ML Away":c.get("odds_f5_ml_away"),
            "F5 Total Line":c.get("odds_f5_total_line"),
            "F5 Over Price":c.get("odds_f5_over_price"),
            "F5 Under Price":c.get("odds_f5_under_price"),
            "Odds Bookmaker":c.get("odds_bookmaker",""),
            "NRFI":s.get("nrfi_lean"),"NRFI Conf":s.get("nrfi_conf"),
            "NRFI IP Note":s.get("nrfi_ip_note",""),
            "K-Prop Targets":" | ".join(s.get("k_prop_targets",[])),
            "BB PROP":" | ".join(
                f"{b['lean']} [{b['conf']}]"
                for b in s.get("bb_prop_signals", [])
            ),
            "TEAM TOTAL":" | ".join(
                f"{t['lean']} [{t['conf']}]"
                for t in s.get("team_total_signals", [])
            ),
            "Fade Notes":" | ".join(s.get("fade_notes",[])),
            "Bullpen Notes":" | ".join(s.get("bullpen_notes",[])),
        })
    out=pd.DataFrame(rows); out.to_csv(filename,index=False)
    print(f"\n✅  Full model output saved → {filename}")
    return out

# ============================================================
# PHASE 7 — RESULTS LOGGER + WIN RATE TRACKER
# ============================================================

RESULTS_FILE = "results_log.csv"

import requests as _requests_early, json as _json_early
GITHUB_TOKEN = "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"  # create at github.com/settings/tokens
GIST_IDS = {
    "daily_matchup_full.csv": "2646eb7878b6b52ac71cff0cdeec67ef",
    "pitcher_targets.csv":    "2c6c7f7f94bdb67e901db9baa7d77697",
    "results_log.csv":        "3c79f6bdd7b4d6d33c5dd0934398be6b",
}
GIST_FILENAMES = {
    "daily_matchup_full.csv": "mlb_matchup.csv",
    "pitcher_targets.csv":    "mlb_targets.csv",
    "results_log.csv":        "mlb_results.csv",
}

def _sync_local_results_log_from_gist():
    """
    Pull the live Gist copy of results_log.csv at script START and merge any
    dashboard-graded changes (Result/PnL/Odds/Stake/Score) into the LOCAL
    file on disk, before the script does anything else with it. This stops
    the script from working off stale local data — e.g. re-treating an
    already-VOIDed or already-graded dashboard row as still PENDING.
    """
    import os
    print("🔄  Pre-sync: checking Gist for dashboard-graded changes...")
    if not os.path.exists(RESULTS_FILE):
        # Fresh environment (e.g. GitHub Actions) — download from Gist first
        print(f"  ℹ  {RESULTS_FILE} not found locally — downloading from Gist...")
        try:
            resp = _requests_early.get(
                f"https://api.github.com/gists/{GIST_IDS['results_log.csv']}",
                headers={"Authorization": f"token {GITHUB_TOKEN}",
                         "Accept": "application/vnd.github.v3+json"},
            )
            if resp.status_code == 200:
                files = resp.json().get("files", {})
                file_obj = files.get(GIST_FILENAMES["results_log.csv"])
                if file_obj and file_obj.get("content","").strip():
                    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
                        f.write(file_obj["content"])
                    print(f"  ✅ Downloaded {RESULTS_FILE} from Gist ({file_obj['content'].count(chr(10))} rows)")
                else:
                    print(f"  ⚠ Gist has no content for {RESULTS_FILE} — starting fresh")
                    return
            else:
                print(f"  ⚠ Could not download from Gist: {resp.status_code} — starting fresh")
                return
        except Exception as e:
            print(f"  ⚠ Download failed: {e} — starting fresh")
            return
    try:
        resp = _requests_early.get(
            f"https://api.github.com/gists/{GIST_IDS['results_log.csv']}",
            headers={"Authorization": f"token {GITHUB_TOKEN}",
                     "Accept": "application/vnd.github.v3+json"},
        )
        if resp.status_code != 200:
            print(f"  ⚠ Could not pull Gist for pre-sync: {resp.status_code}")
            return
        files = resp.json().get("files", {})
        file_obj = files.get(GIST_FILENAMES["results_log.csv"])
        if not file_obj:
            print("  ⚠ Pre-sync: Gist file object not found")
            return
        remote_text = file_obj.get("content", "")
        if not remote_text.strip():
            print("  ⚠ Pre-sync: remote Gist content is empty")
            return

        import io
        remote_df = pd.read_csv(io.StringIO(remote_text))
        local_df  = pd.read_csv(RESULTS_FILE)
        print(f"  Pre-sync: remote has {len(remote_df)} rows, local has {len(local_df)} rows")
        if remote_df.empty or local_df.empty:
            print("  ⚠ Pre-sync: one side is empty, skipping merge")
            return

        def _row_key(row):
            import math
            def _clean(v):
                if v is None: return ""
                s = str(v).strip()
                return "" if s.lower() in ("nan", "none", "") else s
            pitcher = _clean(row.get("Pitcher", ""))
            date    = _clean(row.get("Date", ""))
            matchup = _clean(row.get("Matchup", ""))
            bettype = _clean(row.get("Bet_Type", ""))
            return (date, matchup, bettype, pitcher)

        grading_cols = ["Result", "PnL_Units", "Kelly_Units", "Stake_Dollars",
                         "Odds", "Decimal_Odds", "Implied_Prob",
                         "Daily_Cap_Applied", "Home_Score", "Away_Score"]

        remote_lookup = {_row_key(r): r for _, r in remote_df.iterrows()}
        synced = 0
        unmatched = 0
        for idx, row in local_df.iterrows():
            key = _row_key(row)
            if key in remote_lookup:
                rrow = remote_lookup[key]
                remote_result = str(rrow.get("Result", "PENDING")).strip()
                local_result  = str(row.get("Result", "PENDING")).strip()
                if remote_result != "PENDING" and remote_result != local_result:
                    for col in grading_cols:
                        if col in rrow and col in local_df.columns:
                            local_df.at[idx, col] = rrow[col]
                    synced += 1
            else:
                unmatched += 1

        print(f"  Pre-sync: {synced} row(s) updated, {unmatched} local row(s) had no key match in remote")

        if synced:
            local_df.to_csv(RESULTS_FILE, index=False)
            print(f"🔄  Synced {synced} dashboard-graded row(s) from Gist into local file before run")
        else:
            print("  Pre-sync: nothing to merge (local already matches remote, or no key overlap)")
    except Exception as e:
        print(f"  ⚠ Pre-sync from Gist failed (continuing with local file): {e}")

_sync_local_results_log_from_gist()

RESULTS_COLUMNS = [
    "Date", "Matchup", "Game_PK",
    "Bet_Type",       # ML | F5 | F5_TOTAL | TOTAL | NRFI | K_PROP
    "Lean",           # e.g. "HOME (Yankees)", "OVER", "STRONG NRFI"
    "Confidence",     # HIGH | MED
    "Pitcher",        # K_PROP / NRFI only
    "Odds",           # American odds at time of logging (e.g. +153, -112)
    "Implied_Prob",   # implied probability % derived from Odds
    "Line",           # total/F5 line (e.g. 8.5) — same as Notes for totals
    "Bookmaker",      # which book the odds came from
    "Home_Score", "Away_Score",
    "Result",         # WIN | LOSS | PUSH | PENDING | VOID
    "PnL_Units",      # profit/loss in units — Kelly-sized, filled on grading
    "Kelly_Units",    # staked units after daily cap applied
    "Kelly_Units_Raw",# staked units before daily cap
    "Stake_Dollars",  # dollar stake at $20/unit, rounded to nearest $5
    "Daily_Cap_Applied", # True/False — was the daily cap scaling applied?
    "Notes",          # manual field (actual K total for K_PROP etc.)
]

def _load_results_log():
    try:
        df = pd.read_csv(RESULTS_FILE)
        for col in RESULTS_COLUMNS:
            if col not in df.columns:
                df[col] = None
        def _norm_date(d):
            try: return pd.to_datetime(str(d).strip()).strftime("%Y-%m-%d")
            except: return str(d).strip()
        df["Date"] = df["Date"].apply(_norm_date)
        return df
    except FileNotFoundError:
        return pd.DataFrame(columns=RESULTS_COLUMNS)

def _save_results_log(df):
    df.to_csv(RESULTS_FILE, index=False)

def _american_odds_to_pnl(odds, result, kelly_units=1.0):
    """
    Convert American odds + result to P&L in units, scaled by Kelly stake.
    WIN:  decimal profit × kelly_units.  LOSS: -kelly_units.  PUSH: 0.0.
    Falls back to 1.0u when no Kelly stake stored.
    """
    ku = float(kelly_units) if kelly_units not in (None, "", "nan") else 1.0
    if result not in ("WIN", "LOSS", "PUSH"):
        return None
    if result == "LOSS":  return round(-ku, 4)
    if result == "PUSH":  return 0.0
    if odds is None:      return round(ku * 1.0, 4)   # no odds: treat as even money win
    try:
        o = float(odds)
        if o > 0:  return round((o / 100) * ku, 4)
        else:      return round((100 / abs(o)) * ku, 4)
    except:
        return round(ku * 1.0, 4)


# ── Kelly staking helpers ────────────────────────────────────────────────────

def _american_to_decimal(odds):
    """Convert American odds to decimal."""
    try:
        o = float(odds)
        if o > 0:  return round(1 + o / 100, 4)
        else:      return round(1 + 100 / abs(o), 4)
    except:
        return None

def _calc_kelly_units(quality_score, odds_american, threshold=40):
    """
    Stake sizing in units.

    When odds are available: quarter-Kelly from edge estimate.
    When odds are None (NRFI, K_PROP, BB_PROP, TEAM_TOTAL): quality-based
    synthetic stake — higher quality = larger bet, capped at 3.0u since
    there is no odds-based edge validation for these markets.

    threshold: the quality gate for this market (used to normalise
               the quality-based stake). Defaults to 40.
    """
    if odds_american is None:
        # Quality-based synthetic stake for no-odds markets
        # At threshold: 0.5u minimum; scales to 3.0u at quality 100
        if quality_score < threshold:
            return FLAT_STAKE_UNITS
        norm = min(1.0, (quality_score - threshold) / (100 - threshold))
        stake = 0.5 + norm * 2.5
        # Round to nearest 0.5u, floor 0.5u, cap 3.0u
        stake = max(0.5, min(3.0, round(stake * 2) / 2))
        return stake

    dec = _american_to_decimal(odds_american)
    if dec is None or dec <= 1.0:
        return FLAT_STAKE_UNITS

    # Edge estimate: quality score above threshold scales 0→15% edge
    edge = max(0.0, (quality_score - 40) / 60 * 0.15)

    if dec - 1 <= 0:
        return FLAT_STAKE_UNITS

    kelly_raw = KELLY_FRACTION * (edge / (dec - 1))
    kelly_units = kelly_raw * 100

    # Floor at 0.5u, cap at MAX_KELLY_UNITS
    kelly_units = max(0.5, min(kelly_units, MAX_KELLY_UNITS))
    return round(kelly_units, 2)

def _apply_daily_cap(new_rows):
    """
    If total raw Kelly units across today's new bets exceeds DAILY_STAKE_CAP,
    scale all stakes proportionally. Updates each row's Kelly_Units in place.
    Returns (scale_factor, was_capped).
    """
    total_raw = sum(r.get("Kelly_Units_Raw", 1.0) for r in new_rows)
    if total_raw <= DAILY_STAKE_CAP:
        for r in new_rows:
            r["Kelly_Units"] = r["Kelly_Units_Raw"]
        return 1.0, False

    scale = DAILY_STAKE_CAP / total_raw
    for r in new_rows:
        r["Kelly_Units"] = round(r["Kelly_Units_Raw"] * scale, 2)
    return scale, True


def log_bets_to_results(game_cards):
    log      = _load_results_log()
    new_rows = []

    existing_keys = set()
    if not log.empty:
        for _, row in log.iterrows():
            bt = str(row.get("Bet_Type", ""))
            pitcher = str(row.get("Pitcher", ""))
            if bt in ("K_PROP", "BB_PROP", "TEAM_TOTAL"):
                # Include pitcher/lean for per-pitcher dedup
                existing_keys.add((str(row["Date"]), str(row["Matchup"]), bt, pitcher))
            else:
                existing_keys.add((str(row["Date"]), str(row["Matchup"]), bt))

    for card in game_cards:
        matchup  = card.get("matchup", "")
        game_pk  = card.get("game_pk", "")
        signals  = card.get("signals", {})

        def _add(bet_type, lean, conf, pitcher=None):
            if lean in (None, "NEUTRAL", "NEUTRAL / AVOID", "AVOID — RAIN RISK", "SUPPRESSED"):
                return
            if conf in (None, "LOW", "N/A"):
                return
            key = (today, matchup, bet_type)
            if key in existing_keys:
                return

            # Quality gate — require composite score above threshold
            quality   = score_play_quality(card, bet_type, lean)
            threshold = QUALITY_THRESHOLDS.get(bet_type, 50)
            if quality < threshold:
                print(f"    ⛔ {bet_type} {lean[:30]} quality {quality} < {threshold} — skipped")
                return

            existing_keys.add(key)

            # ── Capture the relevant odds at bet-time ──────────────────
            odds_val    = None
            implied_val = None
            line_val    = None
            bookmaker   = card.get("odds_bookmaker", "")

            if bet_type == "ML":
                lean_is_home = "HOME" in lean
                odds_val = card.get("odds_ml_home") if lean_is_home else card.get("odds_ml_away")
                implied_val = (card.get("odds_ml_home_implied") if lean_is_home
                               else card.get("odds_ml_away_implied"))

            elif bet_type == "F5":
                lean_is_home = "HOME" in lean
                odds_val = card.get("odds_f5_ml_home") if lean_is_home else card.get("odds_f5_ml_away")

            elif bet_type == "F5_TOTAL":
                # F5 UNDER — use under price
                odds_val = card.get("odds_f5_under_price")
                line_val = str(card["odds_f5_total_line"]) if card.get("odds_f5_total_line") else ""

            elif bet_type == "TOTAL":
                is_over  = lean == "OVER"
                odds_val = (card.get("odds_total_over_price") if is_over
                            else card.get("odds_total_under_price"))
                line_val = str(card["odds_total_line"]) if card.get("odds_total_line") else ""

            elif bet_type == "NRFI":
                # NRFI odds not currently fetched — leave blank
                odds_val = None

            # Implied probability from odds
            if odds_val is not None and implied_val is None:
                implied_val = _american_to_implied(odds_val)

            # Notes: store line for totals (used by grader)
            notes_val = line_val or ""

            kelly_raw = _calc_kelly_units(quality, odds_val,
                                          threshold=QUALITY_THRESHOLDS.get(bet_type, 40))
            new_rows.append({
                "Date": today, "Matchup": matchup, "Game_PK": game_pk,
                "Bet_Type": bet_type, "Lean": lean, "Confidence": conf,
                "Pitcher": pitcher or "",
                "Odds": odds_val, "Implied_Prob": implied_val,
                "Line": line_val or "", "Bookmaker": bookmaker,
                "Home_Score": None, "Away_Score": None,
                "Result": "PENDING", "PnL_Units": None,
                "Kelly_Units_Raw": kelly_raw, "Kelly_Units": kelly_raw,
                "Stake_Dollars": None, "Daily_Cap_Applied": False,
                "Notes": notes_val,
            })

        _add("ML",    signals.get("ML_lean"),    signals.get("ML_conf"))
        f5_lean = signals.get("f5_lean", "NEUTRAL")
        if f5_lean not in ("NEUTRAL", "UNDER"):
            _add("F5", f5_lean, signals.get("f5_conf"))
        elif f5_lean == "UNDER":
            _add("F5_TOTAL", "UNDER", signals.get("f5_conf"))
        _add("TOTAL", signals.get("total_lean"),  signals.get("total_conf"))
        nrfi = signals.get("nrfi_lean", "NEUTRAL / AVOID")
        if "NRFI" in nrfi:
            _add("NRFI", nrfi, signals.get("nrfi_conf"))

        # Only log the single BEST K_PROP target per game (highest quality),
        # not every pitcher that clears the threshold — prevents 2 correlated
        # bets on the same game.
        kp_candidates = []
        for kp in signals.get("k_prop_targets", []):
            if kp and kp != "None":
                pitcher_name = kp.split("(")[0].strip()
                quality = score_play_quality(card, "K_PROP", kp)
                kp_candidates.append((quality, kp, pitcher_name))
        kp_candidates.sort(reverse=True)  # highest quality first

        if kp_candidates:
            quality, kp, pitcher_name = kp_candidates[0]
            kp_key = (today, matchup, "K_PROP", pitcher_name)
            threshold = QUALITY_THRESHOLDS.get("K_PROP", 45)
            if kp_key not in existing_keys and quality >= threshold:
                existing_keys.add(kp_key)
                kp_kelly_raw = _calc_kelly_units(quality, None, threshold=QUALITY_THRESHOLDS["K_PROP"])
                new_rows.append({
                    "Date": today, "Matchup": matchup, "Game_PK": game_pk,
                    "Bet_Type": "K_PROP", "Lean": kp, "Confidence": "MED",
                    "Pitcher": pitcher_name,
                    "Odds": None, "Implied_Prob": None,
                    "Line": "", "Bookmaker": "",
                    "Home_Score": None, "Away_Score": None,
                    "Result": "PENDING", "PnL_Units": None,
                    "Kelly_Units_Raw": kp_kelly_raw, "Kelly_Units": kp_kelly_raw,
                    "Stake_Dollars": None, "Daily_Cap_Applied": False,
                    "Notes": "",
                })
            elif kp_key not in existing_keys:
                print(f"    ⛔ K_PROP {pitcher_name} quality {quality} < {threshold} — skipped")

        # ── TEAM_TOTAL logging ────────────────────────────────────────────
        for tt in signals.get("team_total_signals", []):
            lean = tt["lean"]
            conf = tt["conf"]
            tt_pitcher = lean.split("opp: ")[1].split(" ")[0] if "opp:" in lean else lean[:20]
            key  = (today, matchup, "TEAM_TOTAL", tt_pitcher)
            if key in existing_keys:
                continue
            quality   = score_play_quality(card, "TEAM_TOTAL", lean)
            threshold = QUALITY_THRESHOLDS.get("TEAM_TOTAL", 35)
            if quality < threshold:
                print(f"    ⛔ TEAM_TOTAL {lean[:35]} quality {quality} < {threshold} — skipped")
                continue
            existing_keys.add(key)
            tt_kelly = _calc_kelly_units(quality, card.get("odds_total_over_price"),
                                         threshold=QUALITY_THRESHOLDS["TEAM_TOTAL"])
            new_rows.append({
                "Date": today, "Matchup": matchup, "Game_PK": game_pk,
                "Bet_Type": "TEAM_TOTAL", "Lean": lean, "Confidence": conf,
                "Pitcher": lean.split("opp: ")[1].split(" ")[0] if "opp:" in lean else "",
                "Odds": card.get("odds_total_over_price"), "Implied_Prob": None,
                "Line": str(card.get("odds_total_line","") or ""), "Bookmaker": card.get("odds_bookmaker",""),
                "Home_Score": None, "Away_Score": None,
                "Result": "PENDING", "PnL_Units": None,
                "Kelly_Units_Raw": tt_kelly, "Kelly_Units": tt_kelly,
                "Stake_Dollars": None, "Daily_Cap_Applied": False,
                "Notes": "",
            })

        # ── BB_PROP logging ───────────────────────────────────────────────
        # Only log the single BEST BB_PROP signal per game (highest quality),
        # not both pitchers — prevents 2 correlated bets on one game.
        bb_candidates = []
        for bb in signals.get("bb_prop_signals", []):
            quality = score_play_quality(card, "BB_PROP", bb["lean"])
            bb_candidates.append((quality, bb))
        bb_candidates.sort(key=lambda x: x[0], reverse=True)

        if bb_candidates:
            quality, bb = bb_candidates[0]
            lean = bb["lean"]
            conf = bb["conf"]
            pitcher_name = bb["pitcher"]
            key  = (today, matchup, "BB_PROP", pitcher_name)
            threshold = QUALITY_THRESHOLDS.get("BB_PROP", 30)
            if key not in existing_keys and quality >= threshold:
                existing_keys.add(key)
                bb_kelly = _calc_kelly_units(quality, None, threshold=QUALITY_THRESHOLDS["BB_PROP"])
                new_rows.append({
                    "Date": today, "Matchup": matchup, "Game_PK": game_pk,
                    "Bet_Type": "BB_PROP", "Lean": lean, "Confidence": conf,
                    "Pitcher": pitcher_name,
                    "Odds": None, "Implied_Prob": None,
                    "Line": "", "Bookmaker": "",
                    "Home_Score": None, "Away_Score": None,
                    "Result": "PENDING", "PnL_Units": None,
                    "Kelly_Units_Raw": bb_kelly, "Kelly_Units": bb_kelly,
                    "Stake_Dollars": None, "Daily_Cap_Applied": False,
                    "Notes": "Manual odds entry required",
                })
            elif key not in existing_keys:
                print(f"    ⛔ BB_PROP {lean[:35]} quality {quality} < {threshold} — skipped")

    if new_rows:
        # ── Per-game bet cap — max MAX_BETS_PER_GAME bets on the same matchup ─
        # Prevents correlated exposure (TOTAL+F5_TOTAL+NRFI+K_PROP on one    ─
        # game). Keeps the highest-quality bets per game first.              ─
        from collections import defaultdict
        game_counts = defaultdict(list)
        for r in new_rows:
            game_counts[r["Matchup"]].append(r)

        per_game_kept = []
        _conf_rank = {"HIGH": 0, "MED": 1, "LOW": 2}
        for matchup_key, rows in game_counts.items():
            if len(rows) <= MAX_BETS_PER_GAME:
                per_game_kept.extend(rows)
            else:
                rows.sort(key=lambda r: (_conf_rank.get(r.get("Confidence","MED"), 1),
                                          -r.get("Kelly_Units_Raw", 0)))
                kept    = rows[:MAX_BETS_PER_GAME]
                dropped = rows[MAX_BETS_PER_GAME:]
                per_game_kept.extend(kept)
                print(f"\n⚠   Per-game cap ({MAX_BETS_PER_GAME} max) on {matchup_key}:")
                for d in dropped:
                    print(f"    ⛔ Cut (game cap): {d['Bet_Type']} — {d['Lean'][:50]}")
        new_rows = per_game_kept

        # ── Cap total plays per day — keep the MAX_DAILY_PLAYS highest        ─
        # conviction bets. Kelly_Units_Raw correlates directly with quality    ─
        # score (both odds-based and synthetic), so it's a reliable proxy for ─
        # ranking without needing to re-run score_play_quality() here.        ─
        if len(new_rows) > MAX_DAILY_PLAYS:
            # Priority: HIGH confidence first, then by Kelly_Units_Raw desc
            conf_rank = {"HIGH": 0, "MED": 1, "LOW": 2}
            new_rows.sort(key=lambda r: (conf_rank.get(r.get("Confidence","MED"), 1),
                                          -r.get("Kelly_Units_Raw", 0)))
            dropped = new_rows[MAX_DAILY_PLAYS:]
            new_rows = new_rows[:MAX_DAILY_PLAYS]
            print(f"\n⚠   Daily play cap: {len(dropped) + MAX_DAILY_PLAYS} candidates → "
                  f"keeping top {MAX_DAILY_PLAYS} by confidence/quality")
            for d in dropped:
                print(f"    ⛔ Cut (play cap): {d['Bet_Type']} {d['Matchup']} — {d['Lean'][:40]}")

        # ── Apply daily cap and convert to dollars ────────────────────────
        scale, was_capped = _apply_daily_cap(new_rows)
        for r in new_rows:
            # Round dollar stake to nearest $5, minimum $5
            raw_dollars = r["Kelly_Units"] * UNIT_SIZE_AUD
            r["Stake_Dollars"] = max(5, round(raw_dollars / 5) * 5)
            r["Daily_Cap_Applied"] = was_capped

        log = pd.concat([log, pd.DataFrame(new_rows)], ignore_index=True)
        _save_results_log(log)

        # ── Staking summary printout ──────────────────────────────────────
        total_units  = sum(r["Kelly_Units"] for r in new_rows)
        total_dollars = sum(r["Stake_Dollars"] for r in new_rows)
        print(f"\n📝  Phase 7: {len(new_rows)} new bets logged to {RESULTS_FILE}")
        cap_note = f"  ⚠ Daily cap fired: scaled ×{scale:.3f}" if was_capped else ""
        print(f"\n💰  Staking summary (unit=${UNIT_SIZE_AUD:.0f}){cap_note}")
        print(f"    {'Matchup':<35} {'Type':<10} {'Kelly':>6}u  {'Stake':>6}  Odds")
        print(f"    {'-'*75}")
        for r in new_rows:
            ku   = r["Kelly_Units"]
            sd   = r["Stake_Dollars"]
            odds = r["Odds"] if r["Odds"] else "  —"
            lean = (r["Lean"] or "")[:28]
            print(f"    {r['Matchup'][:35]:<35} {r['Bet_Type']:<10} {ku:>5.2f}u  ${sd:<5}  {odds}")
        print(f"    {'-'*75}")
        print(f"    {'TOTAL':<46} {total_units:>5.2f}u  ${total_dollars}")
    else:
        print(f"\n📝  Phase 7: No new bets to log (all already present, neutral, or below quality threshold).")

    return log


# ── Score fetching helpers ────────────────────────────────────────────────────

def _fetch_final_score(game_pk):
    """
    FIX G1: Fetch final score only for COMPLETED games (9+ innings).
    Old version returned scores for in-progress games.
    """
    try:
        url     = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/linescore"
        data    = requests.get(url, timeout=10).json()
        innings = data.get("innings", [])
        if not innings:
            return None, None
        # FIX G1: require at least 9 innings in linescore before grading
        if len(innings) < 9:
            return None, None
        home_total = sum(i.get("home", {}).get("runs", 0) or 0 for i in innings)
        away_total = sum(i.get("away", {}).get("runs", 0) or 0 for i in innings)
        if home_total == 0 and away_total == 0:
            return None, None
        return int(home_total), int(away_total)
    except Exception as e:
        print(f"    Score fetch error (game_pk {game_pk}): {e}")
        return None, None


def _fetch_f5_score(game_pk):
    """
    FIX G2: Pad missing innings — MLB API omits 0-run innings in some responses.
    """
    try:
        url     = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/linescore"
        data    = requests.get(url, timeout=10).json()
        innings = data.get("innings", [])
        if len(innings) < 5:
            return None, None
        # FIX G2: guard against None inning dicts
        home_f5 = sum((i.get("home") or {}).get("runs", 0) or 0 for i in innings[:5])
        away_f5 = sum((i.get("away") or {}).get("runs", 0) or 0 for i in innings[:5])
        return int(home_f5), int(away_f5)
    except Exception as e:
        print(f"    F5 score fetch error (game_pk {game_pk}): {e}")
        return None, None


def _fetch_first_inning_score(game_pk):
    """
    FIX G2 (same guard): Returns (home_r, away_r) for 1st inning (NRFI grading).
    """
    try:
        url     = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/linescore"
        data    = requests.get(url, timeout=10).json()
        innings = data.get("innings", [])
        if not innings:
            return None, None
        first  = innings[0]
        # FIX G2: guard against None
        home_r = (first.get("home") or {}).get("runs", 0) or 0
        away_r = (first.get("away") or {}).get("runs", 0) or 0
        return int(home_r), int(away_r)
    except Exception as e:
        print(f"    NRFI score fetch error (game_pk {game_pk}): {e}")
        return None, None


# ── Grading helpers ───────────────────────────────────────────────────────────

def _grade_ml(lean, home_score, away_score):
    if home_score is None: return "PENDING"
    home_win = home_score > away_score
    away_win = away_score > home_score
    if "HOME" in lean: return "WIN" if home_win else ("PUSH" if home_score == away_score else "LOSS")
    if "AWAY" in lean: return "WIN" if away_win else ("PUSH" if home_score == away_score else "LOSS")
    return "VOID"

def _grade_total(lean, home_score, away_score, line=None):
    if home_score is None: return "PENDING"
    total = home_score + away_score
    try:
        threshold = float(line) if line else None
    except:
        threshold = None
    if threshold is None:
        return "PENDING"   # FIX G4: don't default to wrong 8.5
    if lean == "OVER":
        if total > threshold:  return "WIN"
        if total == threshold: return "PUSH"
        return "LOSS"
    if lean == "UNDER":
        if total < threshold:  return "WIN"
        if total == threshold: return "PUSH"
        return "LOSS"
    return "VOID"

def _grade_f5(lean, home_f5, away_f5, line=None):
    if home_f5 is None: return "PENDING"
    if "HOME" in lean: return "WIN" if home_f5 > away_f5 else ("PUSH" if home_f5 == away_f5 else "LOSS")
    if "AWAY" in lean: return "WIN" if away_f5 > home_f5 else ("PUSH" if home_f5 == away_f5 else "LOSS")
    if lean == "UNDER":
        try: threshold = float(line) if line else 4.5
        except: threshold = 4.5
        total = home_f5 + away_f5
        if total < threshold:  return "WIN"
        if total == threshold: return "PUSH"
        return "LOSS"
    return "VOID"

def _grade_nrfi(home_r1, away_r1):
    if home_r1 is None: return "PENDING"
    return "WIN" if (home_r1 == 0 and away_r1 == 0) else "LOSS"


def fetch_and_log_results(target_date=None):
    """
    Call the morning after a slate to auto-grade all PENDING bets.
    FIX G3: game_pk cast to int to strip pandas float (.0) before API call.
    FIX G4: TOTAL bets without a line in Notes are skipped with a warning
            instead of silently grading against the wrong 8.5 default.
    """
    log = _load_results_log()
    if log.empty:
        print("  No results log found.")
        return log

    grade_date = target_date or (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    pending    = log[(log["Date"] == grade_date) & (log["Result"] == "PENDING")].copy()

    if pending.empty:
        print(f"  No PENDING bets found for {grade_date}.")
        return log

    print(f"\n🔍  Grading {len(pending)} PENDING bets for {grade_date}...")

    score_cache = {}
    f5_cache    = {}
    nrfi_cache  = {}

    graded = 0
    for idx, row in pending.iterrows():
        # FIX G3: strip pandas .0 float suffix from game_pk
        try:
            gp = str(int(float(row.get("Game_PK", "") or 0)))
        except:
            gp = str(row.get("Game_PK", "")).strip()

        bet_type = row.get("Bet_Type", "")
        lean     = str(row.get("Lean", ""))
        notes    = str(row.get("Notes", "")).strip() if pd.notnull(row.get("Notes")) else ""

        if not gp or gp in ("0", "nan", ""):
            print(f"    ⚠  Missing game_pk for {row.get('Matchup','?')} — skipping")
            continue

        if gp not in score_cache:
            score_cache[gp] = _fetch_final_score(gp)
        if gp not in f5_cache:
            f5_cache[gp]    = _fetch_f5_score(gp)
        if gp not in nrfi_cache:
            nrfi_cache[gp]  = _fetch_first_inning_score(gp)

        home_s,  away_s  = score_cache[gp]
        home_f5, away_f5 = f5_cache[gp]
        home_r1, away_r1 = nrfi_cache[gp]

        result = "PENDING"
        if bet_type == "ML":
            result = _grade_ml(lean, home_s, away_s)

        elif bet_type in ("F5", "F5_TOTAL"):
            # Read from Line column first, fall back to Notes
            line_val = str(row.get("Line", "")).strip() if pd.notnull(row.get("Line")) else ""
            if not line_val or not line_val.replace('.','').isdigit():
                line_val = notes
            line = line_val if line_val and line_val.replace('.','').isdigit() else None
            result = _grade_f5(lean, home_f5, away_f5, line=line)

        elif bet_type == "TOTAL":
            # FIX G4: read from Line column first, fall back to Notes
            line_val = str(row.get("Line", "")).strip() if pd.notnull(row.get("Line")) else ""
            if not line_val or not line_val.replace('.','').isdigit():
                line_val = notes  # legacy rows may still use Notes
            if line_val and line_val.replace('.','').isdigit():
                result = _grade_total(lean, home_s, away_s, line=line_val)
            else:
                print(f"    ⚠  No line for TOTAL ({row.get('Matchup','?')}) "
                      f"— add actual line to Line or Notes column in results_log.csv and re-run")
                continue

        elif bet_type == "NRFI":
            result = _grade_nrfi(home_r1, away_r1)

        elif bet_type == "K_PROP":
            result = "PENDING"  # manual — enter actual K total in Notes

        if result != "PENDING":
            log.at[idx, "Home_Score"] = home_s
            log.at[idx, "Away_Score"] = away_s
            log.at[idx, "Result"]     = result
            # Calculate P&L using the odds stored at bet-time, scaled by Kelly stake
            bet_odds   = row.get("Odds")
            kelly_u    = row.get("Kelly_Units") if "Kelly_Units" in row.index else 1.0
            pnl        = _american_odds_to_pnl(bet_odds, result, kelly_units=kelly_u)
            stake_d    = row.get("Stake_Dollars") if "Stake_Dollars" in row.index else None
            log.at[idx, "PnL_Units"] = pnl
            graded += 1
            score_str  = f"{away_s}-{home_s}" if home_s is not None else "?"
            stake_str  = f"${int(stake_d)}" if stake_d not in (None, "", "nan") else f"{kelly_u:.2f}u"
            pnl_str    = f"{pnl:+.2f}u" if pnl is not None else "  ?"
            print(f"    {row['Matchup'][:35]:<35} {bet_type:<8} {lean[:25]:<25} {score_str}  → {result}  [{stake_str}]  {pnl_str}")

    _save_results_log(log)
    print(f"\n✅  Grading complete: {graded} bets graded. Log saved → {RESULTS_FILE}")
    return log


def print_results_summary(days=14, min_bets=3):
    log = _load_results_log()
    if log.empty:
        print("  No results log found.")
        return

    cutoff = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    graded = log[
        (log["Date"] >= cutoff) &
        (log["Result"].isin(["WIN", "LOSS", "PUSH"]))
    ].copy()

    if graded.empty:
        print(f"  No graded results in last {days} days.")
        return

    # Ensure PnL_Units is numeric
    graded["PnL_Units"] = pd.to_numeric(graded.get("PnL_Units", None), errors="coerce")

    def _stats(subset):
        """Returns (win_rate, total_bets, total_pnl) or (None, n, None)."""
        n = len(subset)
        if n < min_bets:
            return None, n, None
        wins  = (subset["Result"] == "WIN").sum() + 0.5 * (subset["Result"] == "PUSH").sum()
        wr    = round(wins / n * 100, 1)
        pnl   = round(subset["PnL_Units"].sum(), 2) if "PnL_Units" in subset.columns else None
        return wr, n, pnl

    def _pnl_str(pnl):
        if pnl is None: return ""
        return f"  P&L: {pnl:+.2f}u"

    divider = "─" * 68
    print(f"\n{'█'*68}")
    print(f"  RESULTS SUMMARY — last {days} days  (min {min_bets} bets to show)")
    print(f"{'█'*68}")

    print(f"\n  BY MARKET\n  {divider}")
    for market in ["ML", "F5", "F5_TOTAL", "TOTAL", "NRFI", "K_PROP"]:
        sub = graded[graded["Bet_Type"] == market]
        wr, n, pnl = _stats(sub)
        if wr is None:
            print(f"  {market:<10} {n} bets  (below min sample)")
        else:
            bar = "█" * int(wr / 5)
            print(f"  {market:<10} {wr:>5.1f}%  ({n} bets){_pnl_str(pnl)}  {bar}")

    print(f"\n  BY CONFIDENCE\n  {divider}")
    for conf in ["HIGH", "MED", "LOW"]:
        sub = graded[graded["Confidence"] == conf]
        wr, n, pnl = _stats(sub)
        if wr is None:
            print(f"  {conf:<6} {n} bets  (below min sample)")
        else:
            bar = "█" * int(wr / 5)
            print(f"  {conf:<6} {wr:>5.1f}%  ({n} bets){_pnl_str(pnl)}  {bar}")

    print(f"\n  BY MARKET × CONFIDENCE\n  {divider}")
    for market in ["ML", "F5", "TOTAL", "NRFI", "K_PROP"]:
        for conf in ["HIGH", "MED"]:
            sub = graded[(graded["Bet_Type"] == market) & (graded["Confidence"] == conf)]
            wr, n, pnl = _stats(sub)
            if wr is not None:
                bar = "█" * int(wr / 5)
                print(f"  {market:<10} {conf:<4}  {wr:>5.1f}%  ({n} bets){_pnl_str(pnl)}  {bar}")

    wr_all, n_all, pnl_all = _stats(graded)
    print(f"\n  {divider}")
    if wr_all is not None:
        print(f"  OVERALL   {wr_all:>5.1f}%  ({n_all} bets){_pnl_str(pnl_all)}")

    # Average odds on winners vs losers
    if "Odds" in graded.columns:
        graded["Odds_num"] = pd.to_numeric(graded["Odds"], errors="coerce")
        winners = graded[graded["Result"] == "WIN"]["Odds_num"].dropna()
        losers  = graded[graded["Result"] == "LOSS"]["Odds_num"].dropna()
        if len(winners) >= 3:
            print(f"  Avg odds — Winners: {winners.mean():.0f}  Losers: {losers.mean():.0f}" if len(losers) >= 3 else f"  Avg odds — Winners: {winners.mean():.0f}")

    print(f"{'█'*68}\n")

    recent = graded.sort_values("Date").tail(10)
    streak = "  LAST 10: " + "  ".join(
        ("✅" if r == "WIN" else "❌" if r == "LOSS" else "➖")
        for r in recent["Result"]
    )
    print(streak)

    # Rolling P&L by date
    if "PnL_Units" in graded.columns and graded["PnL_Units"].notna().any():
        daily = graded.groupby("Date")["PnL_Units"].sum().sort_index()
        cumulative = daily.cumsum()
        print(f"\n  CUMULATIVE P&L ({days}d):")
        for d, cum in cumulative.items():
            day_pnl = daily[d]
            print(f"    {d}  {day_pnl:+.2f}u  (running: {cum:+.2f}u)")
    print()


# ============================================================
# PHASE 9 — EXPORT MANUAL XLSX GRADES BACK TO RESULTS_LOG.CSV
# ============================================================

def export_xlsx_to_results_log(csv_path="results_log.csv", xlsx_path="results_log_template.xlsx"):
    """
    Calls export_xlsx_to_results_log.py to overlay your manually-graded
    Win/Loss/Push/Void results from results_log_template.xlsx onto
    results_log.csv — so the dashboard (which reads results_log.csv via
    the Gist) reflects your manual grading. Your xlsx is authoritative:
    any row you've graded overwrites whatever auto-grading produced.
    Rows still "Pending" in the xlsx are left untouched. Safe to call
    every run (idempotent).
    """
    import subprocess, os, sys

    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "export_xlsx_to_results_log.py")

    if not os.path.exists(script_path):
        print(f"  ⚠  export_xlsx_to_results_log.py not found at {script_path} — skipping")
        return
    if not os.path.exists(xlsx_path):
        print(f"  ⚠  {xlsx_path} not found — skipping")
        return
    if not os.path.exists(csv_path):
        print(f"  ⚠  {csv_path} not found — skipping")
        return

    try:
        result = subprocess.run(
            [sys.executable, script_path, csv_path, xlsx_path],
            capture_output=True, text=True, timeout=60
        )
        if result.stdout:
            for line in result.stdout.strip().splitlines():
                print(f"    {line}")
        if result.returncode != 0:
            print(f"  ⚠  xlsx export failed (exit {result.returncode}): {result.stderr.strip()[:300]}")
    except Exception as e:
        print(f"  ⚠  xlsx export error: {e}")


# ============================================================
# PHASE 8 — AUTO-IMPORT TO RESULTS LOG XLSX
# ============================================================

def auto_import_to_xlsx(csv_path="results_log.csv", xlsx_path="results_log_template.xlsx"):
    """
    Calls import_results_log.py to append new PENDING bets into the
    formatted Excel tracker, preserving manually-graded rows, dropdowns,
    formulas and the summary box. Safe to call every run (idempotent).
    """
    import subprocess, os, sys

    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "import_results_log.py")

    if not os.path.exists(script_path):
        print(f"  ⚠  import_results_log.py not found at {script_path} — skipping xlsx import")
        return
    if not os.path.exists(xlsx_path):
        print(f"  ⚠  {xlsx_path} not found — skipping xlsx import (place the template in the script folder)")
        return
    if not os.path.exists(csv_path):
        print(f"  ⚠  {csv_path} not found — skipping xlsx import")
        return

    try:
        result = subprocess.run(
            [sys.executable, script_path, csv_path, xlsx_path],
            capture_output=True, text=True, timeout=60
        )
        if result.stdout:
            for line in result.stdout.strip().splitlines():
                print(f"    {line}")
        if result.returncode != 0:
            print(f"  ⚠  xlsx import failed (exit {result.returncode}): {result.stderr.strip()[:300]}")
    except Exception as e:
        print(f"  ⚠  xlsx import error: {e}")


# ============================================================
# MAIN RUNNER
# ============================================================
if __name__ == "__main__":

    # ── PHASE 1: Pitcher profiles ─────────────────────────────
    url  = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}&hydrate=probablePitcher"
    data = requests.get(url).json()
    games_today = data.get("dates", [])[0].get("games", []) if data.get("dates") else []

    pitchers = []
    for g in games_today:
        try:
            home = g["teams"]["home"].get("probablePitcher")
            away = g["teams"]["away"].get("probablePitcher")
            if home: pitchers.append(home["fullName"])
            if away: pitchers.append(away["fullName"])
        except:
            continue
    pitchers = list(set(pitchers))
    print(f"Pitchers found: {len(pitchers)}")

    results = []
    for p in pitchers:
        print(f"Processing {p}")
        out = pitcher_profile(p)
        if out: results.append(out)

    df = pd.DataFrame(results)

    for filt_name, filt_df in [
        ("FILTERED TARGET LIST",   df[(df["HardHit%"]>28)&(df["K%"]<22.0)&(df["xwOBA"]>0.330)].sort_values(["HardHit%","xwOBA"],ascending=[False,False])),
        ("K% + BB% TARGET LIST",   df[(df["K%"]>25.0)&(df["BB%"]<8.8)].sort_values(["K%","BB%"],ascending=[False,True])),
        ("ELITE K-PROP TARGETS",   df[(df["K%"]>=25.0)&(df["K-BB%"]>=14.0)&(df["Avg_Pitches_Start"]>=85)].sort_values(["K%","Avg_IP_Start"],ascending=[False,False])),
        ("NRFI TARGETS",           df[(df["TTO_xwOBA"]<0.290)&(df["TTO_K%"]>24.0)].sort_values("TTO_xwOBA")),
    ]:
        print(f"\n{filt_name}\n")
        print(filt_df[["Pitcher","Hand","K%","BB%","K-BB%","xwOBA","HardHit%","Avg_IP_Start"]] if not filt_df.empty else "  (none)")

    df.sort_values("K%",ascending=False).to_csv("daily_pitcher_model.csv",index=False)
    print("\n✅  Phase 1 complete — daily_pitcher_model.csv saved")

    # ── Export pitcher target lists ───────────────────────────
    target_lists = {
        "FILTERED":    df[(df["HardHit%"]>28)&(df["K%"]<22.0)&(df["xwOBA"]>0.330)].sort_values(["HardHit%","xwOBA"],ascending=[False,False]),
        "K_BB":        df[(df["K%"]>25.0)&(df["BB%"]<8.8)].sort_values(["K%","BB%"],ascending=[False,True]),
        "ELITE_KPROP": df[(df["K%"]>=25.0)&(df["K-BB%"]>=14.0)&(df["Avg_Pitches_Start"]>=85)].sort_values(["K%","Avg_IP_Start"],ascending=[False,False]),
        "NRFI":        df[(df["TTO_xwOBA"]<0.290)&(df["TTO_K%"]>24.0)].sort_values("TTO_xwOBA"),
    }
    target_rows = []
    cols = ["Pitcher","Hand","K%","BB%","K-BB%","xwOBA","HardHit%","Avg_IP_Start","TTO_xwOBA","TTO_K%","K%_vL","K%_vR"]
    for list_name, tdf in target_lists.items():
        if not tdf.empty:
            for _, row in tdf.iterrows():
                r = {"List": list_name, "Date": today}
                for c in cols:
                    r[c] = row.get(c)
                target_rows.append(r)
    if target_rows:
        pd.DataFrame(target_rows).to_csv("pitcher_targets.csv", index=False)
        print(f"✅  Pitcher targets saved → pitcher_targets.csv ({len(target_rows)} rows)")
    else:
        pd.DataFrame(columns=["List","Date"]+cols).to_csv("pitcher_targets.csv", index=False)
        print("  No pitcher targets today")

    # ── PHASE 2: Lineups + matchup cards ─────────────────────
    print(f"\n✅  Phase 1 pitcher data loaded: {len(df)} pitchers")
    print("\n🔍  Fetching today's lineups...")
    matchups = get_todays_lineups_with_probables()
    print(f"    {len(matchups)} games found")

    print("\n⚙   Building matchup cards (profiling batters)...")
    game_cards = build_matchup_cards(df, matchups)

    # ── PHASE 3: Park + weather ───────────────────────────────
    print("\n🌤   Fetching park factors + weather...")
    game_cards = apply_park_weather(game_cards)

    # ── PHASE 4: Bullpen analysis ─────────────────────────────
    print("\n🔍  Fetching MLB team IDs...")
    team_id_map = get_team_id_map()
    print(f"    {len(team_id_map)} teams loaded")
    print("\n⚙   Profiling bullpens (3-5 min)...")
    game_cards = apply_bullpen_analysis(game_cards, team_id_map)

    # ── FINAL: resolve any remaining signal conflicts ─────────
    game_cards = apply_final_conflict_resolution(game_cards)

    # ── PHASE 5: Live odds from The Odds API ──────────────────
    game_cards = apply_odds_api(game_cards)

    # ── PRINT + EXPORT ────────────────────────────────────────
    print("\n\n" + "="*62)
    print("  DAILY REPORT — COMPLETE MODEL (P1 + P2 + P3 + P4)")
    print("="*62)
    for card in game_cards:
        print_game_card(card)
        print_environment_card(card)
        print_bullpen_card(card)

    summary = export_full_cards(game_cards)
    print("\n📋  COMPLETE DAILY SUMMARY")
    print(summary[["Matchup","ML Lean","ML Conf","Total Lean","Total Conf",
                   "F5 Lean","F5 Conf","NRFI","NRFI Conf","Home BP Grade","Away BP Grade","Rain Risk"]].to_string(index=False))

    # ── PHASE 7: Log today's bets ─────────────────────────────
    print("\n📝  Logging today's actionable bets...")
    log_bets_to_results(game_cards)

    # ── PHASE 7: Grade yesterday's bets ──────────────────────
    print("\n🔍  Auto-grading yesterday's pending bets...")
    fetch_and_log_results()

    # ── PHASE 7: Print rolling win-rate summary ───────────────
    print_results_summary(days=14)

    # ── MANUAL USAGE REMINDERS ───────────────────────────────
    # To grade a specific date:
    #   fetch_and_log_results("2025-06-10")
    #
    # To see a longer window:
    #   print_results_summary(days=30)
    #
    # If a TOTAL bet has no line in Notes, open results_log.csv,
    # add the actual line (e.g. 8.5) to the Notes column for that
    # row, save, then re-run fetch_and_log_results().

import requests, json

def _pull_gist_csv(gist_id, gist_filename):
    """Fetch the current live content of a Gist file. Returns raw text or None."""
    resp = requests.get(
        f"https://api.github.com/gists/{gist_id}",
        headers={
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json",
        },
    )
    if resp.status_code != 200:
        print(f"  ⚠ Could not pull Gist {gist_filename} for merge: {resp.status_code}")
        return None
    try:
        files = resp.json().get("files", {})
        file_obj = files.get(gist_filename)
        if not file_obj:
            return None
        return file_obj.get("content", "")
    except Exception as e:
        print(f"  ⚠ Error parsing Gist content for {gist_filename}: {e}")
        return None


def _merge_results_log_with_gist(local_df):
    """
    Pull the live Gist version of results_log.csv and merge dashboard-made
    grading changes (Result, PnL_Units, Kelly_Units, Stake_Dollars, Odds,
    Decimal_Odds, Implied_Prob) into the local dataframe before pushing.

    This prevents the script from silently overwriting grades/voids that
    were made directly in the dashboard since the last script run — the
    Gist is the source of truth for anything already pushed there, while
    the local script is the source of truth only for brand-new rows.
    """
    import io
    gist_id       = GIST_IDS["results_log.csv"]
    gist_filename = GIST_FILENAMES["results_log.csv"]
    remote_text   = _pull_gist_csv(gist_id, gist_filename)
    if not remote_text or not remote_text.strip():
        print("  ⚠ No remote results_log.csv found on Gist — pushing local as-is")
        return local_df

    try:
        remote_df = pd.read_csv(io.StringIO(remote_text))
    except Exception as e:
        print(f"  ⚠ Could not parse remote results_log.csv: {e} — pushing local as-is")
        return local_df

    if remote_df.empty:
        return local_df

    # Build a lookup key for each row: (Date, Matchup, Bet_Type, Pitcher)
    # falls back to (Date, Matchup, Bet_Type) for rows with no Pitcher.
    def _row_key(row):
        def _clean(v):
            if v is None: return ""
            s = str(v).strip()
            return "" if s.lower() in ("nan", "none", "") else s
        return (_clean(row.get("Date","")), _clean(row.get("Matchup","")),
                _clean(row.get("Bet_Type","")), _clean(row.get("Pitcher","")))

    grading_cols = ["Result", "PnL_Units", "Kelly_Units", "Stake_Dollars",
                     "Odds", "Decimal_Odds", "Implied_Prob", "Daily_Cap_Applied",
                     "Home_Score", "Away_Score"]

    remote_lookup = {}
    for _, r in remote_df.iterrows():
        remote_lookup[_row_key(r)] = r

    merged_count = 0
    for idx, row in local_df.iterrows():
        key = _row_key(row)
        if key in remote_lookup:
            remote_row = remote_lookup[key]
            remote_result = str(remote_row.get("Result", "PENDING")).strip()
            local_result  = str(row.get("Result", "PENDING")).strip()
            # Only overwrite if the remote has been graded (not still PENDING)
            # and the local copy hasn't already matched it.
            if remote_result != "PENDING" and remote_result != local_result:
                for col in grading_cols:
                    if col in remote_row and col in local_df.columns:
                        local_df.at[idx, col] = remote_row[col]
                merged_count += 1

    if merged_count:
        print(f"  🔄 Synced {merged_count} dashboard-graded row(s) from Gist before pushing")

    return local_df


def push_csv_to_gist(local_filename):
    gist_id       = GIST_IDS[local_filename]
    gist_filename = GIST_FILENAMES[local_filename]

    # For results_log.csv specifically: pull remote first and merge any
    # grading changes made in the dashboard, so we never clobber them.
    if local_filename == "results_log.csv":
        try:
            local_df = pd.read_csv(local_filename)
            local_df = _merge_results_log_with_gist(local_df)
            local_df.to_csv(local_filename, index=False)
        except Exception as e:
            print(f"  ⚠ Merge step failed, pushing local file unmodified: {e}")

    with open(local_filename, "r") as f:
        content = f.read()
    resp = requests.patch(
        f"https://api.github.com/gists/{gist_id}",
        headers={
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json",
        },
        data=json.dumps({"files": {gist_filename: {"content": content}}})
    )
    if resp.status_code == 200:
        print(f"✅ Pushed {gist_filename} to Gist")
    else:
        print(f"❌ Failed {gist_filename}: {resp.status_code} {resp.text[:100]}")

# ── PHASE 9: Push your manual xlsx grades back into results_log.csv ──
print("\n📥  Phase 9: Exporting your manual grades from results_log_template.xlsx...")
export_xlsx_to_results_log()

# Add this at the very end of your __main__ block:
print("\n📤 Pushing CSVs to GitHub Gists...")
for fname in GIST_IDS:
    try:
        push_csv_to_gist(fname)
    except Exception as e:
        print(f"  Error pushing {fname}: {e}")

# ── PHASE 8: Auto-import new bets into Excel tracker ─────────
print("\n📊  Importing new bets into results_log_template.xlsx...")
auto_import_to_xlsx()
