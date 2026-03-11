#!/usr/bin/env python3
"""
Extract CTFM position report (CPM) data from a TACT flight replay log file.

Parses CURTAIN.PROFILE.PREDICTION.TRACE lines from the mono.out log and
produces a CSV with one row per CPM report processed against the CTFM.
"""

import csv
import re
import sys

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LOG_FILE = "/auto/home/ekhassan/Downloads/REPLAY_OKTNT.20251227/BASELINE/mono.out"
OUT_CSV  = "/auto/home/ekhassan/Downloads/REPLAY_OKTNT.20251227/BASELINE/ctfm_positions.csv"

TRACE_TAG = "CURTAIN.PROFILE.PREDICTION.TRACE"

# ---------------------------------------------------------------------------
# Regular expressions
# ---------------------------------------------------------------------------

# 1) Preliminary-check line (only for CPM source)
#    Example:
#    [ CURTAIN.PROFILE.PREDICTION.TRACE : Report CPM/4904N01711E, Track_Time = 2025/12/27 06:53:23 : performing preliminary check, Clock = ... ]
RE_PRELIM = re.compile(
    r"\[ CURTAIN\.PROFILE\.PREDICTION\.TRACE : "
    r"Report CPM/([^,]+),\s*Track_Time\s*=\s*(\S+ \S+)\s*:\s*performing preliminary check"
)

# 2) Rejection line immediately after preliminary check (same CPM position)
#    Example:
#    [ CURTAIN.PROFILE.PREDICTION.TRACE : Report CPM/4909N01641E : REPORTED_POSITION_WITHIN_ADEP_ZONE Position ... ]
#    [ CURTAIN.PROFILE.PREDICTION.TRACE : Report CPM/4859N01634E : TRACK_TIME_TOO_CLOSE ... ]
RE_REJECTION = re.compile(
    r"\[ CURTAIN\.PROFILE\.PREDICTION\.TRACE : "
    r"Report CPM/(\S+)\s*:\s*(\S+)\s+(.*?)\s*\]"
)

# 3) Image block: Source line (start of block)
RE_IMAGE_SOURCE = re.compile(
    r"\[ CURTAIN\.PROFILE\.PREDICTION\.TRACE : \(Source\s+=>\s+(\w+)"
)

# 4) Image block continuation lines (.. prefix)
RE_IMAGE_DURATION = re.compile(
    r"CURTAIN\.PROFILE\.PREDICTION\.TRACE\s*:\s*Duration\s+=>\s*\(SECONDS\s*=>\s*\"([^\"]+)\"\)"
)
RE_IMAGE_LEVEL = re.compile(
    r"CURTAIN\.PROFILE\.PREDICTION\.TRACE\s*:\s*Level\s+=>\s*(\S+)"
)
RE_IMAGE_POSITION = re.compile(
    r"CURTAIN\.PROFILE\.PREDICTION\.TRACE\s*:\s*Position\s+=>\s*"
    r"\(LONGITUDE\s*=>\s*\"([^\"]+)\",\s*LATITUDE\s*=>\s*\"([^\"]+)\"\)"
)

# 5) Deviation line
#    Example:
#    [ CURTAIN.PROFILE.PREDICTION.TRACE : Equivalent_Distance = 1 km, Deviation: Time -00:11:45 Vertical +12 FL Lateral  20 Nm ]
RE_DEVIATION = re.compile(
    r"\[ CURTAIN\.PROFILE\.PREDICTION\.TRACE : "
    r"Equivalent_Distance\s*=\s*(\d+)\s*km,\s*"
    r"Deviation:\s*Time\s+([+-]?\d{2}:\d{2}:\d{2})\s+"
    r"Vertical\s+([+-]?\s*\d+)\s*FL\s+"
    r"Lateral\s+(\d+)\s*Nm\s*\]"
)

# 6) Update decision line
#    Example:
#    [ CURTAIN.PROFILE.PREDICTION.TRACE : Curtain_Report.Updated => (Global_State => NORMAL, TIME_State => UPDATED, ...) ]
RE_UPDATE = re.compile(
    r"\[ CURTAIN\.PROFILE\.PREDICTION\.TRACE : "
    r"Curtain_Report\.Updated\s*=>\s*\("
    r"Global_State\s*=>\s*(\w+),\s*"
    r"TIME_State\s*=>\s*(\w+),\s*"
    r"LEVEL_State\s*=>\s*(\w+),\s*"
    r"LATERAL_State\s*=>\s*(\w+)\)"
)


# ---------------------------------------------------------------------------
# ICAO position parsing
# ---------------------------------------------------------------------------
def parse_icao_position(pos_str):
    """Parse ICAO compact position like '4904N01711E' into (lat, lon) floats.

    Format: DDMMd[N/S]DDDMMd[E/W]
    - First group: 2-digit degrees, 2-digit minutes, optional extra digits, N/S
    - Second group: 3-digit degrees, 2-digit minutes, optional extra digits, E/W
    """
    m = re.match(
        r"(\d{2})(\d{2})(\d*)([NS])(\d{3})(\d{2})(\d*)([EW])",
        pos_str,
    )
    if not m:
        return (None, None)

    lat_deg = int(m.group(1))
    lat_min = int(m.group(2))
    lat_extra = m.group(3)  # fractional minute digits (may be empty)
    lat_sign = 1 if m.group(4) == "N" else -1

    lon_deg = int(m.group(5))
    lon_min = int(m.group(6))
    lon_extra = m.group(7)
    lon_sign = 1 if m.group(8) == "E" else -1

    # If extra digits present, they are tenths/hundredths of minutes
    if lat_extra:
        lat_min_frac = lat_min + int(lat_extra) / (10 ** len(lat_extra))
    else:
        lat_min_frac = float(lat_min)

    if lon_extra:
        lon_min_frac = lon_min + int(lon_extra) / (10 ** len(lon_extra))
    else:
        lon_min_frac = float(lon_min)

    lat = lat_sign * (lat_deg + lat_min_frac / 60.0)
    lon = lon_sign * (lon_deg + lon_min_frac / 60.0)
    return (round(lat, 6), round(lon, 6))


def parse_precise_coord(raw, is_longitude=False):
    """Parse the precise coordinate from the Image block.

    Format is DDD MM SSSSSS followed by sign character (+ or -).
      DDD    = 3-digit degrees (zero-padded)
      MM     = 2-digit minutes
      SSSSSS = 6-digit seconds field (seconds * 10000, i.e. hundredths of arcsecond)

    Examples:
      "01710407364+" => 017 deg 10 min 40.7364 sec  = +17.17799 deg
      "04903422792+" => 049 deg 03 min 42.2792 sec  = +49.06174 deg
      "01644385578+" => 016 deg 44 min 38.5578 sec  = +16.74405 deg
    """
    if not raw:
        return None
    sign_char = raw[-1]
    digits = raw[:-1]
    sign = 1.0 if sign_char == "+" else -1.0

    deg = int(digits[:3])
    minutes = int(digits[3:5])
    sec_raw = int(digits[5:])        # seconds * 10000
    seconds = sec_raw / 10000.0

    result = deg + minutes / 60.0 + seconds / 3600.0
    return round(sign * result, 8)


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------
def extract_ctfm_reports(log_path):
    """Read the log file and yield one dict per CPM report."""

    with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        # Look for a preliminary-check line for CPM
        m_prelim = RE_PRELIM.search(line)
        if not m_prelim:
            i += 1
            continue

        # Found a CPM preliminary check
        icao_pos = m_prelim.group(1).strip()
        track_time = m_prelim.group(2).strip()
        icao_lat, icao_lon = parse_icao_position(icao_pos)

        record = {
            "track_time": track_time,
            "icao_position": icao_pos,
            "icao_lat": icao_lat,
            "icao_lon": icao_lon,
            "precise_lat": None,
            "precise_lon": None,
            "level_fl": None,
            "duration_seconds": None,
            "equiv_distance_km": None,
            "dev_time": None,
            "dev_vertical_fl": None,
            "dev_lateral_nm": None,
            "update_global": None,
            "update_time": None,
            "update_level": None,
            "update_lateral": None,
            "rejection_reason": None,
            "rejection_detail": None,
        }

        i += 1

        # Now scan forward within a reasonable window (say 30 lines) for
        # the associated block of trace lines for this CPM report.
        # We stop when we see another preliminary-check line or run out of
        # trace context.
        scan_limit = min(i + 40, n)
        while i < scan_limit:
            sline = lines[i]

            # Stop if we hit the next preliminary-check (for any source)
            if "performing preliminary check" in sline and TRACE_TAG in sline:
                break

            # Stop if we see Process_Report (end)
            if "Process_Report (end)" in sline and TRACE_TAG in sline:
                i += 1
                break

            # Check for rejection
            m_rej = RE_REJECTION.search(sline)
            if m_rej and TRACE_TAG in sline:
                record["rejection_reason"] = m_rej.group(2)
                record["rejection_detail"] = m_rej.group(3).strip().rstrip("]").strip()
                i += 1
                break  # rejected => no more data for this report

            # Check Image block source line
            m_src = RE_IMAGE_SOURCE.search(sline)
            if m_src:
                # Only process if source is CPM
                source = m_src.group(1)
                if source != "CPM":
                    i += 1
                    continue

            # Image block: Duration
            m_dur = RE_IMAGE_DURATION.search(sline)
            if m_dur:
                record["duration_seconds"] = m_dur.group(1)

            # Image block: Level
            m_lvl = RE_IMAGE_LEVEL.search(sline)
            if m_lvl:
                raw_level = m_lvl.group(1).rstrip(",")
                # Level may contain underscores like 1_000
                record["level_fl"] = raw_level.replace("_", "")

            # Image block: Position
            m_pos = RE_IMAGE_POSITION.search(sline)
            if m_pos:
                raw_lon = m_pos.group(1)
                raw_lat = m_pos.group(2)
                record["precise_lon"] = parse_precise_coord(raw_lon, is_longitude=True)
                record["precise_lat"] = parse_precise_coord(raw_lat, is_longitude=False)

            # Deviation line
            m_dev = RE_DEVIATION.search(sline)
            if m_dev:
                record["equiv_distance_km"] = m_dev.group(1)
                record["dev_time"] = m_dev.group(2)
                record["dev_vertical_fl"] = m_dev.group(3).replace(" ", "")
                record["dev_lateral_nm"] = m_dev.group(4)

            # Update decision line
            m_upd = RE_UPDATE.search(sline)
            if m_upd:
                record["update_global"] = m_upd.group(1)
                record["update_time"] = m_upd.group(2)
                record["update_level"] = m_upd.group(3)
                record["update_lateral"] = m_upd.group(4)

            i += 1

        yield record


def main():
    print(f"Reading log file: {LOG_FILE}")
    records = list(extract_ctfm_reports(LOG_FILE))
    print(f"Extracted {len(records)} CPM reports")

    if not records:
        print("No CPM reports found. Exiting.")
        sys.exit(1)

    fieldnames = [
        "track_time",
        "icao_position",
        "icao_lat",
        "icao_lon",
        "precise_lat",
        "precise_lon",
        "level_fl",
        "duration_seconds",
        "equiv_distance_km",
        "dev_time",
        "dev_vertical_fl",
        "dev_lateral_nm",
        "update_global",
        "update_time",
        "update_level",
        "update_lateral",
        "rejection_reason",
        "rejection_detail",
    ]

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"CSV written to: {OUT_CSV}")

    # Print summary
    rejected = sum(1 for r in records if r["rejection_reason"])
    updated = sum(
        1 for r in records
        if r["update_global"] and r["update_global"] != "NONE_STATE"
    )
    not_updated = len(records) - rejected - updated  # accepted but no dimension updated would still have NORMAL global
    # Actually let's count properly
    accepted = sum(1 for r in records if r["rejection_reason"] is None)
    any_dim_updated = sum(
        1 for r in records
        if any(
            r.get(f"update_{dim}") == "UPDATED"
            for dim in ("time", "level", "lateral")
        )
    )

    print(f"  Accepted: {accepted} ({any_dim_updated} triggered at least one update)")
    print(f"  Rejected: {rejected}")

    # Breakdown of rejection reasons
    if rejected:
        reasons = {}
        for r in records:
            reason = r["rejection_reason"]
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
        print("  Rejection reasons:")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")


if __name__ == "__main__":
    main()
