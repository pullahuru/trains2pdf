from models import RouteDefinition, TrainTrip, StopCall, format_minutes
from typing import List, Tuple, Dict, Set, Optional
from statistics import median
from datetime import date, timedelta
import heapq
import math

"""
table_compiler.py - Selects trains for a timetable table, builds the cells, sorts and paginates.

Key rules
- Direction: a trip's calls are aligned to the table's station order with a longest
  increasing subsequence over stop_sequence, so trains running the opposite way never
  match more than one station.
- Membership: the aligned calls must include >= 2 distinct stations of at least one of the
  table's `filter_groups` (its own line sections); `filter_stations` is a single group.
- Variants of the same train with identical times on this table are merged and their
  run dates united before the operating-day text is computed.
- Connection (feeder) rows show the latest connecting train from that station that
  reaches the junction in time on every day the main train runs.
- Trains arriving from another route do not show a phantom arrival at their first table station.
- Trains terminating or diverging at a station do not show a departure time into a vacuum.
- Trains taking another route bypass intermediate stations with ':'.
"""

DAY_START_MINUTES = 3 * 60          # table "day" starts at 03:00 for sorting
# skipped table station farther than this from the trip path -> not on line
OFF_LINE_THRESHOLD_KM = 20.0

DAY_ABBR = {
    "fi": ["M", "Ti", "K", "To", "P", "L", "S"],
    "sv": ["M", "Ti", "O", "To", "F", "L", "S"],
}


def format_operating_days(days: Set[int], lang: str = "fi") -> str:
    """Formats weekday sets to timetable abbreviations: consecutive runs become ranges (M-P, L-S)."""
    abbrs = DAY_ABBR.get(lang, DAY_ABBR["fi"])
    ds = sorted(d for d in set(days) if 0 <= d <= 6)
    if not ds:
        return ""
    parts: List[str] = []
    start = prev = ds[0]
    for d in ds[1:] + [None]:
        if d is not None and d == prev + 1:
            prev = d
            continue
        parts.append(abbrs[start] if start ==
                     prev else f"{abbrs[start]}-{abbrs[prev]}")
        if d is not None:
            start = prev = d
    return ",".join(parts)


# fewer differing dates than this are not flagged with '*'
IRREGULAR_MIN_DEVIATIONS = 3
# ... unless they exceed this share of the expected dates
IRREGULAR_SHARE = 0.10
# trains running on at most this many dates show the dates
EXPLICIT_DATES_MAX = 2


def summarize_run_dates(run_dates: Set[date], window: Optional[Tuple[date, date]]) -> Tuple[Set[int], bool]:
    """Return (weekdays, irregular).

    A weekday is included when the train runs on at least half of that weekday's dates within
    the span it is active. `irregular` is True when the actual dates differ noticeably from
    "every included weekday throughout the whole window" (partial period, cancellations,
    extra days): at least IRREGULAR_MIN_DEVIATIONS dates and more than IRREGULAR_SHARE.
    """
    if not run_dates:
        return set(), False
    lo, hi = min(run_dates), max(run_dates)
    occ = [0] * 7
    runs = [0] * 7
    d = lo
    while d <= hi:
        occ[d.weekday()] += 1
        if d in run_dates:
            runs[d.weekday()] += 1
        d += timedelta(days=1)
    weekdays = {w for w in range(7) if occ[w] and runs[w] * 2 >= occ[w]}
    if not weekdays:
        weekdays = {x.weekday() for x in run_dates}
    w_lo, w_hi = window if window and window[0] and window[1] else (lo, hi)
    expected = set()
    d = w_lo
    while d <= w_hi:
        if d.weekday() in weekdays:
            expected.add(d)
        d += timedelta(days=1)
    deviations = len(expected ^ set(run_dates))
    irregular = deviations >= IRREGULAR_MIN_DEVIATIONS and deviations > IRREGULAR_SHARE * \
        max(len(expected), 1)
    return weekdays, irregular


NOTE_WORDS = {
    "fi": {"and": "ja", "not": "ei", "also": "myös", "etc": "ym."},
    "sv": {"and": "och", "not": "ej", "also": "även", "etc": "m.fl."},
}
# a gap of missing run days spanning a week or more splits the validity period
PERIOD_SPLIT_DAYS = 7
MAX_LISTED_DATES = 14


def _fmt_date(d: date) -> str:
    return f"{d.day}.{d.month}."


def _fmt_dates(ds) -> str:
    return ", ".join(_fmt_date(d) for d in ds)


def describe_run_days(run_dates: Set[date], window: Optional[Tuple[date, date]],
                      lang: str = "fi") -> Tuple[str, Optional[List[Tuple[str, bool]]]]:
    """Header weekdays plus, for trains that do not run regularly through the whole period,
    a footnote as (text, bold) segments, e.g. [("15.8.-8.1. ", False), ("M-S", True)]
    or [("16.10.-12.12. ", False), ("M-P", True), (", ei 3.11., 4.11.", False)].
    The header row never carries exception marks; the renderer letters the footnotes."""
    if not run_dates:
        return "", None
    words = NOTE_WORDS.get(lang, NOTE_WORDS["fi"])
    dates = sorted(run_dates)
    if len(dates) <= EXPLICIT_DATES_MAX:
        return format_operating_days({d.weekday() for d in dates}, lang), [(_fmt_dates(dates), False)]
    weekdays, irregular = summarize_run_dates(run_dates, window)
    days_txt = format_operating_days(weekdays, lang)
    if not irregular:
        return days_txt, None

    expected = []
    d = dates[0]
    while d <= dates[-1]:
        if d.weekday() in weekdays:
            expected.append(d)
        d += timedelta(days=1)
    # split into validity periods at long gaps; short gaps become "not on" dates
    periods: List[List[date]] = []
    missing: List[date] = []
    gap: List[date] = []
    for d in expected:
        if d in run_dates:
            if gap and periods and (gap[-1] - gap[0]).days + 1 >= PERIOD_SPLIT_DAYS:
                periods.append([d, d])
            elif periods:
                missing.extend(gap)
                periods[-1][1] = d
            else:
                periods.append([d, d])
            gap = []
        else:
            gap.append(d)
    extra = sorted(set(dates) - set(expected))

    def listing(ds):
        txt = _fmt_dates(ds[:MAX_LISTED_DATES])
        return txt + (f" {words['etc']}" if len(ds) > MAX_LISTED_DATES else "")

    span = f" {words['and']} ".join(_fmt_date(
        a) if a == b else f"{_fmt_date(a)}-{_fmt_date(b)}" for a, b in periods)
    segs: List[Tuple[str, bool]] = [(span + " ", False), (days_txt, True)]
    if missing:
        segs.append((f", {words['not']} {listing(missing)}", False))
    if extra:
        segs.append((f", {words['also']} {listing(extra)}", False))
    return days_txt, segs


def runs_partial_period(run_dates: Set[date], window: Optional[Tuple[date, date]]) -> bool:
    """True when the train does not run through the whole timetable period, i.e. its validity
    starts after the first or ends before the last date of the window that falls on one of its
    running weekdays. Cancellations and extra days inside the period do not count: they are
    exceptions shown in the footnote only (issue #3)."""
    if not run_dates or not (window and window[0] and window[1]):
        return False
    dates = sorted(run_dates)
    if len(dates) <= EXPLICIT_DATES_MAX:
        weekdays = {d.weekday() for d in dates}
    else:
        weekdays, _ = summarize_run_dates(run_dates, window)
    regular = [d for d in dates if d.weekday() in weekdays] or dates
    w_lo, w_hi = window
    first_expected = next((w_lo + timedelta(days=i) for i in range(7)
                           if (w_lo + timedelta(days=i)).weekday() in weekdays), w_lo)
    last_expected = next((w_hi - timedelta(days=i) for i in range(7)
                          if (w_hi - timedelta(days=i)).weekday() in weekdays), w_hi)
    return regular[0] > first_expected or regular[-1] < last_expected


def format_run_days(run_dates: Set[date], window: Optional[Tuple[date, date]], lang: str = "fi") -> str:
    """Column header text: explicit dates for rare trains, otherwise weekdays (+ '*' if irregular)."""
    if not run_dates:
        return ""
    if len(run_dates) <= EXPLICIT_DATES_MAX:
        return ",".join(f"{d.day}.{d.month}." for d in sorted(run_dates))
    weekdays, irregular = summarize_run_dates(run_dates, window)
    return format_operating_days(weekdays, lang) + ("*" if irregular else "")


def _lis_positions(main_idx_seq: List[Tuple[int, int]]) -> List[int]:
    """Longest strictly increasing subsequence on main index.
    Input: [(stop_position, main_idx)] in stop order. Returns the chosen stop positions."""
    n = len(main_idx_seq)
    if n == 0:
        return []
    best = [1] * n
    prev = [-1] * n
    for i in range(n):
        for j in range(i):
            if main_idx_seq[j][1] < main_idx_seq[i][1] and best[j] + 1 > best[i]:
                best[i] = best[j] + 1
                prev[i] = j
    i = max(range(n), key=lambda k: (best[k], -k))
    out = []
    while i != -1:
        out.append(main_idx_seq[i][0])
        i = prev[i]
    return out[::-1]


def _km_point_to_segment(p, a, b) -> float:
    """Approximate distance (km) from point p to segment a-b; points are (lat, lon)."""
    lat0 = math.radians((a[0] + b[0] + p[0]) / 3.0)
    kx = 111.32 * math.cos(lat0)
    ky = 110.57
    ax, ay = a[1] * kx, a[0] * ky
    bx, by = b[1] * kx, b[0] * ky
    px, py = p[1] * kx, p[0] * ky
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(
        0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


class CompiledColumn:
    def __init__(self, trip: TrainTrip, days_str: str):
        self.trip = trip
        self.days_str = days_str
        self.times: Dict[int, str] = {}       # keyed by row index
        self.minutes: Dict[int, int] = {}     # main-row times in minutes, keyed by row index
        self.run_dates: Set[date] = set(trip.run_dates)
        self.trip_ids: Set[str] = {trip.trip_id}
        self.sort_key: int = 0
        # footnote segments for irregular running days
        self.note: Optional[List[Tuple[str, bool]]] = None
        # does not run through the whole timetable period -> hatched header
        self.partial: bool = False


class _TripMatch:
    """A trip aligned to a route: main index -> (stop position, call)."""

    def __init__(self, trip: TrainTrip, matched: Dict[int, Tuple[int, StopCall]]):
        self.trip = trip
        self.matched = matched
        self.first_k = min(matched)
        self.last_k = max(matched)


class TableCompiler:
    def __init__(self, max_columns_per_page: int = 8, lang: str = "fi",
                 window: Optional[Tuple[date, date]] = None,
                 station_coords: Optional[Dict[str, Tuple[float, float]]] = None):
        if max_columns_per_page < 1:
            raise ValueError("max_columns_per_page must be >= 1")
        self.max_columns = max_columns_per_page
        self.lang = lang
        self.window = window
        self.coords = station_coords or {}

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _main_codes(route: RouteDefinition) -> List[str]:
        codes: List[str] = []
        for r in route.station_rows:
            if not r.is_connection and r.station_code not in codes:
                codes.append(r.station_code)
        return codes

    def _match_trip(self, trip: TrainTrip, main_index: Dict[str, int]) -> Optional[_TripMatch]:
        seq = [(pos, main_index[s.station_code])
               for pos, s in enumerate(trip.stops) if s.station_code in main_index]
        positions = _lis_positions(seq)
        if len(positions) < 2:
            return None
        matched = {main_index[trip.stops[p].station_code]
            : (p, trip.stops[p]) for p in positions}
        return _TripMatch(trip, matched)

    def _passes_through(self, m: _TripMatch, k: int, code: str) -> bool:
        """Does the trip physically pass table station `code` (main index k) without stopping?"""
        lower = max(j for j in m.matched if j < k)
        upper = min(j for j in m.matched if j > k)
        pa, pb = m.matched[lower][0], m.matched[upper][0]
        p = self.coords.get(code)
        path = [self.coords.get(s.station_code)
                for s in m.trip.stops[pa:pb + 1]]
        if p is None or any(c is None for c in path):
            return pb - pa == 1  # no geometry: assume on line only if no other stops in between
        return min(_km_point_to_segment(p, a, b) for a, b in zip(path, path[1:])) <= OFF_LINE_THRESHOLD_KM

    # ------------------------------------------------------------------ main
    def compile_route(self, route: RouteDefinition, all_trips: List[TrainTrip]) -> List[List[CompiledColumn]]:
        rows = route.station_rows
        main_codes = self._main_codes(route)
        main_index = {c: i for i, c in enumerate(main_codes)}
        groups_cfg = route.filter_groups or [
            route.filter_stations or main_codes]
        unknown = sorted(
            {c for g in groups_cfg for c in g if c not in main_index})
        if unknown:
            print(
                f"[WARN] Table {route.table_id}: filter stations not in table rows: {unknown}")
        filt_groups = [{c for c in g if c in main_index} for g in groups_cfg]

        has_arr_row = {
            r.station_code for r in rows if not r.is_connection and r.is_arrival}
        has_dep_row = {
            r.station_code for r in rows if not r.is_connection and not r.is_arrival}

        # 1. select & build main cells
        groups: Dict[tuple, CompiledColumn] = {}
        group_match: Dict[tuple, _TripMatch] = {}
        for trip in all_trips:
            if (trip.route_type == "109" and not route.include_commuter) or trip.train_type in route.excluded_train_types:
                continue
            m = self._match_trip(trip, main_index)
            if m is None:
                continue
            served = {main_codes[k] for k in m.matched}
            if not any(len(served & g) >= 2 for g in filt_groups):
                continue

            cells: Dict[int, str] = {}
            minutes: Dict[int, int] = {}
            for row_idx, row in enumerate(rows):
                if row.is_connection:
                    continue
                code = row.station_code
                k = main_index[code]
                if k in m.matched:
                    call = m.matched[k][1]
                    is_first_station = (k == m.first_k)
                    is_last_station = (k == m.last_k)
                    paired = code in has_arr_row and code in has_dep_row

                    if is_first_station:
                        # At the train's first station on this table:
                        # The train begins its journey along this line here.
                        # It must not show an arrival time from a route not on this table.
                        if row.is_arrival:
                            t = None if paired else (
                                call.departure if call.departure is not None else call.arrival)
                        else:
                            t = call.departure if call.departure is not None else call.arrival
                    elif is_last_station:
                        # At the train's final station on this table:
                        # The train ends its journey along this line here.
                        # It must not show a departure time into a vacuum.
                        if row.is_arrival:
                            t = call.arrival
                        else:
                            if paired:
                                t = None
                            else:
                                t = call.arrival if call.arrival is not None else call.departure
                    else:
                        # Intermediate station along this table:
                        if row.is_arrival:
                            t = call.arrival
                        else:
                            t = call.departure if (
                                paired or call.departure is not None) else call.arrival

                    cells[row_idx] = format_minutes(t)
                    if t is not None:
                        minutes[row_idx] = t
                elif m.first_k < k < m.last_k:
                    cells[row_idx] = "|" if self._passes_through(
                        m, k, code) else ":"
                else:
                    cells[row_idx] = ""

            sig = (trip.train_type, trip.train_number,
                   tuple(sorted(cells.items())))
            if sig in groups:
                col = groups[sig]
                col.run_dates |= trip.run_dates
                col.trip_ids.add(trip.trip_id)
            else:
                col = CompiledColumn(trip, "")
                col.times = cells
                col.minutes = minutes
                groups[sig] = col
                group_match[sig] = m

        cols = list(groups.values())
        matches = [group_match[s] for s in groups]

        # 2. operating days
        for col in cols:
            col.days_str, col.note = describe_run_days(
                col.run_dates, self.window, self.lang)
            col.partial = bool(col.note) and runs_partial_period(
                col.run_dates, self.window)

        # 3. connection rows
        self._fill_connections(route, cols, matches, all_trips)

        # 4. sort by estimated departure from the table origin, keeping the times on each
        #    row in order wherever two trains share a row (issue #1)
        self._assign_sort_keys(cols, matches, len(main_codes))
        compiled_cols = [cols[i] for i in _order_columns(cols)]

        # 5. paginate
        return [compiled_cols[i:i + self.max_columns] for i in range(0, len(compiled_cols), self.max_columns)]

    def _assign_sort_keys(self, cols: List[CompiledColumn], matches: List[_TripMatch], n_main: int):
        def t_of(call: StopCall) -> int:
            return call.departure if call.departure is not None else call.arrival

        offset: Dict[int, float] = {0: 0.0}
        for k in range(1, n_main):
            samples = []
            for m in matches:
                if k in m.matched:
                    earlier = [j for j in m.matched if j < k]
                    if earlier:
                        j = max(earlier)
                        samples.append(
                            offset[j] + t_of(m.matched[k][1]) - t_of(m.matched[j][1]))
            offset[k] = median(samples) if samples else offset[k - 1]
        for col, m in zip(cols, matches):
            # Clock time at the train's first table station, counted from the 03:00 day start,
            # projected back to the table origin. Not wrapped again: a train starting mid-table
            # early in the morning gets a negative key and is listed before the long-distance
            # trains that reach that station later in the day.
            local = (t_of(m.matched[m.first_k][1]) - DAY_START_MINUTES) % 1440
            col.sort_key = int(round(local - offset[m.first_k]))

    def _covered(self, main_dates: Set[date], other_dates: Set[date], day_shift: int) -> bool:
        """Does the other train cover at least the weekdays the main train runs on (shifted by day_shift)?
        Dates whose counterpart falls outside the book window are skipped."""
        needed = {d + timedelta(days=day_shift) for d in main_dates}
        valid_other = other_dates
        if self.window and self.window[0] and self.window[1]:
            needed = {
                d for d in needed if self.window[0] <= d <= self.window[1]}
            valid_other = {
                d for d in other_dates if self.window[0] <= d <= self.window[1]}

        if not needed or not valid_other:
            return False

        # Weekdays the main train runs on (accounting for day shift)
        main_weekdays, _ = summarize_run_dates(needed, self.window)
        if not main_weekdays:
            main_weekdays = {d.weekday() for d in needed}

        # Weekdays the connecting train runs on
        other_weekdays, _ = summarize_run_dates(valid_other, self.window)
        if not other_weekdays:
            other_weekdays = {d.weekday() for d in valid_other}

        # The connecting train must:
        # 1. Have at least one overlapping operational date in the timetable period.
        # 2. Cover at least all the weekdays the main train operates on.
        return bool(needed & valid_other) and main_weekdays.issubset(other_weekdays)

    def _fill_connections(self, route: RouteDefinition, cols: List[CompiledColumn],
                          matches: List[_TripMatch], all_trips: List[TrainTrip]):
        """Connection rows (red). Two kinds, as in the booklet:
        - feeder row (no .o): placed before the junction's departure row; shows the departure
          time from that station of the latest train that reaches the junction in time;
        - onward row (.o): placed after the junction's arrival row; shows the arrival time at
          that station of the first train leaving the junction after the main train arrives."""
        rows = route.station_rows
        main_index = {c: i for i, c in enumerate(self._main_codes(route))}
        lo, hi = route.min_transfer_minutes, route.max_transfer_wait_minutes
        for row_idx, row in enumerate(rows):
            if not row.is_connection:
                continue
            onward = row.is_arrival
            after = next(
                (r.station_code for r in rows[row_idx + 1:] if not r.is_connection), None)
            before = next((r.station_code for r in reversed(
                rows[:row_idx]) if not r.is_connection), None)
            junction = (before or after) if onward else (after or before)
            conn = row.station_code
            for col in cols:
                col.times[row_idx] = ""
            if junction is None:
                continue

            # other trains between the connection station and the junction, in the right direction
            others = []
            for t in all_trips:
                pos_c = next((i for i, x in enumerate(t.stops)
                             if x.station_code == conn), None)
                pos_j = next((i for i, x in enumerate(t.stops)
                             if x.station_code == junction), None)
                if pos_c is None or pos_j is None:
                    continue
                if onward and pos_j < pos_c:
                    t_j, t_c = t.stops[pos_j].departure, t.stops[pos_c].arrival
                elif not onward and pos_c < pos_j:
                    t_c, t_j = t.stops[pos_c].any_time, t.stops[pos_j].arrival
                else:
                    continue
                if t_j is not None and t_c is not None:
                    others.append((t, t_j, t_c))

            for col, m in zip(cols, matches):
                trip = m.trip
                j_call = next((c for _, c in m.matched.values()
                              if c.station_code == junction), None)
                if j_call is None:
                    continue
                if onward:
                    if j_call.arrival is None or m.first_k >= main_index[junction]:
                        continue  # the train must arrive at the junction along the table
                    own = next((x for x in trip.stops if x.station_code ==
                               conn and x.sequence > j_call.sequence), None)
                    if own is not None:
                        col.times[row_idx] = format_minutes(
                            own.arrival if own.arrival is not None else own.any_time)
                        continue
                else:
                    if j_call.departure is None or m.last_k <= main_index[junction]:
                        continue  # the train must continue along the table from the junction
                    own = next((x for x in trip.stops if x.station_code ==
                               conn and x.sequence < j_call.sequence), None)
                    if own is not None:
                        col.times[row_idx] = format_minutes(own.any_time)
                        continue
                best = None
                for o, t_j, t_c in others:
                    if o.trip_id in col.trip_ids:
                        continue
                    for k in (-1, 0, 1):          # other train's service day = main day + k
                        if onward:
                            wait = k * 1440 + t_j - j_call.arrival
                        else:
                            wait = j_call.departure - (k * 1440 + t_j)
                        if lo <= wait <= hi and self._covered(col.run_dates, o.run_dates, k):
                            if best is None or wait < best[0]:
                                best = (wait, t_c)
                if best:
                    col.times[row_idx] = format_minutes(best[1])


def _order_columns(cols: List[CompiledColumn]) -> List[int]:
    """Column order: by the estimated origin departure (`sort_key`), but two trains that share a
    table row appear in the order of their times on the first row they share. The estimate
    alone fails for trains that run faster or slower than usual (issue #1): a slow train can get an
    earlier key than a train starting mid-table, yet reach that station later.

    Every shared-row rule that agrees with the estimated order holds anyway. The rules that
    contradict it are applied one by one, the clearest (largest time difference) first, and a rule
    is skipped only if it would close a loop with rules already applied (overtaking trains can
    make the rules contradict each other). The result is a topological order that takes the
    lowest estimated key whenever there is a choice, so one contradiction elsewhere in the table
    can no longer pull an unrelated train out of place."""
    n = len(cols)
    base = sorted(range(n), key=lambda i: (
        cols[i].sort_key, _num_key(cols[i].trip.train_number)))
    rank = {i: r for r, i in enumerate(base)}

    # Times counted from the 03:00 day start of the day the train *enters* the table, so an
    # overnight train keeps running past 1440 instead of wrapping to the early morning.
    day_shift = [((min(c.minutes.values()) - DAY_START_MINUTES) // 1440) * 1440
                 if c.minutes else 0 for c in cols]

    def local(i: int, row: int) -> int:
        return cols[i].minutes[row] - DAY_START_MINUTES - day_shift[i]

    succ: List[Set[int]] = [set() for _ in range(n)]
    against: List[Tuple[int, int, int]] = []   # (-time difference, earlier, later)
    for a in range(n):
        for b in range(a + 1, n):
            for row in sorted(cols[a].minutes.keys() & cols[b].minutes.keys()):
                ta, tb = local(a, row), local(b, row)
                if ta != tb:
                    first, second = (a, b) if ta < tb else (b, a)
                    if rank[first] < rank[second]:
                        succ[first].add(second)
                    else:
                        against.append((-abs(ta - tb), first, second))
                    break

    def reaches(src: int, dst: int) -> bool:
        seen, stack = {src}, [src]
        while stack:
            for j in succ[stack.pop()]:
                if j == dst:
                    return True
                if j not in seen:
                    seen.add(j)
                    stack.append(j)
        return False

    for _, first, second in sorted(against, key=lambda e: (e[0], rank[e[1]], rank[e[2]])):
        if not reaches(second, first):
            succ[first].add(second)

    indeg = [0] * n
    for i in range(n):
        for j in succ[i]:
            indeg[j] += 1
    ready = [(rank[i], i) for i in range(n) if indeg[i] == 0]
    heapq.heapify(ready)
    order: List[int] = []
    while ready:
        _, i = heapq.heappop(ready)
        order.append(i)
        for j in succ[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                heapq.heappush(ready, (rank[j], j))
    assert len(order) == n, "column order constraints must be acyclic"
    return order


def _num_key(num: str):
    return (0, int(num), "") if num.isdigit() else (1, 0, num)
