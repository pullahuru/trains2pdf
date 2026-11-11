from datetime import date

import pytest

from conftest import synthetic_route
from gtfs_importer import GTFSImporter, parse_train_name
from models import format_minutes
from renderer import station_sort_key, TimetableBookRenderer
from table_compiler import TableCompiler, format_operating_days, format_run_days, summarize_run_dates, describe_run_days


# ---------------------------------------------------------------- parsing helpers
@pytest.mark.parametrize("s,expected", [
    ("16:05:42", 16 * 60 + 5), ("24:16:00", 24 * 60 + 16), ("00:00:00", 0),
    ("", None), (None, None), ("garbage", None), ("aa:bb:cc", None),
])
def test_parse_time_keeps_hours_past_midnight(s, expected):
    assert GTFSImporter.parse_time(s) == expected


def test_format_minutes_wraps_only_for_display():
    assert format_minutes(1456) == "00.16"
    assert format_minutes(605) == "10.05"
    assert format_minutes(None) == ""


@pytest.mark.parametrize("short,route_short,expected", [
    ("IC 22", "", ("IC", "22")),
    ("E (HL 8192)", "", ("E", "8192")),
    ("PYO 274", "", ("PYO", "274")),
    ("", "HDM 480", ("HDM", "480")),
    ("S", "", ("S", "")),
])
def test_parse_train_name(short, route_short, expected):
    assert parse_train_name(short, route_short, "tripid123") == expected


@pytest.mark.parametrize("days,fi,sv", [
    ({0, 1, 2, 3, 4, 5, 6}, "M-S", "M-S"),
    ({0, 1, 2, 3, 4}, "M-P", "M-Fr"),
    ({0, 1, 2, 3, 4, 5}, "M-L", "M-L"),
    ({0, 1, 2, 3, 4, 6}, "M-P,S", "M-Fr,S"),
    ({4, 6}, "P,S", "Fr,S"),
    ({5, 6}, "L-S", "L-S"),
    ({0, 2, 4, 5}, "M,K,P-L", "M,O,Fr-L"),
    ({2}, "K", "O"),
    (set(), "", ""),
])
def test_format_operating_days(days, fi, sv):
    assert format_operating_days(days, "fi") == fi
    assert format_operating_days(days, "sv") == sv


def test_run_days_explicit_dates_and_irregular_flag():
    window = (date(2027, 1, 4), date(2027, 1, 31))
    assert format_run_days({date(2027, 1, 15)}, window) == "15.1."
    weekdays = {date(2027, 1, d) for d in range(4, 32) if date(2027, 1, d).weekday() < 5}
    assert format_run_days(weekdays, window) == "M-P"
    # first half of the window only -> flagged
    half = {d for d in weekdays if d.day < 16}
    days, irregular = summarize_run_dates(half, window)
    assert days == {0, 1, 2, 3, 4} and irregular
    assert format_run_days(half, window).endswith("*")


def test_station_sort_order_finnish_swedish():
    names = ["Övertorneå", "Ähtäri", "Åbo", "Zz", "aava", "Ylivieska"]
    assert sorted(names, key=station_sort_key) == ["aava", "Ylivieska", "Zz", "Åbo", "Ähtäri", "Övertorneå"]


def test_compress_ranges():
    assert TimetableBookRenderer._compress_ranges(["7", "8", "9", "16", "7", "A"]) == "7-9, 16, A"


# ---------------------------------------------------------------- importer
def test_importer_calendar_and_filters(synthetic_loaded):
    imp, trips = synthetic_loaded
    by_id = {t.trip_id: t for t in trips}
    assert imp.window == (date(2027, 1, 4), date(2027, 1, 31))
    assert "T12" not in by_id, "service outside the window must be dropped"
    t9 = by_id["T9"]
    assert date(2027, 1, 6) not in t9.run_dates, "exception_type=2 must remove the date"
    assert date(2027, 1, 10) in t9.run_dates, "exception_type=1 must add the date"
    assert date(2027, 1, 17) not in t9.run_dates, "an added Sunday must not make every Sunday a run day"
    assert by_id["T11"].run_dates == {date(2027, 1, 15)}
    t1 = by_id["T1"]
    assert t1.stops[0].arrival is None and t1.stops[-1].departure is None
    assert t1.stops[-1].arrival == 25 * 60 + 10
    assert by_id["T6"].train_type == "R" and by_id["T6"].train_number == "9001"
    assert imp.station_names["A"] == {"fi": "Aava", "sv": "Ava"}


# ---------------------------------------------------------------- compiler
@pytest.fixture(scope="module")
def compiled(synthetic_loaded):
    imp, trips = synthetic_loaded
    route = synthetic_route()
    comp = TableCompiler(max_columns_per_page=50, window=imp.window, station_coords=imp.station_coords)
    pages = comp.compile_route(route, trips)
    cols = [c for p in pages for c in p]
    return {f"{c.trip.train_type} {c.trip.train_number}": c for c in cols}, [c for c in cols]


def _cells(col, n=6):
    return [col.times.get(i, "") for i in range(n)]


def test_direction_and_membership(compiled):
    by_name, _ = compiled
    assert "IC 2" not in by_name, "southbound train in northbound table"
    assert "IC 3" not in by_name, "overnight southbound train in northbound table"
    assert "R 9001" not in by_name, "commuter train in long-distance table"
    assert "IC 7" not in by_name, "train touching only the duplicated junction station"
    assert "IC 12" not in by_name


def test_overnight_train_is_included_with_correct_cells(compiled):
    by_name, _ = compiled
    assert _cells(by_name["IC 1"]) == ["22.00", "23.01", "23.50", "", "00.00", "01.10"]


def test_pass_through_vs_off_line(compiled):
    by_name, _ = compiled
    assert _cells(by_name["IC 4"]) == ["06.00", "|", "|", "", "|", "08.00"]
    assert _cells(by_name["IC 5"]) == ["07.00", "", "", "", "", "09.00"], "detour via X must not show '|'"


def test_feeder_connection(compiled):
    by_name, _ = compiled
    assert _cells(by_name["IC 8"]) == ["09.00", "|", "09.45", "09.00", "09.50", "10.30"]


def test_variants_merged_and_days(compiled):
    by_name, cols = compiled
    assert sum(1 for c in cols if c.trip.train_number == "10") == 1
    assert by_name["IC 10"].days_str == "M-P"
    assert by_name["IC 9"].days_str == "M-P"
    assert by_name["IC 11"].days_str == "P", "header shows the weekday, never dates or marks"
    assert by_name["IC 11"].note == [("15.1.", False)]
    assert by_name["IC 1"].days_str == "M-S"
    assert by_name["IC 1"].note is None and by_name["IC 9"].note is None and by_name["IC 10"].note is None


def test_loop_trip_does_not_crash_and_is_monotonic(compiled):
    by_name, _ = compiled
    col = by_name["IC 13"]
    times = [t for t in _cells(col) if t not in ("", "|")]
    mins = [int(t[:2]) * 60 + int(t[3:]) for t in times]
    assert mins == sorted(mins)


def test_sorting(compiled):
    _, cols = compiled
    names = [f"{c.trip.train_type} {c.trip.train_number}" for c in cols]
    assert names.index("IC 14") < names.index("IC 4") < names.index("IC 8")
    assert names[-2:] == ["IC 1", "IC 16"], "late evening overnight trains last"


def test_pagination():
    from table_compiler import TableCompiler as TC
    with pytest.raises(ValueError):
        TC(max_columns_per_page=0)


# ---------------------------------------------------------------- regressions from the review
def test_no_feeder_for_train_leaving_at_junction(compiled):
    by_name, _ = compiled
    assert _cells(by_name["IC 15"]) == ["09.00", "|", "09.45", "", "09.50", ""]


def test_feeder_on_next_service_day_across_window_end(compiled):
    by_name, _ = compiled
    # IC 17 leaves F at 00.00 on the next service day and reaches C 20 min before IC 16 departs
    assert _cells(by_name["IC 16"]) == ["23.00", "|", "00.20", "00.00", "00.30", "01.10"]


def test_arrival_only_row_never_shows_departure(synthetic_loaded):
    from models import RouteDefinition, StationRowConfig
    imp, trips = synthetic_loaded
    route = RouteDefinition("3", "x", "x", [StationRowConfig("A"), StationRowConfig("B", is_arrival=True),
                                            StationRowConfig("D", is_arrival=True)])
    cols = [c for p in TableCompiler(window=imp.window, station_coords=imp.station_coords).compile_route(route, trips)
            for c in p]
    by_name = {f"{c.trip.train_type} {c.trip.train_number}": c for c in cols}
    assert by_name["IC 18"].times[1] == "", "train starting at B has no arrival at B"
    assert by_name["IC 9"].times[1] == "12.30"


def test_default_window_picks_next_period_change():
    rows = ([{"start_date": "20261016"}] * 50 + [{"start_date": "20261213"}] * 100 +
            [{"start_date": "20270628"}] * 60 + [{"start_date": "20270802"}] * 90 + [{"start_date": "20261025"}] * 20)
    f = GTFSImporter.detect_default_window
    assert f(rows, date(2026, 10, 16), date(2027, 12, 31)) == (date(2026, 10, 16), date(2026, 12, 12))
    assert f(rows, date(2027, 1, 10), date(2027, 12, 31)) == (date(2027, 1, 10), date(2027, 6, 27))
    assert f(rows, date(2027, 9, 1), date(2027, 12, 31)) == (date(2027, 9, 1), date(2027, 12, 31))


def test_window_start_only_and_invalid(synthetic_feed):
    imp = GTFSImporter(synthetic_feed)
    imp.load_feed(date(2027, 1, 10))
    assert imp.window == (date(2027, 1, 10), date(2027, 1, 31))
    with pytest.raises(ValueError):
        GTFSImporter(synthetic_feed).load_feed(date(2027, 1, 20), date(2027, 1, 10))


def test_cli_reports_bad_window(synthetic_feed, tmp_path, capsys):
    import main
    with pytest.raises(SystemExit) as e:
        main.main(["--gtfs", synthetic_feed, "--start", "2027-01-20", "--end", "2027-01-10",
                   "--outdir", str(tmp_path)])
    assert e.value.code == 2
    assert "[ERROR]" in capsys.readouterr().out


def test_duplicate_table_ids_rejected(synthetic_loaded, tmp_path):
    from main import compile_book
    from conftest import synthetic_route
    imp, trips = synthetic_loaded
    with pytest.raises(ValueError, match="Duplicate table_id"):
        compile_book(lang="fi", output_pdf=str(tmp_path / "d.pdf"), importer=imp, all_trips=trips,
                     routes=[synthetic_route(), synthetic_route()])


def test_dot_leaders_on_booklet_grid(tmp_path):
    import style_config as sc
    r = TimetableBookRenderer(str(tmp_path / "d.pdf"))
    first_col = sc.MARGIN_LEFT + sc.STATION_COL_WIDTH
    short, _ = r._dot_positions(50.0, first_col, False)
    long_, _ = r._dot_positions(70.0, first_col, False)
    arr, o_x = r._dot_positions(50.0, first_col, True)
    assert short[-1] == long_[-1] == first_col - sc.DOT_LAST_FROM_COLUMN, "all rows end at the same x"
    assert all(abs((b - a) - sc.DOT_PITCH) < 1e-6 for a, b in zip(short, short[1:]))
    assert long_ == short[-len(long_):], "longer names drop dots from the left on the same grid"
    assert short[0] >= 50.0 + sc.DOT_GAP_AFTER_NAME and short[0] - sc.DOT_PITCH < 50.0 + sc.DOT_GAP_AFTER_NAME
    assert arr[-1] == short[-2], "arrival rows give the last dot slot to '.o'"
    assert abs(o_x - (arr[-1] + sc.ARRIVAL_O_AFTER_DOT)) < 1e-6


def test_describe_run_days_footnotes():
    from datetime import timedelta
    w = (date(2026, 10, 16), date(2026, 12, 12))

    def span(a, b, wd=range(7)):
        out, d = set(), a
        while d <= b:
            if d.weekday() in wd:
                out.add(d)
            d += timedelta(days=1)
        return out

    # regular all period -> no note
    assert describe_run_days(span(*w), w) == ("M-S", None)
    # part of the period -> validity range + bold weekdays, like the booklet's "15.8.-8.1. M-S"
    assert describe_run_days(span(date(2026, 11, 1), w[1]), w) == ("M-S", [("1.11.-12.12. ", False), ("M-S", True)])
    # two periods (week-long gap) in Swedish
    days, note = describe_run_days(span(w[0], date(2026, 10, 31)) | span(date(2026, 11, 20), w[1]), w, "sv")
    assert note[0][0] == "16.10.-31.10. och 20.11.-12.12. "
    # cancellations inside the period are listed, extra days too
    d2 = span(date(2026, 11, 2), w[1], range(5)) - {date(2026, 11, 10), date(2026, 11, 11)} | {date(2026, 11, 8)}
    days, note = describe_run_days(d2, w)
    assert days == "M-P"
    assert "".join(t for t, _ in note) == "2.11.-11.12. M-P, ei 10.11., 11.11., myös 8.11."



def test_onward_connection_row(synthetic_loaded):
    from models import RouteDefinition, StationRowConfig
    imp, trips = synthetic_loaded
    route = RouteDefinition("5", "x", "x", [StationRowConfig("A"), StationRowConfig("C", is_arrival=True),
                                            StationRowConfig("F", is_arrival=True, is_connection=True),
                                            StationRowConfig("C"), StationRowConfig("D", is_arrival=True)])
    cols = [c for p in TableCompiler(window=imp.window, station_coords=imp.station_coords).compile_route(route, trips)
            for c in p]
    by_name = {f"{c.trip.train_type} {c.trip.train_number}": c for c in cols}
    # IC 8 reaches C at 09.45; IC 15 leaves C at 09.50 and arrives at F at 10.30
    assert by_name["IC 8"].times[2] == "10.30"
    assert by_name["IC 15"].times[2] == "10.30", "a train that itself continues to F shows its own arrival"
    assert by_name["IC 1"].times[2] == "", "no onward train within the transfer window"


def test_routes_json_loads():
    from main import load_routes
    routes = load_routes()
    assert len(routes) == 46
    for r in routes:
        assert r.title_fi and r.title_sv and r.station_rows
        assert r.filter_groups and all(len(g) >= 2 for g in r.filter_groups), r.table_id


def test_default_window_starts_today(synthetic_feed):
    imp = GTFSImporter(synthetic_feed)
    imp.load_feed(today=date(2027, 1, 12))
    assert imp.window == (date(2027, 1, 12), date(2027, 1, 31)), "past days are not printed"
    imp = GTFSImporter(synthetic_feed)
    imp.load_feed(today=date(2026, 12, 1))          # before the feed starts -> feed start
    assert imp.window[0] == date(2027, 1, 4)


def test_only_feed_backed_services(synthetic_loaded):
    from models import ServiceType
    _, trips = synthetic_loaded
    assert all(t.services <= {ServiceType.BICYCLE} for t in trips)
    assert any(ServiceType.BICYCLE in t.services for t in trips)


def test_hatch_only_when_not_running_whole_period():
    """Issue #3: exceptions inside the period get a footnote but no hatching."""
    from datetime import timedelta
    from table_compiler import runs_partial_period
    w = (date(2026, 12, 13), date(2027, 4, 25))

    def fridays(a, b):
        out, d = set(), a
        while d <= b:
            if d.weekday() == 4:
                out.add(d)
            d += timedelta(days=1)
        return out

    # Fridays 18.12.-23.4. = the whole period, with holiday cancellations
    full = fridays(date(2026, 12, 18), date(2027, 4, 23)) - {
        date(2026, 12, 25), date(2027, 1, 1), date(2027, 3, 26)}
    assert describe_run_days(full, w)[1] is not None, "exceptions get a footnote"
    assert not runs_partial_period(full, w), "but no hatching"
    # starts later or ends earlier than the period -> hatched
    assert runs_partial_period(fridays(date(2027, 1, 8), date(2027, 4, 23)), w)
    assert runs_partial_period(fridays(date(2026, 12, 18), date(2027, 3, 5)), w)
    # rare trains on single dates -> hatched
    assert runs_partial_period({date(2027, 1, 1)}, w)


def test_columns_in_time_order_on_shared_rows():
    """Issue #1: a slow train must not be listed before a train that starts mid-table and
    leaves the shared station earlier, even though its estimated origin departure is earlier."""
    from datetime import timedelta
    from models import RouteDefinition, StationRowConfig, StopCall, TrainTrip
    days = {date(2027, 1, 4) + timedelta(days=i) for i in range(28)}

    def trip(num, calls):
        stops = [StopCall(code, code, arr, dep, seq)
                 for seq, (code, arr, dep) in enumerate(calls, start=1)]
        return TrainTrip(f"T{num}", str(num), "IC", set(days), stops=stops)

    def hm(s):
        return int(s[:2]) * 60 + int(s[3:])

    trips = [
        trip(1, [("A", None, hm("06:00")), ("C", hm("06:30"), None)]),
        trip(2, [("A", None, hm("07:00")), ("C", hm("07:30"), None)]),
        trip(3, [("A", None, hm("08:00")), ("C", hm("09:00"), hm("09:01")),
                 ("D", hm("09:30"), None)]),            # slow: A-C takes 60 min, median 30
        trip(4, [("C", None, hm("08:45")), ("D", hm("09:15"), None)]),
    ]
    route = RouteDefinition(table_id="9", title_fi="A - D", title_sv="A - D",
                            station_rows=[StationRowConfig("A"), StationRowConfig("C"),
                                          StationRowConfig("D", is_arrival=True)])
    tc = TableCompiler(max_columns_per_page=10, window=(date(2027, 1, 4), date(2027, 1, 31)))
    cols = [c for page in tc.compile_route(route, trips) for c in page]
    assert [c.trip.train_number for c in cols] == ["1", "2", "4", "3"]
    for row in range(3):
        times = [c.minutes[row] for c in cols if row in c.minutes]
        assert times == sorted(times), f"row {row} out of order"


def test_overnight_train_not_moved_before_daytime_trains():
    """An overnight train reaching a shared station after 03:00 the next day still comes after a
    daytime train starting there (it reaches the station later, not earlier)."""
    from models import RouteDefinition, StationRowConfig, StopCall, TrainTrip
    from datetime import timedelta
    days = {date(2027, 1, 4) + timedelta(days=i) for i in range(28)}

    def trip(num, calls):
        stops = [StopCall(code, code, arr, dep, seq)
                 for seq, (code, arr, dep) in enumerate(calls, start=1)]
        return TrainTrip(f"T{num}", str(num), "IC", set(days), stops=stops)

    trips = [
        trip(1, [("A", None, 23 * 60), ("B", 28 * 60, 28 * 60 + 1), ("C", 29 * 60, None)]),
        trip(2, [("B", None, 14 * 60), ("C", 15 * 60, None)]),
    ]
    route = RouteDefinition(table_id="9", title_fi="A - C", title_sv="A - C",
                            station_rows=[StationRowConfig("A"), StationRowConfig("B"),
                                          StationRowConfig("C", is_arrival=True)])
    tc = TableCompiler(max_columns_per_page=10, window=(date(2027, 1, 4), date(2027, 1, 31)))
    cols = [c for page in tc.compile_route(route, trips) for c in page]
    assert [c.trip.train_number for c in cols] == ["2", "1"]


def test_contradicting_rules_do_not_pull_unrelated_train_out_of_place():
    """Issue #1 (Lahti, O 1671 before IC 67): a loop of contradicting shared-row rules elsewhere
    must not move a train ahead of a train it clearly follows on a shared row."""
    from types import SimpleNamespace
    from table_compiler import _order_columns

    def col(num, key, minutes):
        return SimpleNamespace(sort_key=key, minutes=minutes,
                               trip=SimpleNamespace(train_number=num))
    # times in minutes after 03:00 (+180 for absolute), rows 0..3
    o = col("1671", 10, {2: 180 + 115})                  # starts mid-table, low estimate
    i = col("67", 20, {0: 180 + 60, 2: 180 + 108})       # leaves row 2 before O 1671
    x = col("1", 30, {0: 180 + 50, 1: 180 + 55})         # x, y, z: overtaking loop, all before IC 67
    y = col("2", 31, {1: 180 + 56, 3: 180 + 60})
    z = col("3", 32, {0: 180 + 45, 3: 180 + 65})
    cols = [o, i, x, y, z]
    order = [cols[k].trip.train_number for k in _order_columns(cols)]
    assert order.index("67") < order.index("1671")
    assert order.index("1") < order.index("67") and order.index("3") < order.index("67")
