"""Every test compares the code against an answer worked out by hand or by theory."""
import numpy as np
import pandas as pd
import pytest

from execbench import schedules as sch
from execbench.benchmark import (TWAP, VWAP_F, VWAP_O, frontier_table, run_all,
                                 total_variation, tracking_table, volume_forecast_table)
from execbench.data import bucketize, trailing_inputs
from execbench.impact import ImpactModel, permanent_impact, temporary_impact
from execbench.shortfall import (impact_cost_bps, path_cost_bps, timing_risk_theory,
                                 vwap_slippage_bps)
from execbench.synthetic import make_bars, true_minute_curve


# ---------- schedules --------------------------------------------------------

def test_twap_is_equal_slices():
    np.testing.assert_allclose(sch.twap(4), [0.25, 0.25, 0.25, 0.25])


def test_vwap_is_proportional_to_curve():
    np.testing.assert_allclose(sch.vwap([10, 30, 60]), [0.1, 0.3, 0.6])


def test_ac_matches_closed_form():
    # 2 equal buckets, kappa=1: holdings 1 -> sinh(0.5)/sinh(1) -> 0
    mid = np.sinh(0.5) / np.sinh(1.0)          # 0.44341
    np.testing.assert_allclose(sch.almgren_chriss(1.0, [1, 1]), [1 - mid, mid])


def test_ac_zero_urgency_is_the_clock_itself():
    curve = np.array([0.5, 0.2, 0.3])
    np.testing.assert_allclose(sch.almgren_chriss(0.0, curve), curve)
    np.testing.assert_allclose(sch.almgren_chriss(1e-6, curve), curve, atol=1e-6)


def test_ac_more_urgency_trades_earlier():
    first_half = [sch.almgren_chriss(k, np.ones(10))[:5].sum() for k in (0, 1, 2, 4, 8)]
    assert all(a < b for a, b in zip(first_half, first_half[1:]))


@pytest.mark.parametrize("k", [0, 0.5, 2, 6, 20])
def test_all_schedules_are_valid_weights(k):
    rng = np.random.default_rng(1)
    curve = rng.random((7, 78)) + 0.01
    for w in (sch.vwap(curve), sch.almgren_chriss(k, curve)):
        assert np.all(w >= 0)
        np.testing.assert_allclose(w.sum(axis=-1), 1.0)


# ---------- shortfall --------------------------------------------------------

def test_flat_prices_cost_nothing():
    assert path_cost_bps([0.5, 0.5], [100.0, 100.0], 100.0) == 0.0


def test_price_jump_costs_exactly_the_jump_for_any_schedule():
    # price is 1% above arrival in every bucket -> 100 bps, whatever the schedule
    prices = np.full(5, 101.0)
    for w in (sch.twap(5), sch.almgren_chriss(3, np.ones(5)), sch.vwap([5, 1, 1, 1, 9])):
        assert path_cost_bps(w, prices, 100.0, side=1) == pytest.approx(100.0)
        assert path_cost_bps(w, prices, 100.0, side=-1) == pytest.approx(-100.0)


def test_path_cost_hand_example():
    # 50% at 101, 50% at 103, arrival 100 -> average 102 -> 200 bps
    assert path_cost_bps([0.5, 0.5], [101.0, 103.0], 100.0) == pytest.approx(200.0)


def test_vwap_slippage_hand_example_and_oracle_is_zero():
    prices, vol = np.array([100.0, 102.0]), np.array([300.0, 100.0])
    # market VWAP = 0.75*100 + 0.25*102 = 100.5 ; TWAP average = 101
    assert vwap_slippage_bps([0.5, 0.5], prices, vol) == pytest.approx(1e4 * 0.5 / 100.5)
    assert vwap_slippage_bps([0.75, 0.25], prices, vol) == pytest.approx(0.0, abs=1e-9)


def test_timing_risk_formula_twap_closed_form():
    # TWAP over n buckets: sum_j ((n-j+1)/n)^2 = (n+1)(2n+1)/(6n)
    n = 78
    assert timing_risk_theory(sch.twap(n), 1.0) == pytest.approx(np.sqrt((n + 1) * (2 * n + 1) / (6 * n)))


def test_timing_risk_simulation_matches_theory():
    rng = np.random.default_rng(7)
    n, sims, s, p0 = 20, 200_000, 0.05, 100.0
    prices = p0 + np.cumsum(rng.normal(0, s, (sims, n)), axis=1)
    for w in (sch.twap(n), sch.almgren_chriss(4.0, np.ones(n))):
        simulated = path_cost_bps(w, prices, np.full(sims, p0)).std()
        theory = 1e4 * timing_risk_theory(w, s) / p0
        assert simulated == pytest.approx(theory, rel=0.01)
    # front-loading must reduce risk
    assert timing_risk_theory(sch.almgren_chriss(4.0, np.ones(n)), s) < timing_risk_theory(sch.twap(n), s)


# ---------- impact -----------------------------------------------------------

def test_temporary_impact_hand_example():
    # participation 4%, sqrt = 0.2 ; 0.1 * 0.02 * 0.2 = 0.0004 = 4 bps
    m = ImpactModel(eta=0.1, beta=0.5, gamma=0.0)
    temp, capped = temporary_impact(np.array([[400.0]]), np.array([[10_000.0]]), np.array([0.02]), m)
    assert temp[0, 0] == pytest.approx(0.0004)
    assert not capped.any()


def test_constant_participation_closed_form():
    # trading 5% of every bucket: cost = eta * sigma * 0.05^beta, whatever the volume shape
    m = ImpactModel(eta=0.142, beta=0.6, gamma=0.0)
    vol = np.array([[900.0, 100.0, 400.0, 600.0]])
    shares = 0.05 * vol
    w = shares / shares.sum()
    temp, _ = temporary_impact(shares, vol, np.array([0.03]), m)
    assert impact_cost_bps(w, temp)[0] == pytest.approx(1e4 * 0.142 * 0.03 * 0.05 ** 0.6)


def test_permanent_cost_is_the_same_for_every_schedule():
    # linear permanent impact: cost = gamma * sigma * (X / ADV) / 2, independent of schedule
    m = ImpactModel(gamma=0.314)
    X, adv, sigma = 50_000.0, 1_000_000.0, 0.02
    expected = 1e4 * 0.314 * sigma * (X / adv) / 2
    for w in (sch.twap(10), sch.almgren_chriss(5, np.ones(10)), sch.vwap(np.arange(1, 11))):
        perm = permanent_impact((w * X)[None, :], np.array([adv]), np.array([sigma]), m)
        assert impact_cost_bps(w[None, :], perm)[0] == pytest.approx(expected)


def test_oracle_vwap_minimises_temporary_impact():
    # under this model no schedule can beat trading in proportion to realised volume
    rng = np.random.default_rng(3)
    m = ImpactModel(eta=0.142, beta=0.5, gamma=0.0)
    vol = rng.random((1, 30)) * 1e5 + 1e3
    X, sigma = 0.05 * vol.sum(), np.array([0.02])

    def cost(w):
        temp, _ = temporary_impact(w * X, vol, sigma, m)
        return impact_cost_bps(w, temp)[0]

    best = cost(vol / vol.sum())
    for _ in range(500):
        w = rng.dirichlet(np.ones(30))[None, :]
        assert cost(w) >= best - 1e-12


def test_zero_volume_bucket_is_capped_and_flagged():
    m = ImpactModel(eta=0.1, beta=0.5, max_participation=1.0)
    temp, capped = temporary_impact(np.array([[100.0, 100.0]]), np.array([[0.0, 1000.0]]),
                                    np.array([0.02]), m)
    assert capped.tolist() == [[True, False]]
    assert np.isfinite(temp).all() and temp[0, 0] == pytest.approx(0.1 * 0.02 * 1.0)


# ---------- data -------------------------------------------------------------

def _tiny_bars():
    # two minutes in the first 5-minute bucket of one day
    return pd.DataFrame({
        "symbol": "X", "timestamp": ["2025-01-02T14:30:00Z", "2025-01-02T14:31:00Z"],
        "open": [10.0, 10.2], "high": [10.3, 10.6], "low": [9.9, 10.1], "close": [10.2, 10.5],
        "volume": [100.0, 300.0], "vwap": [10.0, 10.4]})


def test_bucket_vwap_hand_example():
    p = bucketize(_tiny_bars(), bucket_minutes=5, min_coverage=0.0)["X"]
    # the day is incomplete (no last bucket) so it is dropped, which is itself the check
    assert p.n_dropped == 1 and len(p.dates) == 0
    full = make_bars(("X",), n_days=1, seed=0)
    full.loc[0, ["volume", "vwap"]] = [100.0, 10.0]
    full.loc[1, ["volume", "vwap"]] = [300.0, 10.4]
    full.loc[2:4, "volume"] = 0.0
    p = bucketize(full, bucket_minutes=5)["X"]
    assert p.volume[0, 0] == 400.0
    assert p.price[0, 0] == pytest.approx((100 * 10.0 + 300 * 10.4) / 400)   # 10.3
    assert p.price.shape == (1, 78)


def test_times_are_converted_from_utc_and_session_filtered():
    bars = make_bars(("X",), n_days=3, seed=1, start="2025-07-01")   # summer time: 09:30 NY = 13:30 UTC
    assert bars["timestamp"].iloc[0].endswith("13:30:00Z")
    extra = bars.iloc[[0]].copy()
    extra["timestamp"] = "2025-07-01T12:00:00Z"                      # 08:00 NY, pre-market
    extra["volume"] = 9e12
    p = bucketize(pd.concat([bars, extra]), bucket_minutes=5)["X"]
    assert p.volume.max() < 9e12 and p.price.shape == (3, 78)


def test_half_days_are_dropped():
    bars = make_bars(("X",), n_days=5, seed=2)
    ts = pd.to_datetime(bars["timestamp"], utc=True).dt.tz_convert("America/New_York")
    day3 = ts.dt.strftime("%Y-%m-%d") == sorted(ts.dt.strftime("%Y-%m-%d").unique())[2]
    bars = bars[~(day3 & (ts.dt.hour >= 13))]                        # market closes at 13:00
    p = bucketize(bars, bucket_minutes=5)["X"]
    assert p.n_dropped == 1 and len(p.dates) == 4
    assert np.isfinite(p.logret[1:]).all()


def test_no_lookahead():
    """Changing day d's data must not change day d's inputs, but must change day d+1's."""
    bars = make_bars(("X",), n_days=40, seed=4)
    base = bucketize(bars, bucket_minutes=5)["X"]
    a = trailing_inputs(base, lookback=20)
    d = a["valid"][5]
    ts = pd.to_datetime(bars["timestamp"], utc=True).dt.tz_convert("America/New_York")
    hit = ts.dt.strftime("%Y-%m-%d") == base.dates[d]
    shocked = bars.copy()
    shocked.loc[hit, "volume"] *= np.linspace(5, 0.2, hit.sum())
    shocked.loc[hit, ["open", "high", "low", "close", "vwap"]] *= 1.5
    b = trailing_inputs(bucketize(shocked, bucket_minutes=5)["X"], lookback=20)
    i = 5
    for key in ("adv", "sigma", "curve"):
        np.testing.assert_array_equal(a[key][i], b[key][i])           # day d untouched
        assert not np.allclose(a[key][i + 1], b[key][i + 1])          # day d+1 sees it


def test_trailing_inputs_hand_example():
    bars = make_bars(("X",), n_days=6, seed=5)
    p = bucketize(bars, bucket_minutes=5)["X"]
    t = trailing_inputs(p, lookback=3)
    assert t["valid"].tolist() == [4, 5]                              # day 0 has no return
    total = p.volume.sum(axis=1)
    assert t["adv"][0] == pytest.approx(total[1:4].mean())
    assert t["sigma"][0] == pytest.approx(np.std(p.logret[1:4], ddof=1))
    np.testing.assert_allclose(t["curve"][0], (p.volume[1:4] / total[1:4, None]).mean(axis=0))


# ---------- end to end -------------------------------------------------------

@pytest.fixture(scope="module")
def u_results():
    bars = make_bars(("AAA", "BBB", "CCC"), n_days=120, seed=11, shape="u")
    return run_all(bucketize(bars, bucket_minutes=5), lookback=20)


def test_forecast_recovers_a_real_volume_curve(u_results):
    t = volume_forecast_table(u_results).iloc[0]
    assert t["skill"] > 0.3 and t["reduction_lo"] > 0


def test_forecast_shows_no_skill_when_the_true_curve_is_flat():
    """Guard against manufacturing skill out of noise."""
    bars = make_bars(("AAA", "BBB", "CCC"), n_days=120, seed=12, shape="flat")
    t = volume_forecast_table(run_all(bucketize(bars, bucket_minutes=5), lookback=20)).iloc[0]
    assert t["skill"] < 0.02
    assert t["reduction_hi"] < 0.005


def test_forecast_curve_is_close_to_the_known_truth(u_results):
    bars = make_bars(("AAA",), n_days=120, seed=11, shape="u")
    p = bucketize(bars, bucket_minutes=5)["AAA"]
    fc = trailing_inputs(p, lookback=100)["curve"][-1]
    truth = true_minute_curve("u").reshape(78, 5).sum(axis=1)
    assert total_variation(fc, truth) < 0.02


def test_frontier_ordering_on_synthetic_data(u_results):
    f = frontier_table(u_results)
    f = f[f["size"] == 0.05].set_index("schedule")
    # modelled impact: oracle <= forecast VWAP <= TWAP, and urgency costs more
    assert f.loc[VWAP_O, "temp_bps"] <= f.loc[VWAP_F, "temp_bps"] <= f.loc[TWAP, "temp_bps"]
    assert f.loc[VWAP_F, "temp_bps"] < f.loc["AC k=2", "temp_bps"] < f.loc["AC k=4", "temp_bps"]
    # measured risk falls as urgency rises
    assert f.loc["AC k=4", "timing_risk_bps"] < f.loc["AC k=2", "timing_risk_bps"] < f.loc[VWAP_F, "timing_risk_bps"]
    # permanent cost identical across schedules
    assert f["perm_bps"].max() - f["perm_bps"].min() < 1e-9
    # driftless random walk: measured drift CI must include zero
    assert (f["drift_lo"] < 0).all() and (f["drift_hi"] > 0).all()


def test_timing_risk_end_to_end_matches_theory(u_results):
    # TWAP on a 2% daily-vol random walk: theory sd = sigma_bucket * sqrt((n+1)(2n+1)/(6n))
    n = 78
    theory = 1e4 * 0.02 / np.sqrt(n) * np.sqrt((n + 1) * (2 * n + 1) / (6 * n))
    f = frontier_table(u_results)
    got = f[(f["size"] == 0.05) & (f["schedule"] == TWAP)]["timing_risk_bps"].iloc[0]
    assert got == pytest.approx(theory, rel=0.10)


def test_forecast_vwap_tracks_market_vwap_better_than_twap(u_results):
    t = tracking_table(u_results).iloc[0]
    assert t["abs_slip_vwap_bps"] < t["abs_slip_twap_bps"] and t["reduction_lo"] > 0


# ---------- fetch script (parsing only; the live API is not called in tests) --

def test_fetch_parser_handles_both_response_shapes():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "fetch_alpaca", Path(__file__).resolve().parents[1] / "scripts" / "fetch_alpaca.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    bar = {"t": "2025-01-02T14:30:00Z", "o": 1.0, "h": 2.0, "l": 0.5, "c": 1.5, "v": 100, "n": 7, "vw": 1.2}
    expected = {"symbol": "AAPL", "timestamp": "2025-01-02T14:30:00Z", "open": 1.0, "high": 2.0,
                "low": 0.5, "close": 1.5, "volume": 100, "vwap": 1.2, "trades": 7}
    assert mod.parse_page({"bars": {"AAPL": [bar]}, "next_page_token": None}, "AAPL") == [expected]
    assert mod.parse_page({"bars": [bar]}, "AAPL") == [expected]
    assert mod.parse_page({"bars": None}, "AAPL") == []
    frame = pd.DataFrame(mod.parse_page({"bars": {"AAPL": [bar]}}, "AAPL"))
    assert set(["symbol", "timestamp", "open", "high", "low", "close", "volume", "vwap"]) <= set(frame.columns)


def test_half_day_with_after_hours_trickle_is_dropped():
    """Found on real data: on a 13:00 close, liquid stocks keep printing thin
    after-hours trades until 16:00, so every bucket has volume and the day looks
    complete. It must still be dropped, and the drop must be auditable."""
    bars = make_bars(("X",), n_days=5, seed=2)
    ts = pd.to_datetime(bars["timestamp"], utc=True).dt.tz_convert("America/New_York")
    dates = sorted(ts.dt.strftime("%Y-%m-%d").unique())
    after_close = (ts.dt.strftime("%Y-%m-%d") == dates[2]) & (ts.dt.hour >= 13)
    bars.loc[after_close, "volume"] = np.maximum(1.0, np.round(bars.loc[after_close, "volume"] * 0.003))
    assert (bars.loc[after_close, "volume"] > 0).all()               # every bucket still trades
    p = bucketize(bars, bucket_minutes=5)["X"]
    assert p.n_dropped == 1 and p.dropped_dates == (dates[2],)
    assert dates[2] not in p.dates and len(p.dates) == 4
    # the old rule alone would have kept it
    kept_by_old_rule = bucketize(bars, bucket_minutes=5, min_late_share=0.0)["X"]
    assert kept_by_old_rule.n_dropped == 0


def test_normal_days_survive_the_late_share_rule(u_results):
    bars = make_bars(("AAA", "BBB"), n_days=60, seed=21, shape="flat")  # flat = lowest late share
    assert all(p.n_dropped == 0 for p in bucketize(bars, bucket_minutes=5).values())
