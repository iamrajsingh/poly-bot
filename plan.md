# polyengine: plan & status

_Last updated: 2026-10-21_

## Goal

A 24x7 engine that analyses Polymarket markets, understands their order books, researches
across multiple sources, cross-checks other prediction markets, enters at good prices, and exits
in profit. Risk is kept low by requiring several independent sources to agree.

---

## Done

| Area | Status | Where |
|---|---|---|
| Market discovery and filters | ✅ verified live | `venues/polymarket.py`, `engine.py` |
| Order book analytics (spread, depth, imbalance, microprice, VWAP sweep) | ✅ tested + live | `orderbook.py` |
| **Per-market taker fees** from each market's `feeSchedule` (0% to 7%) | ✅ tested; field checked live | `venues/polymarket.py`, `signals.py`, `portfolio.py` |
| Kalshi series-targeted retrieval + Manifold search | ✅ verified live | `venues/`, `matching.py` |
| Claude same/inverse matcher | ✅ **verified with real API** (~$0.015/check) | `matching.py` |
| Blind Claude forecaster, prediction-market odds sites blocked | ✅ **verified with real API** (~$0.35/forecast) | `research/forecaster.py` |
| **Research triage:** forecast only where another venue disagrees | ✅ tested + live (159 markets → 2 forecasts per cycle) | `engine.py` |
| Manifold minimum-bettor filter + participation weighting | ✅ | `engine.py` |
| Log-odds ensemble + "≥2 sources agree" gate | ✅ tested | `signals.py` |
| Entry timing, ¼-Kelly sizing, risk caps, kill switch, LLM budget | ✅ tested | `signals.py`, `risk.py`, `llm.py` |
| **Maker repricing** (follow when outbid, capped at the edge; cancel when the edge is gone) | ✅ tested | `engine.py` |
| **Neg-risk basket arbitrage** (NO baskets; YES opt-in, never on augmented events) | ✅ tested incl. leg unwind; scanner live | `arbitrage.py`, `engine.py` |
| **Calibration + adaptive weights** (Brier, skill vs market, closer-when-disagreeing) | ✅ tested; needs resolved markets to act | `calibration.py` |
| Paper broker, exits, settlement, SQLite audit trail | ✅ tested | `execution/paper.py`, `portfolio.py`, `storage.py` |
| **Web dashboard**: 5 tabs (Overview, Markets, Trading, Learning, System), full trade history, per-market history | ✅ verified in browser, light + dark | `dashboard.py`, `web/index.html` |
| **Learning loop**: shrunk skill weights, logistic recalibration, Claude track-record feedback, learning forecasts | ✅ tested (recovers a 2× over-confident forecaster); live, awaiting first resolutions | `calibration.py`, `engine.py` |
| **One quote per venue** (a venue's YES/NO pair can't count as two sources) + stricter matcher rules | ✅ tested | `signals.py`, `matching.py` |
| 24x7 loop, graceful shutdown, CLI, Dockerfile | ✅ verified live | `engine.py`, `cli.py` |
| **Network watchdog**: pause on loss (idle heartbeat), backoff probing, catch-up on reconnect | ✅ tested + **live outage simulated** (paused, resumed 8s after reconnect) | `connectivity.py`, `engine.py` |
| **Restart safety**: atomic fills, pre-recorded orders, persisted fair values, single-instance lock | ✅ tested + `kill -9` recovery verified | `storage.py`, `engine.py`, `service.py` |
| **Sleep/wake detection** with catch-up | ✅ tested | `connectivity.py` |
| **launchd service** (`polyengine service install`), caffeinate, auto-restart | ✅ plist validated; ⚠️ not yet installed on this Mac | `service.py` |
| Live GTD order expiry + orphan-order reconcile on restart | ⚠️ written, not exercised against the exchange | `execution/live.py` |
| Live broker (py-clob-client) | ⚠️ written, **never run against the exchange** | `execution/live.py` |

Tests: 31 passing (`.venv/bin/pytest`).

---

## What's left

### P0: before trusting results

1. **Move the API key out of `.env.example`.** It was pasted into the template (which gets
   committed). The engine reads `.env`, where a copy now lives. Blank the template, and rotate the
   key if the template was ever shared or committed.
2. **Paper-trade for 2–4 weeks** with `polyengine run --dashboard`. Watch the funnel, review
   `signals` and `ledger`, and let calibration accumulate resolved markets.
3. **Tune spend from real numbers.** Forecasts cost about $0.35 each. With triage, a full cycle
   used 2 forecasts and about 25 match checks. Set `PE_MAX_LLM_USD_PER_DAY` to match how much
   coverage you want.

### P1: before going live

4. **Exercise the live broker** with $1–5 orders: `post_order`/`get_order` fields, post-only
   rejection, FAK partial fills, neg-risk markets, USDC balance units. Selling also needs the
   CONDITIONAL token allowance (`update_balance_allowance`), which isn't wired yet.
5. **Exact live fill prices** from `get_trades` (fills are currently booked at the limit price).
6. **Position reconciliation:** orphaned orders are now cancelled on restart, but positions
   aren't yet compared against the exchange's records (Data API).
7. **Basket execution on live:** legs are sequential FAK orders. Measure real leg slippage, and
   consider the neg-risk adapter's NO→USDC conversion to free capital before resolution.
8. **Partial maker fills in paper:** fill only the volume that traded through, so paper results aren't optimistic.
9. **Sports in-play risk:** skip entries near `gameStartTime`, and don't hold positions into play unintentionally.
10. **Daily loss metric** from a UTC-midnight equity snapshot.
11. **Alerts** (Telegram/Slack/email) for fills, exits, halts and repeated errors.
12. **Secrets:** wallet key in a secret manager or keychain; use a dedicated low-balance trading wallet.

### P2: more profit

13. **Domain sources:** sportsbook odds for sports (sharp, easy to match, where Kalshi coverage is
    weak), CME FedWatch/FRED for Fed markets, polling averages for elections.
14. **Date-ladder and threshold arbitrage:** "by March" ≤ "by June", "above $X" ≥ "above $Y".
    Needs reliable detection of nested markets.
15. **Event-driven research:** re-research on breaking news that matches a held or watched market.
16. **WebSocket feeds** for books and fills instead of polling.
17. **Cheaper research:** Batch API (50% off) for routine re-research; fewer web searches for
    routine updates. Most of the cost is about 48k input tokens of search results per forecast.
18. **Backtest** timing and exit thresholds on `prices-history`.

- **Learning, next steps:** learn per category (sports vs politics vs crypto) once each has
    30+ resolutions, and tune `PE_MIN_EDGE` from realized edge vs predicted edge.

### P3: operations

19. `git init`, CI, lockfile.
20. Always-on host with a persistent `data/` volume, restart policy and health check.
21. Confirm Polymarket is legally accessible from where the engine runs and trades.

---

## Suggested order

**Now:** P0.1 (key) → `polyengine service install` (paper mode) and watch the dashboard.<br>
**During the paper run:** P1.8–P1.11, P2.13 (sportsbook odds is the biggest coverage gain).<br>
**Before live:** P1.4–P1.7, P1.12, then go live with `PE_MAX_POSITION_USD=5`.
