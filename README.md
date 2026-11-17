# polyengine

A 24x7 research and trading engine for Polymarket. It scans markets, reads order books, researches
each question with Claude and live web search, compares prices on Kalshi and Manifold, and
trades only when several independent sources agree that the market is mispriced. Then it manages
exits.

**Paper trading is the default.** Live trading requires two explicit switches (see below).

## How it decides

```
discover ─► filter ─► match other venues ─► research ─► ensemble ─► edge & timing ─► size ─► execute ─► monitor/exit
 (Gamma)   liquidity   Kalshi + Manifold     news +        log-odds     fees, maker vs   ¼-Kelly   post-only    TP / stop /
           spread,     (fuzzy → Claude       Claude web    weighted     taker, book      + caps    maker, or    thesis-break /
           time, price  confirms same/        search        blend        imbalance,                FAK taker    pre-resolution
                        inverse rules)        (blind)                    momentum
```

| Stage | What it does | Why |
|---|---|---|
| **Discovery** | Top markets by 24h volume from the Gamma API, filtered by liquidity, volume, spread, time to resolution, and price (4¢–96¢). | Thin or near-certain markets aren't worth the risk. |
| **Order book** | Best bid/ask, spread, depth within 5¢, bid/ask imbalance, microprice, VWAP and slippage for a given size. | The true cost of a trade is the walked book plus fees, not the mid. |
| **Cross-venue matching** | Fuzzy retrieval over about 15k Kalshi series plus Manifold search, then Claude reads both sets of resolution rules and labels each candidate `same`, `inverse` or `unrelated`. Cached for 24h. | "Similar" markets with different deadlines or thresholds are the classic false arbitrage. |
| **Research triage** | Claude forecasts (~$0.30 each) run only where another venue already disagrees with Polymarket, biggest disagreement first. Held positions are always re-researched. | Under the "2 sources agree" rule, a forecast anywhere else can't lead to a trade. |
| **Research** | Google News and GDELT headlines, then Claude (`claude-opus-5-5`, adaptive thinking, live web search) produces a probability, a confidence score, and a list of resolution risks. | The forecaster is **blind to the market price**: prediction-market odds sites are blocked in its web search, so it counts as a genuinely independent source rather than an echo of the market. |
| **Ensemble** | Log-odds weighted blend: Polymarket mid (weight 3), Kalshi (2), LLM (1.6 × confidence), Manifold (0.6). | The market price is a strong prior. Fair value only moves far from it when other sources pull hard. |
| **Signal gates** | Net edge ≥ 4¢ after fees, **at least 2 independent sources on the same side**, and a minimum confidence. | No single source, the LLM included, can trigger a trade on its own. |
| **Execution** | Default is a **post-only maker** order at the bid or one tick better: makers pay no fee and earn rebates. It takes liquidity (FAK) only when the edge exceeds the fee plus a 3¢ premium. Resting orders are **repriced** when outbid, never past fair value minus the minimum edge, and cancelled if the edge disappears. | Fees are read **per market** from its `feeSchedule` (`shares × rate × p × (1−p)`). About half of markets are fee-free, so taking liquidity is cheaper there. |
| **Timing** | Defers entry when the book is heavily one-sided against you, when the microprice is below the mid (taker entries), or when price fell more than 8¢ in 6h. | Avoids catching falling knives and trading against informed flow. Deferring costs nothing; it re-checks next cycle. |
| **Sizing** | ¼-Kelly × confidence, capped by $/position, % of bankroll, total exposure, per-event exposure (correlated markets), and 25% of visible depth. | |
| **Exits** | Hard stop (−35%), thesis broken (fresh fair value below the bid), edge captured (bid ≈ fair), take-profit (+25%), and a de-risk exit 2h before resolution if little edge remains. Profit exits rest as makers; stops cross the spread. | |
| **Basket arbitrage** | In neg-risk events (outcomes are mutually exclusive), buying NO on *every* outcome pays at least N−1. If the walked-book cost plus fees is below that, the profit is locked in. Legs execute thinnest first, and excess on any leg is unwound. YES baskets are opt-in and never used on "augmented" events, whose outcome lists aren't exhaustive. | No forecasting risk. Opportunities are brief and appear after news. |
| **Learning loop** | Every source's raw probability is snapshotted hourly and scored when the market resolves (Brier, log loss, skill vs the market, reliability by bucket). From 3 results on, damped by a 20-market prior: **(1)** ensemble weights scale with measured skill (0.25–2×), **(2)** a logistic recalibration corrects systematic over- or under-confidence (typical for LLMs), **(3)** Claude is shown its own track record in later forecasts. Two cheap **learning forecasts** a day on markets resolving within 72h make the outcomes arrive fast. | The system gets more accurate as evidence accumulates, without being swung by a lucky streak. |
| **Guards** | Daily loss limit (5%), `STOP` kill-switch file, daily Claude budget in requests and dollars, 24h match cache, research reruns only after a TTL or a 5¢ price move. | |

Every forecast, signal, order, fill, settlement and LLM call is written to SQLite (`data/polyengine.db`).

## Dashboard

`polyengine run --dashboard` serves a local page at **http://127.0.0.1:8642** (localhost only, read-only). It has five tabs:

- **Overview:** equity, P&L today, win rate, positions, Claude spend vs cap, equity curve, pipeline funnel, daily P&L, opportunities per cycle, and a filterable live activity feed.
- **Markets:** every analysed market with a source strip, fair value, edge and pipeline stage. Clicking one opens a drill-down: stage stepper, source comparison, price and forecast history, order-book depth, Claude's reasoning and sources, cross-venue matches, and every signal and trade on it.
- **Trading:** realized P&L, profit factor, win/loss, drawdown, P&L by strategy and exit reason, positions, open orders, full trade history, and the arbitrage scanner.
- **Learning:** the learning loop's state, calibration curves, weight multipliers and Brier scores over time, the source scorecard, and the exact track record fed back to Claude.
- **System:** network and restarts, cycles, Claude spend per day with cache hit rate, warnings and errors, and every effective setting.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
cp .env.example .env          # add ANTHROPIC_API_KEY
.venv/bin/pytest              # 16 tests
```

## Commands

```bash
polyengine scan                         # tradable markets + book stats (no LLM, no orders)
polyengine analyze <slug-or-id>         # full pipeline on one market, prints the decision as JSON
polyengine analyze <slug> --trade       # ...and place the paper order
polyengine run                          # 24x7 paper trading
polyengine run --dashboard              # ...plus the live web dashboard on http://127.0.0.1:8642
polyengine dashboard                    # dashboard only, over the saved database
polyengine arb                          # scan neg-risk events for basket arbitrage (dry run)
polyengine calibration                  # accuracy of each source against resolved outcomes
polyengine status                       # cash, positions, uPnL, open orders, LLM spend today
touch STOP                              # kill switch: no new entries, cancels resting entry orders
```

Docker: `docker build -t polyengine . && docker run -d --env-file .env -v $PWD/data:/app/data polyengine`

## Running 24/7 on your Mac

```bash
polyengine service install      # start now + at every login, auto-restart after crashes
polyengine service status       # launchd state, last heartbeat, network status
polyengine service logs         # follow data/engine.log
polyengine service stop         # stop until next login (or `service start`)
polyengine service uninstall    # remove it completely
```

What survives what:

| Event | What happens |
|---|---|
| **Wi-Fi / internet drops** | Detected within ~15–45s, even when idle. Trading loops pause instead of erroring. Probes back off 5s → 60s. On reconnect, every loop runs a **catch-up** immediately: fills are booked, markets that resolved meanwhile are settled, and analysis is refreshed. The dashboard shows "Offline since … · paused". |
| **Mac sleeps** | `caffeinate` prevents idle sleep while plugged in. If it sleeps anyway (lid closed), the wake is detected and a catch-up cycle runs. |
| **Crash / kill / power cut** | launchd restarts it within ~30s. Positions, orders, cash, research, matches, fair values and the daily Claude budget all live in SQLite. Fills are written atomically. Nothing is re-bought and no research is re-paid. |
| **Reboot / login** | Starts automatically. |
| **Two copies at once** | A lock file refuses the second copy, so the same book can't be traded twice. |
| **Live mode, long outage** | Resting orders are placed as GTD and **expire on the exchange by themselves**, so stale orders can't sit there while you're offline. On restart, exchange orders the database doesn't know about are cancelled. |

Note: with the lid closed on battery, macOS sleeps regardless. For true 24x7, keep it plugged in (or use an always-on host).

## Going live

Run in paper mode for at least a few weeks first and review `signals` and `ledger` in the DB.
Then:

1. Fund a Polygon wallet with USDC and set allowances. Trading once through polymarket.com does
   this for proxy wallets.
2. In `.env`, set `PE_PM_PRIVATE_KEY`, plus `PE_PM_FUNDER` and `PE_PM_SIGNATURE_TYPE` if you log in
   with email or a browser wallet.
3. Start tiny: `PE_MAX_POSITION_USD=5`.
4. `PE_LIVE_TRADING=true` **and** `polyengine run --live`. Live state uses a separate DB
   (`polyengine.live.db`), so paper and live books never mix.

Check that Polymarket is legally available to you before trading: the global exchange
geo-restricts several jurisdictions.

## Honest limitations

- **Paper fills are approximations.** A resting order fills when the ask reaches it or a trade
  prints through it. Queue position is not modelled.
- **Live fill prices are logged at the limit price**, which is worst-case for takers. Reconcile
  against `get_trades` if you need exact PnL.
- **LLM forecasts are noisy.** That's why they're down-weighted, blind to price, and can never
  trade alone. Track calibration with the `research` table before raising `PE_W_LLM`.
- **Cross-venue coverage is partial.** Kalshi retrieval works well for politics, macro and
  futures-style markets, but often misses single-game sports markets.
- **Manifold matches with fewer than 15 bettors are ignored.** Thin play-money markets aren't an independent signal.
- **Polling, not streaming.** The monitor polls every 20s. For faster reaction, swap in the CLOB
  WebSocket (`wss://ws-subscriptions-clob.polymarket.com/ws/market`).
- No strategy guarantees profit. Prediction markets on liquid questions are often efficient, and
  most cycles should end in "no trade".
