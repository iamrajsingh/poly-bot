"""All tunables live here. Every field can be overridden with an env var (prefix PE_) or .env file."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PE_", env_file=".env", extra="ignore")

    # ---- mode -------------------------------------------------------------
    # Live trading needs BOTH this flag and `--live` on the CLI. Default is paper.
    live_trading: bool = False
    paper_starting_cash: float = 1_000.0
    db_path: str = "data/polyengine.db"
    kill_switch_file: str = "STOP"  # touch this file to halt new entries + cancel orders

    # ---- Polymarket credentials (live only) -------------------------------
    pm_private_key: str = ""
    pm_funder: str = ""  # proxy wallet address if you trade via a Polymarket.com account
    pm_signature_type: int = 1  # 0 = EOA, 1 = email/magic proxy, 2 = browser-wallet proxy
    pm_chain_id: int = 137
    pm_clob_host: str = "https://clob.polymarket.com"
    pm_gamma_host: str = "https://gamma-api.polymarket.com"

    # ---- Claude -----------------------------------------------------------
    # Read from ANTHROPIC_API_KEY (env or .env). Empty -> SDK falls back to its own credential chain.
    anthropic_api_key: str = Field("", validation_alias="ANTHROPIC_API_KEY")
    anthropic_model: str = "claude-opus-5-5"
    # Token economy (measured: a forecast was ~56k input tokens, mostly web-search results re-read
    # inside the search loop). Research-style work is nearly flat across effort levels, so medium.
    forecast_effort: str = "medium"  # low | medium | high | xhigh | max
    match_effort: str = "low"
    match_model: str = ""  # empty = anthropic_model; see `polyengine eval-matcher` before changing
    web_search_max_uses: int = 5  # first forecast on a market
    web_search_refresh_max_uses: int = 3  # re-forecast: only looks for what changed since the last one
    prompt_caching: bool = True  # search results re-read inside the loop bill at the cache-read rate
    news_headlines: int = 10
    match_cache_hours: float = 72  # resolution rules don't change; venue prices are re-quoted every cycle
    # Keep the forecaster independent: odds from venues the ensemble already uses would just be
    # echoed back as fake "agreement".
    forecast_blocked_domains: list[str] = [
        "polymarket.com", "kalshi.com", "manifold.markets", "predictit.org", "metaculus.com",
        "defirate.com", "polymarketanalytics.com", "electionbettingodds.com",
    ]
    max_llm_calls_per_day: int = 300  # hard spend guards (counted per API request)
    max_llm_usd_per_day: float = 10.0
    research_ttl_minutes: int = 480  # re-research a market at most this often...
    research_price_move_trigger: float = 0.05  # ...unless the price moved this much since last research

    # ---- market universe filters -----------------------------------------
    scan_limit: int = 300  # markets pulled from Gamma per discovery pass
    min_liquidity_usd: float = 5_000.0
    min_volume_24h_usd: float = 2_000.0
    min_hours_to_resolution: float = 6.0
    max_days_to_resolution: float = 120.0
    min_price: float = 0.04  # skip near-certain markets: tiny upside, tail risk
    max_price: float = 0.96
    max_spread: float = 0.04
    max_markets_researched_per_cycle: int = 6  # forecasts (~$0.30 each) per analysis cycle
    max_matches_per_cycle: int = 25  # cross-venue match checks (~$0.02 each, cached 24h)
    # Only forecast where other venues already disagree with Polymarket (otherwise the "2 agreeing
    # sources" rule means a forecast can't lead to a trade). Held positions are always re-researched.
    research_requires_venue_signal: bool = True

    # ---- signal / edge ----------------------------------------------------
    # Log-odds ensemble weights. The Polymarket price itself gets a large prior weight:
    # liquid markets are usually right, so we only trade when other sources pull hard.
    w_polymarket: float = 3.0
    w_kalshi: float = 2.0
    w_manifold: float = 0.6
    manifold_min_bettors: int = 15  # thinner play-money markets aren't an independent signal
    w_llm: float = 1.6
    min_edge: float = 0.04  # net edge (after fees/slippage) required to enter
    min_confidence: float = 0.55
    min_agreeing_sources: int = 2  # independent sources that must agree on the direction
    taker_fee_rate: float = 0.05  # fallback when a market has no feeSchedule; fee = shares*rate*p*(1-p)
    taker_edge_premium: float = 0.03  # extra edge (beyond fees) required before crossing the spread

    # ---- basket arbitrage (neg-risk events) ---------------------------------
    arb_enabled: bool = True
    arb_yes_baskets: bool = False  # YES baskets need an exhaustive outcome list; never used on augmented events
    arb_min_return: float = 0.01  # locked-in return per basket after fees
    arb_min_apr: float = 0.20  # capital is tied up until resolution - demand a decent annualised rate
    arb_max_usd: float = 200.0  # per basket
    arb_max_exposure_pct: float = 0.30  # of bankroll across all baskets
    arb_max_legs: int = 12
    arb_scan_events: int = 200
    arb_interval: int = 60

    # ---- calibration ------------------------------------------------------
    calib_adaptive: bool = True  # learn weights + recalibration from resolved markets
    calib_min_markets: int = 3  # start adapting after this many resolutions...
    calib_prior_markets: int = 20  # ...shrunk towards "no change" with this many markets' worth of prior
    calib_feedback_min_markets: int = 8  # resolved forecasts before Claude is shown its track record
    # Learning forecasts: a few cheap forecasts a day on markets that resolve soon, purely so the
    # engine gets outcomes to learn from quickly (~$0.14 each).
    learning_forecasts_per_day: int = 2
    learning_horizon_hours: float = 72
    calibration_interval: int = 3_600
    snapshot_interval_minutes: int = 60

    # ---- maker repricing ----------------------------------------------------
    reprice_enabled: bool = True

    # ---- risk / sizing ----------------------------------------------------
    kelly_fraction: float = 0.25
    max_position_usd: float = 50.0
    max_position_pct_bankroll: float = 0.05
    max_total_exposure_pct: float = 0.60
    max_open_positions: int = 15
    max_event_exposure_pct: float = 0.15  # correlated markets inside one event
    max_daily_loss_pct: float = 0.05
    max_book_participation: float = 0.25  # never take more than this share of visible depth
    min_order_usd: float = 5.0

    # ---- exits --------------------------------------------------------------
    take_profit_pct: float = 0.25  # +25% on position cost
    stop_loss_pct: float = 0.35  # hard stop: -35% on position cost
    exit_edge_buffer: float = 0.01  # exit when bid >= fair - buffer (edge captured)
    thesis_break_edge: float = -0.03  # exit when re-researched edge turns this negative
    exit_hours_before_resolution: float = 2.0

    # ---- execution / timing -----------------------------------------------
    maker_order_ttl_seconds: int = 900
    reprice_threshold: float = 0.01
    adverse_imbalance: float = 0.6  # defer buying when book pressure is this one-sided against us
    momentum_lookback_hours: int = 6
    max_adverse_momentum: float = 0.08  # don't catch a falling knife

    # ---- loop cadence (seconds) -------------------------------------------
    discovery_interval: int = 900
    analysis_interval: int = 300
    monitor_interval: int = 20
    kalshi_refresh_interval: int = 1_800

    http_timeout: float = 20.0
    log_level: str = "INFO"

    def fee_for(self, market) -> tuple[float, float]:
        """(rate, exponent) for a market's taker fee, falling back to the configured default."""
        rate = market.fee_rate if market.fee_rate is not None else self.taker_fee_rate
        return rate, market.fee_exponent

    def weight_for(self, venue: str) -> float:
        return {"polymarket": self.w_polymarket, "kalshi": self.w_kalshi,
                "manifold": self.w_manifold, "llm": self.w_llm}.get(venue, 0.5)


@lru_cache
def get_settings() -> Settings:
    return Settings()
