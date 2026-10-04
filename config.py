"""
Centralized configuration: trading_config.json workers + .env secrets/globals.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

SUPPORTED_TRADING_ASSETS: frozenset[str] = frozenset(
    {"btc", "eth", "sol", "xrp", "doge", "hype", "bnb"}
)
SUPPORTED_WINDOWS: frozenset[str] = frozenset({"5m", "15m"})
WINDOW_SECONDS: dict[str, int] = {"5m": 300, "15m": 900}
MIN_SHARES: int = 5

_ASSET_ALIASES: dict[str, str] = {
    "bitcoin": "btc",
    "ethereum": "eth",
    "solana": "sol",
    "ripple": "xrp",
}


def _fatal(message: str) -> None:
    print(f"❌ [config] {message}", file=sys.stderr)
    sys.exit(1)


def normalize_asset_slug(raw: str) -> str:
    token = (raw or "").strip().lower()
    if not token:
        raise ValueError("empty asset token")
    return _ASSET_ALIASES.get(token, token)


def normalize_window(raw: str) -> str:
    w = (raw or "").strip().lower()
    if w not in SUPPORTED_WINDOWS:
        raise ValueError(f"unsupported window {raw!r}")
    return w


def worker_key(asset: str, window: str) -> str:
    return f"{normalize_asset_slug(asset)}:{normalize_window(window)}"


def _parse_unit_fraction(name: str, value: Any, default: float) -> float:
    """Parse a config fraction in (0, 1] — used for entry threshold and stop-loss pct."""
    raw = value if value is not None else default
    try:
        v = float(raw)
    except (TypeError, ValueError):
        _fatal(f"{name}={raw!r} is not a valid number.")
    if v <= 0 or v > 1 or v != v:
        _fatal(f"{name} must be in (0, 1] (got {raw!r}).")
    return v


def _parse_positive_int(name: str, value: Any) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        _fatal(f"{name}={value!r} is not a valid integer.")
    if v < MIN_SHARES:
        _fatal(f"{name} must be >= {MIN_SHARES} (got {value!r}).")
    return v


def _parse_order_size(name: str, value: Any, default: float) -> float:
    raw = value if value is not None else default
    try:
        v = float(raw)
    except (TypeError, ValueError):
        _fatal(f"{name}={raw!r} is not a valid number.")
    if v < MIN_SHARES or v != v or v in (float("inf"), float("-inf")):
        _fatal(f"{name} must be >= {MIN_SHARES} (got {raw!r}).")
    return v


def _parse_max_shares(name: str, value: Any, default: float) -> float:
    raw = value if value is not None else default
    try:
        v = float(raw)
    except (TypeError, ValueError):
        _fatal(f"{name}={raw!r} is not a valid number.")
    if v < MIN_SHARES or v != v or v in (float("inf"), float("-inf")):
        _fatal(f"{name} must be >= {MIN_SHARES} (got {raw!r}).")
    return v


def _parse_nonnegative_float(name: str, value: Any, default: float) -> float:
    raw = value if value is not None else default
    try:
        v = float(raw)
    except (TypeError, ValueError):
        _fatal(f"{name}={raw!r} is not a valid number.")
    if v < 0 or v != v or v in (float("inf"), float("-inf")):
        _fatal(f"{name} must be >= 0 (got {raw!r}).")
    return v


def _parse_bool_value(name: str, value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _parse_cooldown_ms(name: str, value: Any, default: int) -> int:
    try:
        v = int(value if value is not None else default)
    except (TypeError, ValueError):
        _fatal(f"{name}={value!r} is not a valid integer.")
    if v < 0:
        _fatal(f"{name} must be >= 0 (got {value!r}).")
    return v


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


def _load_env_sizing_overrides() -> Optional[dict[str, float]]:
    """Apply sizing overrides only when ORDER_SIZE_MIN, ORDER_SIZE_MAX, and MAX_SHARES are all set."""
    keys = ("ORDER_SIZE_MIN", "ORDER_SIZE_MAX", "MAX_SHARES")
    raw = {k: os.getenv(k, "").strip() for k in keys}
    set_keys = [k for k, v in raw.items() if v]
    if not set_keys:
        return None
    if len(set_keys) != len(keys):
        _fatal(
            "ORDER_SIZE_MIN, ORDER_SIZE_MAX, and MAX_SHARES must all be set together "
            f"to override sizing (found: {', '.join(set_keys)}). "
            "Omit all three to use trading_config.json defaults."
        )
    out: dict[str, float] = {
        "order_size_min": _parse_order_size("ORDER_SIZE_MIN", raw["ORDER_SIZE_MIN"], 0.0),
        "order_size_max": _parse_order_size("ORDER_SIZE_MAX", raw["ORDER_SIZE_MAX"], 0.0),
        "max_shares": _parse_max_shares("MAX_SHARES", raw["MAX_SHARES"], 0.0),
    }
    optional_max_order = os.getenv("MAX_ORDER_SIZE", "").strip()
    if optional_max_order:
        out["max_order_size"] = _parse_order_size("MAX_ORDER_SIZE", optional_max_order, 0.0)
    return out


ENV_SIZING_OVERRIDES: Optional[dict[str, float]] = _load_env_sizing_overrides()


DRY_RUN_DEFAULT: bool = _parse_bool_env("DRY_RUN_DEFAULT", _parse_bool_env("DRY_MODE", True))


def _cfg_get(raw: dict, defaults: dict, *keys: str, default: Any = None) -> Any:
    """Return the first present value from worker entry or defaults."""
    for key in keys:
        if key in raw and raw[key] is not None:
            return raw[key]
        if key in defaults and defaults[key] is not None:
            return defaults[key]
    return default


@dataclass(frozen=True)
class WorkerConfig:
    asset: str
    window: str
    # Legacy fields kept for dashboard/backward compatibility. Gabagool does not use them for entries/exits.
    momentum_entry_threshold: float = 0.90
    stop_loss_pct: float = 0.35
    gabagool_initial_entry_threshold: float = 0.49
    gabagool_max_pair_cost: float = 0.98
    gabagool_min_profit_margin: float = 0.02
    gabagool_max_unpaired_shares: float = 5.0
    gabagool_min_time_to_resolution: int = 60
    gabagool_price_buffer: float = 0.00
    gabagool_inventory_soft_limit: float = 2.0
    gabagool_hedge_max_pair_cost: float = 0.995
    # Maximum allowed loss per share when completing an already-open pair.
    # This prevents an "imbalance hedge" from turning a temporary loss into
    # a guaranteed large loss (e.g. YES 0.48 + NO 0.60 = 1.08).
    gabagool_max_hedge_loss_per_share: float = 0.005
    gabagool_emergency_unwind_imbalance: float = 3.0
    gabagool_emergency_max_loss_per_share: float = 0.02
    gabagool_endgame_seconds: int = 30
    gabagool_endgame_max_unpaired: float = 1.0
    gabagool_falling_knife_enabled: bool = True
    gabagool_falling_knife_window_seconds: float = 3.0
    gabagool_falling_knife_min_drop: float = 0.035
    gabagool_falling_knife_fast_window_seconds: float = 1.0
    gabagool_falling_knife_fast_drop: float = 0.015
    gabagool_falling_knife_recovery_cents: float = 1.5
    gabagool_falling_knife_cooldown_seconds: float = 5.0
    gabagool_falling_knife_binance_confirm: float = 0.10
    trade_cooldown_ms: int = 3000
    order_size_min: float = 5.0
    order_size_max: float = 5.0
    max_order_size: float = 5.0
    max_shares: float = 20.0
    dry_run: bool = DRY_RUN_DEFAULT
    dry_run_fill_delay_min_ms: int = 200
    dry_run_fill_delay_max_ms: int = 2500
    listener_activate_secs: int = 300
    entry_seconds_left: int = 300
    min_entry_seconds_left: int = 0
    fill_timeout_ms: int = 10000
    fill_poll_ms: int = 400
    enabled: bool = True

    @property
    def interval_seconds(self) -> int:
        return WINDOW_SECONDS[self.window]

    @property
    def key(self) -> str:
        return worker_key(self.asset, self.window)

    def market_slug(self, start_ts: int) -> str:
        return f"{self.asset}-updown-{self.window}-{start_ts}"

    @property
    def order_size(self) -> float:
        return self.order_size_max

    @property
    def random_order_size(self) -> bool:
        return self.order_size_min < self.order_size_max - 1e-9


def _merge_worker_entry(raw: dict, defaults: dict) -> WorkerConfig:
    asset = normalize_asset_slug(str(raw.get("asset", "")))
    if asset not in SUPPORTED_TRADING_ASSETS:
        _fatal(f"Invalid asset {raw.get('asset')!r}. Supported: {sorted(SUPPORTED_TRADING_ASSETS)}")

    try:
        window = normalize_window(str(raw.get("window", "")))
    except ValueError:
        _fatal(
            f"Invalid window {raw.get('window')!r} for {asset}. "
            f"Supported: {sorted(SUPPORTED_WINDOWS)}"
        )

    momentum_entry_threshold = _parse_unit_fraction(
        "momentum_entry_threshold",
        _cfg_get(raw, defaults, "momentum_entry_threshold"),
        float(defaults.get("momentum_entry_threshold", 0.90)),
    )
    env_momentum = os.getenv("MOMENTUM_ENTRY_THRESHOLD", "").strip()
    if env_momentum:
        momentum_entry_threshold = _parse_unit_fraction(
            "MOMENTUM_ENTRY_THRESHOLD", env_momentum, momentum_entry_threshold,
        )
    stop_loss_pct = _parse_unit_fraction(
        "stop_loss_pct",
        _cfg_get(raw, defaults, "stop_loss_pct"),
        float(defaults.get("stop_loss_pct", 0.35)),
    )
    env_stop = os.getenv("STOP_LOSS_PCT", "").strip()
    if env_stop:
        stop_loss_pct = _parse_unit_fraction("STOP_LOSS_PCT", env_stop, stop_loss_pct)
    gabagool_initial_entry_threshold = _parse_unit_fraction(
        "gabagool_initial_entry_threshold",
        _cfg_get(raw, defaults, "gabagool_initial_entry_threshold"),
        float(defaults.get("gabagool_initial_entry_threshold", 0.49)),
    )
    gabagool_max_pair_cost = _parse_unit_fraction(
        "gabagool_max_pair_cost",
        _cfg_get(raw, defaults, "gabagool_max_pair_cost"),
        float(defaults.get("gabagool_max_pair_cost", 0.98)),
    )
    gabagool_min_profit_margin = _parse_unit_fraction(
        "gabagool_min_profit_margin",
        _cfg_get(raw, defaults, "gabagool_min_profit_margin"),
        float(defaults.get("gabagool_min_profit_margin", 0.02)),
    )
    gabagool_max_unpaired_shares = _parse_max_shares(
        "gabagool_max_unpaired_shares",
        _cfg_get(raw, defaults, "gabagool_max_unpaired_shares"),
        float(defaults.get("gabagool_max_unpaired_shares", 5.0)),
    )
    gabagool_min_time_to_resolution = int(
        _cfg_get(raw, defaults, "gabagool_min_time_to_resolution", default=60)
    )
    gabagool_price_buffer = float(
        _cfg_get(raw, defaults, "gabagool_price_buffer", default=0.0)
    )
    gabagool_inventory_soft_limit = _parse_nonnegative_float(
        "gabagool_inventory_soft_limit",
        _cfg_get(raw, defaults, "gabagool_inventory_soft_limit"),
        float(defaults.get("gabagool_inventory_soft_limit", 2.0)),
    )
    gabagool_hedge_max_pair_cost = _parse_unit_fraction(
        "gabagool_hedge_max_pair_cost",
        _cfg_get(raw, defaults, "gabagool_hedge_max_pair_cost"),
        float(defaults.get("gabagool_hedge_max_pair_cost", 0.995)),
    )
    gabagool_max_hedge_loss_per_share = _parse_unit_fraction(
        "gabagool_max_hedge_loss_per_share",
        _cfg_get(raw, defaults, "gabagool_max_hedge_loss_per_share"),
        float(defaults.get("gabagool_max_hedge_loss_per_share", 0.005)),
    )
    gabagool_emergency_unwind_imbalance = _parse_nonnegative_float(
        "gabagool_emergency_unwind_imbalance",
        _cfg_get(raw, defaults, "gabagool_emergency_unwind_imbalance"),
        float(defaults.get("gabagool_emergency_unwind_imbalance", 3.0)),
    )
    gabagool_emergency_max_loss_per_share = _parse_unit_fraction(
        "gabagool_emergency_max_loss_per_share",
        _cfg_get(raw, defaults, "gabagool_emergency_max_loss_per_share"),
        float(defaults.get("gabagool_emergency_max_loss_per_share", 0.02)),
    )
    gabagool_endgame_seconds = int(
        _cfg_get(raw, defaults, "gabagool_endgame_seconds", default=30)
    )
    gabagool_endgame_max_unpaired = _parse_nonnegative_float(
        "gabagool_endgame_max_unpaired",
        _cfg_get(raw, defaults, "gabagool_endgame_max_unpaired"),
        float(defaults.get("gabagool_endgame_max_unpaired", 1.0)),
    )
    if gabagool_min_time_to_resolution < 0:
        _fatal(f"{asset}:{window}: gabagool_min_time_to_resolution must be >= 0")
    if gabagool_price_buffer < 0 or gabagool_price_buffer > 0.05:
        _fatal(f"{asset}:{window}: gabagool_price_buffer must be between 0 and 0.05")
    if gabagool_min_profit_margin >= 1:
        _fatal(f"{asset}:{window}: gabagool_min_profit_margin must be < 1")
    if gabagool_hedge_max_pair_cost > 1.0 + 1e-9:
        _fatal(f"{asset}:{window}: gabagool_hedge_max_pair_cost must be <= 1.0")
    if gabagool_max_hedge_loss_per_share < 0 or gabagool_max_hedge_loss_per_share >= 1.0:
        _fatal(f"{asset}:{window}: gabagool_max_hedge_loss_per_share must be >= 0 and < 1.0")
    if gabagool_inventory_soft_limit > gabagool_max_unpaired_shares + 1e-9:
        _fatal(
            f"{asset}:{window}: gabagool_inventory_soft_limit={gabagool_inventory_soft_limit} "
            f"cannot exceed gabagool_max_unpaired_shares={gabagool_max_unpaired_shares}"
        )
    if gabagool_emergency_unwind_imbalance > gabagool_max_unpaired_shares + 1e-9:
        _fatal(
            f"{asset}:{window}: gabagool_emergency_unwind_imbalance={gabagool_emergency_unwind_imbalance} "
            f"cannot exceed gabagool_max_unpaired_shares={gabagool_max_unpaired_shares}"
        )
    if gabagool_endgame_seconds < 0:
        _fatal(f"{asset}:{window}: gabagool_endgame_seconds must be >= 0")
    if gabagool_endgame_max_unpaired > gabagool_max_unpaired_shares + 1e-9:
        _fatal(
            f"{asset}:{window}: gabagool_endgame_max_unpaired={gabagool_endgame_max_unpaired} "
            f"cannot exceed gabagool_max_unpaired_shares={gabagool_max_unpaired_shares}"
        )
    gabagool_falling_knife_enabled = _parse_bool_value(
        "gabagool_falling_knife_enabled",
        _cfg_get(raw, defaults, "gabagool_falling_knife_enabled", default=True),
        True,
    )
    gabagool_falling_knife_window_seconds = _parse_nonnegative_float(
        "gabagool_falling_knife_window_seconds",
        _cfg_get(raw, defaults, "gabagool_falling_knife_window_seconds", default=3.0),
        3.0,
    )
    gabagool_falling_knife_min_drop = _parse_nonnegative_float(
        "gabagool_falling_knife_min_drop",
        _cfg_get(raw, defaults, "gabagool_falling_knife_min_drop", default=0.035),
        0.035,
    )
    gabagool_falling_knife_fast_window_seconds = _parse_nonnegative_float(
        "gabagool_falling_knife_fast_window_seconds",
        _cfg_get(raw, defaults, "gabagool_falling_knife_fast_window_seconds", default=1.0),
        1.0,
    )
    gabagool_falling_knife_fast_drop = _parse_nonnegative_float(
        "gabagool_falling_knife_fast_drop",
        _cfg_get(raw, defaults, "gabagool_falling_knife_fast_drop", default=0.015),
        0.015,
    )
    gabagool_falling_knife_recovery_cents = _parse_nonnegative_float(
        "gabagool_falling_knife_recovery_cents",
        _cfg_get(raw, defaults, "gabagool_falling_knife_recovery_cents", default=1.5),
        1.5,
    )
    gabagool_falling_knife_cooldown_seconds = _parse_nonnegative_float(
        "gabagool_falling_knife_cooldown_seconds",
        _cfg_get(raw, defaults, "gabagool_falling_knife_cooldown_seconds", default=5.0),
        5.0,
    )
    gabagool_falling_knife_binance_confirm = _parse_nonnegative_float(
        "gabagool_falling_knife_binance_confirm",
        _cfg_get(raw, defaults, "gabagool_falling_knife_binance_confirm", default=0.10),
        0.10,
    )
    if gabagool_falling_knife_window_seconds <= 0:
        _fatal(f"{asset}:{window}: gabagool_falling_knife_window_seconds must be > 0")
    if gabagool_falling_knife_fast_window_seconds <= 0:
        _fatal(f"{asset}:{window}: gabagool_falling_knife_fast_window_seconds must be > 0")
    if gabagool_falling_knife_fast_window_seconds > gabagool_falling_knife_window_seconds:
        _fatal(f"{asset}:{window}: falling-knife fast window cannot exceed main window")
    if gabagool_falling_knife_min_drop <= 0 and gabagool_falling_knife_fast_drop <= 0:
        _fatal(f"{asset}:{window}: at least one falling-knife drop threshold must be > 0")

    if gabagool_max_pair_cost > 1.0 - gabagool_min_profit_margin + 1e-9:
        _fatal(
            f"{asset}:{window}: gabagool_max_pair_cost={gabagool_max_pair_cost} "
            f"must be <= 1 - gabagool_min_profit_margin={1.0 - gabagool_min_profit_margin:.4f}"
        )

    trade_cooldown_ms = _parse_cooldown_ms(
        "trade_cooldown_ms",
        _cfg_get(raw, defaults, "trade_cooldown_ms"),
        int(defaults.get("trade_cooldown_ms", 3000)),
    )

    order_size_fixed = _parse_order_size(
        "order_size",
        _cfg_get(raw, defaults, "order_size", "spread_size"),
        float(_cfg_get(defaults, {}, "order_size", "spread_size", default=5.0)),
    )
    size_min_raw = _cfg_get(raw, defaults, "order_size_min", "spread_size_min")
    size_max_raw = _cfg_get(raw, defaults, "order_size_max", "spread_size_max")
    if size_min_raw is None and size_max_raw is None:
        order_size_min = order_size_max = order_size_fixed
    else:
        order_size_min = _parse_order_size(
            "order_size_min",
            size_min_raw if size_min_raw is not None else order_size_fixed,
            order_size_fixed,
        )
        order_size_max = _parse_order_size(
            "order_size_max",
            size_max_raw if size_max_raw is not None else order_size_fixed,
            order_size_fixed,
        )
    max_order = _parse_order_size(
        "max_order_size",
        _cfg_get(raw, defaults, "max_order_size"),
        float(defaults.get("max_order_size", 5.0)),
    )
    max_shares = _parse_max_shares(
        "max_shares",
        _cfg_get(raw, defaults, "max_shares"),
        float(defaults.get("max_shares", 20.0)),
    )

    if ENV_SIZING_OVERRIDES:
        order_size_min = ENV_SIZING_OVERRIDES["order_size_min"]
        order_size_max = ENV_SIZING_OVERRIDES["order_size_max"]
        max_shares = ENV_SIZING_OVERRIDES["max_shares"]
        if "max_order_size" in ENV_SIZING_OVERRIDES:
            max_order = ENV_SIZING_OVERRIDES["max_order_size"]

    if order_size_min > order_size_max:
        _fatal(
            f"{asset}:{window}: order_size_min ({order_size_min}) "
            f"cannot exceed order_size_max ({order_size_max})"
        )
    max_order = max(max_order, order_size_max)
    if max_order > max_shares:
        _fatal(
            f"{asset}:{window}: max_order_size ({max_order}) "
            f"cannot exceed max_shares ({max_shares})"
        )

    dr_raw = _cfg_get(raw, defaults, "dry_run")
    if dr_raw is None:
        dry_run = DRY_RUN_DEFAULT
    else:
        dry_run = bool(dr_raw)

    dry_min = _parse_cooldown_ms(
        "dry_run_fill_delay_min_ms",
        _cfg_get(raw, defaults, "dry_run_fill_delay_min_ms"),
        int(defaults.get("dry_run_fill_delay_min_ms", 200)),
    )
    dry_max = _parse_cooldown_ms(
        "dry_run_fill_delay_max_ms",
        _cfg_get(raw, defaults, "dry_run_fill_delay_max_ms"),
        int(defaults.get("dry_run_fill_delay_max_ms", 2500)),
    )
    if dry_max < dry_min:
        _fatal(f"{asset}:{window}: dry_run_fill_delay_max_ms must be >= dry_run_fill_delay_min_ms")

    interval = WINDOW_SECONDS[window]
    listener_raw = _cfg_get(raw, defaults, "listener_activate_secs")
    entry_raw = _cfg_get(raw, defaults, "entry_seconds_left")
    env_listener = os.getenv("LISTENER_ACTIVATE_SECONDS", "").strip()
    env_entry = os.getenv("ENTRY_SECONDS_LEFT", "").strip()
    if listener_raw is not None:
        listener_secs = int(listener_raw)
    elif env_listener:
        listener_secs = int(env_listener)
    else:
        listener_secs = interval
    if entry_raw is not None:
        entry_secs = int(entry_raw)
    elif env_entry:
        entry_secs = int(env_entry)
    else:
        entry_secs = interval

    min_entry_raw = _cfg_get(raw, defaults, "min_entry_seconds_left")
    fill_timeout_raw = _cfg_get(
        raw, defaults, "fill_timeout_ms", "spread_fill_timeout_ms",
    )
    fill_poll_raw = _cfg_get(raw, defaults, "fill_poll_ms", "spread_fill_poll_ms")

    min_entry_secs = int(min_entry_raw if min_entry_raw is not None else 0)
    fill_timeout_ms = _parse_cooldown_ms(
        "fill_timeout_ms",
        fill_timeout_raw,
        int(_cfg_get(defaults, {}, "fill_timeout_ms", "spread_fill_timeout_ms", default=10000)),
    )
    fill_poll_ms = _parse_cooldown_ms(
        "fill_poll_ms",
        fill_poll_raw,
        int(_cfg_get(defaults, {}, "fill_poll_ms", "spread_fill_poll_ms", default=400)),
    )

    if min_entry_secs < 0:
        _fatal(f"{asset}:{window}: min_entry_seconds_left must be >= 0")
    if fill_poll_ms > fill_timeout_ms:
        _fatal(f"{asset}:{window}: fill_poll_ms must be <= fill_timeout_ms")

    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        enabled = str(enabled).lower() in ("1", "true", "yes", "on")

    return WorkerConfig(
        asset=asset,
        window=window,
        momentum_entry_threshold=momentum_entry_threshold,
        stop_loss_pct=stop_loss_pct,
        gabagool_initial_entry_threshold=gabagool_initial_entry_threshold,
        gabagool_max_pair_cost=gabagool_max_pair_cost,
        gabagool_min_profit_margin=gabagool_min_profit_margin,
        gabagool_max_unpaired_shares=gabagool_max_unpaired_shares,
        gabagool_min_time_to_resolution=gabagool_min_time_to_resolution,
        gabagool_price_buffer=gabagool_price_buffer,
        gabagool_inventory_soft_limit=gabagool_inventory_soft_limit,
        gabagool_hedge_max_pair_cost=gabagool_hedge_max_pair_cost,
        gabagool_max_hedge_loss_per_share=gabagool_max_hedge_loss_per_share,
        gabagool_emergency_unwind_imbalance=gabagool_emergency_unwind_imbalance,
        gabagool_emergency_max_loss_per_share=gabagool_emergency_max_loss_per_share,
        gabagool_endgame_seconds=gabagool_endgame_seconds,
        gabagool_endgame_max_unpaired=gabagool_endgame_max_unpaired,
        gabagool_falling_knife_enabled=gabagool_falling_knife_enabled,
        gabagool_falling_knife_window_seconds=gabagool_falling_knife_window_seconds,
        gabagool_falling_knife_min_drop=gabagool_falling_knife_min_drop,
        gabagool_falling_knife_fast_window_seconds=gabagool_falling_knife_fast_window_seconds,
        gabagool_falling_knife_fast_drop=gabagool_falling_knife_fast_drop,
        gabagool_falling_knife_recovery_cents=gabagool_falling_knife_recovery_cents,
        gabagool_falling_knife_cooldown_seconds=gabagool_falling_knife_cooldown_seconds,
        gabagool_falling_knife_binance_confirm=gabagool_falling_knife_binance_confirm,
        trade_cooldown_ms=trade_cooldown_ms,
        order_size_min=order_size_min,
        order_size_max=order_size_max,
        max_order_size=max_order,
        max_shares=max_shares,
        dry_run=dry_run,
        dry_run_fill_delay_min_ms=dry_min,
        dry_run_fill_delay_max_ms=dry_max,
        listener_activate_secs=listener_secs,
        entry_seconds_left=entry_secs,
        min_entry_seconds_left=min_entry_secs,
        fill_timeout_ms=fill_timeout_ms,
        fill_poll_ms=fill_poll_ms,
        enabled=enabled,
    )


def load_worker_configs(path: Optional[str] = None) -> Tuple[WorkerConfig, ...]:
    cfg_path = path or os.getenv("TRADING_CONFIG_PATH", "trading_config.json")
    if not os.path.isfile(cfg_path):
        _fatal(f"Trading config not found: {cfg_path}")

    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        _fatal(f"Invalid JSON in {cfg_path}: {e}")
    except OSError as e:
        _fatal(f"Cannot read {cfg_path}: {e}")

    if not isinstance(data, dict):
        _fatal(f"{cfg_path} must be a JSON object.")

    defaults = data.get("defaults") or {}
    workers_raw = data.get("workers")
    if not isinstance(workers_raw, list) or not workers_raw:
        _fatal(f"{cfg_path} must contain a non-empty 'workers' array.")

    seen: set[str] = set()
    out: list[WorkerConfig] = []
    for entry in workers_raw:
        if not isinstance(entry, dict):
            _fatal("Each worker entry must be a JSON object.")
        wc = _merge_worker_entry(entry, defaults)
        if not wc.enabled:
            continue
        if wc.key in seen:
            _fatal(f"Duplicate worker config: {wc.key}")
        seen.add(wc.key)
        out.append(wc)

    if not out:
        _fatal("No enabled workers in trading config.")

    return tuple(out)


WORKER_CONFIGS: Tuple[WorkerConfig, ...] = load_worker_configs()
TRADING_ASSETS: Tuple[str, ...] = tuple(dict.fromkeys(w.asset for w in WORKER_CONFIGS))
TRADING_ASSETS_UPPER: Tuple[str, ...] = tuple(a.upper() for a in TRADING_ASSETS)
ALL_TRACKED_ASSETS = TRADING_ASSETS
TOTAL_BOTS: int = len(WORKER_CONFIGS)


def asset_pnl_filename(asset: str, window: str = "5m") -> str:
    a = normalize_asset_slug(asset)
    w = normalize_window(window)
    return f"{a}_{w}_pnl_history.json"


PNL_FILES: list[str] = [asset_pnl_filename(w.asset, w.window) for w in WORKER_CONFIGS]


def validate_trading_assets() -> Tuple[str, ...]:
    if not TRADING_ASSETS:
        _fatal("No trading assets resolved from worker config.")
    return TRADING_ASSETS


def trading_assets_label(separator: str = " · ") -> str:
    labels = [f"{w.asset.upper()} {w.window}" for w in WORKER_CONFIGS]
    return separator.join(labels)


def _parse_positive_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        _fatal(f"{name}={raw!r} is not a valid number.")
    if value <= 0 or value != value or value in (float("inf"), float("-inf")):
        _fatal(f"{name} must be a positive number (got {raw!r}).")
    return value


def _parse_positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        _fatal(f"{name}={raw!r} is not a valid integer.")
    if value <= 0:
        _fatal(f"{name} must be a positive integer (got {raw!r}).")
    return value


ASSET_MAX_CUMULATIVE_LOSS: float = _parse_positive_float_env(
    "ASSET_MAX_CUMULATIVE_LOSS", 3.00,
)
ASSET_COOLDOWN_MINUTES: int = _parse_positive_int_env("ASSET_COOLDOWN_MINUTES", 30)
ASSET_COOLDOWN_SECONDS: int = ASSET_COOLDOWN_MINUTES * 60


def validate_asset_cooldown_config() -> tuple[float, int]:
    return ASSET_MAX_CUMULATIVE_LOSS, ASSET_COOLDOWN_MINUTES


print(
    f"📌 Workers ({len(WORKER_CONFIGS)}): "
    + ", ".join(f"{w.asset.upper()} {w.window}" for w in WORKER_CONFIGS)
)
print(
    f"🛡️  Asset cooldown: max loss ${ASSET_MAX_CUMULATIVE_LOSS:.2f} | "
    f"cooldown {ASSET_COOLDOWN_MINUTES} min (per asset+window)"
)
print(f"🧪 DRY_RUN_DEFAULT={DRY_RUN_DEFAULT}")
if WORKER_CONFIGS:
    wc0 = WORKER_CONFIGS[0]
    if ENV_SIZING_OVERRIDES:
        size_label = (
            f"{wc0.order_size_min}-{wc0.order_size_max} random"
            if wc0.random_order_size
            else str(wc0.order_size_max)
        )
        print(
            f"📐 Sizing (.env override): order={size_label} | "
            f"max_order={wc0.max_order_size} | max_shares={wc0.max_shares}"
        )
    else:
        print(
            f"📐 Sizing (trading_config.json): order={wc0.order_size_max} fixed | "
            f"max_order={wc0.max_order_size} | max_shares={wc0.max_shares}"
        )
    print(
        f"📈 Momentum: entry>={wc0.momentum_entry_threshold:.2f} | "
        f"stop_loss={wc0.stop_loss_pct:.0%} (env: MOMENTUM_ENTRY_THRESHOLD, STOP_LOSS_PCT)"
    )
