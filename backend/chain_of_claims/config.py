"""Runtime configuration.

All settings come from environment variables so the service can be deployed
without code changes. Model tiering lives here: cheaper/faster models for
mechanical transform stages, the stronger model for the judgment-heavy stages
(warrant audit and fact-check).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


# Model ids. Defaults follow the env aliases the runtime exposes (proxy-friendly),
# then fall back to explicit latest tiers. Override via COC_* env if needed.
TRANSFORM_MODEL = os.environ.get(
    "COC_TRANSFORM_MODEL",
    os.environ.get("ANTHROPIC_DEFAULT_SONNET_MODEL", "claude-sonnet-latest"),
)
JUDGE_MODEL = os.environ.get(
    "COC_JUDGE_MODEL",
    os.environ.get("ANTHROPIC_DEFAULT_OPUS_MODEL", "claude-opus-latest"),
)


@dataclass(frozen=True)
class Settings:
    provider: str = os.environ.get("COC_LLM_PROVIDER", "anthropic")
    anthropic_api_key: str | None = os.environ.get("ANTHROPIC_API_KEY")
    # Proxy/gateway support: some environments (e.g. Claude Code via an AI proxy)
    # authenticate with a bearer auth token against a custom base URL instead of an
    # API key. The Anthropic SDK honours both; we surface them here.
    anthropic_auth_token: str | None = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    anthropic_base_url: str | None = os.environ.get("ANTHROPIC_BASE_URL")
    data_dir: Path = Path(os.environ.get("COC_DATA_DIR", "./coc_data")).resolve()
    db_path: Path = Path(
        os.environ.get("COC_DB_PATH", "./coc_data/chain_of_claims.db")
    ).resolve()
    # Number of independent verifiers in the Stage-4 warrant critical-question panel.
    cq_panel_size: int = int(os.environ.get("COC_CQ_PANEL_SIZE", "3"))
    # Causal-warrant check mode (Stage 4 / 4b):
    #   "off"        -> no causal check (causal_* fields stay None)
    #   "structural" -> Part B only: LLM multi-pair cause->effect extraction
    #                   (replaces the old regex; multi-pair; no evidence needed)
    #   "full"       -> Part B + Part C attribution against grounded evidence
    #                   (degrades to "structural" behaviour when no sources supplied)
    # Back-compat: the retired COC_ENABLE_CAUSAL_CHECK=0 maps to "off".
    causal_check_mode: str = os.environ.get(
        "COC_CAUSAL_CHECK_MODE",
        "structural" if os.environ.get("COC_ENABLE_CAUSAL_CHECK", "1") == "1" else "off",
    )
    # Global toggle to bypass real LLM calls (deterministic stub) for tests/CI/demo.
    offline: bool = os.environ.get("COC_OFFLINE", "0") == "1"

    @property
    def enable_causal_check(self) -> bool:
        """Back-compat alias: any causal work happens unless mode is 'off'."""
        return self.causal_check_mode != "off"

    @property
    def causal_attribution_enabled(self) -> bool:
        """Part C (evidence attribution) runs only in 'full' mode."""
        return self.causal_check_mode == "full"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "uploads").mkdir(parents=True, exist_ok=True)


settings = Settings()
