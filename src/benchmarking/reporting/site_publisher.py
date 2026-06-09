"""Publish a benchmark session to a static site data directory.

Transforms a BenchmarkSession into a pre-aggregated JSON file (stats are
computed here, reusing StatisticalAnalyzer, so the site never re-derives them)
and upserts an entry into a manifest.json. The manifest is the single source of
truth for which sessions are live on the site, so deciding what to publish is
just a matter of running `publish` / `unpublish`.

Layout written into the site (default `<site>/public/benchmarks/`):

    manifest.json            # index of published sessions + which is default
    <session-id>.json        # one pre-aggregated file per published session
    <session-id>_raw.csv     # optional raw data, for a "download" link
"""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from ..config.models import BenchmarkSession
from ..config.settings import Settings
from ..metrics.statistics import StatisticalAnalyzer

# Metrics emitted per (model, provider, workload) row. Keyed by the JSON field
# name the site reads; value is how to pull the raw values off a result.
_METRICS = {
    "ttft_ms": lambda r: r.ttft_ms,
    "latency_ms": lambda r: r.end_to_end_latency_ms,
    "output_tok_per_sec": lambda r: r.tokens_per_sec,
    "inter_token_latency_ms": lambda r: r.inter_token_latency_ms,
    "tokens_generated": lambda r: float(r.tokens_generated),
}

# Stat fields copied from StatisticalAnalyzer into each metric block.
_STAT_FIELDS = ("mean", "median", "p95", "p99", "std_dev", "min", "max", "count", "cv")


class SitePublisher:
    """Write a session's aggregated stats + manifest entry into the site repo."""

    def __init__(self, settings: Settings, results_dir: str):
        self.settings = settings
        self.results_dir = Path(results_dir)
        self.analyzer = StatisticalAnalyzer()

    def _workload_tokens(self) -> Dict[str, int]:
        """Map workload key -> input token size, from config (best effort)."""
        try:
            return {
                name: cfg.tokens for name, cfg in self.settings.get_all_workload_configs().items()
            }
        except Exception:
            return {}

    def _stats(self, values: List[float]) -> Optional[Dict[str, float]]:
        values = [v for v in values if v is not None]
        if not values:
            return None
        s = self.analyzer.analyze_metrics(values)
        return {k: s[k] for k in _STAT_FIELDS}

    def build_payload(
        self, session: BenchmarkSession, label: Optional[str], published_at: str
    ) -> dict:
        """Build the pre-aggregated JSON payload for a session."""
        tokens_by_workload = self._workload_tokens()

        rows: List[dict] = []
        for model in session.models:
            for provider in session.services:
                for workload in session.workloads:
                    subset = [
                        r
                        for r in session.results
                        if r.model_name == model
                        and r.service_name == provider
                        and r.workload_type == workload
                    ]
                    if not subset:
                        continue
                    succeeded = [r for r in subset if r.error is None]
                    error_rate = self.analyzer.calculate_error_rate(
                        len(subset), len(subset) - len(succeeded)
                    )

                    metrics: Dict[str, Optional[dict]] = {}
                    for name, getter in _METRICS.items():
                        metrics[name] = self._stats([getter(r) for r in succeeded])

                    rows.append(
                        {
                            "model": model,
                            "provider": provider,
                            "workload": workload,
                            "tokens": tokens_by_workload.get(workload),
                            "samples": len(succeeded),
                            "error_rate_pct": round(error_rate, 3),
                            "metrics": metrics,
                        }
                    )

        # Workloads sorted by input size so the site can plot a clean sweep.
        workloads = sorted(
            session.workloads,
            key=lambda w: tokens_by_workload.get(w, 0),
        )
        workload_meta = [{"key": w, "tokens": tokens_by_workload.get(w)} for w in workloads]

        gen = self.settings.get_generation_params()

        return {
            "session_id": session.session_id,
            "label": label or session.session_id,
            "published_at": published_at,
            "start_time": session.start_time.isoformat() if session.start_time else None,
            "end_time": session.end_time.isoformat() if session.end_time else None,
            "providers": session.services,
            "models": session.models,
            "workloads": workload_meta,
            "iterations_per_workload": session.iterations_per_workload,
            "generation": {
                "max_tokens": gen.max_tokens,
                "temperature": gen.temperature,
                "top_p": gen.top_p,
            },
            "rows": rows,
        }

    def _load_manifest(self, manifest_path: Path) -> dict:
        if manifest_path.exists():
            with open(manifest_path) as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                return loaded
            raise ValueError(f"Manifest must contain a JSON object: {manifest_path}")
        return {"default": None, "sessions": []}

    def publish(
        self,
        session: BenchmarkSession,
        site_dir: Path,
        label: Optional[str] = None,
        make_default: Optional[bool] = None,
        copy_raw: bool = True,
    ) -> dict:
        """Write the session JSON, optional raw CSV, and upsert the manifest.

        Returns a small dict describing what was written.
        """
        site_dir.mkdir(parents=True, exist_ok=True)
        published_at = datetime.now(timezone.utc).isoformat()

        payload = self.build_payload(session, label, published_at)
        session_file = f"{session.session_id}.json"
        with open(site_dir / session_file, "w") as f:
            json.dump(payload, f, indent=2)

        raw_file = None
        if copy_raw:
            src = self.results_dir / f"session_{session.session_id}_raw.csv"
            if src.exists():
                raw_file = f"{session.session_id}_raw.csv"
                shutil.copyfile(src, site_dir / raw_file)

        manifest_path = site_dir / "manifest.json"
        manifest = self._load_manifest(manifest_path)

        entry = {
            "session_id": session.session_id,
            "label": payload["label"],
            "published_at": published_at,
            "models": session.models,
            "providers": session.services,
            "workloads": [w["key"] for w in payload["workloads"]],
            "iterations_per_workload": session.iterations_per_workload,
            "file": session_file,
            "raw_file": raw_file,
        }

        # Upsert by session_id; newest-published first.
        others = [s for s in manifest["sessions"] if s["session_id"] != session.session_id]
        manifest["sessions"] = [entry] + others

        # Default selection: explicit flag wins; otherwise first publish becomes
        # default, and re-publishing the current default keeps it.
        if make_default is True or not manifest.get("default"):
            manifest["default"] = session.session_id
        elif make_default is False and manifest["default"] == session.session_id:
            manifest["default"] = manifest["sessions"][0]["session_id"]

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        return {
            "session_file": str(site_dir / session_file),
            "raw_file": str(site_dir / raw_file) if raw_file else None,
            "manifest": str(manifest_path),
            "is_default": manifest["default"] == session.session_id,
            "row_count": len(payload["rows"]),
        }

    def unpublish(self, session_id: str, site_dir: Path) -> dict:
        """Remove a session from the site (files + manifest entry)."""
        manifest_path = site_dir / "manifest.json"
        manifest = self._load_manifest(manifest_path)

        before = len(manifest["sessions"])
        removed_entry = next(
            (s for s in manifest["sessions"] if s["session_id"] == session_id), None
        )
        manifest["sessions"] = [s for s in manifest["sessions"] if s["session_id"] != session_id]

        # Delete the data files.
        for fname in [f"{session_id}.json", f"{session_id}_raw.csv"]:
            fpath = site_dir / fname
            if fpath.exists():
                fpath.unlink()

        # Repair default if we removed it.
        if manifest.get("default") == session_id:
            manifest["default"] = (
                manifest["sessions"][0]["session_id"] if manifest["sessions"] else None
            )

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        return {
            "removed": removed_entry is not None,
            "remaining": len(manifest["sessions"]),
            "was_present": before != len(manifest["sessions"]),
        }
