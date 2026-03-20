"""
backend/tests/test_llm_full_run_usage.py
Measures real LLM usage for one full AutoEval-C pipeline run.

Run:
    python -m backend.tests.test_llm_full_run_usage

What it measures:
  - Agent 1 request attempts and successful calls
  - Agent 2 request attempts and successful calls
  - Agent 3 request attempts and successful calls
  - Prompt, completion, and total tokens reported by the provider
  - Rate limit retries and transient retries
  - Retry wait time injected by llm_client.py
  - Peak rolling 60-second request demand
  - Peak rolling 60-second token demand

This is a manual diagnostic tool.
It does NOT run as pytest.
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from dotenv import load_dotenv

from backend.agents.agent1_extractor import get_agent1
from backend.agents.agent2_evaluator import get_agent2
from backend.agents.agent3_feedback import get_agent3
from backend.config.config import DATA_INPUTS_PATH, DATA_OUTPUTS_PATH, LLM_PROVIDER
from backend.rag.rag_pipeline import rag_pipeline
from backend.utils import llm_client
from backend.utils.logger import engine_logger

load_dotenv()


DEFAULT_ASSIGNMENT_ID = "diagnostic_usage_lab"
DEFAULT_STUDENT_ID = "diagnostic_usage_student_01"
DEFAULT_FORCE_REGENERATE = True
DEFAULT_INPUTS_DIRECTORY = Path(DATA_INPUTS_PATH)
DEFAULT_STUDENT_PATH = DEFAULT_INPUTS_DIRECTORY / "students" / "student_01.c"
DEFAULT_DIAGNOSTIC_OUTPUT_DIRECTORY = Path(DATA_OUTPUTS_PATH) / "diagnostics"
DEFAULT_EVALUATION_JSON_SUFFIX = "_evaluation.json"
DEFAULT_EVALUATION_MARKDOWN_SUFFIX = "_evaluation.md"
DEFAULT_USAGE_JSON_NAME = "llm_usage_report.json"
DEFAULT_USAGE_MARKDOWN_NAME = "llm_usage_report.md"
REQUEST_WINDOW_SECONDS = 60
SECONDS_PER_MINUTE = 60
DISPLAY_SEPARATOR = "=" * 78
AGENT_LABELS = ("AGENT 1", "AGENT 2", "AGENT 3")
STATUS_SUCCESS = "success"
STATUS_ERROR = "error"
ERROR_RATE_LIMIT = "rate_limit"
ERROR_TRANSIENT = "transient"
ERROR_OTHER = "other"
UNKNOWN_VALUE = "unknown"


class UsageRecorder:
    """
    Captures raw LLM request events without modifying production agent code.
    """

    def __init__(self) -> None:
        self.events: list[dict] = []
        self.retry_wait_events: list[dict] = []

    def wrap_create(self, agent_label: str, create_callable):
        """
        Returns a wrapper around client.chat.completions.create that records
        timing, usage, and error details for every attempt.
        """

        def wrapped_create(*args, **kwargs):
            started_at = time.time()
            model_name = self._extract_model_name(kwargs)

            try:
                response = create_callable(*args, **kwargs)
                finished_at = time.time()
                usage = self._extract_usage(response)

                event = {
                    "status": STATUS_SUCCESS,
                    "agent": agent_label,
                    "model": model_name or self._extract_response_model(response),
                    "started_at_epoch": started_at,
                    "finished_at_epoch": finished_at,
                    "duration_seconds": round(finished_at - started_at, 6),
                    "prompt_tokens": usage["prompt_tokens"],
                    "completion_tokens": usage["completion_tokens"],
                    "total_tokens": usage["total_tokens"],
                    "usage_reported": usage["usage_reported"],
                    "error_type": None,
                    "error_message": None,
                }
                self.events.append(event)
                return response

            except Exception as exc:
                finished_at = time.time()
                error_text = str(exc)
                event = {
                    "status": STATUS_ERROR,
                    "agent": agent_label,
                    "model": model_name,
                    "started_at_epoch": started_at,
                    "finished_at_epoch": finished_at,
                    "duration_seconds": round(finished_at - started_at, 6),
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                    "usage_reported": False,
                    "error_type": self._classify_error(error_text),
                    "error_message": error_text,
                }
                self.events.append(event)
                raise

        return wrapped_create

    def wrap_sleep(self, original_sleep):
        """
        Records retry wait time injected by llm_client.call_llm_with_retry.
        """

        def wrapped_sleep(seconds: float):
            started_at = time.time()
            original_sleep(seconds)
            finished_at = time.time()
            self.retry_wait_events.append({
                "started_at_epoch": started_at,
                "finished_at_epoch": finished_at,
                "requested_wait_seconds": seconds,
                "actual_wait_seconds": round(finished_at - started_at, 6),
            })

        return wrapped_sleep

    @staticmethod
    def _extract_model_name(kwargs: dict) -> str:
        model_name = kwargs.get("model", UNKNOWN_VALUE)
        if isinstance(model_name, str) and model_name.strip():
            return model_name.strip()
        return UNKNOWN_VALUE

    @staticmethod
    def _extract_response_model(response) -> str:
        model_name = getattr(response, "model", UNKNOWN_VALUE)
        if isinstance(model_name, str) and model_name.strip():
            return model_name.strip()
        return UNKNOWN_VALUE

    @staticmethod
    def _extract_usage(response) -> dict:
        usage = getattr(response, "usage", None)

        prompt_tokens = UsageRecorder._extract_usage_value(usage, "prompt_tokens")
        completion_tokens = UsageRecorder._extract_usage_value(usage, "completion_tokens")
        total_tokens = UsageRecorder._extract_usage_value(usage, "total_tokens")

        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens

        usage_reported = total_tokens is not None or (
            prompt_tokens is not None and completion_tokens is not None
        )

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "usage_reported": usage_reported,
        }

    @staticmethod
    def _extract_usage_value(usage, key: str) -> int | None:
        if usage is None:
            return None

        value = getattr(usage, key, None)

        if value is None and isinstance(usage, dict):
            value = usage.get(key)

        if isinstance(value, int) and value >= 0:
            return value

        return None

    @staticmethod
    def _classify_error(error_text: str) -> str:
        if llm_client._is_rate_limit_error(error_text):
            return ERROR_RATE_LIMIT
        if llm_client._is_transient_error(error_text):
            return ERROR_TRANSIENT
        return ERROR_OTHER


def _read_env_bool(env_name: str, default: bool) -> bool:
    raw_value = os.getenv(env_name)
    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    return normalized in {"1", "true", "yes", "y", "on"}


def _validate_existing_non_empty_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")

    if not path.is_file():
        raise ValueError(f"{label} path is not a file: {path}")

    if path.stat().st_size <= 0:
        raise ValueError(f"{label} file is empty: {path}")


def _epoch_to_iso(epoch_seconds: float | None) -> str | None:
    if epoch_seconds is None:
        return None
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat()


def _build_empty_stats() -> dict:
    return {
        "attempted_requests": 0,
        "successful_requests": 0,
        "failed_attempts": 0,
        "rate_limit_errors": 0,
        "transient_errors": 0,
        "other_errors": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "requests_without_usage": 0,
        "usage_reported_for_all_successes": True,
        "total_request_time_seconds": 0.0,
        "peak_attempts_in_60s": 0,
        "peak_successes_in_60s": 0,
        "peak_total_tokens_in_60s": 0,
        "observed_requests_per_minute": 0.0,
        "observed_successes_per_minute": 0.0,
        "average_total_tokens_per_success": 0.0,
        "models_used": [],
    }


def _compute_window_peak(events: list[dict], value_key: str | None = None) -> int:
    if not events:
        return 0

    peak_value = 0

    for event in events:
        window_start = event["started_at_epoch"]
        window_end = window_start + REQUEST_WINDOW_SECONDS

        if value_key is None:
            current_value = sum(
                1
                for candidate in events
                if candidate["started_at_epoch"] is not None
                and window_start <= candidate["started_at_epoch"] < window_end
            )
        else:
            current_value = sum(
                int(candidate.get(value_key) or 0)
                for candidate in events
                if candidate["started_at_epoch"] is not None
                and window_start <= candidate["started_at_epoch"] < window_end
            )

        if current_value > peak_value:
            peak_value = current_value

    return peak_value


def _build_stats(events: list[dict], run_duration_seconds: float) -> dict:
    stats = _build_empty_stats()

    if run_duration_seconds < 0:
        run_duration_seconds = 0.0

    if not events:
        return stats

    successful_events = [event for event in events if event["status"] == STATUS_SUCCESS]
    failed_events = [event for event in events if event["status"] == STATUS_ERROR]

    stats["attempted_requests"] = len(events)
    stats["successful_requests"] = len(successful_events)
    stats["failed_attempts"] = len(failed_events)
    stats["rate_limit_errors"] = sum(
        1 for event in failed_events if event["error_type"] == ERROR_RATE_LIMIT
    )
    stats["transient_errors"] = sum(
        1 for event in failed_events if event["error_type"] == ERROR_TRANSIENT
    )
    stats["other_errors"] = sum(
        1 for event in failed_events if event["error_type"] == ERROR_OTHER
    )
    stats["prompt_tokens"] = sum(int(event.get("prompt_tokens") or 0) for event in successful_events)
    stats["completion_tokens"] = sum(int(event.get("completion_tokens") or 0) for event in successful_events)
    stats["total_tokens"] = sum(int(event.get("total_tokens") or 0) for event in successful_events)
    stats["requests_without_usage"] = sum(
        1 for event in successful_events if not event.get("usage_reported", False)
    )
    stats["usage_reported_for_all_successes"] = stats["requests_without_usage"] == 0
    stats["total_request_time_seconds"] = round(
        sum(float(event.get("duration_seconds") or 0.0) for event in events),
        6,
    )
    stats["peak_attempts_in_60s"] = _compute_window_peak(events)
    stats["peak_successes_in_60s"] = _compute_window_peak(successful_events)
    stats["peak_total_tokens_in_60s"] = _compute_window_peak(successful_events, "total_tokens")

    if run_duration_seconds > 0:
        stats["observed_requests_per_minute"] = round(
            stats["attempted_requests"] * SECONDS_PER_MINUTE / run_duration_seconds,
            3,
        )
        stats["observed_successes_per_minute"] = round(
            stats["successful_requests"] * SECONDS_PER_MINUTE / run_duration_seconds,
            3,
        )

    if stats["successful_requests"] > 0:
        stats["average_total_tokens_per_success"] = round(
            stats["total_tokens"] / stats["successful_requests"],
            3,
        )

    stats["models_used"] = sorted({
        str(event.get("model", UNKNOWN_VALUE))
        for event in events
        if event.get("model")
    })

    return stats


def _build_usage_report_json(
    assignment_id: str,
    student_id: str,
    force_regenerate: bool,
    input_paths: dict,
    evaluation_output: dict | None,
    recorder: UsageRecorder,
    run_started_at: float,
    run_finished_at: float,
) -> dict:
    run_duration_seconds = max(run_finished_at - run_started_at, 0.0)

    all_events = sorted(recorder.events, key=lambda event: event["started_at_epoch"])
    per_agent_stats = {}

    for agent_label in AGENT_LABELS:
        agent_events = [event for event in all_events if event["agent"] == agent_label]
        per_agent_stats[agent_label] = _build_stats(agent_events, run_duration_seconds)

    overall_stats = _build_stats(all_events, run_duration_seconds)

    retry_wait_summary = {
        "retry_wait_event_count": len(recorder.retry_wait_events),
        "requested_retry_wait_seconds": round(
            sum(
                float(event.get("requested_wait_seconds") or 0.0)
                for event in recorder.retry_wait_events
            ),
            6,
        ),
        "actual_retry_wait_seconds": round(
            sum(
                float(event.get("actual_wait_seconds") or 0.0)
                for event in recorder.retry_wait_events
            ),
            6,
        ),
    }

    phase_summary = {
        "agent1_phase1_completed": bool(
            per_agent_stats["AGENT 1"]["successful_requests"] > 0
            or rag_pipeline.assignment_exists(assignment_id)
        ),
        "agent2_phase2_completed": bool(evaluation_output and evaluation_output.get("json")),
        "agent3_phase3_completed": bool(evaluation_output and evaluation_output.get("markdown")),
        "evaluated_skills_count": len(
            (evaluation_output or {}).get("json", {}).get("skills", [])
        ),
    }

    request_events = []
    for event in all_events:
        request_events.append({
            "status": event["status"],
            "agent": event["agent"],
            "model": event["model"],
            "started_at_utc": _epoch_to_iso(event["started_at_epoch"]),
            "finished_at_utc": _epoch_to_iso(event["finished_at_epoch"]),
            "duration_seconds": event["duration_seconds"],
            "prompt_tokens": event["prompt_tokens"],
            "completion_tokens": event["completion_tokens"],
            "total_tokens": event["total_tokens"],
            "usage_reported": event["usage_reported"],
            "error_type": event["error_type"],
            "error_message": event["error_message"],
        })

    return {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "assignment_id": assignment_id,
            "student_id": student_id,
            "provider": LLM_PROVIDER,
            "force_regenerate": force_regenerate,
            "request_window_seconds": REQUEST_WINDOW_SECONDS,
            "run_started_at_utc": _epoch_to_iso(run_started_at),
            "run_finished_at_utc": _epoch_to_iso(run_finished_at),
            "run_duration_seconds": round(run_duration_seconds, 6),
            "input_files": input_paths,
            "models": {
                "agent1": get_agent1().model,
                "agent2": get_agent2().model,
                "agent3": get_agent3().model,
            },
        },
        "phase_summary": phase_summary,
        "usage_summary": {
            "overall": overall_stats,
            "per_agent": per_agent_stats,
            "retry_wait": retry_wait_summary,
        },
        "request_events": request_events,
        "retry_wait_events": [
            {
                "started_at_utc": _epoch_to_iso(event["started_at_epoch"]),
                "finished_at_utc": _epoch_to_iso(event["finished_at_epoch"]),
                "requested_wait_seconds": event["requested_wait_seconds"],
                "actual_wait_seconds": event["actual_wait_seconds"],
            }
            for event in recorder.retry_wait_events
        ],
        "notes": {
            "token_accuracy_rule": (
                "Token totals are exact only for successful calls where the provider "
                "returned usage.prompt_tokens and usage.completion_tokens."
            ),
            "rpm_accuracy_rule": (
                "Peak rolling 60-second request counts reflect observed demand from this "
                "specific full pipeline run."
            ),
        },
    }


def _build_usage_markdown_report(report_json: dict) -> str:
    metadata = report_json["metadata"]
    overall = report_json["usage_summary"]["overall"]
    per_agent = report_json["usage_summary"]["per_agent"]
    retry_wait = report_json["usage_summary"]["retry_wait"]
    phase_summary = report_json["phase_summary"]

    lines = []
    lines.append("# LLM Usage Diagnostic Report")
    lines.append("")
    lines.append(f"- Assignment ID: `{metadata['assignment_id']}`")
    lines.append(f"- Student ID: `{metadata['student_id']}`")
    lines.append(f"- Provider: `{metadata['provider']}`")
    lines.append(f"- Run duration: `{metadata['run_duration_seconds']}` seconds")
    lines.append(f"- Force regenerate: `{metadata['force_regenerate']}`")
    lines.append("")
    lines.append("## Phase Summary")
    lines.append("")
    lines.append(f"- Agent 1 completed: `{phase_summary['agent1_phase1_completed']}`")
    lines.append(f"- Agent 2 completed: `{phase_summary['agent2_phase2_completed']}`")
    lines.append(f"- Agent 3 completed: `{phase_summary['agent3_phase3_completed']}`")
    lines.append(f"- Evaluated skills: `{phase_summary['evaluated_skills_count']}`")
    lines.append("")
    lines.append("## Overall Usage")
    lines.append("")
    lines.append(f"- Attempted requests: `{overall['attempted_requests']}`")
    lines.append(f"- Successful requests: `{overall['successful_requests']}`")
    lines.append(f"- Failed attempts: `{overall['failed_attempts']}`")
    lines.append(f"- Rate limit errors: `{overall['rate_limit_errors']}`")
    lines.append(f"- Transient errors: `{overall['transient_errors']}`")
    lines.append(f"- Other errors: `{overall['other_errors']}`")
    lines.append(f"- Prompt tokens: `{overall['prompt_tokens']}`")
    lines.append(f"- Completion tokens: `{overall['completion_tokens']}`")
    lines.append(f"- Total tokens: `{overall['total_tokens']}`")
    lines.append(f"- Successful requests without usage: `{overall['requests_without_usage']}`")
    lines.append(f"- Peak attempts in 60s: `{overall['peak_attempts_in_60s']}`")
    lines.append(f"- Peak successes in 60s: `{overall['peak_successes_in_60s']}`")
    lines.append(f"- Peak total tokens in 60s: `{overall['peak_total_tokens_in_60s']}`")
    lines.append(
        f"- Observed attempted RPM: `{overall['observed_requests_per_minute']}`"
    )
    lines.append(
        f"- Observed successful RPM: `{overall['observed_successes_per_minute']}`"
    )
    lines.append(
        f"- Average total tokens per successful call: `{overall['average_total_tokens_per_success']}`"
    )
    lines.append("")
    lines.append("## Retry Wait")
    lines.append("")
    lines.append(
        f"- Retry wait events: `{retry_wait['retry_wait_event_count']}`"
    )
    lines.append(
        f"- Requested retry wait seconds: `{retry_wait['requested_retry_wait_seconds']}`"
    )
    lines.append(
        f"- Actual retry wait seconds: `{retry_wait['actual_retry_wait_seconds']}`"
    )
    lines.append("")
    lines.append("## Per-Agent Breakdown")
    lines.append("")
    lines.append("| Agent | Attempts | Successes | Rate Limit Errors | Prompt Tokens | Completion Tokens | Total Tokens | Peak 60s Requests | Peak 60s Tokens |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    for agent_label in AGENT_LABELS:
        stats = per_agent[agent_label]
        lines.append(
            f"| {agent_label} | {stats['attempted_requests']} | {stats['successful_requests']} | "
            f"{stats['rate_limit_errors']} | {stats['prompt_tokens']} | "
            f"{stats['completion_tokens']} | {stats['total_tokens']} | "
            f"{stats['peak_attempts_in_60s']} | {stats['peak_total_tokens_in_60s']} |"
        )

    return "\n".join(lines)


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _save_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def _print_summary(report_json: dict, usage_json_path: Path, usage_md_path: Path) -> None:
    overall = report_json["usage_summary"]["overall"]
    per_agent = report_json["usage_summary"]["per_agent"]

    print()
    print(DISPLAY_SEPARATOR)
    print("  AutoEval-C Full Pipeline LLM Usage Diagnostic")
    print(DISPLAY_SEPARATOR)
    print()
    print(f"  Assignment ID: {report_json['metadata']['assignment_id']}")
    print(f"  Student ID:    {report_json['metadata']['student_id']}")
    print(f"  Provider:      {report_json['metadata']['provider']}")
    print(f"  Duration:      {report_json['metadata']['run_duration_seconds']:.3f}s")
    print()
    print("  OVERALL")
    print(f"    Attempted requests:      {overall['attempted_requests']}")
    print(f"    Successful requests:     {overall['successful_requests']}")
    print(f"    Rate limit errors:       {overall['rate_limit_errors']}")
    print(f"    Prompt tokens:           {overall['prompt_tokens']}")
    print(f"    Completion tokens:       {overall['completion_tokens']}")
    print(f"    Total tokens:            {overall['total_tokens']}")
    print(f"    Peak attempts in 60s:    {overall['peak_attempts_in_60s']}")
    print(f"    Peak total tokens in 60s:{overall['peak_total_tokens_in_60s']}")
    print()
    print("  PER AGENT")

    for agent_label in AGENT_LABELS:
        stats = per_agent[agent_label]
        print(
            f"    {agent_label:<7} attempts={stats['attempted_requests']} "
            f"successes={stats['successful_requests']} "
            f"tokens={stats['total_tokens']} "
            f"peak60s_req={stats['peak_attempts_in_60s']}"
        )

    print()
    print(f"  JSON report:     {usage_json_path}")
    print(f"  Markdown report: {usage_md_path}")
    print()
    print(DISPLAY_SEPARATOR)
    print()


def run_full_pipeline_usage_diagnostic() -> int:
    assignment_id = os.getenv("AUTOEVAL_DIAGNOSTIC_ASSIGNMENT_ID", DEFAULT_ASSIGNMENT_ID).strip()
    student_id = os.getenv("AUTOEVAL_DIAGNOSTIC_STUDENT_ID", DEFAULT_STUDENT_ID).strip()
    force_regenerate = _read_env_bool(
        "AUTOEVAL_DIAGNOSTIC_FORCE_REGENERATE",
        DEFAULT_FORCE_REGENERATE,
    )

    inputs_directory = Path(
        os.getenv("AUTOEVAL_DIAGNOSTIC_INPUT_DIR", str(DEFAULT_INPUTS_DIRECTORY))
    )
    instructions_path = inputs_directory / "instructions.md"
    starter_path = inputs_directory / "starter_code.c"
    teacher_path = inputs_directory / "teacher_correction_code.c"
    student_path = Path(
        os.getenv("AUTOEVAL_DIAGNOSTIC_STUDENT_PATH", str(DEFAULT_STUDENT_PATH))
    )

    try:
        if not assignment_id:
            raise ValueError("AUTOEVAL_DIAGNOSTIC_ASSIGNMENT_ID cannot be blank.")

        if not student_id:
            raise ValueError("AUTOEVAL_DIAGNOSTIC_STUDENT_ID cannot be blank.")

        _validate_existing_non_empty_file(instructions_path, "instructions")
        _validate_existing_non_empty_file(starter_path, "starter_code")
        _validate_existing_non_empty_file(teacher_path, "teacher_code")
        _validate_existing_non_empty_file(student_path, "student_code")

        input_paths = {
            "instructions": str(instructions_path),
            "starter_code": str(starter_path),
            "teacher_code": str(teacher_path),
            "student_code": str(student_path),
        }

        engine_logger.info(
            "DIAGNOSTIC: Starting full pipeline LLM usage measurement."
        )

        agent1 = get_agent1()
        agent2 = get_agent2()
        agent3 = get_agent3()
        recorder = UsageRecorder()

        evaluation_output = None
        run_started_at = time.time()

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    agent1.client.chat.completions,
                    "create",
                    side_effect=recorder.wrap_create(
                        "AGENT 1",
                        agent1.client.chat.completions.create,
                    ),
                )
            )
            stack.enter_context(
                patch.object(
                    agent2.client.chat.completions,
                    "create",
                    side_effect=recorder.wrap_create(
                        "AGENT 2",
                        agent2.client.chat.completions.create,
                    ),
                )
            )
            stack.enter_context(
                patch.object(
                    agent3.client.chat.completions,
                    "create",
                    side_effect=recorder.wrap_create(
                        "AGENT 3",
                        agent3.client.chat.completions.create,
                    ),
                )
            )
            stack.enter_context(
                patch.object(
                    llm_client.time,
                    "sleep",
                    side_effect=recorder.wrap_sleep(llm_client.time.sleep),
                )
            )

            phase1_success = agent1.run(
                assignment_id=assignment_id,
                instructions_path=str(instructions_path),
                starter_path=str(starter_path),
                teacher_path=str(teacher_path),
                force_regenerate=force_regenerate,
            )

            if not phase1_success:
                raise RuntimeError("Phase 1 failed. No usage report generated.")

            evaluation_result = agent2.run(
                assignment_id=assignment_id,
                student_id=student_id,
                student_path=str(student_path),
            )

            if not evaluation_result:
                raise RuntimeError("Phase 2 failed. No usage report generated.")

            evaluation_output = agent3.run(evaluation_result)

            if not evaluation_output:
                raise RuntimeError("Phase 3 failed. No usage report generated.")

        run_finished_at = time.time()

        output_directory = DEFAULT_DIAGNOSTIC_OUTPUT_DIRECTORY
        evaluation_json_path = output_directory / f"{student_id}{DEFAULT_EVALUATION_JSON_SUFFIX}"
        evaluation_md_path = output_directory / f"{student_id}{DEFAULT_EVALUATION_MARKDOWN_SUFFIX}"
        usage_json_path = output_directory / DEFAULT_USAGE_JSON_NAME
        usage_md_path = output_directory / DEFAULT_USAGE_MARKDOWN_NAME

        _save_json(evaluation_json_path, evaluation_output["json"])
        _save_text(evaluation_md_path, evaluation_output["markdown"])

        usage_report_json = _build_usage_report_json(
            assignment_id=assignment_id,
            student_id=student_id,
            force_regenerate=force_regenerate,
            input_paths=input_paths,
            evaluation_output=evaluation_output,
            recorder=recorder,
            run_started_at=run_started_at,
            run_finished_at=run_finished_at,
        )
        usage_report_markdown = _build_usage_markdown_report(usage_report_json)

        _save_json(usage_json_path, usage_report_json)
        _save_text(usage_md_path, usage_report_markdown)

        engine_logger.info(
            "DIAGNOSTIC: Full pipeline LLM usage measurement complete."
        )
        _print_summary(usage_report_json, usage_json_path, usage_md_path)
        return 0

    except Exception as exc:
        engine_logger.error(f"DIAGNOSTIC: Failed to measure full pipeline usage: {exc}")
        print()
        print(DISPLAY_SEPARATOR)
        print("  AutoEval-C Full Pipeline LLM Usage Diagnostic")
        print(DISPLAY_SEPARATOR)
        print()
        print(f"  ERROR: {exc}")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(run_full_pipeline_usage_diagnostic())
