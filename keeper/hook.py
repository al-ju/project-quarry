"""`keeper hook`: answer Claude Code style hooks for SPARCO (Claude Code) and SPARC (dsh).

The host (Claude Code, or dsh through its dsh-hooks-claude-code bridge) runs a hook command
at each checkpoint and sends one JSON object on stdin. This module turns that into a short
conversation, runs the subject's checks, and answers in the hook format:

- exit 0, no output: allow.
- exit 0 with JSON ``hookSpecificOutput``: "ask" (Al decides) or extra context for a helper.
- exit 2 with the alert on stderr: block. The host shows the reason to the model.

Fail closed: any problem inside the Keeper (bad input, missing subject, broken rule) is a
block. A host treats any exit other than 0 or 2 as "not blocking", so ``keeper-hook.sh``
wraps this command and turns a crash into exit 2 as well.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from keeper.guard import CheckReport, check, write_log
from keeper.layout import load_subject, require_root

ALLOW, BLOCK = 0, 2
_MAX_USER_MESSAGES = 20
_MAX_TRANSCRIPT_BYTES = 20_000_000

EVENTS = {
    "pre-tool": "PreToolUse",
    "post-tool": "PostToolUse",
    "stop": "Stop",
    "subagent-start": "SubagentStart",
}


@dataclass
class HookAnswer:
    exit_code: int
    stdout: str = ""
    stderr: str = ""


def run_hook(event: str, root: Path, subject: str, payload_text: str) -> HookAnswer:
    """Answer one hook call. Never raises: every failure becomes a block."""

    try:
        root = require_root(root)
        load_subject(root, subject)
        payload = json.loads(payload_text) if payload_text.strip() else {}
        if not isinstance(payload, dict):
            raise ValueError("the hook input is not a JSON object")
        if event == "pre-tool":
            return _pre_tool(root, subject, payload)
        if event == "post-tool":
            return _post_tool(root, subject, payload)
        if event == "stop":
            return _stop(root, subject, payload)
        if event == "subagent-start":
            return _subagent_start(root, subject)
        raise ValueError(f"unknown hook event '{event}'")
    except Exception as error:  # noqa: BLE001 - fail closed
        return HookAnswer(
            BLOCK,
            stderr=(
                f"ALERT: The Keeper could not check this step ({EVENTS.get(event, event)}), "
                "so it is blocked.\n\n"
                "What happened, step by step:\n"
                f"1. Problem: {type(error).__name__}: {error}\n"
                "2. Stopped. Nothing was retried or worked around. Waiting for Al.\n\n"
                "In simple terms: the guard couldn't do its check, and when it can't be sure, "
                "it says stop."
            ),
        )


# --- checkpoints --------------------------------------------------------------------------


def _pre_tool(root: Path, subject: str, payload: dict[str, Any]) -> HookAnswer:
    call = _tool_call(payload)
    messages = [*_user_messages(payload), call]
    report = check(root, subject, {"messages": messages})
    _log(root, report, "PreToolUse")
    if report.verdict == "PASS":
        return HookAnswer(ALLOW)
    if report.stop_kind == "rule" and all(item.action == "ask" for item in report.broken):
        reason = report.alert_text().replace("ALERT: Keeper stopped", "Keeper asks Al about", 1)
        output = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "ask",
                "permissionDecisionReason": reason,
            }
        }
        return HookAnswer(ALLOW, stdout=json.dumps(output, ensure_ascii=False))
    return HookAnswer(BLOCK, stderr=report.alert_text())


def _post_tool(root: Path, subject: str, payload: dict[str, Any]) -> HookAnswer:
    call = _tool_call(payload)
    result = {
        "role": "tool",
        "tool_call_id": call["tool_calls"][0]["id"],
        "content": _text(payload.get("tool_response")),
    }
    _record_step(root, subject, payload)
    report = check(root, subject, {"messages": [*_user_messages(payload), call, result]})
    _log(root, report, "PostToolUse")
    if report.verdict == "PASS":
        return HookAnswer(ALLOW)
    return HookAnswer(BLOCK, stderr=report.alert_text())


def _stop(root: Path, subject: str, payload: dict[str, Any]) -> HookAnswer:
    reply = _text(payload.get("last_assistant_message")) or _last_reply(payload)
    if not reply:
        # dsh's bridge sends no reply text yet. Nothing to check is not a broken rule, and
        # blocking here would only force another turn, so it is logged and allowed.
        _log_line(root, subject, "Stop · SKIPPED (the host sent no reply text)")
        return HookAnswer(ALLOW)
    if payload.get("stop_hook_active") is True:
        return HookAnswer(ALLOW)
    marker = _stop_marker(root, payload, reply)
    if marker.exists():
        # Already sent back once for this exact reply: don't loop.
        _log_line(root, subject, "Stop · ALLOWED (already sent back once)")
        return HookAnswer(ALLOW)
    messages = [*_user_messages(payload), {"role": "assistant", "content": reply}]
    report = check(root, subject, {"messages": messages})
    _log(root, report, "Stop")
    if report.verdict == "PASS":
        return HookAnswer(ALLOW)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("sent back once\n", encoding="utf-8")
    return HookAnswer(
        BLOCK,
        stderr=report.alert_text() + "\n\nFix the reply so it follows these rules, then finish.",
    )


def _subagent_start(root: Path, subject: str) -> HookAnswer:
    rules = (load_subject(root, subject).path / "RULES.md").read_text(encoding="utf-8")
    output = {
        "hookSpecificOutput": {
            "hookEventName": "SubagentStart",
            "additionalContext": f'Rules from the Keeper for subject "{subject}":\n\n{rules}',
        }
    }
    _log_line(root, subject, "SubagentStart · RULES.md handed to the helper")
    return HookAnswer(ALLOW, stdout=json.dumps(output, ensure_ascii=False))


# --- payload helpers ----------------------------------------------------------------------


def _tool_call(payload: dict[str, Any]) -> dict[str, Any]:
    name = payload.get("tool_name")
    if not isinstance(name, str) or not name:
        raise ValueError("the hook input has no tool_name")
    arguments = payload.get("tool_input", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            arguments = {"input": arguments}
    if not isinstance(arguments, dict):
        arguments = {"input": arguments}
    call_id = str(payload.get("tool_use_id") or "keeper-call")
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
            }
        ],
    }


def _user_messages(payload: dict[str, Any]) -> list[dict[str, str]]:
    """Al's recent messages from the transcript, so "approved" earlier in the chat counts."""

    messages = []
    for entry in _transcript(payload):
        if entry.get("type") != "user":
            continue
        message = entry.get("message")
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        text = _text(message.get("content"))
        if text:
            messages.append({"role": "user", "content": text})
    return messages[-_MAX_USER_MESSAGES:] or [
        {"role": "user", "content": "(no user text available)"}
    ]


def _last_reply(payload: dict[str, Any]) -> str:
    for entry in reversed(_transcript(payload)):
        message = entry.get("message")
        if entry.get("type") == "assistant" and isinstance(message, dict):
            text = _text(message.get("content"))
            if text:
                return text
    return ""


def _transcript(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Read a Claude Code style JSONL transcript. Missing or unreadable means none."""

    path_text = payload.get("transcript_path")
    if not isinstance(path_text, str) or not path_text:
        return []
    path = Path(path_text).expanduser()
    try:
        if not path.is_file() or path.stat().st_size > _MAX_TRANSCRIPT_BYTES:
            return []
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    entries = []
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    return entries


def _text(content: Any) -> str:
    """Text only: tool-result blocks inside a user turn are not Al's words."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def _record_step(root: Path, subject: str, payload: dict[str, Any]) -> None:
    from keeper import logbook

    job_id = logbook.active_job(root, subject)
    if job_id is None:
        return
    try:
        logbook.record_step(
            root,
            subject,
            job_id,
            tool=str(payload.get("tool_name")),
            input=json.dumps(payload.get("tool_input", {}), ensure_ascii=False, default=str),
            output=_text(payload.get("tool_response")),
        )
    except logbook.LogbookError:
        return  # the active job was closed by hand; recording is best-effort


def _stop_marker(root: Path, payload: dict[str, Any], reply: str) -> Path:
    session = str(payload.get("session_id") or "no-session")
    digest = hashlib.sha256(f"{session}\n{reply}".encode()).hexdigest()[:24]
    return root / "log" / ".sent-back" / digest


def _log(root: Path, report: CheckReport, source: str) -> None:
    with contextlib.suppress(OSError):
        write_log(root, report, source=source)


def _log_line(root: Path, subject: str, text: str) -> None:
    from datetime import UTC, datetime

    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    try:
        path = root / "log" / f"{stamp[:10]}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"- {stamp} · {subject} · {text}\n")
    except OSError:
        pass
