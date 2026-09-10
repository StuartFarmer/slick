"""Small direct UI used by the coding agent in headless mode."""

import json
import re


def safe_text(value):
    """Remove terminal control sequences while preserving ordinary text."""
    text = str(value)
    text = re.sub(r"(?:\x1b\]|\x9d).*?(?:\x07|\x1b\\|\x9c|$)", "", text, flags=re.S)
    text = re.sub(
        r"(?:\x1b[PX^_]|\x90|\x98|\x9e|\x9f).*?(?:\x1b\\|\x9c|$)",
        "",
        text,
        flags=re.S,
    )
    text = re.sub(r"(?:\x1b\[|\x9b)[0-?]*[ -/]*[@-~]", "", text)
    text = re.sub(r"\x1b[ -/]*[@-~]", "", text)
    return re.sub(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]", "", text)


def write(text):
    print(safe_text(text))


def content_text(content):
    if isinstance(content, dict | list):
        return json.dumps(content, ensure_ascii=False, indent=2)
    return str(content)


def check_text(check, baseline=False):
    command = check["command"]
    state = "timed out" if command.get("timed_out") else f"exit {command.get('exit_code')}"
    line = f"{'Baseline' if baseline else 'Check'} {check['name']}: {state}"
    details = [value for key in ("stdout", "stderr") if (value := command.get(key))]
    if command.get("truncated"):
        details.append("[output truncated]")
    output = "\n".join(details)
    return f"{line}\n{output}" if output else line


def completion_text(result):
    summary = (
        f"{result['status'].capitalize()} · {result['turns']} turns · "
        f"{result['tool_calls']} tools · {result['repairs']} repairs"
    )
    return f"{summary}\n{result['answer']}" if result.get("answer") else summary


class ConsoleUI:
    """Readable, inert output with conservative command decisions."""

    def __init__(self, write=write):
        self.write = write

    def _line(self, text):
        self.write(safe_text(text))

    def user(self, text):
        self._line(f"You: {text}")

    def assistant(self, text):
        self._line(f"Agent: {text}")

    def status(self, text):
        self._line(f"Status: {text}")

    def tool(self, name, content, is_error=False):
        label = "Tool error" if is_error else "Tool"
        self._line(f"{label} ({name}): {content_text(content)}")

    def checks(self, results, baseline=False):
        for check in results:
            self._line(check_text(check, baseline))

    def completed(self, result):
        self._line(completion_text(result))

    async def approve(self, request):
        self._line(
            "Denied command: "
            f"argv={json.dumps(request['argv'], ensure_ascii=False)} "
            f"cwd={request['cwd']} timeout={request['timeout']}s"
        )
        return "deny"
