import os
import re
import subprocess


def get_display_size() -> tuple[int, int]:
    session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()

    match session_type:
        case "wayland":
            output = subprocess.check_output(["wlr-randr"], text=True)
            match = re.search(
                r"^\s*(\d+)x(\d+)\s+px.*\bcurrent\b", output, re.MULTILINE
            )
        case "x11":
            output = subprocess.check_output(["xrandr", "--current"], text=True)
            match = re.search(r" connected.*?(\d+)x(\d+)\+", output)
        case _:
            raise RuntimeError(
                f"Could not determine display type; XDG_SESSION_TYPE must be 'wayland' or 'x11'. Got {session_type}"
            )

    if match is None:
        raise RuntimeError(
            f"Could not parse display resolution from command output:\n{output}"
        )

    return int(match.group(1)), int(match.group(2))
