import re
from datetime import datetime

INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
NAME_MAX_LENGTH = 20


def sanitize_name_component(raw: str) -> str:
    """Strip characters invalid in Windows file/folder names and truncate to NAME_MAX_LENGTH."""
    return INVALID_FILENAME_CHARS.sub("", raw)[:NAME_MAX_LENGTH]


def build_timestamped_name(prefix: str | None, default_prefix: str, when: datetime | None = None) -> str:
    """Build '<prefix><DD-MM-YYYY>_<HH-MM>' (no extension).

    If prefix is falsy, uses default_prefix verbatim. Otherwise prefix is
    sanitized and truncated to NAME_MAX_LENGTH first.
    """
    when = when or datetime.now()
    timestamp = when.strftime("%d-%m-%Y_%H-%M")
    clean_prefix = sanitize_name_component(prefix) if prefix else default_prefix
    return f"{clean_prefix}{timestamp}"
