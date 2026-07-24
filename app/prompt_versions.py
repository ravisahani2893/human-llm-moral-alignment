import hashlib
from pathlib import Path

from app.prompts import build_prompt as _current_build_prompt

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _load_versioned_builder(filename: str):
    """
    app/prompts/v1.txt and v2.txt are saved as .txt but are actually full
    Python source defining a build_prompt function — load and exec them in
    an isolated namespace to get the function object without touching the
    current live prompt in app/prompts.py.
    """
    source = (PROMPTS_DIR / filename).read_text()
    namespace: dict = {}
    exec(compile(source, filename, "exec"), namespace)
    return namespace["build_prompt"]


_VERSION_BUILDERS = {
    "v1": _load_versioned_builder("v1.txt"),
    "v2": _load_versioned_builder("v2.txt"),
    "current": _current_build_prompt,
}


def get_prompt_builder(version: str):
    if version not in _VERSION_BUILDERS:
        raise ValueError(f"Unknown prompt version {version!r}. Expected one of {list(_VERSION_BUILDERS)}")
    return _VERSION_BUILDERS[version]


def available_versions() -> list[str]:
    return list(_VERSION_BUILDERS.keys())


def compute_prompt_hash(version: str) -> str:
    """
    A short fingerprint of the actual rendered prompt template for a
    version — not just the version label — so that even if a version's
    underlying template changes later, past results can be checked against
    the exact template text that produced them, not just a name that may
    now mean something different.
    """
    builder = get_prompt_builder(version)
    template_text = builder("{{SCENARIO_PLACEHOLDER}}")
    return hashlib.sha256(template_text.encode()).hexdigest()[:12]
