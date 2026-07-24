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
