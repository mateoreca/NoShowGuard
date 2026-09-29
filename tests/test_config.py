from noshow_guard import __version__
from noshow_guard.config import PATHS, SEED


def test_version_is_defined() -> None:
    assert __version__ == "0.1.0"


def test_paths_are_inside_project_root() -> None:
    assert PATHS.data_raw.is_relative_to(PATHS.root)
    assert PATHS.raw_csv.parent == PATHS.data_raw
    assert (PATHS.root / "pyproject.toml").exists()


def test_seed_is_fixed() -> None:
    assert SEED == 42
