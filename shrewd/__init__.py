"""Turn LLM judgments into a small, fast, local text model for one fixed task."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shrewd._autotune import autotune as autotune
    from shrewd.decide import Choice as Choice
    from shrewd.decide import Noul as Noul
    from shrewd.decide import Score as Score
    from shrewd.decisions import Decisions as Decisions
    from shrewd.evaluate import DistillResult as DistillResult
    from shrewd.evaluate import Finding as Finding
    from shrewd.project import Project as Project
    from shrewd.students import load as load

__version__ = "0.1.0"

__all__ = [
    "Choice", "Decisions", "DistillResult", "Finding", "Noul", "Project", "Score",
    "autotune", "load",
]


def __dir__():
    return sorted(set(globals()) | set(__all__))


class _needs_teacher_extra:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if isinstance(exc, ImportError) and exc.name in ("litellm", "gepa"):
            raise ImportError(
                f"training needs {exc.name}, which is not installed: "
                'pip install "shrewd[teacher]" (inference-only installs can skip it)'
            ) from None


# Lazy so that `from shrewd import load` works in inference environments
# where litellm/gepa are not installed.
def __getattr__(name):
    if name == "Project":
        with _needs_teacher_extra():
            from shrewd.project import Project

        return Project
    if name == "autotune":
        # the submodule is named _autotune so this attribute never collides with it
        # (a same-named submodule would shadow the function on from-imports)
        with _needs_teacher_extra():
            from shrewd._autotune import autotune

        return autotune
    if name == "load":
        from shrewd.students import load

        return load
    if name in ("Choice", "Noul", "Score"):
        from shrewd import decide

        return getattr(decide, name)
    if name == "Decisions":
        with _needs_teacher_extra():
            from shrewd.decisions import Decisions

        return Decisions
    if name in ("DistillResult", "Finding"):
        from shrewd import evaluate

        return getattr(evaluate, name)
    raise AttributeError(f"module 'shrewd' has no attribute {name!r}")
