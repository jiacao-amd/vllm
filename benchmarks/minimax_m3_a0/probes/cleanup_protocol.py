"""Untimed cleanup orchestration; no device imports or implicit resource releases."""

from collections.abc import Callable
import traceback

CleanupAction = tuple[str, Callable[[], object]]


def run_cleanup(
    actions: list[CleanupAction], primary_error: BaseException | None,
) -> None:
    """Attempt every action; annotate a primary error or raise all cleanup failures.

    The caller must propagate an existing primary error, normally by invoking this
    helper in a finally block. Actions retain responsibility for ownership and
    quiescence checks; this helper never force-frees resources or waives checks.
    """
    failures: list[tuple[str, BaseException]] = []
    for name, action in actions:
        try:
            action()
        except BaseException as error:
            failures.append((name, error))
    if not failures:
        return
    if primary_error is not None:
        for name, error in failures:
            detail = "".join(traceback.format_exception_only(error)).rstrip()
            BaseException.add_note(primary_error, f"Cleanup action {name!r} failed:\n{detail}")
        return
    for name, error in failures:
        BaseException.add_note(error, f"Cleanup action: {name!r}")
    # The constructor returns ExceptionGroup when all members are Exceptions.
    raise BaseExceptionGroup("Cleanup actions failed", [error for _, error in failures])
