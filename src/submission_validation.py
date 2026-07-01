from pathlib import Path

from validate_submission import validate_submission as organizer_validate_submission


def validate_submission(path: str | Path) -> list[str]:
    path = Path(path)
    errors = []
    if not path.exists():
        errors.append(f"submission does not exist: {path}")
    if path.suffix.lower() != ".csv":
        errors.append("submission path must end with .csv")
    if errors:
        return errors
    return organizer_validate_submission(path)


def require_valid_submission(path: str | Path) -> None:
    errors = validate_submission(path)
    if errors:
        raise ValueError("submission validation failed: " + "; ".join(errors))
