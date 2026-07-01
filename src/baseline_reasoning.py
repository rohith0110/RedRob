def _clean(value) -> str:
    # pandas read_csv turns empty cells into float NaN; treat NaN/"nan" as empty.
    if value is None or isinstance(value, float):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def reasoning_for(row: dict) -> str:
    positive = _clean(row.get("top_positive_evidence")) or "limited relevant evidence found"
    negative = _clean(row.get("top_negative_evidence"))
    # Only upper-case the first character; capitalize() would lowercase "Python", "AI", etc.
    positive = positive[0].upper() + positive[1:]
    base = f"{positive}, which is relevant to a Senior AI Engineer ranking role."
    if negative:
        return f"{base} Weakness: {negative}."
    return base
