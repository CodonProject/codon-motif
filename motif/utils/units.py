from typing import Any

#: Decimal units (not KiB/MiB): aligned with the existing `params / 1e6` spelling.
UNITS = ('', 'K', 'M', 'B', 'T')


def format_count(count: Any, decimals: int = 3, space: bool = True) -> str:
    '''
    Format a number as a short unit-suffixed string such as `'242.700 M'`.

    * values smaller than 1000 in absolute value are printed as plain integers
      (so the less readable `'0.512 K'` never appears);
    * past T (1e12) the unit stops growing.

    Args:
        count (Any): Number to format (int or float).
        decimals (int): Number of decimal places (3 by default, matching the
            `:.3f M` spelling used by `describe()`).
        space (bool): Whether to keep a space between the value and the unit
            (`'1.387 M'` vs `'1.387M'`).

    Returns:
        str: The formatted string.
    '''
    value = float(count)
    if abs(value) < 1000:
        return str(int(round(value)))

    unit = 0
    while abs(value) >= 1000 and unit < len(UNITS) - 1:
        value /= 1000.0
        unit += 1
    return f'{value:.{decimals}f}{" " if space else ""}{UNITS[unit]}'


def count_params(module: Any, trainable_only: bool = False) -> int:
    '''
    Parameter count of `module` as an int.

    Args:
        module (Any): Any object exposing `parameters()`.
        trainable_only (bool): Count only the parameters with
            `requires_grad=True`.

    Returns:
        int: The total number of parameters.
    '''
    if trainable_only:
        return int(sum(p.numel() for p in module.parameters() if p.requires_grad))
    return int(sum(p.numel() for p in module.parameters()))
