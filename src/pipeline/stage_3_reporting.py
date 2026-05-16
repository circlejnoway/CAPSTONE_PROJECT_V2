"""
stage_3_reporting.py
--------------------
Reporting utilities for nutrition summaries and export formats.
"""

from typing import Dict, Any, List


def build_nutrition_report(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Format nutrition data into a report dictionary."""
    return {
        "items": items,
        "total_calories": sum(item.get("calories", 0) for item in items),
        "total_protein": sum(item.get("protein", 0) for item in items),
    }
