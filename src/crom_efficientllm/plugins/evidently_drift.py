from __future__ import annotations
from typing import List

try:
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report
    import pandas as pd
except Exception as e:  # pragma: no cover
    raise RuntimeError("evidently not installed. Install extras: pip install .[plugins]") from e

def drift_report(ref: List[List[float]], cur: List[List[float]]):
    ref_df = pd.DataFrame(ref)
    cur_df = pd.DataFrame(cur)
    rep = Report(metrics=[DataDriftPreset()])
    rep.run(reference_data=ref_df, current_data=cur_df)
    return rep
