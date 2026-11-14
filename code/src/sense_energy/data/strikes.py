"""Doctor strike days in England inside the demand record.

National walkouts called by the British Medical Association: fifteen rounds of junior doctors
(resident doctors from September 2024) and four rounds of consultants. A round starts at 07:00 on
its first day and ends at 07:00 on the day after its last full day, so the strike days listed
here are the calendar days on which the walkout covered the working day. On these days elective
operations and outpatient appointments were cancelled in large numbers, which makes them a shock
to hospital activity that the weather and the calendar do not explain.

Sources: BMA announcements of each round; NHS England industrial-action notices (2023 rounds);
the government's "Resident doctor strikes: everything you need to know" pages (2025 and 2026
rounds; the April 2026 walkout is counted there as the fifteenth round, which this list
reproduces). One of those pages prints the autumn 2025 round as 14-19 October; the BMA
announcement and the December 2025 page give 14-19 November, a Friday start like the other rounds.

Not included: the Royal College of Nursing and ambulance strikes of December 2022 to May 2023,
which were held at lists of trusts and services that differ by day.
"""

from __future__ import annotations

import pandas as pd

#: (first strike day, last strike day, staff group)
ROUNDS = (
    ("2023-03-13", "2023-03-15", "junior doctors"),
    ("2023-04-11", "2023-04-14", "junior doctors"),
    ("2023-06-14", "2023-06-16", "junior doctors"),
    ("2023-07-13", "2023-07-17", "junior doctors"),
    ("2023-07-20", "2023-07-21", "consultants"),
    ("2023-08-11", "2023-08-14", "junior doctors"),
    ("2023-08-24", "2023-08-25", "consultants"),
    ("2023-09-19", "2023-09-20", "consultants"),
    ("2023-09-20", "2023-09-22", "junior doctors"),
    ("2023-10-02", "2023-10-04", "consultants"),
    ("2023-10-02", "2023-10-04", "junior doctors"),
    ("2023-12-20", "2023-12-22", "junior doctors"),
    ("2024-01-03", "2024-01-08", "junior doctors"),
    ("2024-02-24", "2024-02-28", "junior doctors"),
    ("2024-06-27", "2024-07-01", "junior doctors"),
    ("2025-07-25", "2025-07-29", "junior doctors"),
    ("2025-11-14", "2025-11-18", "junior doctors"),
    ("2025-12-17", "2025-12-21", "junior doctors"),
    ("2026-04-07", "2026-04-12", "junior doctors"),
)


def strike_days() -> pd.DataFrame:
    """One row per strike day: ``date``, ``junior`` and ``consultant`` flags and the round's
    first day (``round``)."""
    rows = []
    for first, last, group in ROUNDS:
        for d in pd.date_range(first, last, freq="D"):
            rows.append({"date": d, "group": group, "round": pd.Timestamp(first)})
    r = pd.DataFrame(rows)
    out = (
        r.assign(junior=r["group"].eq("junior doctors"), consultant=r["group"].eq("consultants"))
        .groupby("date")
        .agg(junior=("junior", "max"), consultant=("consultant", "max"), round=("round", "min"))
        .reset_index()
    )
    return out
