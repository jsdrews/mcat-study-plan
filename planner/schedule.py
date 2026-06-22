"""
Generate a week-by-week MCAT study schedule.

Based on research showing:
  - 4-6 focused hours/day is optimal (quality drops beyond 6h)
  - 300-500 total hours is the target range for competitive scores
  - Spaced repetition > massed practice (2x exam success rate)
  - Practice tests provide biggest score gains in first 4-5 FLs
  - Retrieval practice > rereading (61% vs 40% retention at 1 week)

Schedule is split into three phases:
  Phase 1 — Content Review (~40% of time)
  Phase 2 — Practice & Integration (~40% of time)
  Phase 3 — Test Readiness (~20% of time)
"""

from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum

import pandas as pd


class Phase(Enum):
    CONTENT = "Content Review"
    PRACTICE = "Practice & Integration"
    TEST_READY = "Test Readiness"


# MCAT content areas with approximate weight
CONTENT_AREAS = {
    "Bio/Biochem": {
        "section": "Biological and Biochemical Foundations (B/B)",
        "weight": 0.30,
        "topics": [
            "Amino acids & proteins",
            "Enzyme kinetics",
            "Molecular biology (DNA/RNA)",
            "Cell biology & signaling",
            "Organ systems",
            "Metabolism (glycolysis, TCA, ETC)",
            "Genetics & evolution",
            "Microbiology",
        ],
    },
    "Chem/Phys": {
        "section": "Chemical and Physical Foundations (C/P)",
        "weight": 0.25,
        "topics": [
            "General chemistry (bonding, reactions, equilibrium)",
            "Organic chemistry (reactions, mechanisms, spectroscopy)",
            "Physics (mechanics, fluids, circuits, optics)",
            "Thermodynamics & kinetics",
            "Acid/base chemistry",
            "Electrochemistry",
        ],
    },
    "Psych/Soc": {
        "section": "Psychological, Social, and Biological Foundations (P/S)",
        "weight": 0.25,
        "topics": [
            "Learning & memory",
            "Cognition & consciousness",
            "Emotion & stress",
            "Identity & personality",
            "Social structures & demographics",
            "Health disparities",
        ],
    },
    "CARS": {
        "section": "Critical Analysis and Reasoning Skills",
        "weight": 0.20,
        "topics": [
            "Daily passage practice (humanities)",
            "Daily passage practice (social sciences)",
            "Argument analysis & rhetoric",
            "Timing drills",
        ],
    },
}


@dataclass
class StudyDay:
    date: date
    week: int
    day_of_week: str
    phase: Phase
    hours: float
    focus: str
    activities: list[str]
    is_rest: bool = False
    is_fl: bool = False  # full-length practice test day


@dataclass
class StudyWeek:
    week_number: int
    phase: Phase
    start_date: date
    end_date: date
    days: list[StudyDay]
    total_hours: float
    cumulative_hours: float
    theme: str


def generate_schedule(
    test_date: date,
    start_date: date | None = None,
    hours_per_day: float = 5.0,
    rest_days_per_week: int = 1,
) -> list[StudyWeek]:
    """Generate a week-by-week study schedule.

    Args:
        test_date: MCAT exam date.
        start_date: When to start studying. Defaults to today.
        hours_per_day: Target focused hours per study day (recommended: 4-6).
        rest_days_per_week: Full rest days per week (recommended: 1).

    Returns:
        List of StudyWeek objects with daily plans.
    """
    if start_date is None:
        start_date = date.today()

    total_days = (test_date - start_date).days
    if total_days <= 0:
        raise ValueError("Test date must be in the future")

    total_weeks = total_days // 7
    if total_weeks < 4:
        raise ValueError("Need at least 4 weeks for a meaningful schedule")

    # Phase allocation
    content_weeks = max(2, round(total_weeks * 0.40))
    practice_weeks = max(2, round(total_weeks * 0.40))
    test_ready_weeks = max(1, total_weeks - content_weeks - practice_weeks)

    # Adjust if rounding pushed us over
    while content_weeks + practice_weeks + test_ready_weeks > total_weeks:
        if content_weeks > practice_weeks:
            content_weeks -= 1
        else:
            practice_weeks -= 1

    study_days_per_week = 7 - rest_days_per_week
    content_topics = _build_topic_rotation()

    weeks = []
    cumulative = 0.0
    fl_count = 0  # full-length practice test counter

    for week_num in range(1, total_weeks + 1):
        week_start = start_date + timedelta(weeks=week_num - 1)
        week_end = week_start + timedelta(days=6)

        # Determine phase
        if week_num <= content_weeks:
            phase = Phase.CONTENT
            theme = _content_theme(week_num, content_weeks)
        elif week_num <= content_weeks + practice_weeks:
            phase = Phase.PRACTICE
            practice_week = week_num - content_weeks
            theme = _practice_theme(practice_week, practice_weeks)
        else:
            phase = Phase.TEST_READY
            remaining = total_weeks - week_num + 1
            theme = _test_ready_theme(remaining)

        days = []
        week_hours = 0.0

        for day_offset in range(7):
            day_date = week_start + timedelta(days=day_offset)
            dow = day_date.strftime("%A")

            # Rest day (default Sunday, or last day of week)
            if day_offset >= study_days_per_week:
                days.append(StudyDay(
                    date=day_date, week=week_num, day_of_week=dow,
                    phase=phase, hours=0, focus="Rest",
                    activities=["Rest day — no studying. Walk, socialize, sleep."],
                    is_rest=True,
                ))
                continue

            # Full-length practice test days
            is_fl_day = False
            if phase == Phase.PRACTICE and day_offset == 5:  # Saturday of practice weeks
                is_fl_day = True
                fl_count += 1
            elif phase == Phase.TEST_READY and day_offset == 2:  # midweek FL
                is_fl_day = True
                fl_count += 1

            if is_fl_day:
                days.append(StudyDay(
                    date=day_date, week=week_num, day_of_week=dow,
                    phase=phase, hours=7.5,
                    focus=f"Full-Length #{fl_count}",
                    activities=[
                        f"AAMC Full-Length Practice Test #{fl_count} (6h 15m, real conditions)",
                        "Review missed questions (1h)",
                        "Log score and note weak areas",
                    ],
                    is_fl=True,
                ))
                week_hours += 7.5
                continue

            # Regular study day
            focus, activities = _daily_plan(
                phase, week_num, day_offset, content_topics, content_weeks
            )
            days.append(StudyDay(
                date=day_date, week=week_num, day_of_week=dow,
                phase=phase, hours=hours_per_day, focus=focus,
                activities=activities,
            ))
            week_hours += hours_per_day

        cumulative += week_hours
        weeks.append(StudyWeek(
            week_number=week_num, phase=phase,
            start_date=week_start, end_date=week_end,
            days=days, total_hours=round(week_hours, 1),
            cumulative_hours=round(cumulative, 1),
            theme=theme,
        ))

    return weeks


def schedule_to_dataframe(weeks: list[StudyWeek]) -> pd.DataFrame:
    """Convert schedule to a flat DataFrame for analysis/export."""
    rows = []
    for week in weeks:
        for day in week.days:
            rows.append({
                "date": day.date,
                "week": day.week,
                "day": day.day_of_week,
                "phase": day.phase.value,
                "hours": day.hours,
                "focus": day.focus,
                "activities": " | ".join(day.activities),
                "is_rest": day.is_rest,
                "is_full_length": day.is_fl,
                "cumulative_hours": week.cumulative_hours,
            })
    return pd.DataFrame(rows)


def schedule_summary(weeks: list[StudyWeek]) -> pd.DataFrame:
    """Weekly summary table."""
    rows = []
    for w in weeks:
        rows.append({
            "week": w.week_number,
            "dates": f"{w.start_date.strftime('%b %d')} – {w.end_date.strftime('%b %d')}",
            "phase": w.phase.value,
            "theme": w.theme,
            "hours": w.total_hours,
            "cumulative": w.cumulative_hours,
        })
    return pd.DataFrame(rows)


# ── Internal helpers ─────────────────────────────────────────────────────────

def _build_topic_rotation() -> list[tuple[str, str]]:
    """Build a flat list of (area, topic) pairs for content phase rotation."""
    topics = []
    for area, info in CONTENT_AREAS.items():
        for topic in info["topics"]:
            topics.append((area, topic))
    return topics


def _content_theme(week: int, total_content_weeks: int) -> str:
    areas = list(CONTENT_AREAS.keys())
    # Cycle through content areas
    idx = (week - 1) % len(areas)
    return f"Focus: {areas[idx]}"


def _practice_theme(practice_week: int, total_practice_weeks: int) -> str:
    if practice_week <= 2:
        return "Section-level practice + weak area review"
    return "Mixed practice + full-length tests"


def _test_ready_theme(remaining: int) -> str:
    if remaining > 1:
        return "Final full-lengths + targeted weak spots"
    return "Light review + rest before test day"


def _daily_plan(
    phase: Phase,
    week: int,
    day_offset: int,
    topics: list[tuple[str, str]],
    content_weeks: int,
) -> tuple[str, list[str]]:
    """Return (focus_label, activity_list) for a regular study day."""

    if phase == Phase.CONTENT:
        # Rotate through topics
        topic_idx = ((week - 1) * 6 + day_offset) % len(topics)
        area, topic = topics[topic_idx]

        activities = [
            f"Content review: {topic} (2h)",
            "Anki/flashcard review — spaced repetition (30min)",
            f"Practice questions: {area} (1.5h)",
            "CARS daily passage practice (1h)",
        ]
        return f"{area}: {topic}", activities

    elif phase == Phase.PRACTICE:
        # Alternate between section practice and mixed sets
        areas = list(CONTENT_AREAS.keys())
        area = areas[day_offset % len(areas)]

        if day_offset < 3:
            activities = [
                f"Timed section practice: {area} (1.5h)",
                "Review & error log (45min)",
                "Anki review — spaced repetition (30min)",
                "CARS timed passage set (1h)",
                "Weak area targeted review (1h)",
            ]
            return f"Section practice: {area}", activities
        else:
            activities = [
                "Mixed question set — all sections (2h)",
                "Detailed review of missed questions (1h)",
                "Anki review (30min)",
                "CARS passage set (1h)",
            ]
            return "Mixed practice", activities

    else:  # TEST_READY
        if day_offset == 0:
            activities = [
                "Review last FL — rework missed questions (2h)",
                "Anki — high-yield cards only (30min)",
                "Light CARS practice (1h)",
            ]
            return "FL review + weak spots", activities
        elif day_offset < 4:
            activities = [
                "Targeted review: weakest content area (1.5h)",
                "Quick-hit practice questions (1h)",
                "Anki review (30min)",
                "CARS passage (45min)",
            ]
            return "Targeted weak areas", activities
        else:
            activities = [
                "Light content review — skim notes only (1h)",
                "Anki — final review (30min)",
                "Relax, sleep well, prepare for test day",
            ]
            return "Light review + rest", activities
