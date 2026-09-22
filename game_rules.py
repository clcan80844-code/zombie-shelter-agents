"""Deterministic game rules for the zombie shelter simulation.

The LLM can propose actions and narrate outcomes, but this module owns the
numbers that change the world. Keeping those rules in one place makes the game
testable, replayable, and much harder for a model response to accidentally
break.
"""

from copy import deepcopy
from dataclasses import dataclass
from random import Random
from typing import Iterable, Mapping


DAILY_FOOD_COST = 4
DAILY_WATER_COST = 4
DEFAULT_AGENT_STATE = {
    "stamina": 100,
    "stress": 10,
    "health": 100,
    "injury_status": "健康",
    "injury_days": 0,
    "is_alive": True,
    "cause_of_death": "",
}


@dataclass(frozen=True)
class SearchOutcome:
    event_type: str
    event_desc: str
    food_found: int
    water_found: int
    bullets_used: int
    ammo_found: int


@dataclass(frozen=True)
class HazardReport:
    agent_states: dict[str, dict[str, object]]
    incidents: list[str]


def normalize_agent_states(
    agent_states: Mapping[str, Mapping[str, object]]
) -> dict[str, dict[str, object]]:
    """Migrate old saves and clamp all persistent life-state fields."""

    normalized: dict[str, dict[str, object]] = {}
    for name, raw_state in agent_states.items():
        state = {**DEFAULT_AGENT_STATE, **dict(raw_state)}
        state["stamina"] = max(0, min(100, int(state.get("stamina", 100))))
        state["stress"] = max(0, min(100, int(state.get("stress", 0))))
        state["health"] = max(0, min(100, int(state.get("health", 100))))
        state["injury_days"] = max(0, int(state.get("injury_days", 0)))
        state["is_alive"] = bool(state.get("is_alive", True)) and state["health"] > 0
        if not state["is_alive"]:
            state["health"] = 0
            state["injury_status"] = "死亡"
        elif state["injury_status"] not in {"健康", "轻伤", "重伤"}:
            state["injury_status"] = "健康"
        normalized[name] = state
    return normalized


def roll_search_outcome(search_count: int, rng: Random | None = None) -> SearchOutcome:
    """Roll one bounded physical outcome for the day's search actions."""

    rng = rng or Random()
    if search_count <= 0:
        return SearchOutcome(
            event_type="QUIET",
            event_desc="【避而不出】今天没有人外出搜寻，避难所暂时保持安静。",
            food_found=0,
            water_found=0,
            bullets_used=0,
            ammo_found=0,
        )

    event_roll = rng.random()
    if event_roll < 0.20:
        return SearchOutcome("EMPTY", "【空手而归】搜寻地点已被扫荡一空，毫无收获。", 0, 0, 0, 0)
    if event_roll < 0.50:
        return SearchOutcome(
            "AMBUSH",
            "【丧尸伏击】搜寻引发尸群攻击！幸存者们陷入苦战。",
            rng.randint(0, 1),
            rng.randint(0, 1),
            rng.randint(1, 2),
            1 if rng.random() < 0.10 else 0,
        )
    if event_roll < 0.90:
        multiplier = 2 if search_count >= 2 else 1
        ammo_found = rng.randint(1, 2) if rng.random() < 0.35 else 0
        return SearchOutcome(
            "NORMAL",
            "【平稳搜寻】幸存者们找到了一些残存的食物和水。",
            rng.randint(1, 2) * multiplier,
            rng.randint(1, 2) * multiplier,
            0,
            ammo_found,
        )
    return SearchOutcome(
        "JACKPOT",
        "【大丰收】搜寻队找到了一间尚未被破坏的储藏室。",
        rng.randint(3, 5),
        rng.randint(3, 5),
        0,
        rng.randint(2, 4) if rng.random() < 0.70 else 0,
    )


def apply_daily_resources(
    resources: Mapping[str, int],
    outcome: SearchOutcome,
    stolen_food: int = 0,
) -> dict[str, int]:
    """Apply survival costs and bounded physical gains to the shared stock."""

    available_bullets = max(0, int(resources.get("bullets", 0)))
    actual_bullets_used = min(
        available_bullets, max(0, int(outcome.bullets_used))
    )
    return {
        "food_cans": max(
            0,
            int(resources.get("food_cans", 0))
            - DAILY_FOOD_COST
            + outcome.food_found
            - max(0, stolen_food),
        ),
        "water_bottles": max(
            0,
            int(resources.get("water_bottles", 0))
            - DAILY_WATER_COST
            + outcome.water_found,
        ),
        "bullets": max(
            0,
            available_bullets
            - actual_bullets_used
            + max(0, outcome.ammo_found),
        ),
    }


def resolve_exploration_hazards(
    agent_states: Mapping[str, Mapping[str, object]],
    daily_actions: Iterable[Mapping[str, str]],
    outcome: SearchOutcome,
    available_bullets: int,
    rng: Random | None = None,
) -> HazardReport:
    """Apply bounded injury/death risk to agents who actually explored."""

    rng = rng or Random()
    updated = normalize_agent_states(deepcopy(agent_states))
    incidents: list[str] = []
    base_risk = {
        "EMPTY": 0.08,
        "AMBUSH": 0.42,
        "NORMAL": 0.10,
        "JACKPOT": 0.06,
        "QUIET": 0.0,
    }.get(outcome.event_type, 0.10)

    for action in daily_actions:
        if action["action"] not in {"SEARCH", "JOINT_SEARCH"}:
            continue
        state = updated.get(action["agent"])
        if state is None or not state["is_alive"] or rng.random() >= base_risk:
            continue

        damage = rng.randint(18, 38)
        if outcome.event_type == "AMBUSH":
            damage = rng.randint(35, 70)
            if available_bullets <= 0:
                damage += 15
            elif outcome.bullets_used > 0:
                damage = max(10, damage - 12)

        state["health"] = max(0, int(state["health"]) - damage)
        death_roll = 0.06 if outcome.event_type != "AMBUSH" else 0.14
        if state["health"] <= 0 or (
            outcome.event_type == "AMBUSH"
            and int(state["health"]) < 25
            and rng.random() < death_roll
        ):
            state["health"] = 0
            state["is_alive"] = False
            state["injury_status"] = "死亡"
            state["injury_days"] = 0
            state["cause_of_death"] = "探索过程中遭遇致命伤"
            incidents.append(f"☠️ {action['agent']} 在探索中受到致命伤，已经死亡。")
        else:
            state["injury_days"] = rng.randint(1, 3)
            state["injury_status"] = "重伤" if damage >= 45 else "轻伤"
            incidents.append(
                f"🩸 {action['agent']} 在探索中受伤（{state['injury_status']}，生命 {state['health']}/100）。"
            )

    return HazardReport(updated, incidents)


def apply_status_effects(
    agent_states: Mapping[str, Mapping[str, int]],
    daily_actions: Iterable[Mapping[str, str]],
    event_type: str,
    is_crisis: bool,
) -> dict[str, dict[str, int]]:
    """Update stamina and stress without allowing values outside 0..100."""

    updated = normalize_agent_states(deepcopy(agent_states))
    for state in updated.values():
        if state["is_alive"] and state["injury_status"] != "健康":
            state["injury_days"] = max(0, int(state["injury_days"]) - 1)
            state["health"] = min(100, int(state["health"]) + 8)
            if int(state["health"]) >= 90:
                state["health"] = 100
                state["injury_days"] = 0
                state["injury_status"] = "健康"

    for log in daily_actions:
        name = log["agent"]
        action = log["action"]
        state = updated.setdefault(name, dict(DEFAULT_AGENT_STATE))
        if not state.get("is_alive", True):
            continue
        stamina = int(state.get("stamina", 100))
        stress = int(state.get("stress", 0))

        if action in {"SEARCH", "JOINT_SEARCH"}:
            stamina -= 25
            stress += 35 if event_type == "AMBUSH" else 10
            if event_type == "AMBUSH":
                stamina -= 15
            if state.get("injury_status") in {"轻伤", "重伤"}:
                stamina -= 10
                stress += 5
        elif action == "REST":
            stamina += 35
            stress -= 20
        elif action == "GUARD":
            stamina -= 10
            stress += 5
        elif action == "STEAL_RESOURCE":
            stamina -= 10
            stress += 30

        if is_crisis:
            stress += 10

        state["stamina"] = max(0, min(100, stamina))
        state["stress"] = max(0, min(100, stress))

    return updated
