"""Scenario generator for QuRescue: drones, requests and events for three disaster modes."""
from dataclasses import dataclass, field
import numpy as np

# Which drone type can serve which request type
COMPAT = {
    "medical": {"medical", "stranded"},
    "cargo": {"supply", "stranded"},
    "sensor": {"gas_leak", "survey"},
}

REQUEST_STYLE = {
    "medical":  ("Medical emergency", "#d62728"),
    "stranded": ("Stranded people",   "#ff7f0e"),
    "supply":   ("Supply drop",       "#1f77b4"),
    "gas_leak": ("Gas leak",          "#9467bd"),
    "survey":   ("Survey / search",   "#2ca02c"),
}


@dataclass
class Drone:
    id: int
    kind: str          # medical | cargo | sensor
    pos: tuple
    range_km: float


@dataclass
class Request:
    id: int
    kind: str
    pos: tuple
    people: int
    urgency: float     # 0..1
    label: str = ""


@dataclass
class Scenario:
    mode: str
    title: str
    base: tuple
    drones: list
    requests: list
    hazard_zones: list = field(default_factory=list)  # (x, y, radius) for drawing
    event_name: str = "Escalate"


def _jitter(rng, p, s=0.4):
    return (float(p[0] + rng.normal(0, s)), float(p[1] + rng.normal(0, s)))


def make_scenario(mode: str = "flood", seed: int = 7) -> Scenario:
    rng = np.random.default_rng(seed)
    base = (1.0, 1.0)

    if mode == "flood":
        title = "Flood relief: rising water, cut-off villages"
        fleet = ["medical", "cargo", "cargo", "sensor"]
        reqs = [
            ("medical", (6.5, 7.0), 1, 0.95, "Injured man, Village A"),
            ("medical", (3.0, 6.0), 2, 0.55, "Fever cases, relief camp 2"),
            ("stranded", (8.0, 4.5), 6, 0.85, "Family on rooftop"),
            ("stranded", (3.5, 8.5), 4, 0.70, "People on embankment"),
            ("supply", (8.5, 8.0), 200, 0.40, "Village B: no clean water"),
            ("supply", (5.5, 2.5), 150, 0.35, "Village C: food running out"),
            ("gas_leak", (5.0, 3.5), 50, 0.70, "Exposed gas main, near homes"),
            ("survey", (8.0, 9.0), 0, 0.30, "Unknown: check for survivors"),
        ]
        hazards = [(7.0, 6.5, 2.6), (4.0, 8.0, 1.4)]
        event = "Water rising"
    elif mode == "earthquake":
        title = "Earthquake response: mountain villages"
        fleet = ["medical", "medical", "cargo", "sensor"]
        reqs = [
            ("medical", (7.5, 8.0), 2, 0.90, "Collapsed school"),
            ("medical", (4.0, 6.5), 1, 0.70, "Injured shepherd"),
            ("stranded", (8.5, 4.0), 12, 0.60, "Road blocked, families cut off"),
            ("supply", (6.0, 9.5), 300, 0.55, "Tents & blankets needed"),
            ("gas_leak", (5.5, 4.5), 80, 0.75, "Gas smell in bazaar"),
            ("survey", (9.0, 9.0), 0, 0.40, "Landslide area: search"),
        ]
        hazards = [(7.5, 7.5, 2.0)]
        event = "Aftershock"
    elif mode == "border":
        title = "Border post: casualty evacuation & resupply (non-lethal)"
        fleet = ["medical", "cargo", "sensor"]
        reqs = [
            ("medical", (8.5, 8.0), 1, 0.95, "Wounded soldier, Post 3"),
            ("medical", (6.0, 9.0), 2, 0.65, "Injured patrol"),
            ("stranded", (9.0, 5.0), 4, 0.55, "Patrol lost in snow"),
            ("supply", (5.5, 6.5), 20, 0.50, "Post 2: food & medicine low"),
            ("survey", (7.0, 4.0), 0, 0.45, "Sector sweep for lost patrol"),
        ]
        hazards = [(8.0, 8.0, 1.6)]
        event = "New casualty report"
    else:
        raise ValueError(mode)

    drones = [Drone(i, k, _jitter(rng, base, 0.3), 22.0) for i, k in enumerate(fleet)]
    requests = [Request(i, k, _jitter(rng, p, 0.3), n, u, lab)
                for i, (k, p, n, u, lab) in enumerate(reqs)]
    return Scenario(mode, title, base, drones, requests, hazards, event)


def escalate(sc: Scenario, seed: int = 0) -> Scenario:
    """Situation changes: urgencies shift (water rises / aftershock / new casualty)."""
    rng = np.random.default_rng(seed)
    new_reqs = []
    for r in sc.requests:
        u = r.urgency
        if r.kind in ("stranded", "medical", "gas_leak"):
            u = min(1.0, u + rng.uniform(0.1, 0.3))
        else:
            u = max(0.05, u - rng.uniform(0.0, 0.15))
        new_reqs.append(Request(r.id, r.kind, r.pos, r.people, round(u, 2), r.label))
    return Scenario(sc.mode, sc.title, sc.base, sc.drones, new_reqs,
                    sc.hazard_zones, sc.event_name)


def dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))
