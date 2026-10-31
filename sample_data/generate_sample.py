"""Generate the synthetic sample dataset (deterministic).

The data is SYNTHETIC: responses are assembled from templates so the
platform can be demonstrated and tested. It is not real research data.
Built-in temporal patterns: performance complaints rise after month 6;
support complaints fall over the year; pricing concerns spike in month 9.

Usage: python sample_data/generate_sample.py [n_responses]
"""
from __future__ import annotations

import csv
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

SEED = 20240101

TEMPLATES = {
    "navigation": [
        "I cannot find the settings page easily.",
        "The menus are confusing and the navigation is difficult.",
        "Finding the export option in the menus takes far too long.",
        "The navigation menu hides important settings.",
        "It is difficult to navigate between projects and settings.",
        "The user interface is confusing, especially the main menu.",
        "I get lost in the menus every time I need the settings.",
    ],
    "performance": [
        "The application takes too long to load.",
        "Pages load slowly and the app freezes during large uploads.",
        "Performance is poor; the dashboard is very slow to load.",
        "The app is slow and crashes when I open large reports.",
        "Loading times have become much slower after the update.",
        "Search results load slowly and the page sometimes freezes.",
    ],
    "support": [
        "Customer support took a week to respond to my ticket.",
        "The support team was slow and did not solve my problem.",
        "I waited days for customer support to reply.",
        "Support response time is too long for urgent issues.",
        "The customer support agent was helpful and solved the issue quickly.",
        "Customer support was friendly and responded within an hour.",
    ],
    "pricing": [
        "The subscription price is too expensive for small teams.",
        "The new pricing plan is expensive and confusing.",
        "I think the price increase is not justified by the features.",
        "The annual subscription price is too high compared to competitors.",
        "Pricing tiers are confusing and the premium plan is expensive.",
    ],
    "reporting": [
        "The reporting features are excellent and easy to share.",
        "I love the export to PDF reports feature.",
        "The charts in the reports are clear and useful for my team.",
        "Custom reports save me hours every week.",
        "Report templates are useful but need more chart options.",
    ],
    "onboarding": [
        "The onboarding tutorial was clear and helpful.",
        "Getting started was easy thanks to the onboarding guide.",
        "The onboarding emails helped me learn the main features.",
        "Onboarding took too long and the tutorial skipped important steps.",
        "The setup wizard made onboarding simple for new team members.",
    ],
    "mobile": [
        "The mobile app is missing many features from the desktop version.",
        "The mobile app crashes when I try to upload photos.",
        "I would like offline mode in the mobile app.",
        "Notifications on the mobile app are not reliable.",
        "The mobile app is great for quick checks on the go.",
    ],
    "collaboration": [
        "Sharing projects with my team is easy and reliable.",
        "Team collaboration features like comments are very useful.",
        "Real-time collaboration works well for our remote team.",
        "Permissions for team members are hard to manage.",
        "I wish collaboration supported external guests.",
    ],
}

FILLERS = ["Overall it is okay.", "Thanks.", "Please fix this soon.", "Otherwise I am satisfied.", "", "", "", ""]
REGIONS = ["North", "South", "East", "West"]


def theme_weights(month: int) -> dict[str, float]:
    w = {"navigation": 1.0, "performance": 0.6, "support": 1.2, "pricing": 0.6, "reporting": 0.8,
         "onboarding": 0.7, "mobile": 0.6, "collaboration": 0.6}
    if month >= 6:
        w["performance"] = 0.6 + 0.35 * (month - 5)
    w["support"] = max(0.3, 1.4 - 0.1 * month)
    if month == 9:
        w["pricing"] = 2.2
    return w


def generate(n: int = 1200) -> list[dict]:
    rng = random.Random(SEED)
    start = datetime(2024, 1, 1)
    rows = []
    for i in range(n):
        day = rng.randrange(0, 366)
        ts = start + timedelta(days=day, hours=rng.randrange(8, 20), minutes=rng.randrange(60))
        month = ts.month
        w = theme_weights(month)
        themes = list(w)
        k = rng.choices([1, 2, 3], weights=[0.45, 0.4, 0.15])[0]
        chosen = []
        while len(chosen) < k:
            t = rng.choices(themes, weights=[w[x] for x in themes])[0]
            if t not in chosen:
                chosen.append(t)
        # correlated pairs: navigation <-> onboarding; performance <-> mobile
        if "navigation" in chosen and rng.random() < 0.35 and "onboarding" not in chosen:
            chosen.append("onboarding")
        if "performance" in chosen and rng.random() < 0.35 and "mobile" not in chosen:
            chosen.append("mobile")
        sentences = [rng.choice(TEMPLATES[t]) for t in chosen]
        filler = rng.choice(FILLERS)
        if filler:
            sentences.append(filler)
        text = " ".join(sentences)
        if i % 97 == 0:
            text = ""  # missing responses to exercise validation
        rows.append({
            "respondent_id": f"R{rng.randrange(1, n // 2 + 1):05d}",
            "response_text": text,
            "created_at": ts.strftime("%Y-%m-%d %H:%M"),
            "region": rng.choice(REGIONS),
            "primary_topic": chosen[0] if rng.random() < 0.5 else "",
        })
    rows.append(dict(rows[5]))  # one exact duplicate
    return rows


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
    out = Path(__file__).with_name("customer_feedback_synthetic.csv")
    rows = generate(n)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
