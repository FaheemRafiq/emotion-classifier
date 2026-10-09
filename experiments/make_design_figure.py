"""Greyscale flow diagram of the safety guard and the weekly check-in (FYP report, Chapter 3).

Usage: uv run python experiments/make_design_figure.py -> reports/figures/intent2act_safety_checkin_flow.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "reports/figures/intent2act_safety_checkin_flow.png"
INK, FILL, DARK = "#111111", "#f2f2f2", "#d0d0d0"


def box(ax, x, y, w, h, text, fill=FILL, diamond=False, bold=False):
    if diamond:
        ax.add_patch(plt.Polygon([(x, y + h / 2), (x + w / 2, y + h), (x + w, y + h / 2), (x + w / 2, y)],
                                 closed=True, fc="white", ec=INK, lw=1.1))
    else:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.08", fc=fill, ec=INK, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.6, color=INK,
            fontweight="bold" if bold else "normal", wrap=True)
    return (x, y, w, h)


def arrow(ax, p, q, label=None, lx=0.0, ly=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, lw=1.0, color=INK, shrinkA=1, shrinkB=1))
    if label:
        ax.text((p[0] + q[0]) / 2 + lx, (p[1] + q[1]) / 2 + ly, label, fontsize=8, color=INK, style="italic")


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 8.2), dpi=200)
    ax.set_xlim(0, 11)
    ax.set_ylim(0.7, 8.2)
    ax.axis("off")

    # ---- (a) safety guard
    ax.text(0.2, 7.95, "(a) Safety guard on every chat message", fontsize=10.5, fontweight="bold", color=INK)
    box(ax, 1.4, 7.15, 2.6, 0.5, "User sends a message", fill=DARK, bold=True)
    box(ax, 0.2, 6.15, 1.55, 0.65, "Safety check\n(gpt-4o-mini,\n5 harm categories)")
    box(ax, 1.95, 6.15, 1.5, 0.65, "Emotion model\n(DeBERTa-v3, local)")
    box(ax, 3.65, 6.15, 1.5, 0.65, "System prompt\n(pipelines + memory)")
    for x in (0.97, 2.7, 4.4):
        arrow(ax, (2.7, 7.15), (x, 6.8))
    ax.text(3.0, 5.92, "run in parallel; each fails open", fontsize=7.8, style="italic", color="#333333")
    box(ax, 1.55, 4.75, 2.3, 0.95, "Crisis\n(self-harm / suicide)?", diamond=True)
    arrow(ax, (0.97, 6.15), (2.7, 5.7))
    box(ax, 0.1, 3.3, 2.05, 0.95, "Fixed crisis reply with\nhelplines (LLM not called)\nSafetyEvent: ESCALATED\nadmins notified", fill="white")
    arrow(ax, (1.9, 5.22), (1.1, 4.25), "yes", -0.45, 0.0)
    box(ax, 2.55, 3.4, 2.35, 0.8, "Risk MEDIUM/HIGH?\nadd Safety Guidance to\nprompt (SafetyEvent: GUIDED)", fill="white")
    arrow(ax, (3.5, 5.22), (3.7, 4.2), "no", 0.1, 0.0)
    box(ax, 2.55, 2.25, 2.35, 0.7, "Mentor LLM streams its reply\n(emotion signal + guidance)", fill=DARK)
    arrow(ax, (3.7, 3.4), (3.7, 2.95))
    box(ax, 2.55, 1.0, 2.35, 0.85, "After streaming: check the reply\nunsafe? SafetyEvent: FLAGGED\nadmins review it", fill="white")
    arrow(ax, (3.7, 2.25), (3.7, 1.85))
    box(ax, 0.1, 1.0, 2.05, 0.85, "Admin panel:\nSafety Events page\n(human oversight)", fill=FILL)
    arrow(ax, (2.55, 1.42), (2.15, 1.42))
    arrow(ax, (1.1, 3.3), (1.1, 1.85))

    # ---- (b) weekly check-in
    ax.plot([5.55, 5.55], [0.6, 8.0], color="#999999", lw=0.8, ls="--")
    ax.text(5.8, 7.95, "(b) Weekly check-in and plan editing", fontsize=10.5, fontweight="bold", color=INK)
    steps = [
        ("Monday, at the user's reminder time\n(scheduler; once per week, deduplicated)", DARK),
        ("Collect last 7 days from proof-verified tasks:\napproved, rejected, expired, pending; streak", FILL),
        ("Mentor writes the check-in: one win, one obstacle,\nONE concrete change to upcoming days, yes/no question", FILL),
        ("Saved as a mentor chat message + in-app\nnotification + push (quiet hours, daily cap)", FILL),
        ("User agrees in chat (\"yes, make it easier\")", DARK),
        ("Agent calls get_challenge_plan, then\nupdate_challenge_for_user with the agreed days", FILL),
        ("Backend edits only days not yet started;\nstarted / submitted / finished days are refused", "white"),
    ]
    y = 7.1
    for i, (txt, fill) in enumerate(steps):
        box(ax, 6.2, y, 4.4, 0.62, txt, fill=fill, bold=fill == DARK)
        if i:
            arrow(ax, (8.4, y + 0.62 + 0.36), (8.4, y + 0.62))
        y -= 0.98
    fig.tight_layout()
    fig.savefig(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
