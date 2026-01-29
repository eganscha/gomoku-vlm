#!/usr/bin/env python3


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path("plots")
OUTPUT_FORMAT = "png"
DPI = 220
FIGSIZE = (11.0, 6.0)
TITLE_PREFIX = ""


def setup_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": DPI,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.titlepad": 12,
            "figure.autolayout": False,
        }
    )


def save_fig(fig, filename_stem: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"{filename_stem}.{OUTPUT_FORMAT}"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def grouped_bar(
    title: str,
    x_labels: list[str],
    series: dict[str, list[float]],
    ylabel: str = "Accuracy",
    ylim: tuple[float, float] = (0.0, 1.0),
    annotate: bool = True,
    filename_stem: str = "grouped_bar",
) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE)

    keys = list(series.keys())
    data = np.array([series[k] for k in keys], dtype=float)
    n_series, n_x = data.shape

    x = np.arange(n_x)
    width = 0.8 / max(n_series, 1)

    for i, k in enumerate(keys):
        bars = ax.bar(
            x + (i - (n_series - 1) / 2) * width, data[i], width=width, label=k
        )
        if annotate:
            for b in bars:
                h = b.get_height()
                ax.annotate(
                    f"{h:.3f}",
                    (b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_title(f"{TITLE_PREFIX}{title}")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.legend(loc="upper left", frameon=True)
    return save_fig(fig, filename_stem)


def sorted_delta_bar(
    title: str,
    items: list[tuple[str, float]],
    ylabel: str = "Δ Accuracy",
    filename_stem: str = "delta_bar",
) -> Path:
    items = sorted(items, key=lambda t: t[1])
    labels = [k for k, _ in items]
    vals = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(11.5, max(5.5, 0.35 * len(labels))))

    y = np.arange(len(labels))
    bars = ax.barh(y, vals)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(ylabel)
    ax.set_title(f"{TITLE_PREFIX}{title}")

    ax.axvline(0.0, linewidth=1)

    for b in bars:
        w = b.get_width()
        ax.annotate(
            f"{w:+.3f}",
            (w, b.get_y() + b.get_height() / 2),
            xytext=(5 if w >= 0 else -5, 0),
            textcoords="offset points",
            ha="left" if w >= 0 else "right",
            va="center",
            fontsize=9,
        )

    return save_fig(fig, filename_stem)


def line_plot(
    title: str,
    x_labels: list[str],
    series: dict[str, list[float]],
    ylabel: str = "Accuracy",
    ylim: tuple[float, float] = (0.0, 1.0),
    filename_stem: str = "line_plot",
) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(x_labels))

    for name, ys in series.items():
        ax.plot(x, ys, marker="o", linewidth=2, label=name)

    ax.set_title(f"{TITLE_PREFIX}{title}")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.legend(loc="upper left", frameon=True)
    return save_fig(fig, filename_stem)


SUMMARY_PRE_POST_VISUAL = {
    "Pre-Train": {"all": 0.160, "perception": 0.207, "strategy": 0.003},
    "Post-Visual": {"all": 0.284, "perception": 0.367, "strategy": 0.007},
}

SUMMARY_POST_VISUAL_POST_STRATEGY = {
    "Post-Visual": {"all": 0.284, "perception": 0.367, "strategy": 0.007},
    "Post-Strategy": {"all": 0.238, "perception": 0.285, "strategy": 0.080},
}

SUMMARY_CURRICULUM = {
    "Step 1": {"all": 0.165, "perception": 0.213, "strategy": 0.003},
    "Step 2": {"all": 0.301, "perception": 0.410, "strategy": 0.030},
    "Step 3": {"all": 0.307, "perception": 0.414, "strategy": 0.040},
    "Step 4": {"all": 0.249, "perception": 0.323, "strategy": 0.003},
}

VISUAL_PRE_POST_VISUAL: dict[str, tuple[list[str], list[float], list[float]]] = {
    "color_at_position": (
        ["Q1", "Q2", "Q3", "Q4"],
        [0.640, 0.680, 0.400, 0.200],
        [0.960, 0.760, 0.640, 0.520],
    ),
    "count_black_stones": (
        ["Q101", "Q102", "Q103", "Q104"],
        [0.040, 0.000, 0.000, 0.000],
        [0.600, 0.120, 0.040, 0.200],
    ),
    "count_white_stones": (
        ["Q201", "Q202", "Q203", "Q204"],
        [0.120, 0.040, 0.000, 0.000],
        [0.680, 0.160, 0.040, 0.200],
    ),
    "count_empty_intersections": (
        ["Q301", "Q302", "Q303", "Q304"],
        [0.000, 0.000, 0.000, 0.040],
        [0.040, 0.000, 0.000, 0.080],
    ),
    "three_in_a_row": (
        ["Q401", "Q402", "Q403", "Q404"],
        [0.000, 0.200, 0.080, 0.000],
        [0.680, 0.160, 0.160, 0.080],
    ),
    "four_in_a_row": (
        ["Q501", "Q502", "Q503", "Q504"],
        [0.000, 0.000, 0.040, 0.040],
        [0.600, 0.200, 0.080, 0.080],
    ),
    "determine_who_won": (
        ["Q601", "Q602", "Q603", "Q604"],
        [0.480, 0.480, 0.120, 0.480],
        [0.520, 0.640, 0.440, 0.320],
    ),
    "can_you_win": (
        ["Q701", "Q702", "Q703", "Q704"],
        [0.480, 0.640, 0.520, 0.520],
        [0.720, 0.920, 0.560, 0.960],
    ),
    "can_you_lose": (
        ["Q801", "Q802", "Q803", "Q804"],
        [0.520, 0.680, 0.280, 0.560],
        [0.600, 0.680, 0.560, 0.680],
    ),
    "print_board_matrix": (
        ["Q901", "Q902", "Q903", "Q904"],
        [0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000],
    ),
    "win_next_turn": (
        ["Q1101", "Q1102", "Q1103", "Q1104"],
        [0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.040, 0.000],
    ),
    "best_next_move": (
        ["Q1201", "Q1202", "Q1203", "Q1204"],
        [0.040, 0.000, 0.000, 0.000],
        [0.040, 0.000, 0.000, 0.000],
    ),
    "reason_next_move": (
        ["Q1301", "Q1302", "Q1303", "Q1304"],
        [0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000],
    ),
}

VISUAL_POST_VISUAL_POST_STRATEGY: dict[
    str, tuple[list[str], list[float], list[float]]
] = {
    "color_at_position": (
        ["Q1", "Q2", "Q3", "Q4"],
        [0.960, 0.760, 0.640, 0.520],
        [0.960, 0.800, 0.640, 0.480],
    ),
    "count_black_stones": (
        ["Q101", "Q102", "Q103", "Q104"],
        [0.600, 0.120, 0.040, 0.200],
        [0.120, 0.000, 0.080, 0.080],
    ),
    "count_white_stones": (
        ["Q201", "Q202", "Q203", "Q204"],
        [0.680, 0.160, 0.040, 0.200],
        [0.120, 0.040, 0.000, 0.040],
    ),
    "count_empty_intersections": (
        ["Q301", "Q302", "Q303", "Q304"],
        [0.040, 0.000, 0.000, 0.080],
        [0.000, 0.000, 0.000, 0.040],
    ),
    "three_in_a_row": (
        ["Q401", "Q402", "Q403", "Q404"],
        [0.680, 0.160, 0.160, 0.080],
        [0.520, 0.160, 0.000, 0.000],
    ),
    "four_in_a_row": (
        ["Q501", "Q502", "Q503", "Q504"],
        [0.600, 0.200, 0.080, 0.080],
        [0.560, 0.240, 0.120, 0.000],
    ),
    "determine_who_won": (
        ["Q601", "Q602", "Q603", "Q604"],
        [0.520, 0.640, 0.440, 0.320],
        [0.440, 0.520, 0.640, 0.480],
    ),
    "can_you_win": (
        ["Q701", "Q702", "Q703", "Q704"],
        [0.720, 0.920, 0.560, 0.960],
        [0.520, 0.680, 0.600, 0.560],
    ),
    "can_you_lose": (
        ["Q801", "Q802", "Q803", "Q804"],
        [0.600, 0.680, 0.560, 0.680],
        [0.440, 0.520, 0.520, 0.480],
    ),
    "print_board_matrix": (
        ["Q901", "Q902", "Q903", "Q904"],
        [0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000],
    ),
    "win_next_turn": (
        ["Q1101", "Q1102", "Q1103", "Q1104"],
        [0.000, 0.000, 0.040, 0.000],
        [0.160, 0.280, 0.240, 0.280],
    ),
    "best_next_move": (
        ["Q1201", "Q1202", "Q1203", "Q1204"],
        [0.040, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000],
    ),
    "reason_next_move": (
        ["Q1301", "Q1302", "Q1303", "Q1304"],
        [0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000],
    ),
}

CURRICULUM_FOCUS = {
    "color_at_position": [0.670, 0.660, 0.710, 0.730],
    "count_black_stones": [0.030, 0.000, 0.000, 0.110],
    "count_white_stones": [0.070, 0.000, 0.000, 0.100],
    "count_empty_intersections": [0.010, 0.000, 0.000, 0.010],
    "three_in_a_row": [0.050, 0.170, 0.180, 0.220],
    "four_in_a_row": [0.060, 0.230, 0.200, 0.210],
    "determine_who_won": [0.220, 0.000, 0.000, 0.500],
    "can_you_win": [0.480, 0.520, 0.510, 0.820],
    "can_you_lose": [0.540, 0.470, 0.470, 0.530],
    "print_board_matrix": [0.000, 0.000, 0.000, 0.000],
}


def mean(xs: list[float]) -> float:
    return float(np.mean(np.array(xs, dtype=float))) if xs else float("nan")


def focus_means_from_variants(
    table: dict[str, tuple[list[str], list[float], list[float]]],
) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for focus, (_labels, a, b) in table.items():
        out[focus] = (mean(a), mean(b))
    return out


def main() -> None:
    setup_style()

    grouped_bar(
        title="Overall Summary — Pre-Train vs Post-Visual",
        x_labels=["all", "perception", "strategy"],
        series={
            "Pre-Train": [
                SUMMARY_PRE_POST_VISUAL["Pre-Train"][k]
                for k in ["all", "perception", "strategy"]
            ],
            "Post-Visual": [
                SUMMARY_PRE_POST_VISUAL["Post-Visual"][k]
                for k in ["all", "perception", "strategy"]
            ],
        },
        ylim=(0.0, 0.45),
        filename_stem="summary_pre_vs_post_visual",
    )

    grouped_bar(
        title="Overall Summary — Post-Visual vs Post-Strategy",
        x_labels=["all", "perception", "strategy"],
        series={
            "Post-Visual": [
                SUMMARY_POST_VISUAL_POST_STRATEGY["Post-Visual"][k]
                for k in ["all", "perception", "strategy"]
            ],
            "Post-Strategy": [
                SUMMARY_POST_VISUAL_POST_STRATEGY["Post-Strategy"][k]
                for k in ["all", "perception", "strategy"]
            ],
        },
        ylim=(0.0, 0.45),
        filename_stem="summary_post_visual_vs_post_strategy",
    )

    line_plot(
        title="Curriculum Learning — Overall Summary",
        x_labels=["Step 1", "Step 2", "Step 3", "Step 4"],
        series={
            "all": [
                SUMMARY_CURRICULUM[s]["all"]
                for s in ["Step 1", "Step 2", "Step 3", "Step 4"]
            ],
            "perception": [
                SUMMARY_CURRICULUM[s]["perception"]
                for s in ["Step 1", "Step 2", "Step 3", "Step 4"]
            ],
            "strategy": [
                SUMMARY_CURRICULUM[s]["strategy"]
                for s in ["Step 1", "Step 2", "Step 3", "Step 4"]
            ],
        },
        ylim=(0.0, 0.50),
        filename_stem="curriculum_summary_lines",
    )

    means_pre_post = focus_means_from_variants(VISUAL_PRE_POST_VISUAL)
    delta_items = [(focus, post - pre) for focus, (pre, post) in means_pre_post.items()]
    sorted_delta_bar(
        title="Visual Fine-Tuning — Δ Accuracy by Focus (mean over variants)",
        items=delta_items,
        ylabel="Δ (Post-Visual − Pre-Train)",
        filename_stem="delta_by_focus_visual_ft",
    )

    means_postV_postS = focus_means_from_variants(VISUAL_POST_VISUAL_POST_STRATEGY)
    delta_items2 = [
        (focus, postS - postV) for focus, (postV, postS) in means_postV_postS.items()
    ]
    sorted_delta_bar(
        title="Strategy Fine-Tuning — Δ Accuracy by Focus (mean over variants)",
        items=delta_items2,
        ylabel="Δ (Post-Strategy − Post-Visual)",
        filename_stem="delta_by_focus_strategy_ft",
    )

    for focus, (labels, pre, post) in VISUAL_PRE_POST_VISUAL.items():
        grouped_bar(
            title=f"Pre-Train vs Post-Visual — {focus}",
            x_labels=labels,
            series={
                "Pre-Train": pre,
                "Post-Visual": post,
            },
            ylim=(0.0, 1.05),
            filename_stem=f"variants_pre_vs_post_visual__{focus}",
            annotate=False,  # cleaner for many bars
        )

    for focus, (labels, postV, postS) in VISUAL_POST_VISUAL_POST_STRATEGY.items():
        grouped_bar(
            title=f"Post-Visual vs Post-Strategy — {focus}",
            x_labels=labels,
            series={
                "Post-Visual": postV,
                "Post-Strategy": postS,
            },
            ylim=(0.0, 1.05),
            filename_stem=f"variants_post_visual_vs_post_strategy__{focus}",
            annotate=False,
        )

    line_plot(
        title="Curriculum Learning — Focus Accuracies Across Steps",
        x_labels=["Step 1", "Step 2", "Step 3", "Step 4"],
        series=CURRICULUM_FOCUS,
        ylim=(0.0, 1.0),
        filename_stem="curriculum_focus_trends",
    )

    print(f"Done. Plots written to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
