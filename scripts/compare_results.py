from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
TRUE_DIR = WORKSPACE / "model-photogrammetry" / "sim2blender" / "ground_truth"


def angle_summary(name: str, errors: pd.Series) -> str:
    """Строка статистики ошибки одного угла."""
    return (
        f"{name}: "
        f"mean_abs={errors.abs().mean():.4f} deg, "
        f"median_abs={errors.abs().median():.4f} deg, "
        f"rmse={np.sqrt(np.mean(errors**2)):.4f} deg, "
        f"max_abs={errors.abs().max():.4f} deg"
    )


def compare_angles(
    true_path: Path,
    measured_path: Path,
) -> tuple[pd.DataFrame, list[str]]:
    """Сравнение истинных и измеренных углов по кадрам."""
    true_angles = pd.read_csv(true_path)
    measured_angles = pd.read_csv(measured_path)

    merged = true_angles.merge(
        measured_angles,
        on="frame",
        suffixes=("_true", "_meas"),
    )

    angles = pd.DataFrame(
        {
            "frame": merged["frame"],
            "t": merged["t_true"],
            "phi_true_deg": merged["phi_deg_true"],
            "phi_meas_deg": merged["phi_deg_meas"],
            "dphi_deg": merged["phi_deg_meas"] - merged["phi_deg_true"],
            "theta_true_deg": merged["theta_deg_true"],
            "theta_meas_deg": merged["theta_deg_meas"],
            "dtheta_deg": merged["theta_deg_meas"] - merged["theta_deg_true"],
        }
    )

    lines = [
        "Углы",
        f"  кадров: {len(angles)}",
        "  " + angle_summary("phi", angles["dphi_deg"]),
        "  " + angle_summary("theta", angles["dtheta_deg"]),
    ]
    return angles, lines


def save_angle_plot(angles: pd.DataFrame, output_dir: Path) -> None:
    """Сохранение графика углов и их ошибок."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(angles["t"], angles["phi_true_deg"], label="true")
    axes[0].plot(angles["t"], angles["phi_meas_deg"], label="measured")
    axes[0].set_ylabel("phi, deg")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(angles["t"], angles["theta_true_deg"], label="true")
    axes[1].plot(angles["t"], angles["theta_meas_deg"], label="measured")
    axes[1].set_ylabel("theta, deg")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(angles["t"], angles["dphi_deg"], label="dphi")
    axes[2].plot(angles["t"], angles["dtheta_deg"], label="dtheta")
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_xlabel("time, s")
    axes[2].set_ylabel("error, deg")
    axes[2].legend()
    axes[2].grid(True)

    fig.tight_layout()
    fig.savefig(output_dir / "angles.png", dpi=150)
    plt.close(fig)


def save_error_hist(angles: pd.DataFrame, output_dir: Path) -> None:
    """Сохранение гистограмм ошибок dphi и dtheta."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, column, title, label, color in (
        (axes[0], "dphi_deg", "phi error", "dphi, deg", "tab:blue"),
        (axes[1], "dtheta_deg", "theta error", "dtheta, deg", "tab:orange"),
    ):
        ax.hist(
            angles[column],
            bins=30,
            color=color,
            edgecolor="white",
            linewidth=0.8,
        )
        ax.axvline(0.0, color="black", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel(label)
        ax.set_ylabel("frames")
        ax.grid(True, alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_dir / "angle_errors_hist.png", dpi=150)
    plt.close(fig)


def compare(
    camera: str = "cam1",
    true_dir: Path | None = None,
    measured_dir: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Сравнение углов и сохранение результатов.

    params:
        camera: имя камеры
        true_dir: папка с rotor_angles_true.csv
        measured_dir: папка с rotor_angles_measured.csv
        output_dir: папка для результатов

    returns:
        output_dir: путь к папке с angles.csv, angles.png,
            angle_errors_hist.png и summary.txt
    """
    true_dir = true_dir or TRUE_DIR
    measured_dir = measured_dir or (
        ROOT
        / "output"
        / "videos"
        / f"model_{camera}"
        / "measurements"
        / camera
    )
    output_dir = output_dir or ROOT / "output" / "comparison" / camera
    output_dir.mkdir(parents=True, exist_ok=True)

    angles, angle_lines = compare_angles(
        true_dir / "rotor_angles_true.csv",
        measured_dir / "rotor_angles_measured.csv",
    )

    angles.to_csv(
        output_dir / "angles.csv", index=False, float_format="%.9f"
    )
    save_angle_plot(angles, output_dir)
    save_error_hist(angles, output_dir)

    summary = "\n".join(
        [f"Сравнение углов, {camera}", *angle_lines]
    )
    (output_dir / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary)
    print(f"\nРезультаты: {output_dir}")
    return output_dir
