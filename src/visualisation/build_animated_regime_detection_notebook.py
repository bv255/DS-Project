from textwrap import dedent

import nbformat as nbf


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


nb = nbf.v4.new_notebook()
nb.cells = [
    md(
        """
        # Animated Regime Detection

        This notebook builds a GIF-style animation of bull/bear regime detection over the aligned 1-day test window.

        Outputs:
        - `reports/figures/model_results_visualisation/regime_detection_full_test_window.gif`
        - `reports/figures/model_results_visualisation/regime_detection_full_test_window_last_frame.png`
        - `reports/figures/model_results_visualisation/regime_detection_recent_test_period.gif`
        - `reports/figures/model_results_visualisation/regime_detection_recent_test_period_last_frame.png`
        """
    ),
    code(
        """
        from pathlib import Path

        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from matplotlib.animation import FuncAnimation, PillowWriter

        plt.rcParams["figure.dpi"] = 140
        plt.rcParams["savefig.bbox"] = "tight"


        def find_repo_root(start: Path) -> Path:
            current = start.resolve()
            for candidate in [current, *current.parents]:
                if (candidate / ".git").exists() or (candidate / "README.md").exists():
                    return candidate
            raise FileNotFoundError("Could not locate repo root")


        ROOT = find_repo_root(Path.cwd())
        REPORTS_DIR = ROOT / "reports"
        FIGURES_DIR = REPORTS_DIR / "figures" / "model_results_visualisation"
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        MODEL_COLORS = {
            "Logistic": "#355C7D",
            "XGBoost": "#C06C84",
            "PatchTST": "#F67280",
            "LSTM": "#6C5B7B",
        }
        """
    ),
    code(
        """
        def load_model_predictions(model_name: str, path: Path) -> pd.DataFrame:
            df = pd.read_csv(path)
            rename_map = {}
            if "date" in df.columns:
                rename_map["date"] = "Date"
            if "y_prob" in df.columns:
                rename_map["y_prob"] = "y_pred_prob"
            df = df.rename(columns=rename_map)
            df["Date"] = pd.to_datetime(df["Date"])
            df["model"] = model_name
            return df[["Date", "y_true", "y_pred", "y_pred_prob", "model"]]


        prediction_specs = {
            "Logistic": REPORTS_DIR / "trained_logistic" / "basecase_1day" / "test_predictions.csv",
            "XGBoost": REPORTS_DIR / "trained_xgboost" / "basecase_1day" / "test_predictions.csv",
            "PatchTST": REPORTS_DIR / "trained_patchtst" / "results_lag_1" / "predictions.csv",
            "LSTM": REPORTS_DIR / "trained_ltsm" / "results_lag_1" / "predictions.csv",
        }

        prediction_frames = [load_model_predictions(model_name, path) for model_name, path in prediction_specs.items()]
        predictions = pd.concat(prediction_frames, ignore_index=True)

        starts = predictions.groupby("model")["Date"].min()
        ends = predictions.groupby("model")["Date"].max()
        common_start = starts.max()
        common_end = ends.min()

        predictions = predictions[(predictions["Date"] >= common_start) & (predictions["Date"] <= common_end)].copy()
        predictions = predictions.sort_values(["model", "Date"]).reset_index(drop=True)

        truth = (
            predictions[predictions["model"] == "Logistic"][["Date", "y_true"]]
            .drop_duplicates("Date")
            .sort_values("Date")
            .reset_index(drop=True)
        )

        price = pd.read_csv(ROOT / "data" / "labeled_dataset.csv", usecols=["Date", "GSPC"])
        price["Date"] = pd.to_datetime(price["Date"])
        price = price[(price["Date"] >= common_start) & (price["Date"] <= common_end)].copy()
        price = price.sort_values("Date").reset_index(drop=True)

        timeline = truth.merge(price, on="Date", how="inner")
        predictions = predictions.merge(timeline[["Date"]], on="Date", how="inner")

        print("Common test window:", common_start.date(), "to", common_end.date())
        print("Rows in aligned animation data:", len(timeline))
        """
    ),
    md(
        """
        ## Build Frames

        The animation reveals the price path and model bull probabilities through time. Green background means bull, red means bear.
        """
    ),
    code(
        """
        def draw_regime_background(ax, frame_df, ymin, ymax):
            regime_values = frame_df["y_true"].to_numpy()
            dates = frame_df["Date"].to_numpy()
            start_idx = 0
            for idx in range(1, len(frame_df) + 1):
                if idx == len(frame_df) or regime_values[idx] != regime_values[start_idx]:
                    start_date = pd.Timestamp(dates[start_idx])
                    end_date = pd.Timestamp(dates[idx - 1])
                    color = "#D8F3DC" if regime_values[start_idx] == 1 else "#F8D7DA"
                    ax.axvspan(start_date, end_date, color=color, alpha=0.45, lw=0)
                    start_idx = idx
            ax.set_ylim(ymin, ymax)


        def render_animation(timeline_df, prediction_df, gif_name, frame_name, title_prefix, frame_step, min_history):
            frame_indices = list(range(min_history, len(timeline_df), frame_step))
            if frame_indices[-1] != len(timeline_df) - 1:
                frame_indices.append(len(timeline_df) - 1)

            gif_path = FIGURES_DIR / gif_name
            frame_path = FIGURES_DIR / frame_name
            window_start = timeline_df["Date"].min().date()
            window_end = timeline_df["Date"].max().date()
            fig, (ax_price, ax_prob) = plt.subplots(
                2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
            )

            def update(frame_pos: int):
                cutoff = frame_indices[frame_pos]
                current = timeline_df.iloc[: cutoff + 1].copy()
                ax_price.clear()
                ax_prob.clear()

                draw_regime_background(ax_price, current, current["GSPC"].min() * 0.98, current["GSPC"].max() * 1.02)
                ax_price.plot(current["Date"], current["GSPC"], color="#111111", linewidth=2.3)
                ax_price.set_title(
                    f"{title_prefix}: {window_start} to {current['Date'].iloc[-1].date()}",
                    fontsize=15,
                )
                ax_price.set_ylabel("S&P 500")
                ax_price.grid(alpha=0.25)

                for model_name, model_df in prediction_df.groupby("model"):
                    model_current = model_df[model_df["Date"] <= current["Date"].iloc[-1]].copy()
                    ax_prob.plot(
                        model_current["Date"],
                        model_current["y_pred_prob"],
                        label=model_name,
                        color=MODEL_COLORS[model_name],
                        linewidth=1,
                    )

                ax_prob.plot(current["Date"], current["y_true"], color="#008000", alpha=0.45, linewidth=1.4, label="True regime")
                # ax_prob.axhline(0.5, color="#777777", linestyle="--", linewidth=1)
                ax_prob.set_ylim(-0.05, 1.05)
                ax_prob.set_ylabel("Bull probability")
                ax_prob.set_xlabel("Date")
                ax_prob.grid(alpha=0.25)
                ax_prob.legend(loc="lower left", ncol=5, fontsize=9)
                ax_prob.xaxis.set_major_locator(mdates.YearLocator())
                ax_prob.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
                ax_prob.grid(False)

                fig.tight_layout()
                return []

            animation = FuncAnimation(fig, update, frames=len(frame_indices), interval=140, blit=False)
            animation.save(gif_path, writer=PillowWriter(fps=8))
            update(len(frame_indices) - 1)
            fig.savefig(frame_path)
            plt.close(fig)
            # return gif_path, frame_path
            return FIGURES_DIR


        recent_start = pd.Timestamp("2025-01-01")
        recent_timeline = timeline[timeline["Date"] >= recent_start].copy().reset_index(drop=True)
        recent_predictions = predictions[predictions["Date"] >= recent_start].copy().reset_index(drop=True)

        full_outputs = render_animation(
            timeline_df=timeline,
            prediction_df=predictions,
            gif_name="regime_detection_full_test_window.gif",
            frame_name="regime_detection_full_test_window_last_frame.png",
            title_prefix="S&P 500 Bull/Bear Regime Detection (Full Test Window)",
            frame_step=10,
            min_history=60,
        )

        recent_outputs = render_animation(
            timeline_df=recent_timeline,
            prediction_df=recent_predictions,
            gif_name="regime_detection_recent_test_period.gif",
            frame_name="regime_detection_recent_test_period_last_frame.png",
            title_prefix="S&P 500 Bull/Bear Regime Detection (Recent Test Period)",
            frame_step=5,
            min_history=30,
        )

        full_outputs, recent_outputs
        """
    ),
]

nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.11",
    },
}

with open("notebooks/06_data_visualisation/animated_regime_detection.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
