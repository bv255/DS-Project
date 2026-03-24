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
        # Model Results Visualisation

        This notebook consolidates the aligned `1day`, `5days`, and `20days` forecasting outputs for:
        - Logistic regression
        - XGBoost
        - PatchTST
        - LSTM

        It reads the exported artifacts under `reports/` and writes refreshed static figures to
        `reports/figures/model_results_visualisation/`.
        """
    ),
    code(
        """
        from pathlib import Path
        import math

        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        import numpy as np
        import pandas as pd
        import seaborn as sns
        from IPython.display import display
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

        sns.set_theme(style="whitegrid", context="talk")
        plt.rcParams["figure.dpi"] = 140
        plt.rcParams["savefig.bbox"] = "tight"


        def find_repo_root(start: Path) -> Path:
            current = start.resolve()
            for candidate in [current, *current.parents]:
                if (candidate / ".git").exists() or (candidate / "README.md").exists():
                    return candidate
            raise FileNotFoundError("Could not locate repo root from notebook location")


        ROOT = find_repo_root(Path.cwd())
        REPORTS_DIR = ROOT / "reports"
        FIGURES_DIR = REPORTS_DIR / "figures" / "model_results_visualisation"
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        HORIZONS = ["1day", "5days", "20days"]
        MODEL_ORDER = ["Logistic", "XGBoost", "PatchTST", "LSTM"]
        MODEL_COLORS = {
            "Logistic": "#355C7D",
            "XGBoost": "#C06C84",
            "PatchTST": "#F67280",
            "LSTM": "#6C5B7B",
        }


        def save_current_figure(name: str) -> Path:
            path = FIGURES_DIR / name
            plt.savefig(path)
            return path
        """
    ),
    code(
        """
        PREDICTION_SPECS = {
            "Logistic": {
                "1day": REPORTS_DIR / "trained_logistic" / "basecase_1day" / "test_predictions.csv",
                "5days": REPORTS_DIR / "trained_logistic" / "basecase_5days" / "test_predictions.csv",
                "20days": REPORTS_DIR / "trained_logistic" / "basecase_20days" / "test_predictions.csv",
            },
            "XGBoost": {
                "1day": REPORTS_DIR / "trained_xgboost" / "basecase_1day" / "test_predictions.csv",
                "5days": REPORTS_DIR / "trained_xgboost" / "basecase_5days" / "test_predictions.csv",
                "20days": REPORTS_DIR / "trained_xgboost" / "basecase_20days" / "test_predictions.csv",
            },
            "PatchTST": {
                "1day": REPORTS_DIR / "trained_patchtst" / "results_lag_1" / "predictions.csv",
                "5days": REPORTS_DIR / "trained_patchtst" / "results_lag_5" / "predictions.csv",
                "20days": REPORTS_DIR / "trained_patchtst" / "results_lag_20" / "predictions.csv",
            },
            "LSTM": {
                "1day": REPORTS_DIR / "trained_ltsm" / "results_lag_1" / "predictions.csv",
                "5days": REPORTS_DIR / "trained_ltsm" / "results_lag_5" / "predictions.csv",
                "20days": REPORTS_DIR / "trained_ltsm" / "results_lag_20" / "predictions.csv",
            },
        }

        SPLIT_LABELS = {model: "test" for model in MODEL_ORDER}


        def normalise_prediction_frame(df: pd.DataFrame, model: str, horizon: str) -> pd.DataFrame:
            frame = df.copy()

            rename_map = {}
            if "date" in frame.columns:
                rename_map["date"] = "Date"
            if "y_prob" in frame.columns:
                rename_map["y_prob"] = "y_pred_prob"
            frame = frame.rename(columns=rename_map)

            required = ["Date", "y_true", "y_pred", "y_pred_prob"]
            missing = [col for col in required if col not in frame.columns]
            if missing:
                raise KeyError(f"{model} {horizon} missing columns: {missing}")

            frame["Date"] = pd.to_datetime(frame["Date"])
            frame["model"] = model
            frame["horizon"] = horizon
            frame["split"] = SPLIT_LABELS[model]
            return frame[["Date", "y_true", "y_pred", "y_pred_prob", "model", "split", "horizon"]]


        def load_all_predictions() -> pd.DataFrame:
            frames = []
            for model, horizon_map in PREDICTION_SPECS.items():
                for horizon, path in horizon_map.items():
                    if not path.exists():
                        raise FileNotFoundError(
                            f"Missing {path}. Rerun the {model} {horizon} notebook export cell to produce test_predictions.csv."
                        )
                    df = pd.read_csv(path)
                    frames.append(normalise_prediction_frame(df, model=model, horizon=horizon))

            combined = pd.concat(frames, ignore_index=True)
            combined["horizon"] = pd.Categorical(combined["horizon"], categories=HORIZONS, ordered=True)
            combined["model"] = pd.Categorical(combined["model"], categories=MODEL_ORDER, ordered=True)
            return combined.sort_values(["horizon", "model", "Date"]).reset_index(drop=True)


        def summarise_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
            rows = []
            for (model, horizon, split), df in predictions.groupby(["model", "horizon", "split"], observed=True):
                rows.append(
                    {
                        "model": model,
                        "horizon": horizon,
                        "split": split,
                        "accuracy": accuracy_score(df["y_true"], df["y_pred"]),
                        "precision": precision_score(df["y_true"], df["y_pred"], zero_division=0),
                        "recall": recall_score(df["y_true"], df["y_pred"], zero_division=0),
                        "f1_score": f1_score(df["y_true"], df["y_pred"], zero_division=0),
                        "roc_auc": roc_auc_score(df["y_true"], df["y_pred_prob"]),
                        "rows": len(df),
                        "start_date": df["Date"].min(),
                        "end_date": df["Date"].max(),
                    }
                )
            metrics = pd.DataFrame(rows)
            metrics["horizon"] = pd.Categorical(metrics["horizon"], categories=HORIZONS, ordered=True)
            metrics["model"] = pd.Categorical(metrics["model"], categories=MODEL_ORDER, ordered=True)
            return metrics.sort_values(["horizon", "model"]).reset_index(drop=True)


        def load_cross_validation_artifacts():
            logistic_test = pd.read_csv(REPORTS_DIR / "trained_logistic" / "cross_validation" / "test_metrics.csv")
            logistic_test["model"] = "Logistic"
            logistic_test["horizon"] = "1day"

            logistic_cv = pd.read_csv(REPORTS_DIR / "trained_logistic" / "cross_validation" / "cv_summary.csv")
            logistic_cv["model"] = "Logistic"
            logistic_cv["horizon"] = "1day"

            xgb_test = pd.read_csv(REPORTS_DIR / "trained_xgboost" / "cross_validation_1day" / "test_metrics.csv")
            xgb_test["model"] = "XGBoost"
            xgb_test["horizon"] = "1day"

            xgb_grid = pd.read_csv(REPORTS_DIR / "trained_xgboost" / "cross_validation_1day" / "cv_summary.csv")
            xgb_grid["model"] = "XGBoost"
            xgb_grid["horizon"] = "1day"

            return logistic_test, logistic_cv, xgb_test, xgb_grid


        def intersect_test_windows(predictions: pd.DataFrame) -> pd.DataFrame:
            aligned_frames = []
            for horizon in HORIZONS:
                subset = predictions[predictions["horizon"] == horizon].copy()
                starts = subset.groupby("model", observed=True)["Date"].min()
                ends = subset.groupby("model", observed=True)["Date"].max()
                common_start = starts.max()
                common_end = ends.min()

                if common_start > common_end:
                    raise ValueError(f"No common test window available for {horizon}")

                aligned = subset[(subset["Date"] >= common_start) & (subset["Date"] <= common_end)].copy()
                aligned["common_start"] = common_start
                aligned["common_end"] = common_end
                aligned_frames.append(aligned)

            combined = pd.concat(aligned_frames, ignore_index=True)
            combined["horizon"] = pd.Categorical(combined["horizon"], categories=HORIZONS, ordered=True)
            combined["model"] = pd.Categorical(combined["model"], categories=MODEL_ORDER, ordered=True)
            return combined.sort_values(["horizon", "model", "Date"]).reset_index(drop=True)


        def load_training_curves():
            xgb_frames = []
            for horizon in HORIZONS:
                df = pd.read_csv(REPORTS_DIR / "trained_xgboost" / f"basecase_{horizon}" / "learning_curve.csv")
                df["model"] = "XGBoost"
                df["horizon"] = horizon
                xgb_frames.append(df)

            patch_frames = []
            lag_map = {"1day": "results_lag_1", "5days": "results_lag_5", "20days": "results_lag_20"}
            for horizon, folder in lag_map.items():
                df = pd.read_csv(REPORTS_DIR / "trained_patchtst" / folder / "training_history.csv")
                df["model"] = "PatchTST"
                df["horizon"] = horizon
                patch_frames.append(df)

            return pd.concat(xgb_frames, ignore_index=True), pd.concat(patch_frames, ignore_index=True)


        def draw_confusion_artifact(ax, title: str, csv_path: Path | None = None, image_path: Path | None = None):
            if csv_path is not None and csv_path.exists():
                df = pd.read_csv(csv_path, index_col=0)
                sns.heatmap(df, annot=True, fmt="g", cmap="Blues", cbar=False, ax=ax)
                ax.set_title(title)
                ax.set_xlabel("")
                ax.set_ylabel("")
                return

            if image_path is not None and image_path.exists():
                img = mpimg.imread(image_path)
                ax.imshow(img)
                ax.set_title(title)
                ax.axis("off")
                return

            ax.text(0.5, 0.5, "Missing artifact", ha="center", va="center")
            ax.set_title(title)
            ax.axis("off")


        predictions = load_all_predictions()
        aligned_predictions = intersect_test_windows(predictions)
        metrics = summarise_metrics(aligned_predictions)
        logistic_cv_test, logistic_cv_folds, xgb_cv_test, xgb_cv_grid = load_cross_validation_artifacts()
        xgb_learning_curves, patchtst_training = load_training_curves()

        display(metrics)
        """
    ),
    md(
        """
        ## Unified Forecast Metrics

        These metrics are recomputed directly from the exported held-out `test_predictions.csv` or equivalent
        files and restricted to the common overlapping test window for each horizon.
        """
    ),
    code(
        """
        summary_cols = ["model", "split", "horizon", "accuracy", "precision", "recall", "f1_score", "roc_auc", "rows", "start_date", "end_date"]
        display(metrics[summary_cols].style.format({col: "{:.3f}" for col in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]}))
        """
    ),
    code(
        """
        plot_df = metrics.melt(
            id_vars=["model", "horizon", "split"],
            value_vars=["accuracy", "precision", "recall", "roc_auc"],
            var_name="metric",
            value_name="value",
        )

        fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=True, sharey=True)
        for ax, metric_name in zip(axes.flat, ["accuracy", "precision", "recall", "roc_auc"]):
            subset = plot_df[plot_df["metric"] == metric_name]
            sns.barplot(
                data=subset,
                x="horizon",
                y="value",
                hue="model",
                hue_order=MODEL_ORDER,
                palette=MODEL_COLORS,
                ax=ax,
            )
            ax.set_title(metric_name.replace("_", " ").title())
            ax.set_xlabel("")
            ax.set_ylabel("Score")
            ax.set_ylim(0, 1.0)
            if ax is axes.flat[0]:
                ax.legend(title="")
            else:
                ax.get_legend().remove()

        fig.suptitle("Forecast Metrics Across Aligned Horizons", y=1.02, fontsize=18)
        fig.text(0.5, -0.01, "Logistic/XGBoost use validation exports; PatchTST/LSTM use test exports.", ha="center", fontsize=11)
        plt.tight_layout()
        save_current_figure("aligned_forecast_metrics.png")
        plt.show()
        """
    ),
    md(
        """
        ## Horizon-Wise Probability Timelines

        The first view keeps a compact recent slice for readability. The second view shows the full aligned test
        window for each horizon using the same visual structure.
        """
    ),
    code(
        """
        fig, axes = plt.subplots(len(HORIZONS), 1, figsize=(16, 13), sharex=False)
        if len(HORIZONS) == 1:
            axes = [axes]
        
        for ax, horizon in zip(axes, HORIZONS):
            subset = aligned_predictions[aligned_predictions["horizon"] == horizon].copy()
            for model in MODEL_ORDER:
                df = subset[subset["model"] == model].sort_values("Date").tail(180)
                ax.plot(df["Date"], df["y_pred_prob"], label=model, color=MODEL_COLORS[model], linewidth=2)
        
            truth = subset[subset["model"] == "Logistic"].sort_values("Date").tail(180)
            ax.plot(truth["Date"], truth["y_true"], color="#008000", alpha=0.5, linewidth=1.8, label="True regime")
            # ax.axhline(0.5, color="#888888", linestyle="--", linewidth=1)
            ax.set_ylim(-0.05, 1.05)
            common_start = subset["common_start"].iloc[0].date()
            common_end = subset["common_end"].iloc[0].date()
            ax.set_title(f"{horizon} predicted bull probability ({common_start} to {common_end})")
            ax.set_ylabel("Probability")
            ax.legend(ncol=5, fontsize=12, loc="lower left")
            ax.grid(False)
        
        axes[-1].set_xlabel("Date")
        fig.suptitle("Prediction Timelines on Common Test Windows", y=1.01, fontsize=18)
        plt.tight_layout()
        save_current_figure("prediction_timelines_by_horizon.png")
        plt.show()
        """
    ),
    code(
        """
        fig, axes = plt.subplots(len(HORIZONS), 1, figsize=(16, 13), sharex=False)
        if len(HORIZONS) == 1:
            axes = [axes]
        
        for ax, horizon in zip(axes, HORIZONS):
            subset = aligned_predictions[aligned_predictions["horizon"] == horizon].copy()
            for model in MODEL_ORDER:
                df = subset[subset["model"] == model].sort_values("Date")
                ax.plot(df["Date"], df["y_pred_prob"], label=model, color=MODEL_COLORS[model], linewidth=1)
        
            truth = subset[subset["model"] == "Logistic"].sort_values("Date")
            ax.plot(truth["Date"], truth["y_true"], color="#008000", alpha=0.5, linewidth=1.8, label="True regime")
            # ax.axhline(0.5, color="#888888", linestyle="--", linewidth=1)
            ax.set_ylim(-0.05, 1.05)
            common_start = subset["common_start"].iloc[0].date()
            common_end = subset["common_end"].iloc[0].date()
            ax.set_title(f"{horizon} full test window ({common_start} to {common_end})")
            ax.set_ylabel("Probability")
            ax.legend(ncol=5, fontsize=9, loc="upper left")
            ax.grid(False)
        
        axes[-1].set_xlabel("Date")
        fig.suptitle("Prediction Timelines Across the Full Common Test Window", y=1.01, fontsize=18)
        plt.tight_layout()
        save_current_figure("prediction_timelines_full_test_window.png")
        plt.show()
        """
    ),
    md(
        """
        ## Model Ranking by Horizon

        This view focuses on the strongest headline metric for classification quality.
        """
    ),
    code(
        """
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True)
        for ax, horizon in zip(axes, HORIZONS):
            subset = metrics[metrics["horizon"] == horizon].sort_values("roc_auc", ascending=True)
            sns.barplot(
                data=subset,
                x="roc_auc",
                y="model",
                hue="model",
                hue_order=MODEL_ORDER,
                palette=MODEL_COLORS,
                legend=False,
                ax=ax,
            )
            ax.set_title(f"{horizon} ROC-AUC")
            ax.set_xlabel("ROC-AUC")
            ax.set_ylabel("")
            ax.set_xlim(0, 1.0)

        plt.tight_layout()
        save_current_figure("roc_auc_ranking_by_horizon.png")
        plt.show()
        """
    ),
    md(
        """
        ## Class Balance in the Common Test Window

        This chart shows the bull/bear composition in the aligned comparison window for each horizon.
        It helps explain why some horizons are harder to model and why precision/recall tradeoffs can shift.
        """
    ),
    code(
        """
        class_balance = (
            aligned_predictions[aligned_predictions["model"] == "Logistic"][["Date", "horizon", "y_true"]]
            .drop_duplicates(subset=["Date", "horizon"])
            .assign(regime=lambda df: np.where(df["y_true"] == 1, "bull", "bear"))
            .groupby(["horizon", "regime"], observed=True)
            .size()
            .reset_index(name="count")
        )
        class_balance["horizon"] = pd.Categorical(class_balance["horizon"], categories=HORIZONS, ordered=True)

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
        regime_palette = {"bull": "#2E8B57", "bear": "#B22222"}

        for ax, horizon in zip(axes, HORIZONS):
            subset = class_balance[class_balance["horizon"] == horizon].copy()
            total = subset["count"].sum()
            subset["share"] = subset["count"] / total
            sns.barplot(
                data=subset,
                x="regime",
                y="share",
                hue="regime",
                palette=regime_palette,
                legend=False,
                ax=ax,
            )
            ax.set_title(horizon)
            ax.set_xlabel("")
            ax.set_ylabel("Share of aligned test window")
            ax.set_ylim(0, 1.0)

        fig.suptitle("Regime Class Balance by Horizon", y=1.03, fontsize=18)
        plt.tight_layout()
        save_current_figure("class_balance_by_horizon.png")
        plt.show()
        """
    ),
    md(
        """
        ## 1-Day Feature Signal Views

        The linear and tree baselines still benefit from a direct feature-level readout.
        """
    ),
    code(
        """
        logistic_coef = pd.read_csv(REPORTS_DIR / "trained_logistic" / "basecase_1day" / "coefficients.csv")
        logistic_coef = logistic_coef[logistic_coef["feature"] != "const"].copy()
        logistic_coef["abs_coefficient"] = logistic_coef["coefficient"].abs()
        logistic_top = logistic_coef.nlargest(10, "abs_coefficient").sort_values("coefficient")

        xgb_importance = pd.read_csv(REPORTS_DIR / "trained_xgboost" / "basecase_1day" / "feature_importance.csv")
        xgb_top = xgb_importance.nlargest(10, "importance").sort_values("importance")

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        sns.barplot(data=logistic_top, x="coefficient", y="feature", color=MODEL_COLORS["Logistic"], ax=axes[0])
        axes[0].set_title("Logistic 1-day coefficients")
        axes[0].set_xlabel("Coefficient")
        axes[0].set_ylabel("")

        sns.barplot(data=xgb_top, x="importance", y="feature", color=MODEL_COLORS["XGBoost"], ax=axes[1])
        axes[1].set_title("XGBoost 1-day feature importance")
        axes[1].set_xlabel("Importance")
        axes[1].set_ylabel("")

        plt.tight_layout()
        save_current_figure("model_signal_views_1day.png")
        plt.show()
        """
    ),
    md(
        """
        ## Training Diagnostics

        Only XGBoost and PatchTST currently export training-curve style artifacts in a structured CSV format.
        """
    ),
    code(
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        for idx, horizon in enumerate(HORIZONS):
            xgb_df = xgb_learning_curves[xgb_learning_curves["horizon"] == horizon]
            patch_df = patchtst_training[patchtst_training["horizon"] == horizon]

            ax = axes[0, idx]
            ax.plot(xgb_df["iteration"], xgb_df["train_logloss"], label="Train", color=MODEL_COLORS["Logistic"])
            ax.plot(xgb_df["iteration"], xgb_df["val_logloss"], label="Validation", color=MODEL_COLORS["XGBoost"])
            ax.set_title(f"XGBoost {horizon}")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Log loss")
            if idx == 0:
                ax.legend()

            ax = axes[1, idx]
            ax.plot(patch_df["epoch"], patch_df["train_loss"], label="Train loss", color=MODEL_COLORS["PatchTST"])
            ax2 = ax.twinx()
            ax2.plot(patch_df["epoch"], patch_df["val_auc"], label="Validation AUC", color=MODEL_COLORS["LSTM"])
            ax.set_title(f"PatchTST {horizon}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Train loss")
            ax2.set_ylabel("Validation AUC")

        fig.suptitle("Training Diagnostics by Horizon", y=1.02, fontsize=18)
        plt.tight_layout()
        save_current_figure("training_diagnostics.png")
        plt.show()
        """
    ),
    md(
        """
        ## Cross-Validation Artifacts

        Logistic regression still exports fold-level cross-validation metrics. The current XGBoost export for
        `1day` is a grid-search summary, so it is shown separately rather than forced into a fold chart.
        """
    ),
    code(
        """
        logistic_cv_plot = logistic_cv_folds.copy()
        logistic_cv_plot["fold"] = pd.to_numeric(logistic_cv_plot["fold"], errors="coerce")
        logistic_cv_plot = logistic_cv_plot.dropna(subset=["fold"]).copy()
        logistic_cv_plot["fold"] = logistic_cv_plot["fold"].astype(int)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        sns.barplot(
            data=pd.concat([logistic_cv_test, xgb_cv_test], ignore_index=True),
            x="model",
            y="roc_auc",
            hue="model",
            palette=MODEL_COLORS,
            legend=False,
            ax=axes[0],
        )
        axes[0].set_title("1-day CV held-out ROC-AUC")
        axes[0].set_xlabel("")
        axes[0].set_ylabel("ROC-AUC")
        axes[0].set_ylim(0, 1.0)

        sns.lineplot(
            data=logistic_cv_plot,
            x="fold",
            y="accuracy",
            marker="o",
            color=MODEL_COLORS["Logistic"],
            ax=axes[1],
        )
        axes[1].set_title("Logistic fold accuracy")
        axes[1].set_xlabel("Fold")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_ylim(0, 1.0)

        xgb_grid_top = xgb_cv_grid.sort_values("mean_test_score", ascending=False).head(8).copy()
        xgb_grid_top["label"] = [f"Rank {rank}" for rank in xgb_grid_top["rank_test_score"]]
        sns.barplot(data=xgb_grid_top, x="mean_test_score", y="label", color=MODEL_COLORS["XGBoost"], ax=axes[2])
        axes[2].set_title("XGBoost grid-search top scores")
        axes[2].set_xlabel("Mean CV score")
        axes[2].set_ylabel("")

        plt.tight_layout()
        save_current_figure("cross_validation_overview.png")
        plt.show()
        """
    ),
    md(
        """
        ## Confusion Matrix Gallery

        The forecasting artifacts are now aligned enough that the confusion matrices can be shown in a single grid.
        """
    ),
    code(
        """
        confusion_specs = [
            ("Logistic 1-day", REPORTS_DIR / "trained_logistic" / "basecase_1day" / "confusion_matrix.csv", None),
            ("Logistic 5-day", REPORTS_DIR / "trained_logistic" / "basecase_5days" / "confusion_matrix.csv", None),
            ("Logistic 20-day", REPORTS_DIR / "trained_logistic" / "basecase_20days" / "confusion_matrix.csv", None),
            ("XGBoost 1-day", REPORTS_DIR / "trained_xgboost" / "basecase_1day" / "confusion_matrix.csv", None),
            ("XGBoost 5-day", REPORTS_DIR / "trained_xgboost" / "basecase_5days" / "confusion_matrix.csv", None),
            ("XGBoost 20-day", REPORTS_DIR / "trained_xgboost" / "basecase_20days" / "confusion_matrix.csv", None),
            ("PatchTST 1-day", REPORTS_DIR / "trained_patchtst" / "results_lag_1" / "confusion_matrix.csv", None),
            ("PatchTST 5-day", REPORTS_DIR / "trained_patchtst" / "results_lag_5" / "confusion_matrix.csv", None),
            ("PatchTST 20-day", REPORTS_DIR / "trained_patchtst" / "results_lag_20" / "confusion_matrix.csv", None),
            ("LSTM 1-day", None, REPORTS_DIR / "trained_ltsm" / "results_lag_1" / "confusion_matrix.png"),
            ("LSTM 5-day", None, REPORTS_DIR / "trained_ltsm" / "results_lag_5" / "confusion_matrix.png"),
            ("LSTM 20-day", None, REPORTS_DIR / "trained_ltsm" / "results_lag_20" / "confusion_matrix.png"),
        ]

        fig, axes = plt.subplots(4, 3, figsize=(16, 18))
        for ax, (title, csv_path, image_path) in zip(axes.ravel(), confusion_specs):
            draw_confusion_artifact(ax, title, csv_path=csv_path, image_path=image_path)

        plt.suptitle("Confusion Matrix Gallery", y=1.01, fontsize=18)
        plt.tight_layout()
        save_current_figure("confusion_matrix_gallery.png")
        plt.show()
        """
    ),
    md(
        """
        ## Saved Diagnostic Gallery

        These are the original exported single-model diagnostic views, preserved here as a quick report appendix.
        """
    ),
    code(
        """
        image_specs = [
            ("Logistic 1-day timeline", REPORTS_DIR / "trained_logistic" / "basecase_1day" / "prediction_timeline.png"),
            ("XGBoost 1-day timeline", REPORTS_DIR / "trained_xgboost" / "basecase_1day" / "prediction_timeline.png"),
            ("PatchTST 1-day prediction view", REPORTS_DIR / "trained_patchtst" / "results_lag_1" / "prediction_visualisation.png"),
            ("LSTM 1-day training curves", REPORTS_DIR / "trained_ltsm" / "results_lag_1" / "training_curves.png"),
            ("PatchTST 5-day prediction view", REPORTS_DIR / "trained_patchtst" / "results_lag_5" / "prediction_visualisation.png"),
            ("LSTM 20-day training curves", REPORTS_DIR / "trained_ltsm" / "results_lag_20" / "training_curves.png"),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        for ax, (title, image_path) in zip(axes.ravel(), image_specs):
            if image_path.exists():
                img = mpimg.imread(image_path)
                ax.imshow(img)
                ax.set_title(title)
                ax.axis("off")
            else:
                ax.text(0.5, 0.5, "Missing artifact", ha="center", va="center")
                ax.set_title(title)
                ax.axis("off")

        plt.suptitle("Saved Diagnostic Gallery", y=1.02, fontsize=18)
        plt.tight_layout()
        save_current_figure("saved_diagnostic_gallery.png")
        plt.show()
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

with open("notebooks/06_data_visualisation/model_results_visualisation.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
