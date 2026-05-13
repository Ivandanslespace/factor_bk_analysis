import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class FactorStabilityAnalyzer:
    """
    Outil d'analyse pour des facteurs et combinaisons de facteurs.
    Le DataFrame d'entrée doit contenir :
    - un index de dates
    - des colonnes = noms de facteurs
    - des valeurs = ratio cumulatif vs benchmark (strictement positif)
    """

    def __init__(self, df: pd.DataFrame, ann_factor: int = 252):
        # Initialisation de l'objet
        self.ann_factor = ann_factor
        self.df = self._prepare_df(df)
        self.factor_map = self._build_factor_map()

    @staticmethod
    def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
        # Nettoyage et harmonisation du DataFrame
        x = df.copy()
        x.index = pd.to_datetime(x.index)
        x = x.sort_index()
        x = x.replace([np.inf, -np.inf], np.nan)
        x = x.ffill()
        x = x.where(x > 0)
        x = x.dropna(axis=1, how="all")
        return x

    def _build_factor_map(self) -> pd.DataFrame:
        # Construction d'une table descriptive des facteurs
        records = []
        for col in self.df.columns:
            is_combo = "_x_" in col
            parts = col.split("_x_") if is_combo else [col]
            records.append(
                {
                    "factor": col,
                    "is_combo": is_combo,
                    "n_parts": len(parts),
                    "parts": parts,
                }
            )
        return pd.DataFrame(records).set_index("factor")

    @staticmethod
    def _max_drawdown_from_array(arr: np.ndarray) -> float:
        # Calcul du drawdown maximal d'une série cumulative
        x = np.asarray(arr, dtype=float)
        if len(x) < 2 or np.any(~np.isfinite(x)) or x[0] <= 0:
            return np.nan
        x = x / x[0]
        peak = np.maximum.accumulate(x)
        dd = x / peak - 1.0
        return float(np.min(dd))

    @staticmethod
    def _ulcer_index_from_array(arr: np.ndarray) -> float:
        # Calcul de l'Ulcer Index
        x = np.asarray(arr, dtype=float)
        if len(x) < 2 or np.any(~np.isfinite(x)) or x[0] <= 0:
            return np.nan
        x = x / x[0]
        peak = np.maximum.accumulate(x)
        dd = x / peak - 1.0
        return float(np.sqrt(np.mean((dd * 100.0) ** 2)))

    @staticmethod
    def _monotonicity_from_array(arr: np.ndarray) -> float:
        # Part des variations journalières positives en log
        x = np.asarray(arr, dtype=float)
        if len(x) < 3 or np.any(~np.isfinite(x)) or np.any(x <= 0):
            return np.nan
        r = np.diff(np.log(x))
        if len(r) == 0:
            return np.nan
        return float(np.mean(r > 0))

    @staticmethod
    def _new_high_share_from_array(arr: np.ndarray) -> float:
        # Part du temps passé sur un plus haut historique
        x = np.asarray(arr, dtype=float)
        if len(x) < 2 or np.any(~np.isfinite(x)):
            return np.nan
        peak = np.maximum.accumulate(x)
        return float(np.mean(np.isclose(x, peak, rtol=1e-10, atol=1e-12)))

    @staticmethod
    def _trend_slope_r2_from_array(arr: np.ndarray):
        # Régression linéaire simple sur le log de la courbe
        x = np.asarray(arr, dtype=float)
        if len(x) < 5 or np.any(~np.isfinite(x)) or np.any(x <= 0):
            return np.nan, np.nan
        y = np.log(x)
        t = np.arange(len(y), dtype=float)
        slope, intercept = np.polyfit(t, y, 1)
        y_hat = slope * t + intercept
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return float(slope), float(r2)

    @staticmethod
    def _safe_rank_pct(s: pd.Series, ascending: bool = True) -> pd.Series:
        # Conversion d'une métrique en percentile de rang
        if ascending:
            return s.rank(pct=True, ascending=True)
        return s.rank(pct=True, ascending=False)

    def _series(self, col: str) -> pd.Series:
        # Extraction propre d'une série
        return self.df[col].dropna()

    def full_period_stats(self) -> pd.DataFrame:
        # Statistiques globales sur toute la période
        rows = []
        for col in self.df.columns:
            s = self._series(col)
            if len(s) < 10:
                continue

            rel = s / s.iloc[0]
            log_ret = np.log(rel).diff().dropna()
            n = len(log_ret)

            total_return = rel.iloc[-1] - 1.0
            cagr = rel.iloc[-1] ** (self.ann_factor / max(n, 1)) - 1.0 if n > 0 else np.nan
            ann_vol = log_ret.std() * np.sqrt(self.ann_factor) if len(log_ret) > 1 else np.nan

            downside = log_ret[log_ret < 0]
            downside_std = downside.std() * np.sqrt(self.ann_factor) if len(downside) > 1 else np.nan
            sortino = (log_ret.mean() * self.ann_factor / downside_std) if pd.notna(downside_std) and downside_std > 0 else np.nan

            mdd = self._max_drawdown_from_array(rel.values)
            ulcer = self._ulcer_index_from_array(rel.values)
            calmar = cagr / abs(mdd) if pd.notna(mdd) and mdd < 0 else np.nan
            monotonicity = self._monotonicity_from_array(rel.values)
            new_high_share = self._new_high_share_from_array(rel.values)
            slope, r2 = self._trend_slope_r2_from_array(rel.values)

            rows.append(
                {
                    "factor": col,
                    "is_combo": self.factor_map.loc[col, "is_combo"],
                    "n_obs": len(s),
                    "start": s.index.min(),
                    "end": s.index.max(),
                    "final_ratio": s.iloc[-1],
                    "total_return": total_return,
                    "cagr": cagr,
                    "ann_vol_log": ann_vol,
                    "max_drawdown": mdd,
                    "ulcer_index": ulcer,
                    "sortino_log": sortino,
                    "calmar": calmar,
                    "monotonicity": monotonicity,
                    "new_high_share": new_high_share,
                    "trend_slope_log_daily": slope,
                    "trend_r2": r2,
                }
            )

        out = pd.DataFrame(rows).set_index("factor").sort_values("cagr", ascending=False)
        return out

    def rolling_analysis(self, window: int = 252):
        # Analyse glissante des performances
        rolling_return = pd.DataFrame(index=self.df.index, columns=self.df.columns, dtype=float)
        rolling_cagr = pd.DataFrame(index=self.df.index, columns=self.df.columns, dtype=float)
        rolling_mdd = pd.DataFrame(index=self.df.index, columns=self.df.columns, dtype=float)
        rolling_mono = pd.DataFrame(index=self.df.index, columns=self.df.columns, dtype=float)
        rolling_new_high = pd.DataFrame(index=self.df.index, columns=self.df.columns, dtype=float)

        for col in self.df.columns:
            s = self._series(col)
            if len(s) <= window:
                continue

            rolling_return[col] = s / s.shift(window) - 1.0
            rolling_cagr[col] = (s / s.shift(window)) ** (self.ann_factor / window) - 1.0

            rolling_mdd[col] = s.rolling(window).apply(
                lambda x: self._max_drawdown_from_array(np.asarray(x, dtype=float)),
                raw=True,
            )
            rolling_mono[col] = s.rolling(window).apply(
                lambda x: self._monotonicity_from_array(np.asarray(x, dtype=float)),
                raw=True,
            )
            rolling_new_high[col] = s.rolling(window).apply(
                lambda x: self._new_high_share_from_array(np.asarray(x, dtype=float)),
                raw=True,
            )

        return {
            "rolling_return": rolling_return,
            "rolling_cagr": rolling_cagr,
            "rolling_mdd": rolling_mdd,
            "rolling_monotonicity": rolling_mono,
            "rolling_new_high": rolling_new_high,
        }

    def stability_ranking(
        self,
        windows=(126, 252, 504),
        drawdown_limit=0.15,
        min_obs=252,
    ) -> pd.DataFrame:
        # Classement des facteurs selon la stabilité multi-fenêtre
        base = self.full_period_stats()
        base = base[base["n_obs"] >= min_obs].copy()

        all_scores = []

        for w in windows:
            ra = self.rolling_analysis(window=w)
            rr = ra["rolling_return"]
            rc = ra["rolling_cagr"]
            rm = ra["rolling_mdd"]
            rmo = ra["rolling_monotonicity"]
            rnh = ra["rolling_new_high"]

            rows = []
            for col in base.index:
                x_rr = rr[col].dropna()
                x_rc = rc[col].dropna()
                x_rm = rm[col].dropna()
                x_rmo = rmo[col].dropna()
                x_rnh = rnh[col].dropna()

                if len(x_rr) == 0:
                    continue

                rows.append(
                    {
                        "factor": col,
                        f"win_{w}_pct_positive": (x_rr > 0).mean(),
                        f"win_{w}_pct_cagr_positive": (x_rc > 0).mean() if len(x_rc) else np.nan,
                        f"win_{w}_worst_return": x_rr.min(),
                        f"win_{w}_p10_return": x_rr.quantile(0.10),
                        f"win_{w}_median_cagr": x_rc.median() if len(x_rc) else np.nan,
                        f"win_{w}_median_mdd": x_rm.median() if len(x_rm) else np.nan,
                        f"win_{w}_low_dd_share": (x_rm > -drawdown_limit).mean() if len(x_rm) else np.nan,
                        f"win_{w}_median_monotonicity": x_rmo.median() if len(x_rmo) else np.nan,
                        f"win_{w}_median_new_high": x_rnh.median() if len(x_rnh) else np.nan,
                    }
                )

            tmp = pd.DataFrame(rows).set_index("factor")
            all_scores.append(tmp)

        if len(all_scores) == 0:
            return base

        score_df = pd.concat(all_scores, axis=1)
        out = base.join(score_df, how="left")

        # Score agrégé : plus il est élevé, plus le facteur est "stable"
        score_parts = []

        for w in windows:
            cols_needed = [
                f"win_{w}_pct_positive",
                f"win_{w}_p10_return",
                f"win_{w}_median_mdd",
                f"win_{w}_low_dd_share",
                f"win_{w}_median_monotonicity",
                f"win_{w}_median_new_high",
            ]
            existing = [c for c in cols_needed if c in out.columns]
            if len(existing) == 0:
                continue

            part = pd.DataFrame(index=out.index)
            if f"win_{w}_pct_positive" in out.columns:
                part["a"] = self._safe_rank_pct(out[f"win_{w}_pct_positive"], ascending=True)
            if f"win_{w}_p10_return" in out.columns:
                part["b"] = self._safe_rank_pct(out[f"win_{w}_p10_return"], ascending=True)
            if f"win_{w}_median_mdd" in out.columns:
                part["c"] = self._safe_rank_pct(out[f"win_{w}_median_mdd"], ascending=True)
            if f"win_{w}_low_dd_share" in out.columns:
                part["d"] = self._safe_rank_pct(out[f"win_{w}_low_dd_share"], ascending=True)
            if f"win_{w}_median_monotonicity" in out.columns:
                part["e"] = self._safe_rank_pct(out[f"win_{w}_median_monotonicity"], ascending=True)
            if f"win_{w}_median_new_high" in out.columns:
                part["f"] = self._safe_rank_pct(out[f"win_{w}_median_new_high"], ascending=True)

            out[f"stability_score_{w}"] = part.mean(axis=1)
            score_parts.append(out[f"stability_score_{w}"])

        # Score global long terme
        if len(score_parts) > 0:
            out["stability_score"] = pd.concat(score_parts, axis=1).mean(axis=1)
        else:
            out["stability_score"] = np.nan

        # Bonus pour une courbe globale propre
        out["stability_score"] = (
            0.70 * out["stability_score"].fillna(0)
            + 0.10 * self._safe_rank_pct(out["monotonicity"], ascending=True).fillna(0)
            + 0.10 * self._safe_rank_pct(out["trend_r2"], ascending=True).fillna(0)
            + 0.10 * self._safe_rank_pct(out["calmar"], ascending=True).fillna(0)
        )

        out = out.sort_values("stability_score", ascending=False)
        return out

    def yearly_returns(self) -> pd.DataFrame:
        # Rendement calendaire annuel du ratio cumulatif
        yearly_level = self.df.resample("Y").last()
        yearly_ret = yearly_level.pct_change()
        yearly_ret.index = yearly_ret.index.year
        return yearly_ret

    def horizon_success_matrix(self, horizons=(63, 126, 252, 504)) -> pd.DataFrame:
        # Part des fenêtres glissantes positives selon plusieurs horizons
        rows = []
        for col in self.df.columns:
            s = self._series(col)
            if len(s) < max(horizons) + 5:
                continue

            record = {"factor": col}
            for h in horizons:
                r = (s / s.shift(h) - 1.0).dropna()
                record[f"positive_rate_{h}d"] = (r > 0).mean() if len(r) else np.nan
                record[f"p10_return_{h}d"] = r.quantile(0.10) if len(r) else np.nan
                record[f"worst_return_{h}d"] = r.min() if len(r) else np.nan
            rows.append(record)

        return pd.DataFrame(rows).set_index("factor")

    def start_date_robustness(self, min_horizon: int = 252) -> pd.DataFrame:
        # Robustesse du choix de la date de départ jusqu'à aujourd'hui
        rows = []
        for col in self.df.columns:
            s = self._series(col)
            if len(s) <= min_horizon:
                continue

            ann_returns = []
            total_returns = []

            for i in range(0, len(s) - min_horizon):
                sub = s.iloc[i:]
                n = len(sub) - 1
                if n < min_horizon:
                    continue
                rel = sub.iloc[-1] / sub.iloc[0]
                total_returns.append(rel - 1.0)
                ann_returns.append(rel ** (self.ann_factor / n) - 1.0)

            if len(ann_returns) == 0:
                continue

            ann_returns = pd.Series(ann_returns)
            total_returns = pd.Series(total_returns)

            rows.append(
                {
                    "factor": col,
                    "start_date_win_rate_to_today": (total_returns > 0).mean(),
                    "start_date_p10_ann_return_to_today": ann_returns.quantile(0.10),
                    "start_date_median_ann_return_to_today": ann_returns.median(),
                    "start_date_worst_ann_return_to_today": ann_returns.min(),
                    "start_date_p10_total_return_to_today": total_returns.quantile(0.10),
                    "start_date_worst_total_return_to_today": total_returns.min(),
                }
            )

        out = pd.DataFrame(rows).set_index("factor")
        return out.sort_values("start_date_win_rate_to_today", ascending=False)

    def synergy_analysis(self, min_obs: int = 252, window_for_path: int = 252) -> pd.DataFrame:
        # Analyse de l'effet de synergie des facteurs combinés
        records = []

        for combo in self.factor_map.index[self.factor_map["is_combo"]]:
            parts = self.factor_map.loc[combo, "parts"]

            # Vérification de l'existence des facteurs simples
            if not all(p in self.df.columns for p in parts):
                continue

            aligned = pd.concat(
                [self.df[combo]] + [self.df[p] for p in parts],
                axis=1,
                join="inner",
            ).dropna()

            if len(aligned) < min_obs:
                continue

            combo_s = aligned.iloc[:, 0]
            part_df = aligned.iloc[:, 1:]
            part_names = list(part_df.columns)

            # Baseline géométrique : moyenne des logs
            geo_baseline = np.exp(np.log(part_df).mean(axis=1))

            # Baseline "meilleur composant"
            best_baseline = part_df.max(axis=1)

            combo_rel = combo_s / combo_s.iloc[0]
            geo_rel = geo_baseline / geo_baseline.iloc[0]
            best_rel = best_baseline / best_baseline.iloc[0]

            combo_mdd = self._max_drawdown_from_array(combo_rel.values)
            combo_mono = self._monotonicity_from_array(combo_rel.values)

            part_mdds = []
            part_monos = []
            for p in part_names:
                p_rel = part_df[p] / part_df[p].iloc[0]
                part_mdds.append(self._max_drawdown_from_array(p_rel.values))
                part_monos.append(self._monotonicity_from_array(p_rel.values))

            avg_part_mdd = np.nanmean(part_mdds)
            avg_part_mono = np.nanmean(part_monos)

            edge_terminal_vs_geo_log = np.log(combo_rel.iloc[-1]) - np.log(geo_rel.iloc[-1])
            edge_terminal_vs_best_log = np.log(combo_rel.iloc[-1]) - np.log(best_rel.iloc[-1])

            share_above_geo = (combo_rel > geo_rel).mean()
            share_above_best = (combo_rel > best_rel).mean()

            # Lift de stabilité sur fenêtre glissante
            combo_roll = (combo_s / combo_s.shift(window_for_path) - 1.0).dropna()
            parts_roll = []
            for p in part_names:
                r = (part_df[p] / part_df[p].shift(window_for_path) - 1.0).dropna()
                parts_roll.append(r)

            avg_positive_rate_parts = np.nan
            if len(parts_roll) > 0:
                tmp = pd.concat(parts_roll, axis=1)
                avg_positive_rate_parts = (tmp > 0).mean().mean()

            combo_positive_rate = (combo_roll > 0).mean() if len(combo_roll) else np.nan

            records.append(
                {
                    "combo": combo,
                    "parts": " + ".join(part_names),
                    "n_parts": len(part_names),
                    "n_obs": len(aligned),
                    "combo_final_ratio": combo_s.iloc[-1],
                    "geo_baseline_final_ratio": geo_baseline.iloc[-1],
                    "best_single_final_ratio": best_baseline.iloc[-1],
                    "edge_terminal_vs_geo_log": edge_terminal_vs_geo_log,
                    "edge_terminal_vs_best_log": edge_terminal_vs_best_log,
                    "share_above_geo_path": share_above_geo,
                    "share_above_best_path": share_above_best,
                    "combo_monotonicity": combo_mono,
                    "avg_parts_monotonicity": avg_part_mono,
                    "monotonicity_lift": combo_mono - avg_part_mono,
                    "combo_max_drawdown": combo_mdd,
                    "avg_parts_max_drawdown": avg_part_mdd,
                    "drawdown_lift": combo_mdd - avg_part_mdd,
                    f"combo_positive_rate_{window_for_path}d": combo_positive_rate,
                    f"avg_parts_positive_rate_{window_for_path}d": avg_positive_rate_parts,
                    f"positive_rate_lift_{window_for_path}d": combo_positive_rate - avg_positive_rate_parts if pd.notna(combo_positive_rate) and pd.notna(avg_positive_rate_parts) else np.nan,
                }
            )

        out = pd.DataFrame(records)
        if len(out) == 0:
            return out

        out = out.set_index("combo")

        # Score synthétique de synergie
        out["synergy_score"] = (
            0.25 * self._safe_rank_pct(out["edge_terminal_vs_geo_log"], ascending=True).fillna(0)
            + 0.20 * self._safe_rank_pct(out["share_above_geo_path"], ascending=True).fillna(0)
            + 0.15 * self._safe_rank_pct(out["share_above_best_path"], ascending=True).fillna(0)
            + 0.15 * self._safe_rank_pct(out["monotonicity_lift"], ascending=True).fillna(0)
            + 0.15 * self._safe_rank_pct(out["drawdown_lift"], ascending=True).fillna(0)
            + 0.10 * self._safe_rank_pct(out[f"positive_rate_lift_{window_for_path}d"], ascending=True).fillna(0)
        )

        out = out.sort_values("synergy_score", ascending=False)
        return out

    def build_master_table(
        self,
        windows=(126, 252, 504),
        drawdown_limit=0.15,
        min_obs=252,
        min_horizon=252,
    ) -> pd.DataFrame:
        # Fusion de toutes les tables importantes
        stability = self.stability_ranking(
            windows=windows,
            drawdown_limit=drawdown_limit,
            min_obs=min_obs,
        )
        start_rob = self.start_date_robustness(min_horizon=min_horizon)
        horizon = self.horizon_success_matrix(horizons=windows)

        master = stability.join(start_rob, how="left").join(horizon, how="left")

        # Score final orienté "stabilité avant performance brute"
        master["robust_uptrend_score"] = (
            0.45 * self._safe_rank_pct(master["stability_score"], ascending=True).fillna(0)
            + 0.20 * self._safe_rank_pct(master["start_date_win_rate_to_today"], ascending=True).fillna(0)
            + 0.15 * self._safe_rank_pct(master["start_date_p10_ann_return_to_today"], ascending=True).fillna(0)
            + 0.10 * self._safe_rank_pct(master["calmar"], ascending=True).fillna(0)
            + 0.10 * self._safe_rank_pct(master["trend_r2"], ascending=True).fillna(0)
        )

        master = master.sort_values("robust_uptrend_score", ascending=False)
        return master

    def plot_top_stable_factors(self, master_table: pd.DataFrame, top_n: int = 12, figsize=(14, 7)):
        # Graphique des meilleurs facteurs stables
        top_cols = master_table.head(top_n).index.tolist()

        plt.figure(figsize=figsize)
        for col in top_cols:
            s = self._series(col)
            rel = s / s.iloc[0]
            plt.plot(rel.index, rel.values, linewidth=1.8, alpha=0.9, label=col)

        plt.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
        plt.title(f"Top {top_n} facteurs par robust_uptrend_score")
        plt.ylabel("Ratio cumulatif normalisé")
        plt.xlabel("Date")
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.show()

    def plot_yearly_heatmap(self, master_table: pd.DataFrame, top_n: int = 20, figsize=(16, 8)):
        # Carte de chaleur des rendements annuels
        top_cols = master_table.head(top_n).index.tolist()
        yearly = self.yearly_returns()[top_cols].T

        plt.figure(figsize=figsize)
        sns.heatmap(
            yearly,
            cmap="RdYlGn",
            center=0,
            annot=False,
            linewidths=0.3,
            cbar_kws={"label": "Rendement annuel du ratio"},
        )
        plt.title(f"Heatmap annuelle des top {top_n} facteurs stables")
        plt.xlabel("Année")
        plt.ylabel("Facteur")
        plt.tight_layout()
        plt.show()

    def plot_synergy_scatter(self, synergy_table: pd.DataFrame, top_n: int = 40, figsize=(10, 7)):
        # Nuage de points pour visualiser la synergie
        if synergy_table is None or len(synergy_table) == 0:
            print("Aucune combinaison exploitable.")
            return

        x = synergy_table.head(top_n).copy()

        plt.figure(figsize=figsize)
        plt.scatter(
            x["share_above_geo_path"],
            x["edge_terminal_vs_geo_log"],
            s=120 * (x["synergy_score"].fillna(0) + 0.2),
            alpha=0.75,
        )

        for name, row in x.iterrows():
            plt.text(
                row["share_above_geo_path"] + 0.002,
                row["edge_terminal_vs_geo_log"],
                name,
                fontsize=8,
            )

        plt.axvline(0.5, color="gray", linestyle="--", linewidth=1.0)
        plt.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
        plt.xlabel("Part du temps au-dessus de la baseline géométrique")
        plt.ylabel("Edge terminal vs baseline géométrique (log)")
        plt.title("Carte de synergie des facteurs combinés")
        plt.tight_layout()
        plt.show()


# =========================
# Exemple d'utilisation
# =========================

# analyseur = FactorStabilityAnalyzer(df, ann_factor=252)

# table principale
# master = analyseur.build_master_table(
#     windows=(126, 252, 504),
#     drawdown_limit=0.15,
#     min_obs=252,
#     min_horizon=252
# )

# analyse de synergie
# synergy = analyseur.synergy_analysis(
#     min_obs=252,
#     window_for_path=252
# )

# affichage des meilleurs facteurs "stables"
# print(master.head(20)[[
#     "final_ratio",
#     "cagr",
#     "max_drawdown",
#     "monotonicity",
#     "stability_score",
#     "start_date_win_rate_to_today",
#     "robust_uptrend_score"
# ]])

# affichage des meilleures combinaisons
# print(synergy.head(20)[[
#     "parts",
#     "edge_terminal_vs_geo_log",
#     "share_above_geo_path",
#     "monotonicity_lift",
#     "drawdown_lift",
#     "synergy_score"
# ]])

# visualisations
# analyseur.plot_top_stable_factors(master, top_n=12)
# analyseur.plot_yearly_heatmap(master, top_n=20)
# analyseur.plot_synergy_scatter(synergy, top_n=30)
