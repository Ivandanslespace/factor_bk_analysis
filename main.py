import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class FactorStabilityAnalyzer:
    """
    Outil d'analyse pour des facteurs simples et combinés.

    Hypothèses :
    - index : dates
    - colonnes : noms des facteurs
    - valeurs : ratio cumulatif vs benchmark, strictement positif

    Exemples :
    - facteur simple : 'PCT ROE'
    - facteur combiné : 'CROSS_PCT ROE_x_PCT PE LTM'
    - facteur combiné 3 briques : 'CROSS_A_x_B_x_C'
    """

    def __init__(self, df: pd.DataFrame, ann_factor: int = 252):
        # Initialisation de l'analyseur
        self.ann_factor = ann_factor
        self.df = self._prepare_df(df)
        self.factor_map = self._build_factor_map()

    @staticmethod
    def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
        # Préparation propre du DataFrame
        x = df.copy()
        x.index = pd.to_datetime(x.index)
        x = x.sort_index()
        x = x.replace([np.inf, -np.inf], np.nan)
        x = x.ffill()
        x = x.where(x > 0)
        x = x.dropna(axis=1, how="all")
        return x

    @staticmethod
    def _normalize_spaces(name: str) -> str:
        # Normalisation des espaces
        return re.sub(r"\s+", " ", str(name)).strip()

    @classmethod
    def _clean_factor_token(cls, name: str) -> str:
        # Nettoyage d'un morceau de nom de facteur
        x = cls._normalize_spaces(name)
        x = re.sub(r"^(CROSS_|COMBO_)+", "", x)
        x = cls._normalize_spaces(x)
        return x

    def _parse_factor_parts(self, col: str):
        # Découpage intelligent d'un nom de facteur
        col = self._normalize_spaces(col)

        if "_x_" not in col:
            return [self._clean_factor_token(col)]

        parts = col.split("_x_")
        parts = [self._clean_factor_token(p) for p in parts]
        return parts

    def _build_factor_map(self) -> pd.DataFrame:
        # Construction de la table descriptive des facteurs
        records = []

        for col in self.df.columns:
            is_combo = "_x_" in str(col)
            parts = self._parse_factor_parts(col)

            records.append(
                {
                    "factor": col,
                    "is_combo": is_combo,
                    "n_parts": len(parts),
                    "parts": parts,
                }
            )

        return pd.DataFrame(records).set_index("factor")

    def _series(self, col: str) -> pd.Series:
        # Extraction propre d'une série
        return self.df[col].dropna()

    @staticmethod
    def _safe_rank_pct(s: pd.Series, ascending: bool = True) -> pd.Series:
        # Conversion d'une métrique en percentile
        return s.rank(pct=True, ascending=ascending)

    @staticmethod
    def _max_drawdown_from_array(arr: np.ndarray) -> float:
        # Calcul du drawdown maximal
        x = np.asarray(arr, dtype=float)
        if len(x) < 2 or np.any(~np.isfinite(x)) or np.any(x <= 0):
            return np.nan

        x = x / x[0]
        peak = np.maximum.accumulate(x)
        dd = x / peak - 1.0
        return float(np.min(dd))

    @staticmethod
    def _ulcer_index_from_array(arr: np.ndarray) -> float:
        # Calcul de l'Ulcer Index
        x = np.asarray(arr, dtype=float)
        if len(x) < 2 or np.any(~np.isfinite(x)) or np.any(x <= 0):
            return np.nan

        x = x / x[0]
        peak = np.maximum.accumulate(x)
        dd = x / peak - 1.0
        return float(np.sqrt(np.mean((dd * 100.0) ** 2)))

    @staticmethod
    def _monotonicity_from_array(arr: np.ndarray) -> float:
        # Part des rendements log positifs
        x = np.asarray(arr, dtype=float)
        if len(x) < 3 or np.any(~np.isfinite(x)) or np.any(x <= 0):
            return np.nan

        r = np.diff(np.log(x))
        if len(r) == 0:
            return np.nan
        return float(np.mean(r > 0))

    @staticmethod
    def _new_high_share_from_array(arr: np.ndarray) -> float:
        # Part du temps passée sur un plus haut historique
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
    def _drawdown_series_from_array(arr: np.ndarray) -> np.ndarray:
        # Série de drawdown
        x = np.asarray(arr, dtype=float)
        if len(x) == 0 or np.any(~np.isfinite(x)) or np.any(x <= 0):
            return np.array([])

        x = x / x[0]
        peak = np.maximum.accumulate(x)
        dd = x / peak - 1.0
        return dd

    @classmethod
    def _pct_underwater_from_array(cls, arr: np.ndarray) -> float:
        # Part du temps sous le précédent sommet
        dd = cls._drawdown_series_from_array(arr)
        if len(dd) == 0:
            return np.nan
        return float(np.mean(dd < -1e-12))

    @classmethod
    def _underwater_stats_from_array(cls, arr: np.ndarray) -> dict:
        # Statistiques de temps underwater et de récupération
        dd = cls._drawdown_series_from_array(arr)

        if len(dd) == 0:
            return {
                "pct_underwater": np.nan,
                "max_underwater_duration": np.nan,
                "avg_underwater_duration": np.nan,
                "max_recovery_duration": np.nan,
                "avg_recovery_duration": np.nan,
                "n_complete_recoveries": 0,
                "currently_underwater": np.nan,
            }

        underwater = dd < -1e-12
        n = len(underwater)

        pct_underwater = float(np.mean(underwater))
        currently_underwater = bool(underwater[-1])

        segments = []
        in_seg = False
        start = None

        for i, flag in enumerate(underwater):
            if flag and not in_seg:
                start = i
                in_seg = True
            elif (not flag) and in_seg:
                end = i - 1
                segments.append((start, end))
                in_seg = False

        if in_seg:
            segments.append((start, n - 1))

        underwater_durations = []
        recovery_durations = []

        for s, e in segments:
            underwater_durations.append(e - s + 1)

            if e < n - 1 and underwater[e + 1] == False:
                trough_idx = s + int(np.argmin(dd[s:e + 1]))
                recovery_idx = e + 1
                recovery_durations.append(recovery_idx - trough_idx)

        max_underwater_duration = float(np.max(underwater_durations)) if len(underwater_durations) else 0.0
        avg_underwater_duration = float(np.mean(underwater_durations)) if len(underwater_durations) else 0.0
        max_recovery_duration = float(np.max(recovery_durations)) if len(recovery_durations) else np.nan
        avg_recovery_duration = float(np.mean(recovery_durations)) if len(recovery_durations) else np.nan

        return {
            "pct_underwater": pct_underwater,
            "max_underwater_duration": max_underwater_duration,
            "avg_underwater_duration": avg_underwater_duration,
            "max_recovery_duration": max_recovery_duration,
            "avg_recovery_duration": avg_recovery_duration,
            "n_complete_recoveries": int(len(recovery_durations)),
            "currently_underwater": currently_underwater,
        }

    def debug_combo_mapping(self, max_rows: int = 50) -> pd.DataFrame:
        # Diagnostic du mapping des facteurs combinés
        rows = []

        combo_names = self.factor_map.index[self.factor_map["is_combo"]]

        for combo in combo_names:
            parts = self.factor_map.loc[combo, "parts"]
            exists = [p in self.df.columns for p in parts]

            rows.append(
                {
                    "combo": combo,
                    "parts": parts,
                    "n_parts": len(parts),
                    "all_parts_found": all(exists),
                    "missing_parts": [p for p, ok in zip(parts, exists) if not ok],
                }
            )

        out = pd.DataFrame(rows)
        if len(out) == 0:
            return out

        return out.head(max_rows)

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
            sortino = (
                (log_ret.mean() * self.ann_factor / downside_std)
                if pd.notna(downside_std) and downside_std > 0
                else np.nan
            )

            mdd = self._max_drawdown_from_array(rel.values)
            ulcer = self._ulcer_index_from_array(rel.values)
            calmar = cagr / abs(mdd) if pd.notna(mdd) and mdd < 0 else np.nan
            monotonicity = self._monotonicity_from_array(rel.values)
            new_high_share = self._new_high_share_from_array(rel.values)
            slope, r2 = self._trend_slope_r2_from_array(rel.values)
            uw_stats = self._underwater_stats_from_array(rel.values)

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
                    "pct_underwater": uw_stats["pct_underwater"],
                    "max_underwater_duration": uw_stats["max_underwater_duration"],
                    "avg_underwater_duration": uw_stats["avg_underwater_duration"],
                    "max_recovery_duration": uw_stats["max_recovery_duration"],
                    "avg_recovery_duration": uw_stats["avg_recovery_duration"],
                    "n_complete_recoveries": uw_stats["n_complete_recoveries"],
                    "currently_underwater": uw_stats["currently_underwater"],
                }
            )

        out = pd.DataFrame(rows).set_index("factor")
        out = out.sort_values("cagr", ascending=False)
        return out

    def rolling_analysis(self, window: int = 252):
        # Analyse glissante des performances
        rolling_return = pd.DataFrame(index=self.df.index, columns=self.df.columns, dtype=float)
        rolling_cagr = pd.DataFrame(index=self.df.index, columns=self.df.columns, dtype=float)
        rolling_mdd = pd.DataFrame(index=self.df.index, columns=self.df.columns, dtype=float)
        rolling_mono = pd.DataFrame(index=self.df.index, columns=self.df.columns, dtype=float)
        rolling_new_high = pd.DataFrame(index=self.df.index, columns=self.df.columns, dtype=float)
        rolling_underwater = pd.DataFrame(index=self.df.index, columns=self.df.columns, dtype=float)

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

            rolling_underwater[col] = s.rolling(window).apply(
                lambda x: self._pct_underwater_from_array(np.asarray(x, dtype=float)),
                raw=True,
            )

        return {
            "rolling_return": rolling_return,
            "rolling_cagr": rolling_cagr,
            "rolling_mdd": rolling_mdd,
            "rolling_monotonicity": rolling_mono,
            "rolling_new_high": rolling_new_high,
            "rolling_pct_underwater": rolling_underwater,
        }

    def stability_ranking(
        self,
        windows=(126, 252, 504),
        drawdown_limit=0.15,
        min_obs=252,
    ) -> pd.DataFrame:
        # Classement multi-fenêtre orienté stabilité
        base = self.full_period_stats()
        base = base[base["n_obs"] >= min_obs].copy()

        if len(base) == 0:
            return base

        all_scores = []

        for w in windows:
            ra = self.rolling_analysis(window=w)
            rr = ra["rolling_return"]
            rc = ra["rolling_cagr"]
            rm = ra["rolling_mdd"]
            rmo = ra["rolling_monotonicity"]
            rnh = ra["rolling_new_high"]
            ruw = ra["rolling_pct_underwater"]

            rows = []

            for col in base.index:
                x_rr = rr[col].dropna() if col in rr.columns else pd.Series(dtype=float)
                x_rc = rc[col].dropna() if col in rc.columns else pd.Series(dtype=float)
                x_rm = rm[col].dropna() if col in rm.columns else pd.Series(dtype=float)
                x_rmo = rmo[col].dropna() if col in rmo.columns else pd.Series(dtype=float)
                x_rnh = rnh[col].dropna() if col in rnh.columns else pd.Series(dtype=float)
                x_ruw = ruw[col].dropna() if col in ruw.columns else pd.Series(dtype=float)

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
                        f"win_{w}_median_pct_underwater": x_ruw.median() if len(x_ruw) else np.nan,
                    }
                )

            tmp = pd.DataFrame(rows)
            if len(tmp) > 0:
                tmp = tmp.set_index("factor")
                all_scores.append(tmp)

        if len(all_scores) == 0:
            base["stability_score"] = np.nan
            return base

        score_df = pd.concat(all_scores, axis=1)
        out = base.join(score_df, how="left")

        score_parts = []

        for w in windows:
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

            if f"win_{w}_median_pct_underwater" in out.columns:
                part["g"] = self._safe_rank_pct(-out[f"win_{w}_median_pct_underwater"], ascending=True)

            if part.shape[1] > 0:
                out[f"stability_score_{w}"] = part.mean(axis=1)
                score_parts.append(out[f"stability_score_{w}"])

        if len(score_parts) > 0:
            out["stability_score"] = pd.concat(score_parts, axis=1).mean(axis=1)
        else:
            out["stability_score"] = np.nan

        out["stability_score"] = (
            0.55 * out["stability_score"].fillna(0)
            + 0.10 * self._safe_rank_pct(out["monotonicity"], ascending=True).fillna(0)
            + 0.10 * self._safe_rank_pct(out["trend_r2"], ascending=True).fillna(0)
            + 0.10 * self._safe_rank_pct(out["calmar"], ascending=True).fillna(0)
            + 0.10 * self._safe_rank_pct(-out["pct_underwater"], ascending=True).fillna(0)
            + 0.05 * self._safe_rank_pct(-out["avg_recovery_duration"].fillna(out["avg_recovery_duration"].max()), ascending=True).fillna(0)
        )

        out = out.sort_values("stability_score", ascending=False)
        return out

    def yearly_returns(self) -> pd.DataFrame:
        # Rendements calendaires annuels
        yearly_level = self.df.resample("YE").last()
        yearly_ret = yearly_level.pct_change()
        yearly_ret.index = yearly_ret.index.year
        return yearly_ret

    def horizon_success_matrix(self, horizons=(63, 126, 252, 504)) -> pd.DataFrame:
        # Taux de succès selon plusieurs horizons
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

        out = pd.DataFrame(rows).set_index("factor")
        return out

    def start_date_robustness(self, min_horizon: int = 252) -> pd.DataFrame:
        # Robustesse au choix de la date de départ
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

        out = pd.DataFrame(rows)
        if len(out) == 0:
            return pd.DataFrame()

        out = out.set_index("factor")
        out = out.sort_values("start_date_win_rate_to_today", ascending=False)
        return out

    def synergy_analysis(self, min_obs: int = 252, window_for_path: int = 252) -> pd.DataFrame:
        # Analyse de la synergie des facteurs combinés
        records = []

        combo_names = self.factor_map.index[self.factor_map["is_combo"]]

        for combo in combo_names:
            parts = self.factor_map.loc[combo, "parts"]

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

            geo_baseline = np.exp(np.log(part_df).mean(axis=1))
            best_baseline = part_df.max(axis=1)

            combo_rel = combo_s / combo_s.iloc[0]
            geo_rel = geo_baseline / geo_baseline.iloc[0]
            best_rel = best_baseline / best_baseline.iloc[0]

            combo_mdd = self._max_drawdown_from_array(combo_rel.values)
            combo_mono = self._monotonicity_from_array(combo_rel.values)
            combo_uw = self._underwater_stats_from_array(combo_rel.values)

            part_mdds = []
            part_monos = []
            part_pct_uw = []
            part_avg_rec = []

            for p in part_names:
                p_rel = part_df[p] / part_df[p].iloc[0]
                p_uw = self._underwater_stats_from_array(p_rel.values)

                part_mdds.append(self._max_drawdown_from_array(p_rel.values))
                part_monos.append(self._monotonicity_from_array(p_rel.values))
                part_pct_uw.append(p_uw["pct_underwater"])
                part_avg_rec.append(p_uw["avg_recovery_duration"])

            avg_part_mdd = np.nanmean(part_mdds)
            avg_part_mono = np.nanmean(part_monos)
            avg_part_pct_uw = np.nanmean(part_pct_uw)
            avg_part_avg_rec = np.nanmean(part_avg_rec)

            edge_terminal_vs_geo_log = np.log(combo_rel.iloc[-1]) - np.log(geo_rel.iloc[-1])
            edge_terminal_vs_best_log = np.log(combo_rel.iloc[-1]) - np.log(best_rel.iloc[-1])

            share_above_geo = (combo_rel > geo_rel).mean()
            share_above_best = (combo_rel > best_rel).mean()

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
                    "combo_pct_underwater": combo_uw["pct_underwater"],
                    "avg_parts_pct_underwater": avg_part_pct_uw,
                    "underwater_lift": combo_uw["pct_underwater"] - avg_part_pct_uw,
                    "combo_avg_recovery_duration": combo_uw["avg_recovery_duration"],
                    "avg_parts_avg_recovery_duration": avg_part_avg_rec,
                    "recovery_lift": combo_uw["avg_recovery_duration"] - avg_part_avg_rec,
                    f"combo_positive_rate_{window_for_path}d": combo_positive_rate,
                    f"avg_parts_positive_rate_{window_for_path}d": avg_positive_rate_parts,
                    f"positive_rate_lift_{window_for_path}d": (
                        combo_positive_rate - avg_positive_rate_parts
                        if pd.notna(combo_positive_rate) and pd.notna(avg_positive_rate_parts)
                        else np.nan
                    ),
                }
            )

        out = pd.DataFrame(records)

        if len(out) == 0:
            return pd.DataFrame()

        out = out.set_index("combo")

        out["synergy_score"] = (
            0.20 * self._safe_rank_pct(out["edge_terminal_vs_geo_log"], ascending=True).fillna(0)
            + 0.15 * self._safe_rank_pct(out["share_above_geo_path"], ascending=True).fillna(0)
            + 0.15 * self._safe_rank_pct(out["monotonicity_lift"], ascending=True).fillna(0)
            + 0.15 * self._safe_rank_pct(-out["drawdown_lift"], ascending=True).fillna(0)
            + 0.15 * self._safe_rank_pct(-out["underwater_lift"], ascending=True).fillna(0)
            + 0.10 * self._safe_rank_pct(-out["recovery_lift"], ascending=True).fillna(0)
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

        master = stability.copy()

        if start_rob is not None and len(start_rob) > 0:
            master = master.join(start_rob, how="left")

        if horizon is not None and len(horizon) > 0:
            master = master.join(horizon, how="left")

        def rank_col(col, asc=True):
            # Classement d'une colonne existante
            if col not in master.columns:
                return pd.Series(index=master.index, data=np.nan)
            return self._safe_rank_pct(master[col], ascending=asc)

        def rank_series(s, asc=True):
            # Classement d'une série calculée à la volée
            s = pd.Series(s, index=master.index)
            return self._safe_rank_pct(s, ascending=asc)

        pct_underwater_rank = (
            rank_series(-master["pct_underwater"], True)
            if "pct_underwater" in master.columns
            else pd.Series(index=master.index, data=np.nan)
        )

        avg_recovery_rank = (
            rank_series(-master["avg_recovery_duration"].fillna(master["avg_recovery_duration"].max()), True)
            if "avg_recovery_duration" in master.columns
            else pd.Series(index=master.index, data=np.nan)
        )

        master["robust_uptrend_score"] = (
            0.30 * rank_col("stability_score", True).fillna(0)
            + 0.15 * rank_col("start_date_win_rate_to_today", True).fillna(0)
            + 0.10 * rank_col("start_date_p10_ann_return_to_today", True).fillna(0)
            + 0.08 * rank_col("calmar", True).fillna(0)
            + 0.08 * rank_col("trend_r2", True).fillna(0)
            + 0.08 * rank_col("new_high_share", True).fillna(0)
            + 0.08 * rank_col("monotonicity", True).fillna(0)
            + 0.07 * pct_underwater_rank.fillna(0)
            + 0.06 * avg_recovery_rank.fillna(0)
        )

        master = master.sort_values("robust_uptrend_score", ascending=False)
        return master

    def _maybe_save_html(self, fig, file_name=None, save_html=False):
        # Sauvegarde optionnelle en HTML interactif
        if save_html and file_name is not None:
            fig.write_html(file_name, include_plotlyjs="cdn")
        return fig

    def plot_top_stable_factors(self, master_table: pd.DataFrame, top_n: int = 12, save_html=False, file_name="top_stable_factors.html"):
        # Courbes interactives des meilleurs facteurs stables
        top_cols = master_table.head(top_n).index.tolist()

        plot_df = []
        for col in top_cols:
            s = self._series(col)
            rel = s / s.iloc[0]
            tmp = pd.DataFrame({
                "Date": rel.index,
                "Ratio": rel.values,
                "Factor": col
            })
            plot_df.append(tmp)

        plot_df = pd.concat(plot_df, axis=0)

        fig = px.line(
            plot_df,
            x="Date",
            y="Ratio",
            color="Factor",
            title=f"Top {top_n} facteurs stables"
        )

        fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
        fig.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Ratio normalisé")

        self._maybe_save_html(fig, file_name=file_name, save_html=save_html)
        fig.show()
        return fig

    def plot_yearly_heatmap(self, master_table: pd.DataFrame, top_n: int = 20, save_html=False, file_name="yearly_heatmap.html"):
        # Heatmap interactive des rendements annuels
        top_cols = master_table.head(top_n).index.tolist()
        yearly = self.yearly_returns()[top_cols].T

        fig = px.imshow(
            yearly,
            aspect="auto",
            color_continuous_scale="RdYlGn",
            origin="lower",
            labels=dict(x="Année", y="Facteur", color="Rendement"),
            title=f"Heatmap annuelle des top {top_n} facteurs stables"
        )

        fig.update_xaxes(side="bottom")
        fig.update_layout(hovermode="closest")

        self._maybe_save_html(fig, file_name=file_name, save_html=save_html)
        fig.show()
        return fig

    def plot_synergy_scatter(
        self,
        synergy_table: pd.DataFrame,
        top_n: int = 30,
        label_n: int = 12,
        save_html=False,
        file_name="synergy_scatter.html"
    ):
        # Nuage de points interactif pour la synergie
        if synergy_table is None or len(synergy_table) == 0:
            print("Aucune combinaison exploitable.")
            return None

        x = synergy_table.head(top_n).copy()
        x = x.reset_index()

        x["label"] = ""
        x.loc[:min(label_n - 1, len(x) - 1), "label"] = x.loc[:min(label_n - 1, len(x) - 1), "combo"]

        fig = px.scatter(
            x,
            x="monotonicity_lift",
            y="edge_terminal_vs_geo_log",
            size="synergy_score",
            color="underwater_lift",
            hover_name="combo",
            hover_data=[
                "parts",
                "share_above_geo_path",
                "drawdown_lift",
                "underwater_lift",
                "recovery_lift",
                "synergy_score"
            ],
            color_continuous_scale="RdYlGn_r",
            title="Carte de synergie : stabilité vs gain terminal"
        )

        fig.update_traces(
            text=x["label"],
            textposition="top center",
            marker=dict(sizemode="area", sizeref=2.0 * max(x["synergy_score"].max(), 1e-6) / (40.0 ** 2))
        )

        fig.add_vline(x=0.0, line_dash="dash", line_color="gray")
        fig.add_hline(y=0.0, line_dash="dash", line_color="gray")

        fig.update_layout(hovermode="closest")
        fig.update_xaxes(title_text="Monotonicity lift")
        fig.update_yaxes(title_text="Terminal edge log")

        self._maybe_save_html(fig, file_name=file_name, save_html=save_html)
        fig.show()
        return fig

    def plot_underwater_profile(
        self,
        master_table: pd.DataFrame,
        top_n: int = 15,
        save_html=False,
        file_name="underwater_profile.html"
    ):
        # Profil interactif underwater des meilleurs facteurs
        cols = master_table.head(top_n).index.tolist()

        plot_df = master_table.loc[cols, [
            "pct_underwater",
            "max_drawdown",
            "avg_recovery_duration"
        ]].copy()

        plot_df["factor"] = plot_df.index
        plot_df = plot_df.sort_values("pct_underwater", ascending=True)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=plot_df["pct_underwater"],
            y=plot_df["factor"],
            name="Pct underwater",
            orientation="h"
        ))

        fig.add_trace(go.Bar(
            x=plot_df["max_drawdown"],
            y=plot_df["factor"],
            name="Max drawdown",
            orientation="h",
            visible="legendonly"
        ))

        fig.add_trace(go.Bar(
            x=plot_df["avg_recovery_duration"],
            y=plot_df["factor"],
            name="Avg recovery",
            orientation="h",
            visible="legendonly"
        ))

        fig.update_layout(
            barmode="group",
            title=f"Profil underwater des top {top_n} facteurs",
            hovermode="y unified"
        )
        fig.update_xaxes(title_text="Valeur")
        fig.update_yaxes(title_text="Facteur")

        self._maybe_save_html(fig, file_name=file_name, save_html=save_html)
        fig.show()
        return fig

    def plot_single_factor_diagnostics(
        self,
        factor_name: str,
        save_html=False,
        file_name="single_factor_diagnostics.html"
    ):
        # Diagnostic interactif détaillé d'un facteur unique
        s = self._series(factor_name)
        rel = s / s.iloc[0]
        peak = rel.cummax()
        dd = rel / peak - 1.0

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=rel.index,
            y=rel.values,
            mode="lines",
            name="Ratio"
        ))

        fig.add_trace(go.Scatter(
            x=peak.index,
            y=peak.values,
            mode="lines",
            name="High-water mark",
            line=dict(dash="dash")
        ))

        fig.add_trace(go.Scatter(
            x=dd.index,
            y=dd.values,
            mode="lines",
            name="Underwater",
            yaxis="y2"
        ))

        fig.update_layout(
            title=f"Diagnostic interactif : {factor_name}",
            hovermode="x unified",
            yaxis=dict(title="Ratio"),
            yaxis2=dict(
                title="Drawdown",
                overlaying="y",
                side="right",
                showgrid=False
            )
        )
        fig.update_xaxes(title_text="Date")

        self._maybe_save_html(fig, file_name=file_name, save_html=save_html)
        fig.show()
        return fig
