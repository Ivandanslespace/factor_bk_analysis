analyseur = FactorStabilityAnalyzer(df, ann_factor=252)

master = analyseur.build_master_table(
    windows=(126, 252, 504),
    drawdown_limit=0.15,
    min_obs=252,
    min_horizon=252
)

synergy = analyseur.synergy_analysis(
    min_obs=252,
    window_for_path=252
)

analyseur.plot_top_stable_factors(master, top_n=12, save_html=True, file_name="top_stable_factors.html")
analyseur.plot_yearly_heatmap(master, top_n=20, save_html=True, file_name="yearly_heatmap.html")
analyseur.plot_synergy_scatter(synergy, top_n=30, label_n=12, save_html=True, file_name="synergy_scatter.html")
analyseur.plot_underwater_profile(master, top_n=15, save_html=True, file_name="underwater_profile.html")
analyseur.plot_single_factor_diagnostics(master.index[0], save_html=True, file_name="single_factor_diag.html")
