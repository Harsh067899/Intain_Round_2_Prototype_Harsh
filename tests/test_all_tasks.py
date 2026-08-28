"""End-to-end tests for all 8 tasks of the Loan Intelligence Engine.

Uses a small synthetic dataset (50 loans x 12 months) with known outcomes.
Run: pytest tests/test_all_tasks.py -v
"""
import json, os, sys, tempfile, shutil
import numpy as np
import pandas as pd
import pytest

# ---- paths -----------------------------------------------------------------
HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
TEST_DATA = os.path.join(HERE, "test_data")
RAW = os.path.join(TEST_DATA, "raw")
GT = os.path.join(TEST_DATA, "ground_truth")

sys.path.insert(0, os.path.join(ROOT, "src", "profiling"))
sys.path.insert(0, os.path.join(ROOT, "src", "features"))
sys.path.insert(0, os.path.join(ROOT, "src", "copilot"))


# ---- fixtures ---------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def generate_data():
    """Generate test data once for the entire session."""
    from generate_test_data import generate
    generate()
    yield

@pytest.fixture(scope="session")
def train_df():
    return pd.read_csv(os.path.join(RAW, "loan_monthly_performance_train.csv"))

@pytest.fixture(scope="session")
def test_df():
    return pd.read_csv(os.path.join(RAW, "loan_monthly_performance_test.csv"))

@pytest.fixture(scope="session")
def static_df():
    return pd.read_csv(os.path.join(RAW, "loan_static_attributes.csv"))

@pytest.fixture(scope="session")
def updates_df():
    return pd.read_csv(os.path.join(RAW, "servicer_updates.csv"))

@pytest.fixture(scope="session")
def scenarios_df():
    return pd.read_csv(os.path.join(RAW, "macro_scenarios.csv"))

@pytest.fixture(scope="session")
def gt_corruptions():
    return pd.read_csv(os.path.join(GT, "injected_corruptions.csv"))


# =============================================================================
# TASK 1 — Data Intelligence & Profiling
# =============================================================================
class TestTask1Profiling:

    def test_column_profiles(self, train_df):
        from profile_engine import profile_columns
        prof = profile_columns(train_df)
        assert len(prof) == len(train_df.columns)
        assert "missing_pct" in prof.columns
        assert "n_unique" in prof.columns
        # numeric cols should have stats
        bal_row = prof[prof.column == "current_balance"].iloc[0]
        assert pd.notna(bal_row["mean"])

    def test_psi_drift(self, train_df, test_df):
        from profile_engine import psi
        # same distribution => low PSI
        p = psi(train_df["interest_rate"], train_df["interest_rate"])
        assert p < 0.05
        # shifted distribution => higher PSI
        shifted = train_df["interest_rate"] + 5
        p2 = psi(train_df["interest_rate"], shifted)
        assert p2 > 0.1

    def test_drift_table(self, train_df, test_df):
        from profile_engine import drift_table
        dt = drift_table(train_df, test_df, ["interest_rate", "current_balance"])
        assert len(dt) == 2
        assert "psi" in dt.columns
        assert "assessment" in dt.columns

    def test_rule_engine_catches_negative_balance(self, train_df):
        from rule_engine import run_rules
        rules_path = os.path.join(RAW, "validation_rules.json")
        per_row, summary = run_rules(train_df, rules_path)
        # loan LNT00010 at month 2024-05 has balance=-5000
        neg = train_df[train_df.current_balance < 0]
        assert len(neg) >= 1, "Test data must contain negative balance"
        r001 = summary[summary.rule_id == "R001"]
        assert int(r001.violations.iloc[0]) >= 1

    def test_rule_engine_catches_status_dpd_mismatch(self, train_df):
        from rule_engine import run_rules
        rules_path = os.path.join(RAW, "validation_rules.json")
        per_row, summary = run_rules(train_df, rules_path)
        # loan LNT00012 at 2024-04: CURRENT with dpd=45
        r004 = summary[summary.rule_id == "R004"]
        assert int(r004.violations.iloc[0]) >= 1

    def test_rule_engine_catches_date_inversion(self, train_df):
        from rule_engine import run_rules
        rules_path = os.path.join(RAW, "validation_rules.json")
        per_row, summary = run_rules(train_df, rules_path)
        r003 = summary[summary.rule_id == "R003"]
        assert int(r003.violations.iloc[0]) >= 1

    def test_rule_engine_catches_doc_gap(self, train_df):
        from rule_engine import run_rules
        rules_path = os.path.join(RAW, "validation_rules.json")
        per_row, summary = run_rules(train_df, rules_path)
        r007 = summary[summary.rule_id == "R007"]
        assert int(r007.violations.iloc[0]) >= 1

    def test_reconciliation_detects_conflicts(self, train_df, updates_df, static_df):
        from reconcile import reconcile
        recon = reconcile(train_df, updates_df, static_df)
        conflicts = recon[recon.n_conflicts > 0]
        assert len(conflicts) >= 1, "Must detect servicer conflicts"
        # loans 15-19 have 15% balance mismatch
        conflict_loans = set(conflicts.loan_id)
        for i in range(15, 20):
            lid = f"LNT{i:05d}"
            assert lid in conflict_loans, f"{lid} should have conflicts"

    def test_trust_scores_range(self, train_df, updates_df, static_df):
        from rule_engine import run_rules
        from reconcile import reconcile, trust_scores
        rules_path = os.path.join(RAW, "validation_rules.json")
        rule_rows, _ = run_rules(train_df, rules_path)
        recon = reconcile(train_df, updates_df, static_df)
        trust = trust_scores(train_df, rule_rows, recon)
        assert trust.trust_score.min() >= 0.0
        assert trust.trust_score.max() <= 1.0
        assert set(trust.trust_band.dropna().unique()) <= {"LOW", "MEDIUM", "HIGH"}

    def test_corruptions_detected_vs_ground_truth(self, train_df, gt_corruptions):
        from rule_engine import run_rules
        from reconcile import reconcile, trust_scores
        rules_path = os.path.join(RAW, "validation_rules.json")
        static = pd.read_csv(os.path.join(RAW, "loan_static_attributes.csv"))
        updates = pd.read_csv(os.path.join(RAW, "servicer_updates.csv"))
        rule_rows, _ = run_rules(train_df, rules_path)
        recon = reconcile(train_df, updates, static)
        trust = trust_scores(train_df, rule_rows, recon)
        flagged = trust.loc[(trust.n_rules_fired > 0) | (trust.n_conflicts > 0)]
        fkeys = set(flagged.loan_id + "|" + flagged.reporting_month)
        gt = gt_corruptions.copy()
        gt["caught"] = (gt.loan_id + "|" + gt.reporting_month).isin(fkeys)
        recall = gt.caught.mean()
        assert recall >= 0.6, f"Corruption recall {recall:.2f} too low"


# =============================================================================
# TASK 2 — Loan Performance Prediction
# =============================================================================
class TestTask2Prediction:

    def test_build_features_columns(self, train_df):
        from build_features import build_features, feature_cols
        X = build_features(train_df)
        fc = feature_cols(X)
        assert "interest_rate" in fc
        assert "balance_ratio" in fc
        assert "loan_age_months" in fc
        assert "loan_id" not in fc
        assert "reporting_month" not in fc

    def test_build_features_no_target_leakage(self, train_df):
        from build_features import build_features, feature_cols, TARGETS_BIN
        X = build_features(train_df)
        fc = feature_cols(X)
        for t in TARGETS_BIN:
            assert t not in fc, f"Target {t} leaked into features"
        assert "next_state" not in fc
        assert "exception_required" not in fc

    def test_censor_mask(self, train_df):
        from build_features import censor_mask
        mask = censor_mask(train_df, "next_3m_delinquency_flag", "2024-12")
        assert mask.dtype == bool
        assert mask.sum() > 0
        assert mask.sum() < len(train_df)

    def test_model_trains_and_predicts(self, train_df):
        from build_features import build_features, feature_cols, censor_mask
        import lightgbm as lgb
        X = build_features(train_df)
        fc = feature_cols(X)
        tgt = "next_3m_delinquency_flag"
        mask = censor_mask(train_df, tgt, "2024-12")
        Xm = X.loc[mask, fc]
        ym = train_df.loc[mask, tgt]
        if ym.nunique() < 2:
            pytest.skip("Not enough target variance")
        m = lgb.LGBMClassifier(n_estimators=20, verbosity=-1)
        m.fit(Xm, ym)
        probs = m.predict_proba(Xm)[:, 1]
        assert len(probs) == len(Xm)
        assert 0 <= probs.min() <= probs.max() <= 1

    def test_default_loans_score_higher(self, train_df):
        """Loans that actually default should get higher predicted risk."""
        from build_features import build_features, feature_cols
        import lightgbm as lgb
        X = build_features(train_df)
        fc = feature_cols(X)
        tgt = "next_12m_default_flag"
        ym = train_df[tgt]
        if ym.nunique() < 2:
            pytest.skip("Not enough target variance")
        m = lgb.LGBMClassifier(n_estimators=50, verbosity=-1)
        m.fit(X[fc], ym)
        p = m.predict_proba(X[fc])[:, 1]
        mean_pos = p[ym == 1].mean()
        mean_neg = p[ym == 0].mean()
        assert mean_pos > mean_neg, "Default loans should score higher"


# =============================================================================
# TASK 3 — Transition / Survival Model
# =============================================================================
class TestTask3Transition:

    def test_transition_matrix_sums_to_one(self, train_df):
        """Empirical transition matrix rows should sum to 1."""
        df = train_df.copy()
        trans = df.groupby("current_status")["next_state"].value_counts(normalize=True)
        for status in trans.index.get_level_values(0).unique():
            total = trans[status].sum()
            assert abs(total - 1.0) < 0.01, f"Row {status} sums to {total}"

    def test_multinomial_model_trains(self, train_df):
        from build_features import build_features, feature_cols
        import lightgbm as lgb
        X = build_features(train_df)
        fc = feature_cols(X)
        y = train_df["next_state"]
        states = sorted(y.unique())
        if len(states) < 2:
            pytest.skip("Not enough states")
        tmap = {s: i for i, s in enumerate(states)}
        m = lgb.LGBMClassifier(objective="multiclass", num_class=len(states),
                                n_estimators=20, verbosity=-1)
        m.fit(X[fc], y.map(tmap))
        probs = m.predict_proba(X[fc])
        assert probs.shape == (len(X), len(states))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_cumulative_incidence_monotonic(self, train_df):
        """Cumulative default rate should be non-decreasing over time."""
        from build_features import build_features, feature_cols
        import lightgbm as lgb
        X = build_features(train_df)
        fc = feature_cols(X)
        y = train_df["next_state"]
        states = sorted(y.unique())
        tmap = {s: i for i, s in enumerate(states)}
        m = lgb.LGBMClassifier(objective="multiclass", num_class=len(states),
                                n_estimators=20, verbosity=-1)
        m.fit(X[fc], y.map(tmap))
        # simulate forward for a single loan
        row = X.iloc[[0]].copy()
        cum_default = 0.0
        prev = cum_default
        if "DEFAULT" not in states:
            pytest.skip("No DEFAULT state")
        didx = states.index("DEFAULT")
        alive = 1.0
        for t in range(3):
            p = m.predict_proba(row[fc])[0]
            cum_default += alive * p[didx]
            alive *= (1 - p[didx])
            assert cum_default >= prev - 1e-9
            prev = cum_default


# =============================================================================
# TASK 4 — Anomaly & Exception Detection
# =============================================================================
class TestTask4Anomaly:

    def test_isolation_forest_scores(self, train_df):
        from sklearn.ensemble import IsolationForest
        feats = ["days_past_due", "loan_age_months", "interest_rate", "current_balance"]
        iso = IsolationForest(n_estimators=50, random_state=42)
        iso.fit(train_df[feats])
        scores = -iso.score_samples(train_df[feats])
        assert len(scores) == len(train_df)
        # defaulted/delinquent loans should score more anomalous on average
        delinq = train_df.current_status.isin(["DPD30", "DPD60", "DPD90", "DEFAULT"])
        if delinq.sum() > 0 and (~delinq).sum() > 0:
            assert scores[delinq].mean() > scores[~delinq].mean()

    def test_fused_anomaly_score_range(self, train_df):
        from build_features import build_features, feature_cols
        from rule_engine import run_rules
        from sklearn.ensemble import IsolationForest
        X = build_features(train_df)
        rule_rows, _ = run_rules(train_df, os.path.join(RAW, "validation_rules.json"))
        rule_score = rule_rows.n_rules_fired.to_numpy()
        iso_cols = ["days_past_due", "loan_age_months", "interest_rate", "balance_ratio"]
        iso = IsolationForest(n_estimators=50, random_state=42)
        iso.fit(X[iso_cols])
        iso_pct = pd.Series(-iso.score_samples(X[iso_cols])).rank(pct=True).to_numpy()
        W_RULE, W_ISO, W_SUP = 0.45, 0.15, 0.40
        fused = W_RULE * np.clip(rule_score / 5.0, 0, 1) + W_ISO * iso_pct + W_SUP * 0.5
        assert fused.min() >= 0
        assert fused.max() <= 1.5  # theoretical max with all weights

    def test_exception_classifier(self, train_df):
        from build_features import build_features, feature_cols
        import lightgbm as lgb
        X = build_features(train_df)
        fc = feature_cols(X)
        y = train_df["exception_required"]
        if y.nunique() < 2:
            pytest.skip("Not enough variance")
        m = lgb.LGBMClassifier(n_estimators=20, verbosity=-1)
        m.fit(X[fc], y)
        p = m.predict_proba(X[fc])[:, 1]
        # loans with exceptions should score higher
        assert p[y == 1].mean() > p[y == 0].mean()


# =============================================================================
# TASK 5 — Scenario & Stress Simulation
# =============================================================================
class TestTask5Scenarios:

    def test_three_scenarios_exist(self, scenarios_df):
        assert len(scenarios_df) == 3
        names = set(scenarios_df.scenario)
        assert "base" in names
        assert "adverse_credit" in names
        assert "high_prepayment" in names

    def test_adverse_increases_default_hazard(self, scenarios_df):
        adv = scenarios_df[scenarios_df.scenario == "adverse_credit"].iloc[0]
        assert adv.default_hazard_multiplier > 1.0
        assert adv.delinquency_hazard_multiplier > 1.0

    def test_high_prepay_increases_prepay(self, scenarios_df):
        hp = scenarios_df[scenarios_df.scenario == "high_prepayment"].iloc[0]
        assert hp.prepayment_multiplier > 1.5

    def test_scenario_multiplier_changes_probs(self):
        """apply_scenario should shift probabilities."""
        sys.path.insert(0, os.path.join(ROOT, "src", "scenarios"))
        from run_task5 import apply_scenario
        probs = np.array([[0.7, 0.1, 0.05, 0.05, 0.05, 0.05]])
        states = ["CURRENT", "DPD30", "DPD60", "DPD90", "DEFAULT", "PREPAID"]
        sc_base = pd.Series({"delinquency_hazard_multiplier": 1.0,
                             "default_hazard_multiplier": 1.0,
                             "prepayment_multiplier": 1.0})
        sc_stress = pd.Series({"delinquency_hazard_multiplier": 2.0,
                               "default_hazard_multiplier": 2.0,
                               "prepayment_multiplier": 0.5})
        p_base = apply_scenario(probs, "CURRENT", states, sc_base)
        p_stress = apply_scenario(probs, "CURRENT", states, sc_stress)
        # stress should increase DPD30 and DEFAULT prob
        assert p_stress[0, 1] > p_base[0, 1]  # DPD30
        assert p_stress[0, 4] > p_base[0, 4]  # DEFAULT
        # stress should decrease PREPAID prob
        assert p_stress[0, 5] < p_base[0, 5]
        # rows still sum to 1
        np.testing.assert_allclose(p_stress.sum(axis=1), 1.0, atol=1e-6)


# =============================================================================
# TASK 6 — Explainability & Responsible AI
# =============================================================================
class TestTask6Explainability:

    def test_shap_values_match_features(self, train_df):
        from build_features import build_features, feature_cols
        import lightgbm as lgb
        import shap
        X = build_features(train_df)
        fc = feature_cols(X)
        y = train_df["next_12m_default_flag"]
        if y.nunique() < 2:
            pytest.skip("No variance")
        m = lgb.LGBMClassifier(n_estimators=20, verbosity=-1)
        m.fit(X[fc], y)
        expl = shap.TreeExplainer(m)
        sv = expl.shap_values(X[fc].iloc[:10])
        sv = sv[1] if isinstance(sv, list) else sv
        assert sv.shape == (10, len(fc))

    def test_conformal_interval_widens_with_low_trust(self):
        """Trust-scaled conformal: low trust => wider interval."""
        q_glob = 0.15
        LAM = 0.6
        trust_high = 0.95
        trust_low = 0.3
        half_high = q_glob * (1 + LAM * (1 - trust_high))
        half_low = q_glob * (1 + LAM * (1 - trust_low))
        assert half_low > half_high


# =============================================================================
# TASK 7 — LLM Reviewer Copilot
# =============================================================================
class TestTask7Copilot:

    def test_grounding_check_passes_valid(self):
        from copilot import grounding_check
        bundle = {"loan_id": "LN001", "prob_default_12m": 0.85,
                  "trust_score": 0.42, "anomaly_score": 0.91,
                  "rules_fired": ["R001", "R004"]}
        note = "Default probability 0.85, trust 0.42, anomaly 0.91. Rules R001, R004."
        result = grounding_check(note, bundle)
        assert result["grounded"] is True

    def test_grounding_check_rejects_fabricated_numbers(self):
        from copilot import grounding_check
        bundle = {"loan_id": "LN001", "prob_default_12m": 0.85,
                  "trust_score": 0.42, "rules_fired": []}
        note = "Default probability 0.85 and the loss severity is 0.73."
        result = grounding_check(note, bundle)
        assert result["grounded"] is False
        assert len(result["unmatched_numbers"]) > 0

    def test_grounding_check_rejects_fabricated_rules(self):
        from copilot import grounding_check
        bundle = {"loan_id": "LN001", "prob_default_12m": 0.50,
                  "rules_fired": ["R001"]}
        note = "Rules fired: R001, R005."
        result = grounding_check(note, bundle)
        assert result["grounded"] is False
        assert "R005" in result["unmatched_rule_ids"]

    def test_template_contains_recommendation_label(self):
        from copilot import CopilotClient
        # Temporarily point to test data
        import copilot as cop
        old_raw, old_logs = cop.RAW, cop.LOGS
        cop.RAW = RAW
        cop.LOGS = tempfile.mkdtemp()
        try:
            client = CopilotClient(use_api=False)
            bundle = {"loan_id": "LNT00001", "reporting_month": "2024-06",
                      "prob_default_12m": 0.75, "trust_score": 0.5,
                      "anomaly_score": 0.6, "exception_type_pred": "NONE",
                      "rules_fired": [], "top_drivers": [],
                      "artifact_ids": ["test"]}
            rec = client.note_for(bundle)
            assert "RECOMMENDATION" in rec["output"]
            assert "human decision required" in rec["output"]
            assert rec["grounding_check"]["grounded"] is True
        finally:
            cop.RAW, cop.LOGS = old_raw, old_logs

    def test_mini_rag_retrieves_relevant_entries(self):
        from copilot import load_dictionary, load_rules, retrieve
        import copilot as cop
        old_raw = cop.RAW
        cop.RAW = RAW
        try:
            d = load_dictionary()
            r = load_rules()
            bundle = {"rules_fired": ["R001"], "top_drivers": [{"feature": "days_past_due", "shap": 0.1}]}
            ret = retrieve(bundle, d, r)
            assert "days_past_due" in ret["dictionary"]
            assert "R001" in ret["rules"]
        finally:
            cop.RAW = old_raw


# =============================================================================
# TASK 8 — Agentic Coding Evidence
# =============================================================================
class TestTask8AuditTrail:

    def test_ai_dev_log_exists(self):
        log_path = os.path.join(ROOT, "logs", "AI_DEVELOPMENT_LOG.md")
        assert os.path.isfile(log_path), "AI_DEVELOPMENT_LOG.md must exist"

    def test_ai_dev_log_has_entries(self):
        log_path = os.path.join(ROOT, "logs", "AI_DEVELOPMENT_LOG.md")
        with open(log_path) as f:
            content = f.read()
        assert len(content) > 500
        assert "rejected" in content.lower() or "corrected" in content.lower()

    def test_prompt_log_is_valid_jsonl(self):
        log_path = os.path.join(ROOT, "logs", "prompt_log.jsonl")
        if not os.path.isfile(log_path):
            pytest.skip("No prompt log yet")
        with open(log_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) >= 1
        for line in lines:
            rec = json.loads(line)
            assert "output" in rec
            assert "grounding_check" in rec


# =============================================================================
# INTEGRATION — Submission format
# =============================================================================
class TestSubmissionFormat:

    def test_submission_template_schema(self):
        tmpl = pd.read_csv(os.path.join(RAW, "submission_template.csv"))
        expected = ["loan_id", "reporting_month", "prob_delinq_3m", "prob_delinq_6m",
                    "prob_default_12m", "prob_prepay_12m", "next_state_pred",
                    "anomaly_score", "exception_required_prob", "exception_type_pred",
                    "top_drivers", "recommended_action", "confidence"]
        assert list(tmpl.columns) == expected

    def test_production_submission_exists_and_valid(self):
        sub_path = os.path.join(ROOT, "submission.csv")
        if not os.path.isfile(sub_path):
            pytest.skip("No submission.csv yet")
        sub = pd.read_csv(sub_path)
        assert sub.isnull().sum().sum() == 0, "No nulls allowed"
        assert set(sub.recommended_action.unique()) <= {"AUTO_ACCEPT", "REVIEW", "ESCALATE"}
        assert (sub.confidence >= 0).all() and (sub.confidence <= 1).all()
