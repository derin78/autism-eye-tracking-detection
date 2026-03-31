"""
ML Classifier for autism screening based on gaze features.
Uses a Random Forest trained on research-based reference data.

IMPORTANT: This is a SCREENING TOOL for research purposes only.
It is NOT a medical diagnosis.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ═══════════════════════════════════════════════════════════════════
# Reference data based on published research thresholds
# ═══════════════════════════════════════════════════════════════════
def _generate_reference_data(n_typical=100, n_atypical=100, seed=42):
    """
    Generate synthetic training data based on research findings:

    Typical gaze patterns:
    - Face attention ratio: 0.55 – 0.85  (high focus on faces)
    - More fixations on face region
    - Fewer saccades away from face

    Atypical (ASD-indicative) patterns:
    - Face attention ratio: 0.15 – 0.45  (lower focus on faces)
    - More fixations on objects/periphery
    - More saccades between regions

    Includes overlapping "borderline" samples for realistic evaluation.
    """
    rng = np.random.RandomState(seed)

    # Core typical samples (80%)
    n_core_typ = int(n_typical * 0.8)
    n_border_typ = n_typical - n_core_typ
    # Core atypical samples (80%)
    n_core_atyp = int(n_atypical * 0.8)
    n_border_atyp = n_atypical - n_core_atyp

    # Features: [face_ratio, face_fixation_count_norm, object_fixation_count_norm,
    #            saccade_count_norm, gaze_variability]

    # Core typical: clear pattern
    core_typical = np.column_stack([
        rng.uniform(0.60, 0.85, n_core_typ),    # face_ratio
        rng.uniform(0.55, 0.90, n_core_typ),    # face_fixation_count_norm
        rng.uniform(0.10, 0.40, n_core_typ),    # object_fixation_count_norm
        rng.uniform(0.10, 0.35, n_core_typ),    # saccade_count_norm
        rng.uniform(0.05, 0.20, n_core_typ),    # gaze_variability
    ])
    # Borderline typical: overlapping zone
    border_typical = np.column_stack([
        rng.uniform(0.45, 0.60, n_border_typ),  # overlapping face_ratio
        rng.uniform(0.40, 0.60, n_border_typ),
        rng.uniform(0.35, 0.55, n_border_typ),
        rng.uniform(0.25, 0.50, n_border_typ),
        rng.uniform(0.15, 0.35, n_border_typ),
    ])
    typical = np.vstack([core_typical, border_typical])

    # Core atypical: clear pattern
    core_atypical = np.column_stack([
        rng.uniform(0.15, 0.40, n_core_atyp),   # face_ratio
        rng.uniform(0.10, 0.40, n_core_atyp),
        rng.uniform(0.55, 0.90, n_core_atyp),
        rng.uniform(0.45, 0.80, n_core_atyp),
        rng.uniform(0.30, 0.60, n_core_atyp),
    ])
    # Borderline atypical: overlapping zone
    border_atypical = np.column_stack([
        rng.uniform(0.40, 0.55, n_border_atyp),  # overlapping face_ratio
        rng.uniform(0.35, 0.55, n_border_atyp),
        rng.uniform(0.40, 0.60, n_border_atyp),
        rng.uniform(0.30, 0.55, n_border_atyp),
        rng.uniform(0.20, 0.40, n_border_atyp),
    ])
    atypical = np.vstack([core_atypical, border_atypical])

    X = np.vstack([typical, atypical])
    y = np.array([0] * n_typical + [1] * n_atypical)  # 0=typical, 1=atypical
    return X, y


FEATURE_NAMES = [
    "face_attention_ratio",
    "face_fixation_count_norm",
    "object_fixation_count_norm",
    "saccade_count_norm",
    "gaze_variability",
]


class AutismGazeClassifier:
    """
    Random Forest classifier for autism screening based on gaze features.
    """

    DISCLAIMER = (
        "⚠️ IMPORTANT DISCLAIMER: This is a research screening tool ONLY.\n"
        "It does NOT provide a medical diagnosis of Autism Spectrum Disorder.\n"
        "Results should be interpreted by qualified professionals.\n"
        "Please consult a healthcare provider for proper evaluation."
    )

    def __init__(self):
        if not HAS_SKLEARN:
            raise ImportError(
                "scikit-learn is required: pip install scikit-learn"
            )
        self.clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self._trained = False
        self._train_baseline()

    def _train_baseline(self):
        """Train on synthetic reference data."""
        X, y = _generate_reference_data()
        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y)
        self._trained = True

    def extract_features(self, test_results):
        """
        Extract classifier features from test session results.

        Parameters
        ----------
        test_results : dict
            Output from test_session.run_test_session()

        Returns
        -------
        numpy array of shape (1, 5) — feature vector
        """
        per_image = test_results["per_image"]
        if not per_image:
            return None

        ratios = [r["face_attention_ratio"] for r in per_image]
        face_samples = [r["face_samples"] for r in per_image]
        obj_samples = [r["object_samples"] for r in per_image]
        total_samples = [r["total_samples"] for r in per_image]

        overall_face_ratio = np.mean(ratios)

        # Normalized fixation counts
        total = sum(total_samples) or 1
        face_fix_norm = sum(face_samples) / total
        obj_fix_norm = sum(obj_samples) / total

        # Saccade-like metric: how much the face ratio varies between images
        saccade_norm = np.std(ratios) if len(ratios) > 1 else 0

        # Gaze variability: coefficient of variation of face attention
        gaze_var = (np.std(ratios) / (np.mean(ratios) + 1e-6))

        features = np.array([[
            overall_face_ratio,
            face_fix_norm,
            obj_fix_norm,
            saccade_norm,
            gaze_var,
        ]])

        return features

    def classify(self, test_results):
        """
        Run classification on test results.

        Returns
        -------
        dict with:
            risk_score (0–100): probability of atypical pattern
            classification: "Typical" / "Moderate Risk" / "High Risk"
            features: extracted feature values
            feature_names: corresponding names
            disclaimer: medical disclaimer text
        """
        features = self.extract_features(test_results)
        if features is None:
            return {"error": "No valid test data", "disclaimer": self.DISCLAIMER}

        X_scaled = self.scaler.transform(features)
        proba = self.clf.predict_proba(X_scaled)[0]  # [prob_typical, prob_atypical]
        risk_score = proba[1] * 100  # Probability of atypical

        # Also use threshold-based classification as a cross-check
        face_ratio = test_results["overall_face_attention_ratio"]
        if face_ratio < config.RISK_THRESHOLD_HIGH:
            threshold_class = "High Risk"
        elif face_ratio < config.RISK_THRESHOLD_MODERATE:
            threshold_class = "Moderate Risk"
        else:
            threshold_class = "Typical"

        # ML-based classification
        if risk_score >= 65:
            ml_class = "High Risk"
        elif risk_score >= 40:
            ml_class = "Moderate Risk"
        else:
            ml_class = "Typical"

        # Final classification: combine both signals
        risk_levels = {"Typical": 0, "Moderate Risk": 1, "High Risk": 2}
        combined = max(risk_levels[threshold_class], risk_levels[ml_class])
        classification = {0: "Typical", 1: "Moderate Risk", 2: "High Risk"}[combined]

        return {
            "risk_score": round(risk_score, 1),
            "classification": classification,
            "ml_classification": ml_class,
            "threshold_classification": threshold_class,
            "face_attention_ratio": face_ratio,
            "features": features[0].tolist(),
            "feature_names": FEATURE_NAMES,
            "disclaimer": self.DISCLAIMER,
        }
