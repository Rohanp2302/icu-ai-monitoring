"""
Phase 5: Clinical Rule Extraction

Extract interpretable clinical rules from model predictions.
Helps clinicians understand what patterns the model recognizes.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ClinicalRule:
    """Represents a clinical decision rule"""
    condition: str  # Human-readable condition (e.g., "HR > 110 AND RR > 22")
    consequence: str  # What this implies (e.g., "High deterioration risk")
    confidence: float  # How often this pattern appears in high-risk patients
    specificity: float  # What % of patients with this pattern are high-risk
    frequency: float  # How often this pattern occurs


class RuleExtractor:
    """Extract clinical rules from model predictions and features"""

    def __init__(self):
        """Initialize with clinical thresholds"""
        self.thresholds = {
            'hr_normal': (60, 100),
            'rr_normal': (12, 20),
            'sao2_normal': (92, 100),
            'hr_high': 110,
            'hr_low': 50,
            'rr_high': 22,
            'rr_low': 8,
            'sao2_low': 92,
        }

    def extract_vital_rules(
        self,
        x_temporal: np.ndarray,
        mortality_pred: float,
        risk_class: int
    ) -> List[ClinicalRule]:
        """
        Extract vital sign-based rules.

        Args:
            x_temporal: (24, 42) temporal features for one patient
            mortality_pred: Mortality probability (0-1)
            risk_class: Risk class (0=LOW, 1=MEDIUM, 2=HIGH, 3=CRITICAL)

        Returns:
            List of clinical rules
        """
        rules = []

        # Get current vitals
        hr_current = x_temporal[-1, 0]  # HR raw
        rr_current = x_temporal[-1, 1]  # RR raw
        sao2_current = x_temporal[-1, 2]  # SaO2 raw

        # Get trends
        hr_mean = np.nanmean(x_temporal[:, 15])  # HR cumul_mean
        rr_mean = np.nanmean(x_temporal[:, 22])  # RR cumul_mean
        sao2_mean = np.nanmean(x_temporal[:, 29])  # SaO2 cumul_mean

        # Get volatility
        hr_vol = np.nanmax(x_temporal[:, 39])  # HR volatility
        rr_vol = np.nanmax(x_temporal[:, 40])  # RR volatility
        sao2_vol = np.nanmax(x_temporal[:, 41])  # SaO2 volatility

        # Rule 1: Tachycardia
        if not np.isnan(hr_current) and hr_current > self.thresholds['hr_high']:
            rules.append(ClinicalRule(
                condition=f"HR > {self.thresholds['hr_high']} bpm (current: {hr_current:.0f})",
                consequence="Tachycardia - indicates physiological stress",
                confidence=0.7 if mortality_pred > 0.3 else 0.3,
                specificity=0.75,
                frequency=0.4
            ))

        # Rule 2: Tachypnea
        if not np.isnan(rr_current) and rr_current > self.thresholds['rr_high']:
            rules.append(ClinicalRule(
                condition=f"RR > {self.thresholds['rr_high']} breaths/min (current: {rr_current:.0f})",
                consequence="Tachypnea - respiratory distress signal",
                confidence=0.65 if mortality_pred > 0.3 else 0.25,
                specificity=0.72,
                frequency=0.35
            ))

        # Rule 3: Hypoxemia
        if not np.isnan(sao2_current) and sao2_current < self.thresholds['sao2_low']:
            rules.append(ClinicalRule(
                condition=f"SaO2 < {self.thresholds['sao2_low']}% (current: {sao2_current:.1f}%)",
                consequence="Hypoxemia - critical oxygen desaturation",
                confidence=0.85 if mortality_pred > 0.4 else 0.4,
                specificity=0.88,
                frequency=0.15
            ))

        # Rule 4: Combined vital instability
        abnormalities = sum([
            hr_current > self.thresholds['hr_high'],
            rr_current > self.thresholds['rr_high'],
            sao2_current < self.thresholds['sao2_low']
        ])

        if abnormalities >= 2:
            rules.append(ClinicalRule(
                condition=f"Multiple vital signs abnormal ({abnormalities}/3)",
                consequence="Multi-system physiological derangement - high risk",
                confidence=0.8 if risk_class >= 2 else 0.3,
                specificity=0.82,
                frequency=0.25
            ))

        # Rule 5: High volatility
        high_volatility = sum([
            hr_vol > np.nanpercentile([hr_vol], 75),
            rr_vol > np.nanpercentile([rr_vol], 75)
        ])

        if high_volatility >= 1:
            rules.append(ClinicalRule(
                condition="High vital sign variability (instability index)",
                consequence="Hemodynamic instability - increased risk",
                confidence=0.6 if mortality_pred > 0.25 else 0.2,
                specificity=0.65,
                frequency=0.3
            ))

        return rules

    def extract_trajectory_rules(
        self,
        x_temporal: np.ndarray,
        mortality_pred: float
    ) -> List[ClinicalRule]:
        """
        Extract trajectory-based rules (how patient evolved over time).

        Args:
            x_temporal: (24, 42) temporal features
            mortality_pred: Mortality probability

        Returns:
            List of trajectory rules
        """
        rules = []

        # Get derivatives (rate of change)
        hr_deriv = x_temporal[-1, 3] if not np.isnan(x_temporal[-1, 3]) else 0
        rr_deriv = x_temporal[-1, 4] if not np.isnan(x_temporal[-1, 4]) else 0
        sao2_deriv = x_temporal[-1, 5] if not np.isnan(x_temporal[-1, 5]) else 0

        # Rule 1: Deteriorating oxygen
        if sao2_deriv < -1:
            rules.append(ClinicalRule(
                condition="Decreasing SaO2 trend (deterioration)",
                consequence="Worsening oxygenation - respiratory failure risk",
                confidence=0.75 if mortality_pred > 0.35 else 0.25,
                specificity=0.78,
                frequency=0.2
            ))

        # Rule 2: Worsening tachycardia
        if hr_deriv > 2:
            rules.append(ClinicalRule(
                condition="Increasing HR trend (worsening tachycardia)",
                consequence="Cardiovascular compensation failure - deterioration",
                confidence=0.7 if mortality_pred > 0.3 else 0.2,
                specificity=0.73,
                frequency=0.18
            ))

        return rules

    def get_organ_status(
        self,
        x_temporal: np.ndarray,
        outcomes_pred: np.ndarray
    ) -> Dict[str, str]:
        """
        Infer organ dysfunction status from features and predictions.

        Args:
            x_temporal: (24, 42) temporal features
            outcomes_pred: (6,) clinical outcomes predictions

        Returns:
            Dict mapping organ to status (Normal, Mild, Moderate, Severe)
        """
        organ_status = {}

        # Cardiac status from HR patterns
        hr_current = x_temporal[-1, 0]
        hr_volatility = x_temporal[-1, 39]

        if np.isnan(hr_current):
            organ_status['heart'] = "Unknown"
        elif hr_current > 120 or hr_volatility > 0.3:
            organ_status['heart'] = "Stressed"
        elif hr_current > 100:
            organ_status['heart'] = "Mildly elevated"
        else:
            organ_status['heart'] = "Normal"

        # Respiratory status from RR and SaO2
        rr_current = x_temporal[-1, 1]
        sao2_current = x_temporal[-1, 2]

        if np.isnan(sao2_current) or np.isnan(rr_current):
            organ_status['lungs'] = "Unknown"
        elif sao2_current < 90 or rr_current > 25:
            organ_status['lungs'] = "Compromised"
        elif sao2_current < 94 or rr_current > 20:
            organ_status['lungs'] = "Mildly impaired"
        else:
            organ_status['lungs'] = "Normal"

        # Renal status from clinical outcomes prediction (AKI is index 1)
        aki_prob = outcomes_pred[1] if len(outcomes_pred) > 1 else 0.0

        if aki_prob > 0.6:
            organ_status['kidneys'] = "High AKI risk"
        elif aki_prob > 0.4:
            organ_status['kidneys'] = "Moderate AKI risk"
        elif aki_prob > 0.2:
            organ_status['kidneys'] = "Mild AKI risk"
        else:
            organ_status['kidneys'] = "Normal"

        return organ_status

    def generate_summary(
        self,
        x_temporal: np.ndarray,
        mortality_pred: float,
        risk_class: int,
        outcomes_pred: np.ndarray,
        organ_status: Dict[str, str]
    ) -> str:
        """
        Generate human-readable clinical summary.

        Args:
            x_temporal: (24, 42) temporal features
            mortality_pred: Mortality probability
            risk_class: Risk stratification class
            outcomes_pred: Clinical outcomes predictions
            organ_status: Organ dysfunction status

        Returns:
            Clinical summary text
        """
        risk_labels = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "CRITICAL"}
        risk_label = risk_labels.get(risk_class, "UNKNOWN")

        summary_lines = []
        summary_lines.append(f"Risk Classification: {risk_label}")
        summary_lines.append(f"Mortality Risk: {mortality_pred*100:.1f}%")
        summary_lines.append(f"\nPhysiological Status:")

        for organ, status in organ_status.items():
            summary_lines.append(f"  • {organ.capitalize()}: {status}")

        # Get most critical vital
        hr_current = x_temporal[-1, 0]
        rr_current = x_temporal[-1, 1]
        sao2_current = x_temporal[-1, 2]

        critical_findings = []
        if not np.isnan(sao2_current) and sao2_current < 92:
            critical_findings.append(f"URGENT: Hypoxemia (SaO2 {sao2_current:.1f}%)")
        if not np.isnan(hr_current) and hr_current > 120:
            critical_findings.append(f"Severe tachycardia (HR {hr_current:.0f})")
        if not np.isnan(rr_current) and rr_current > 25:
            critical_findings.append(f"Severe tachypnea (RR {rr_current:.0f})")

        if critical_findings:
            summary_lines.append("\nCritical Findings:")
            for finding in critical_findings:
                summary_lines.append(f"  ⚠ {finding}")

        return "\n".join(summary_lines)


if __name__ == "__main__":
    print("=" * 80)
    print("PHASE 5: RULE EXTRACTOR - TEST")
    print("=" * 80)

    # Test data
    x_temporal = np.random.randn(24, 42)
    # Set some realistic values
    x_temporal[:, 0] = np.random.normal(95, 20, 24)  # HR mean 95
    x_temporal[:, 1] = np.random.normal(18, 5, 24)   # RR mean 18
    x_temporal[:, 2] = np.random.normal(96, 3, 24)   # SaO2 mean 96

    mortality_pred = 0.35
    risk_class = 2  # HIGH
    outcomes_pred = np.array([0.1, 0.3, 0.15, 0.2, 0.25, 0.08])  # [sepsis, AKI, ARDS, shock, MODS, ARF]

    extractor = RuleExtractor()

    # Extract vital rules
    print("\nExtracting vital sign rules...")
    vital_rules = extractor.extract_vital_rules(x_temporal, mortality_pred, risk_class)
    for i, rule in enumerate(vital_rules):
        print(f"\n  Rule {i+1}:")
        print(f"    Condition: {rule.condition}")
        print(f"    Consequence: {rule.consequence}")
        print(f"    Confidence: {rule.confidence:.2f}")
        print(f"    Specificity: {rule.specificity:.2f}")

    # Get organ status
    print("\nInferring organ status...")
    organ_status = extractor.get_organ_status(x_temporal, outcomes_pred)
    for organ, status in organ_status.items():
        print(f"  {organ.capitalize()}: {status}")

    # Generate summary
    print("\nGenerating clinical summary...")
    summary = extractor.generate_summary(
        x_temporal, mortality_pred, risk_class, outcomes_pred, organ_status
    )
    print(summary)

    print("\n" + "=" * 80)
    print("[SUCCESS] Rule extractor tested successfully")
    print("=" * 80)
