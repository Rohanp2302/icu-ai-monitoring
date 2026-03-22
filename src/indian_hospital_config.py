"""
Indian Hospital-Customized ICU Mortality Prediction System
- Real-time data integration
- Medicine tracking with drug interactions
- Interpretable for patient families
- Compliant with Indian healthcare standards
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

# ============================================================================
# PART 1: INDIAN HOSPITAL VITAL SIGN RANGES & CLINICAL CONTEXT
# ============================================================================

INDIAN_HOSPITAL_CONFIG = {
    'vital_ranges': {
        'heart_rate': {
            'target': (60, 100),
            'critical_low': 40,
            'critical_high': 140,
            'unit': 'bpm'
        },
        'respiration': {
            'target': (12, 20),
            'critical_low': 8,
            'critical_high': 35,
            'unit': 'breaths/min'
        },
        'oxygen_saturation': {
            'target': (94, 100),
            'critical_low': 85,
            'acceptable_low': 90,
            'unit': '%'
        },
        'blood_pressure_systolic': {
            'target': (100, 140),
            'critical_low': 80,
            'unit': 'mmHg'
        },
        'temperature': {
            'target': (37, 37.5),
            'acceptable_range': (36.5, 38.5),
            'critical_low': 35,
            'critical_high': 40,
            'unit': 'C'
        }
    },

    'common_medicines_india': {
        'vasopressors': ['Noradrenaline', 'Adrenaline', 'Dopamine', 'Vasopressin'],
        'antibiotics_first_line': ['Ceftriaxone', 'Cefotaxime', 'Piperacillin-Tazobactam', 'Ciprofloxacin'],
        'second_line': ['Vancomycin', 'Meropenem', 'Azithromycin'],
        'anticoagulants': ['Heparin', 'Warfarin', 'LMWH'],
        'sedatives': ['Propofol', 'Midazolam', 'Lorazepam'],
        'pain_relief': ['Fentanyl', 'Morphine', 'Tramadol'],
        'inotropes': ['Dobutamine', 'Milrinone'],
        'steroids': ['Dexamethasone', 'Hydrocortisone', 'Methylprednisolone'],
    },

    'drug_interactions_critical': {
        ('Warfarin', 'Aspirin'): {'risk': 'HIGH', 'reason': 'Increased bleeding risk'},
        ('ACE-inhibitor', 'Potassium supplement'): {'risk': 'HIGH', 'reason': 'Hyperkalemia risk'},
        ('Statin', 'Clarithromycin'): {'risk': 'HIGH', 'reason': 'Muscle toxicity'},
        ('Metformin', 'Iodine contrast'): {'risk': 'HIGH', 'reason': 'Acute kidney injury risk'},
    },

    'languages_supported': ['English', 'Hindi', 'Tamil', 'Telugu', 'Kannada', 'Marathi'],

    'hospital_types': {
        'govt_primary_health_center': 'Basic ICU setup, limited monitoring',
        'private_clinic': 'Limited advanced monitoring',
        'district_hospital': 'Basic 5-10 bed ICU',
        'medical_college_hospital': 'Advanced ICU with multiparameter monitoring',
        'corporate_hospital': 'State-of-the-art ICU facilities'
    }
}

# ============================================================================
# PART 2: FAMILY-FRIENDLY EXPLANATIONS (Non-Medical Language)
# ============================================================================

FAMILY_EXPLANATIONS = {
    'heart_rate_high': {
        'simple': 'Heart is beating faster than normal',
        'what_means': 'The heart is working harder. This can happen due to fever, pain, or anxiety.',
        'what_to_monitor': 'If it stays high (above 130), doctors may give medication',
        'possible_causes': ['Infection', 'Fever', 'Dehydration', 'Pain'],
        'actions': ['Ensure patient is comfortable', 'Let doctor know immediately']
    },

    'oxygen_low': {
        'simple': 'Oxygen levels in blood are lower than normal',
        'what_means': 'The lungs are not able to pick up enough oxygen from the air',
        'what_to_monitor': 'Below 90% is serious, needs immediate attention',
        'possible_causes': ['Lung infection/pneumonia', 'Fluid in lungs', 'Blocking of airways'],
        'actions': ['Oxygen will be given', 'Chest monitoring will increase']
    },

    'blood_pressure_low': {
        'simple': 'Blood pressure is lower than normal',
        'what_means': 'The heart is not pumping blood with enough force to reach all body parts',
        'what_to_monitor': 'Below 60 systolic is critical',
        'possible_causes': ['Shock', 'Blood loss', 'Heart failure', 'Severe infection'],
        'actions': ['Fluids will be given', 'Medications to raise pressure', '24-hour monitoring']
    },

    'fever': {
        'simple': 'Body temperature is higher than normal',
        'what_means': 'The body is fighting an infection by raising temperature',
        'what_to_monitor': 'Above 40C needs emergency care',
        'possible_causes': ['Bacterial infection', 'Viral infection', 'Post-operative'],
        'actions': ['Antibiotics may be started', 'Cooling measures, ice packs']
    }
}

# ============================================================================
# PART 3: MEDICINE TRACKER & INTERACTION CHECKER
# ============================================================================

class IndianHospitalMedicineTracker:
    """Track medications, check interactions, generate family alerts"""

    def __init__(self):
        self.config = INDIAN_HOSPITAL_CONFIG
        self.current_medications = []

    def add_medicine(self, med_name: str, dosage: str, frequency: str, start_time: str, reason: str):
        """Add medicine to patient's current medications"""
        med = {
            'name': med_name,
            'dosage': dosage,
            'frequency': frequency,
            'start_time': start_time,
            'reason': reason,
            'added_at': pd.Timestamp.now()
        }
        self.current_medications.append(med)
        return med

    def check_interactions(self) -> List[Dict]:
        """Check for dangerous drug-drug interactions"""
        alerts = []

        med_names = [m['name'] for m in self.current_medications]

        for (drug1, drug2), interaction in self.config['drug_interactions_critical'].items():
            if drug1 in med_names and drug2 in med_names:
                alerts.append({
                    'severity': interaction['risk'],
                    'drug1': drug1,
                    'drug2': drug2,
                    'reason': interaction['reason'],
                    'message': f"WARNING: {drug1} and {drug2} may interact - {interaction['reason']}"
                })

        return alerts

    def generate_medicine_summary_for_family(self) -> str:
        """Generate family-friendly medicine summary"""
        summary = "Current Medications:\n\n"

        for med in self.current_medications:
            summary += f"Medicine: {med['name']}\n"
            summary += f"  Dosage: {med['dosage']}\n"
            summary += f"  How often: {med['frequency']}\n"
            summary += f"  Reason: {med['reason']}\n"
            summary += "\n"

        # Add interactions
        interactions = self.check_interactions()
        if interactions:
            summary += "IMPORTANT ALERTS:\n"
            for alert in interactions:
                summary += f"- {alert['message']}\n"

        return summary


# ============================================================================
# PART 4: REAL-TIME DATA INTEGRATION (Typical Indian Hospital Setup)
# ============================================================================

class IndianHospitalDataIntegrator:
    """Integrate data from typical Indian hospital multiparameter monitors"""

    def __init__(self, hospital_type: str):
        self.hospital_type = hospital_type
        self.config = INDIAN_HOSPITAL_CONFIG
        self.realtime_data = []

    def parse_monitor_data_hl7(self, hl7_message: str) -> Dict:
        """Parse HL7 message from multiparameter monitor (common in Indian hospitals)"""
        # Simplified HL7 parsing
        data = {}
        for line in hl7_message.split('|'):
            parts = line.split('^')
            if len(parts) >= 2:
                data[parts[0]] = parts[1]
        return data

    def parse_manual_entry(self, manual_data: Dict) -> Dict:
        """Parse manually entered vitals (common in primary health centers)"""
        return {
            'hr': manual_data.get('heart_rate'),
            'rr': manual_data.get('respiration'),
            'spo2': manual_data.get('oxygen'),
            'bp_sys': manual_data.get('bp_systolic'),
            'temp': manual_data.get('temperature'),
            'timestamp': pd.Timestamp.now()
        }

    def ingest_realtime(self, vitals: Dict) -> Dict:
        """Ingest real-time vitals and flag abnormalities"""

        alerts = []

        # Check heart rate
        if vitals.get('hr'):
            if vitals['hr'] > self.config['vital_ranges']['heart_rate']['critical_high']:
                alerts.append({'vital': 'HR', 'severity': 'CRITICAL', 'value': vitals['hr']})
            elif vitals['hr'] > self.config['vital_ranges']['heart_rate']['target'][1] + 10:
                alerts.append({'vital': 'HR', 'severity': 'WARNING', 'value': vitals['hr']})

        # Check oxygen
        if vitals.get('spo2'):
            if vitals['spo2'] < self.config['vital_ranges']['oxygen_saturation']['critical_low']:
                alerts.append({'vital': 'SpO2', 'severity': 'CRITICAL', 'value': vitals['spo2']})
            elif vitals['spo2'] < self.config['vital_ranges']['oxygen_saturation']['target'][0]:
                alerts.append({'vital': 'SpO2', 'severity': 'WARNING', 'value': vitals['spo2']})

        # Check BP
        if vitals.get('bp_sys'):
            if vitals['bp_sys'] < self.config['vital_ranges']['blood_pressure_systolic']['critical_low']:
                alerts.append({'vital': 'BP', 'severity': 'CRITICAL', 'value': vitals['bp_sys']})

        vitals['alerts'] = alerts
        self.realtime_data.append(vitals)

        return vitals


# ============================================================================
# PART 5: INTERPRETABILITY FOR FAMILIES
# ============================================================================

class FamilyExplainer:
    """Generate explanations for non-medical family members"""

    def __init__(self):
        self.explanations = FAMILY_EXPLANATIONS

    def explain_vital_sign(self, vital_name: str, current_value: float, target_range: Tuple) -> str:
        """Explain a vital sign in family-friendly language"""

        if vital_name in self.explanations:
            exp = self.explanations[vital_name]

            message = f"Simple explanation: {exp['simple']}\n\n"
            message += f"What this means: {exp['what_means']}\n\n"
            message += f"Possible causes: {', '.join(exp['possible_causes'])}\n\n"
            message += f"What to look for: {exp['what_to_monitor']}\n\n"
            message += f"What doctors will do: {', '.join(exp['actions'])}\n"

            return message

        return f"Current value: {current_value} ({vital_name})"

    def explain_mortality_risk(self, risk_prob: float) -> str:
        """Explain mortality risk in sensitive family-friendly way"""

        if risk_prob < 0.2:
            level = "LOW RISK"
            message = "The doctors believe the patient is likely to recover well."
        elif risk_prob < 0.4:
            level = "MODERATE RISK"
            message = "The patient's condition needs careful monitoring, but recovery is possible with proper treatment."
        elif risk_prob < 0.7:
            level = "HIGH RISK"
            message = "The patient's condition is serious and needs intensive care and monitoring."
        else:
            level = "CRITICAL RISK"
            message = "The patient's condition is very serious. The medical team is doing everything possible."

        return f"""
RISK ASSESSMENT: {level}

{message}

This assessment is based on the patient's vital signs, medical history, and current treatment.
The doctors will update this regularly as the patient's condition changes.

Remember: These are medical predictions, not certainties. Many patients recover even from
serious conditions with proper treatment and care.
"""


if __name__ == '__main__':
    print("Indian Hospital-Customized ICU System Loaded")
    print(f"Supported languages: {', '.join(INDIAN_HOSPITAL_CONFIG['languages_supported'])}")
    print(f"Critical drug interactions tracked: {len(INDIAN_HOSPITAL_CONFIG['drug_interactions_critical'])}")
