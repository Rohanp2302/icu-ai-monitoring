"""
COMPLETE HOSPITAL SYSTEM INTEGRATION

Integrates all modules:
1. Mortality Predictor (ML model)
2. Medication Tracking
3. Patient Communication Engine
4. India-Specific Feature Extractor

Creates a complete interpretable ML system for Indian hospitals
"""

import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

# Import all modules
from mortality_predictor import MortalityPredictor
from medication_tracking_module import PatientMedicationRecord, MedicationEffectivenessTracker
from patient_communication_engine import RiskCommunicator, ProgressTracker, GuidelinesCommunicator
from india_specific_feature_extractor import IndianHospitalAdapter


class ComprehensivePatientSystem:
    """Complete patient management system for Indian hospitals"""
    
    def __init__(self, patient_id: str, patient_name: str, gender: str = 'M'):
        """Initialize complete system for a patient"""
        
        self.patient_id = patient_id
        self.patient_name = patient_name
        self.gender = gender
        
        # Initialize all modules
        self.mortality_predictor = MortalityPredictor()
        self.medication_records = PatientMedicationRecord(patient_id)
        self.medication_tracker = MedicationEffectivenessTracker(patient_id)
        self.risk_communicator = RiskCommunicator()
        self.progress_tracker = ProgressTracker(patient_id)
        self.guidelines = GuidelinesCommunicator()
        self.india_adapter = IndianHospitalAdapter()
        
        print(f"✓ Initialized system for {patient_name} ({patient_id})")
    
    def process_patient_data(self, patient_data: dict) -> dict:
        """
        Process all patient data and generate comprehensive report
        
        Expected patient_data keys:
        - features (156-element array for mortality prediction)
        - lab_values (dict of lab names and values)
        - symptoms (list of symptoms)
        - medications (list of current medications)
        - condition (primary diagnosis)
        """
        
        print("\n" + "="*80)
        print("PROCESSING PATIENT DATA")
        print("="*80)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'patient_id': self.patient_id,
            'patient_name': self.patient_name,
            'gender': self.gender
        }
        
        # STEP 1: Mortality Risk Prediction
        print("\n[STEP 1] Running mortality prediction...")
        features = patient_data.get('features')
        if features is not None:
            mortality_prob = self.mortality_predictor.predict(features)[0]
            mortality_interpretation = self.mortality_predictor.predict_with_interpretation(features)
            
            report['mortality_prediction'] = {
                'probability': float(mortality_prob),
                'risk_level': mortality_interpretation['risk_level'],
                'interpretation': mortality_interpretation['interpretation'],
                'recommendation': mortality_interpretation['recommendation']
            }
            print(f"  Mortality Risk: {mortality_prob*100:.1f}% ({mortality_interpretation['risk_level']})")
        
        # STEP 2: India-Specific Analysis
        print("\n[STEP 2] Performing India-specific analysis...")
        india_data = patient_data.copy()
        india_data['gender'] = self.gender
        india_analysis = self.india_adapter.analyze_patient(india_data)
        
        report['india_analysis'] = {
            'lab_classification': india_analysis['lab_classification'],
            'disease_mapping': [
                {'name': d.get('name'), 'prevalence': d.get('prevalence_india')}
                for d in india_analysis['disease_mapping']
            ],
            'alerts': india_analysis['india_specific_alerts'],
            'cost_estimate': india_analysis['cost_estimate']
        }
        print(f"  Identified {len(india_analysis['disease_mapping'])} disease patterns")
        print(f"  Generated {len(india_analysis['india_specific_alerts'])} India-specific alerts")
        
        # STEP 3: Medication Management
        print("\n[STEP 3] Managing medications...")
        meds = patient_data.get('medications', [])
        for med in meds:
            self.medication_records.add_medication(
                med.get('name'),
                med.get('dose', 'As per doctor'),
                med.get('frequency', 'As per doctor'),
                med.get('start_date', datetime.now().strftime('%Y-%m-%d')),
                med.get('reason', '')
            )
        
        med_summary = self.medication_records.get_medication_summary()
        report['medications'] = {
            'total_active': med_summary['total_active_medications'],
            'categories': med_summary['categories'],
            'warnings': med_summary['warnings'],
            'monitoring_needs': med_summary['monitoring_needed']
        }
        print(f"  Managing {med_summary['total_active_medications']} medications")
        if med_summary['warnings']:
            print(f"  ⚠️  Drug interactions detected: {len(med_summary['warnings'])}")
        
        # STEP 4: Family-Friendly Communication
        print("\n[STEP 4] Generating family communication...")
        
        if features is not None:
            family_msg = self.risk_communicator.get_family_message(
                mortality_prob,
                patient_data.get('condition', 'health condition')
            )
            report['family_communication'] = family_msg
            print(f"  Risk level: {family_msg['emoji']} {family_msg['risk_level']}")
        
        # STEP 5: Progress Tracking Setup
        print("\n[STEP 5] Setting up progress tracking...")
        self.progress_tracker.log_daily_progress(
            datetime.now().strftime('%Y-%m-%d'),
            mortality_prob if features is not None else 0.5,
            patient_data.get('vital_status', 'Being assessed')
        )
        print("  Progress tracking initialized")
        
        # STEP 6: Generate Daily Summary
        print("\n[STEP 6] Creating daily patient summary...")
        summary_data = {
            'name': self.patient_name,
            'condition': patient_data.get('condition', 'Under medical care'),
            'mortality_probability': mortality_prob if features is not None else 0.5,
            'vital_status': patient_data.get('vital_status', 'Stable'),
            'nutrition_status': patient_data.get('nutrition_status', 'As per doctor'),
            'pain_level': patient_data.get('pain_level', 'Controlled'),
            'medicine_count': len(meds),
            'care_focus': patient_data.get('care_focus', 'Recovery'),
            'next_steps': patient_data.get('next_steps', 'Continue current treatment'),
            'trend': patient_data.get('trend', 'stable'),
            'medicines': meds
        }
        
        daily_summary = self.risk_communicator.create_daily_summary(summary_data)
        report['daily_summary'] = daily_summary
        
        return report
    
    def generate_complete_report(self, patient_data: dict, save_to_file: bool = True) -> str:
        """Generate complete integrated report"""
        
        # Process all data
        report_data = self.process_patient_data(patient_data)
        
        # Create formatted report
        report_text = self._format_complete_report(report_data)
        
        # Save if requested
        if save_to_file:
            output_dir = Path('results/patient_reports')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            filename = output_dir / f"{self.patient_id}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report_text)
            
            print(f"\n✓ Report saved to: {filename}")
        
        return report_text
    
    def _format_complete_report(self, report_data: dict) -> str:
        """Format report for display/printing"""
        
        report = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║              COMPREHENSIVE PATIENT ASSESSMENT REPORT                       ║
║                    INDIA-SPECIFIC SYSTEM v1.0                             ║
╚════════════════════════════════════════════════════════════════════════════╝

Generated: {report_data.get('timestamp', 'N/A')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATIENT INFORMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Patient ID: {report_data.get('patient_id', 'N/A')}
Patient Name: {report_data.get('patient_name', 'N/A')}
Gender: {report_data.get('gender', 'N/A')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MORTALITY RISK ASSESSMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
        
        mortality = report_data.get('mortality_prediction', {})
        if mortality:
            report += f"""
Risk Probability: {mortality.get('probability', 0)*100:.1f}%
Risk Level: {mortality.get('risk_level', 'N/A')}
Interpretation: {mortality.get('interpretation', 'N/A')}

Recommended Actions:
"""
            for rec in mortality.get('recommendation', []):
                report += f"  • {rec}\n"
        
        report += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INDIA-SPECIFIC ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Clinical Alerts:
"""
        
        india = report_data.get('india_analysis', {})
        for alert in india.get('alerts', []):
            report += f"  {alert}\n"
        
        cost = india.get('cost_estimate', {})
        report += f"""

Cost Estimation (Indian INR):
  • Daily Cost: ₹{cost.get('daily_cost', 0):,.0f}
  • Estimated Total: ₹{cost.get('total_estimated', 0):,.0f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MEDICATION MANAGEMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
        
        meds = report_data.get('medications', {})
        report += f"Total Active Medications: {meds.get('total_active', 0)}\n"
        report += f"Categories: {', '.join(meds.get('categories', []))}\n"
        
        if meds.get('warnings'):
            report += f"\n⚠️  Drug Interactions:\n"
            for warning in meds.get('warnings', []):
                report += f"  • {warning.get('medication_1')} ↔ {warning.get('medication_2')}\n"
        
        report += f"\nMonitoring Required:\n"
        for monitor in meds.get('monitoring_needs', [])[:5]:
            report += f"  ✓ {monitor}\n"
        
        report += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FAMILY COMMUNICATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
        
        family = report_data.get('family_communication', {})
        if family:
            report += f"{family.get('emoji', '')} {family.get('main_message', '')}\n"
            report += f"\nWhat to Expect: {family.get('what_to_expect', '')}\n"
            report += f"Monitoring Level: {family.get('monitoring_level', '')}\n"
        
        report += report_data.get('daily_summary', '')
        
        report += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SYSTEM INFORMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This report was generated using:
  ✓ RandomForest Mortality Prediction (AUC: 0.8835)
  ✓ Indian Hospital-Specific Customization
  ✓ Medication Interaction Checking
  ✓ Family-Friendly Communication Engine

All predictions are clinical decision support tools.
Final medical decisions rest with qualified physicians.

╚════════════════════════════════════════════════════════════════════════════╝
"""
        
        return report


def demo_complete_system():
    """Demo of complete integrated system"""
    
    print("="*80)
    print("COMPLETE HOSPITAL SYSTEM INTEGRATION - DEMO")
    print("="*80)
    
    # Create system for a patient
    system = ComprehensivePatientSystem(
        patient_id='ICU_20260409_001',
        patient_name='Rajesh Kumar Singh',
        gender='M'
    )
    
    # Load sample feature data
    print("\n[LOADING] Sample patient data...")
    
    try:
        enhanced_df = pd.read_csv('results/trajectory_features/combined_features_with_trajectory.csv')
        feature_cols = [c for c in enhanced_df.columns if c not in ['patientunitstayid', 'mortality']]
        sample_features = enhanced_df[feature_cols].iloc[0].values
    except:
        print("  Note: Using synthetic features for demo")
        sample_features = np.random.randn(156)  # 156 features
    
    # Create patient data
    patient_data = {
        'features': sample_features,
        'condition': 'Pneumonia with sepsis',
        'vital_status': 'Stable',
        'nutrition_status': 'NPO (nothing by mouth)',
        'pain_level': 'Controlled (2/10)',
        'care_focus': 'Treating infection and improving oxygenation',
        'next_steps': 'Continue antibiotics for 7 more days',
        'trend': 'improving',
        'lab_values': {
            'hemoglobin': 10.5,
            'platelets': 150000,
            'bilirubin': 1.2,
            'creatinine': 1.1,
            'albumin': 3.5,
            'ast': 35,
            'alt': 40,
            'wbc': 8500
        },
        'medications': [
            {
                'name': 'ceftriaxone',
                'dose': '2g IV',
                'frequency': '12 hourly',
                'start_date': '2026-04-09',
                'reason': 'Bacterial infection'
            },
            {
                'name': 'dopamine',
                'dose': '5-10 mcg/kg/min',
                'frequency': 'Continuous',
                'start_date': '2026-04-09',
                'reason': 'Blood pressure support'
            },
            {
                'name': 'insulin',
                'dose': 'Variable',
                'frequency': 'As per glucose',
                'start_date': '2026-04-09',
                'reason': 'Glycemic control'
            }
        ],
        'symptoms': ['fever', 'cough', 'shortness of breath']
    }
    
    # Generate report
    report = system.generate_complete_report(patient_data, save_to_file=True)
    
    # Print report
    print(report)
    
    return system


if __name__ == '__main__':
    system = demo_complete_system()
    print("\n✨ Complete system integration SUCCESSFUL!")
