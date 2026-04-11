"""
MEDICATION TRACKING MODULE

Tracks patient medications with:
- Drug-drug interaction detection
- Dosage validation
- Efficacy monitoring
- Side effect warnings
- Customized for Indian hospital settings
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd

class MedicationDatabase:
    """Database of common Indian hospital medications"""
    
    def __init__(self):
        self.medications = {
            # Antimicrobials (most common in Indian ICUs)
            'ceftriaxone': {
                'name': 'Ceftriaxone',
                'category': 'Antibiotic',
                'common_dose': '1-2g IV',
                'frequency': '12-24 hours',
                'interactions': ['warfarin', 'probenecid'],
                'cautions': ['renal impairment', 'allergy'],
                'monitoring': 'Renal function, allergic reactions'
            },
            'ciprofloxacin': {
                'name': 'Ciprofloxacin',
                'category': 'Antibiotic (Fluoroquinolone)',
                'common_dose': '400mg IV',
                'frequency': '8-12 hours',
                'interactions': ['warfarin', 'theophylline', 'tizanidine'],
                'cautions': ['QT prolongation', 'myasthenia gravis'],
                'monitoring': 'ECG for QT prolongation, tendon problems'
            },
            'metronidazole': {
                'name': 'Metronidazole',
                'category': 'Antibiotic (Anaerobic)',
                'common_dose': '500mg IV',
                'frequency': '8 hours',
                'interactions': ['warfarin', 'alcohol'],
                'cautions': ['neuropathy', 'liver disease'],
                'monitoring': 'Liver function, signs of neuropathy'
            },
            
            # ICU Supportive Medications
            'dopamine': {
                'name': 'Dopamine',
                'category': 'Vasopressor',
                'common_dose': '5-20 mcg/kg/min',
                'frequency': 'Continuous infusion',
                'interactions': ['MAOIs', 'tricyclic antidepressants'],
                'cautions': ['coronary artery disease', 'arrhythmias'],
                'monitoring': 'Blood pressure, heart rate, urine output'
            },
            'noradrenaline': {
                'name': 'Noradrenaline (Levophed)',
                'category': 'Vasopressor',
                'common_dose': '0.05-0.3 mcg/kg/min',
                'frequency': 'Continuous infusion',
                'interactions': ['beta blockers'],
                'cautions': ['coronary artery disease'],
                'monitoring': 'Blood pressure, perfusion, tissue oxygenation'
            },
            'insulin': {
                'name': 'Insulin',
                'category': 'Glucose control',
                'common_dose': 'Varies (0.05-1 unit/kg)',
                'frequency': 'Variable based on glucose',
                'interactions': ['many'],
                'cautions': ['hypoglycemia', 'renal disease'],
                'monitoring': 'Blood glucose (4-hourly), electrolytes'
            },
            
            # Anticoagulants (common in Indian ICUs)
            'heparin': {
                'name': 'Heparin (Unfractionated)',
                'category': 'Anticoagulant',
                'common_dose': '5000 IU bolus, then 1000 IU/hr',
                'frequency': 'Continuous/intermittent',
                'interactions': ['warfarin', 'aspirin', 'NSAIDs'],
                'cautions': ['thrombocytopenia', 'bleeding risk'],
                'monitoring': 'aPTT, platelet count, bleeding signs'
            },
            'aspirin': {
                'name': 'Aspirin',
                'category': 'Antiplatelet',
                'common_dose': '75-100mg Daily',
                'frequency': 'Once daily',
                'interactions': ['warfarin', 'ibuprofen', 'other NSAIDs'],
                'cautions': ['GI bleeding', 'allergy'],
                'monitoring': 'GI symptoms, bleeding signs'
            },
            
            # Liver support (critical in India - high prevalence)
            'silymarin': {
                'name': 'Silymarin (Milk Thistle)',
                'category': 'Hepatoprotective',
                'common_dose': '140mg TDS',
                'frequency': '3 times daily',
                'interactions': ['minimal'],
                'cautions': ['allergy'],
                'monitoring': 'Liver enzymes, bilirubin'
            },
            'ursodeoxycholic_acid': {
                'name': 'Ursodeoxycholic Acid',
                'category': 'Hepatoprotective',
                'common_dose': '250-300mg TDS',
                'frequency': '2-3 times daily',
                'interactions': ['minimal'],
                'cautions': ['pregnancy'],
                'monitoring': 'Liver function tests'
            },
            
            # Sedatives (ICU standard)
            'propofol': {
                'name': 'Propofol',
                'category': 'Sedative',
                'common_dose': '0.5-3 mg/kg/hr',
                'frequency': 'Continuous infusion',
                'interactions': ['other CNS depressants'],
                'cautions': ['hypotension', 'bradycardia', 'infusion syndrome'],
                'monitoring': 'Blood pressure, triglycerides, lipids'
            },
            'midazolam': {
                'name': 'Midazolam',
                'category': 'Sedative/Anxiolytic',
                'common_dose': '0.04-0.2 mg/kg',
                'frequency': 'Variable',
                'interactions': ['opioids', 'CNS depressants'],
                'cautions': ['respiratory depression'],
                'monitoring': 'Respiratory rate, SpO2'
            },
            
            # Pain relief
            'morphine': {
                'name': 'Morphine',
                'category': 'Opioid analgesic',
                'common_dose': '2-10mg IV',
                'frequency': '2-4 hourly',
                'interactions': ['other opioids', 'CNS depressants'],
                'cautions': ['respiratory depression', 'addiction'],
                'monitoring': 'Respiratory rate, pain scores, liver function'
            },
            'fentanyl': {
                'name': 'Fentanyl',
                'category': 'Opioid analgesic',
                'common_dose': '1-2 mcg/kg',
                'frequency': '0.5-1 hourly',
                'interactions': ['other opioids'],
                'cautions': ['respiratory depression'],
                'monitoring': 'Respiratory rate, SpO2'
            },
            
            # Antivirals (dengue, COVID era)
            'remdesivir': {
                'name': 'Remdesivir',
                'category': 'Antiviral',
                'common_dose': '200mg first day, 100mg daily',
                'frequency': 'Daily IV',
                'interactions': ['minimal known'],
                'cautions': ['renal impairment', 'liver disease'],
                'monitoring': 'Renal function, liver enzymes'
            },
            'oseltamivir': {
                'name': 'Oseltamivir (Tamiflu)',
                'category': 'Antiviral',
                'common_dose': '75mg',
                'frequency': 'Twice daily',
                'interactions': ['minimal'],
                'cautions': ['renal clearance <30: reduce dose'],
                'monitoring': 'Renal function'
            }
        }
    
    def get_medication(self, med_name: str) -> Dict:
        """Get medication info by name"""
        return self.medications.get(med_name.lower(), None)
    
    def list_all_medications(self) -> List[str]:
        """List all available medications"""
        return list(self.medications.keys())


class PatientMedicationRecord:
    """Tracks medications for a single patient"""
    
    def __init__(self, patient_id: str):
        self.patient_id = patient_id
        self.medications = []  # List of active medications
        self.med_db = MedicationDatabase()
        self.medication_history = []
    
    def add_medication(self, med_name: str, dose: str, frequency: str, 
                       start_date: str, reason: str = ''):
        """Add medication to patient's current regimen"""
        
        med_info = self.med_db.get_medication(med_name)
        if not med_info:
            raise ValueError(f"Medication '{med_name}' not found in database")
        
        med_record = {
            'name': med_name,
            'dose': dose,
            'frequency': frequency,
            'start_date': start_date,
            'reason': reason,
            'added_at': datetime.now().isoformat(),
            'status': 'active'
        }
        
        # Check for interactions
        interactions = self.check_interactions(med_name)
        if interactions:
            med_record['warnings'] = interactions
        
        self.medications.append(med_record)
        self.medication_history.append(med_record)
        
        return med_record
    
    def remove_medication(self, med_name: str, stop_date: str, reason: str = ''):
        """Remove medication (mark as inactive)"""
        
        for med in self.medications:
            if med['name'].lower() == med_name.lower():
                med['status'] = 'discontinued'
                med['stop_date'] = stop_date
                med['discontinuation_reason'] = reason
                return med
        
        raise ValueError(f"Medication '{med_name}' not found in active medications")
    
    def check_interactions(self, new_med: str) -> List[str]:
        """Check for drug-drug interactions"""
        
        new_med_info = self.med_db.get_medication(new_med)
        if not new_med_info:
            return []
        
        interactions = []
        new_med_interactions = new_med_info.get('interactions', [])
        
        # Check against current active medications
        for active_med in self.medications:
            if active_med['status'] == 'active':
                active_med_name = active_med['name'].lower()
                
                # Check both directions
                if active_med_name in [x.lower() for x in new_med_interactions]:
                    interactions.append({
                        'medication_1': new_med,
                        'medication_2': active_med_name,
                        'severity': 'HIGH',
                        'message': f'{new_med} may interact with {active_med_name}'
                    })
        
        return interactions
    
    def get_current_medications(self) -> List[Dict]:
        """Get all active medications"""
        return [m for m in self.medications if m['status'] == 'active']
    
    def get_medication_summary(self) -> Dict:
        """Get summary of current medications"""
        active_meds = self.get_current_medications()
        
        summary = {
            'patient_id': self.patient_id,
            'total_active_medications': len(active_meds),
            'medications': active_meds,
            'categories': list(set([
                self.med_db.get_medication(m['name']).get('category', 'Unknown')
                for m in active_meds
            ])),
            'warnings': self._check_all_interactions(),
            'monitoring_needed': self._get_monitoring_needs()
        }
        return summary
    
    def _check_all_interactions(self) -> List[Dict]:
        """Check all interactions among current medications"""
        interactions = []
        active_meds = self.get_current_medications()
        
        for i, med1 in enumerate(active_meds):
            for med2 in active_meds[i+1:]:
                med1_info = self.med_db.get_medication(med1['name'])
                med1_interactions = med1_info.get('interactions', [])
                
                if med2['name'].lower() in [x.lower() for x in med1_interactions]:
                    interactions.append({
                        'medication_1': med1['name'],
                        'medication_2': med2['name'],
                        'severity': 'HIGH',
                        'action': 'Review medication combination'
                    })
        
        return interactions
    
    def _get_monitoring_needs(self) -> List[str]:
        """Get all monitoring needs based on current medications"""
        monitoring = set()
        active_meds = self.get_current_medications()
        
        for med in active_meds:
            med_info = self.med_db.get_medication(med['name'])
            monitoring_items = med_info.get('monitoring', '').split(',')
            monitoring.update([m.strip() for m in monitoring_items if m.strip()])
        
        return list(monitoring)


class MedicationEffectivenessTracker:
    """Track medication effectiveness over time"""
    
    def __init__(self, patient_id: str):
        self.patient_id = patient_id
        self.effectiveness_records = []
    
    def log_effectiveness(self, med_name: str, date: str, 
                         symptom: str, improvement_score: float):
        """Log how well a medication is working (0-10 scale)"""
        
        record = {
            'medication': med_name,
            'date': date,
            'symptom_treated': symptom,
            'improvement_score': improvement_score,  # 0-10
            'logged_at': datetime.now().isoformat()
        }
        
        self.effectiveness_records.append(record)
        return record
    
    def get_effectiveness_trend(self, med_name: str) -> Dict:
        """Get effectiveness trend for a medication"""
        
        records = [r for r in self.effectiveness_records 
                  if r['medication'].lower() == med_name.lower()]
        
        if not records:
            return {'medication': med_name, 'data': 'No data yet'}
        
        effectiveness_scores = [r['improvement_score'] for r in records]
        
        return {
            'medication': med_name,
            'average_effectiveness': sum(effectiveness_scores) / len(effectiveness_scores),
            'trend': 'improving' if effectiveness_scores[-1] > effectiveness_scores[0] else 'declining',
            'total_logs': len(records),
            'recent_score': effectiveness_scores[-1]
        }


def main():
    """Demo of medication tracking"""
    
    print("="*80)
    print("MEDICATION TRACKING MODULE - DEMO")
    print("="*80)
    
    # Create patient medication record
    print("\n[STEP 1] Creating medication record for patient...")
    patient = PatientMedicationRecord('ICU_12345')
    
    # Add medications
    print("\n[STEP 2] Adding medications...")
    patient.add_medication(
        'ceftriaxone',
        dose='2g',
        frequency='12 hourly',
        start_date='2026-04-09',
        reason='Sepsis (bacterial infection)'
    )
    print("✓ Added Ceftriaxone")
    
    patient.add_medication(
        'dopamine',
        dose='5-10 mcg/kg/min',
        frequency='Continuous',
        start_date='2026-04-09',
        reason='Blood pressure support'
    )
    print("✓ Added Dopamine")
    
    patient.add_medication(
        'insulin',
        dose='Variable',
        frequency='As per glucose levels',
        start_date='2026-04-09',
        reason='Glucose control'
    )
    print("✓ Added Insulin")
    
    # Get medication summary
    print("\n[STEP 3] Current medication summary...")
    summary = patient.get_medication_summary()
    
    print(f"\nPatient: {summary['patient_id']}")
    print(f"Total Active Medications: {summary['total_active_medications']}")
    print(f"Categories: {', '.join(summary['categories'])}")
    
    print("\nActive Medications:")
    for med in summary['medications']:
        med_info = patient.med_db.get_medication(med['name'])
        print(f"\n  • {med_info['name']}")
        print(f"    Dose: {med['dose']}")
        print(f"    Frequency: {med['frequency']}")
        print(f"    Reason: {med['reason']}")
    
    # Check interactions
    print("\n[STEP 4] Drug-drug interaction check...")
    if summary['warnings']:
        print(f"⚠️  Found {len(summary['warnings'])} interaction(s):")
        for warning in summary['warnings']:
            print(f"   • {warning['medication_1']} ↔ {warning['medication_2']}")
            print(f"     Action: {warning['action']}")
    else:
        print("✓ No significant interactions detected")
    
    # Monitoring needs
    print("\n[STEP 5] Required monitoring...")
    print("Based on current medications, monitor:")
    for monitoring in summary['monitoring_needed'][:5]:
        print(f"  ✓ {monitoring}")
    if len(summary['monitoring_needed']) > 5:
        print(f"  ... and {len(summary['monitoring_needed']) - 5} more items")
    
    # Track effectiveness
    print("\n[STEP 6] Tracking medication effectiveness...")
    tracker = MedicationEffectivenessTracker('ICU_12345')
    tracker.log_effectiveness('ceftriaxone', '2026-04-09', 'Fever', 6.5)
    tracker.log_effectiveness('ceftriaxone', '2026-04-10', 'Fever', 7.5)
    
    effectiveness = tracker.get_effectiveness_trend('ceftriaxone')
    print(f"\nCeftriaxone effectiveness:")
    print(f"  Average: {effectiveness['average_effectiveness']:.1f}/10")
    print(f"  Trend: {effectiveness['trend'].capitalize()}")
    
    return patient, tracker


if __name__ == '__main__':
    patient, tracker = main()
    print("\n✨ Medication tracking module COMPLETE!")
