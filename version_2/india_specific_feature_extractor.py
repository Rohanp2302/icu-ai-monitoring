"""
INDIA-SPECIFIC FEATURE EXTRACTOR

Customizes the ML model for Indian hospital settings by:
- Adjusting lab value reference ranges (Indian standards)
- Adding common Indian diseases
- Considering resource constraints
- Including cost-aware recommendations
- Adapting to Indian medication practices
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class IndianLabReferences:
    """Indian hospital lab value reference ranges"""
    
    def __init__(self):
        # References adjusted for Indian populations
        # Most Indian labs follow these ranges
        self.references = {
            'hemoglobin': {
                'male_normal': (13.5, 17.5),  # g/dL
                'female_normal': (12.0, 15.5),
                'critical_low': 7.0,
                'critical_high': 20.0,
                'unit': 'g/dL'
            },
            'hematocrit': {
                'male_normal': (40, 54),  # %
                'female_normal': (36, 46),
                'critical_low': 20,
                'critical_high': 60,
                'unit': '%'
            },
            'wbc': {
                'normal': (4000, 11000),  # /μL
                'critical_low': 2000,
                'critical_high': 30000,
                'unit': '/μL'
            },
            'platelets': {
                'normal': (150000, 400000),  # /μL
                'critical_low': 50000,  # Risk of bleeding
                'unit': '/μL'
            },
            'glucose': {
                'fasting_normal': (70, 100),  # mg/dL
                'random_normal': (70, 140),
                'critical_low': 50,
                'critical_high': 400,
                'unit': 'mg/dL'
            },
            'creatinine': {
                'male_normal': (0.7, 1.3),  # mg/dL
                'female_normal': (0.6, 1.1),
                'critical': 4.0,
                'unit': 'mg/dL'
            },
            'urea': {
                'normal': (7, 20),  # mg/dL (Indian labs often use this)
                'critical': 100,
                'unit': 'mg/dL'
            },
            'bilirubin': {
                'total_normal': (0.1, 1.2),  # mg/dL
                'direct_normal': (0.0, 0.3),
                'critical': 10.0,  # Hepatic encephalopathy risk
                'unit': 'mg/dL'
            },
            'albumin': {
                'normal': (3.5, 5.0),  # g/dL
                'critical_low': 2.0,  # Malnutrition/liver disease
                'unit': 'g/dL'
            },
            'ast': {
                'normal': (5, 40),  # IU/L
                'high': 100,
                'unit': 'IU/L'
            },
            'alt': {
                'normal': (7, 45),  # IU/L
                'high': 100,
                'unit': 'IU/L'
            },
            'sodium': {
                'normal': (136, 145),  # mEq/L
                'critical_low': 125,
                'critical_high': 155,
                'unit': 'mEq/L'
            },
            'potassium': {
                'normal': (3.5, 5.0),  # mEq/L
                'critical_low': 2.5,
                'critical_high': 6.5,
                'unit': 'mEq/L'
            },
            'calcium': {
                'normal': (8.5, 10.5),  # mg/dL
                'critical_low': 6.5,
                'critical_high': 13.0,
                'unit': 'mg/dL'
            },
            'phosphate': {
                'normal': (2.5, 4.5),  # mg/dL
                'unit': 'mg/dL'
            }
        }
    
    def get_reference(self, lab_name: str) -> Dict:
        """Get reference range for a lab"""
        return self.references.get(lab_name.lower(), {})
    
    def classify_value(self, lab_name: str, value: float, gender: str = 'M') -> str:
        """Classify lab value as normal/high/low/critical"""
        
        ref = self.get_reference(lab_name)
        if not ref:
            return 'unknown'
        
        lab_lower = lab_name.lower()
        
        # Handle gender-specific ranges
        if gender.upper() == 'F' and f'{gender.upper()}_normal' in str(ref):
            normal_range = ref.get(f'{gender.lower()}_normal')
        elif gender.upper() == 'M' and f'{gender.upper()}_normal' in str(ref):
            normal_range = ref.get(f'{gender.lower()}_normal')
        else:
            normal_range = ref.get('normal') or ref.get(f'{lab_lower}_normal')
        
        if not normal_range:
            return 'unknown'
        
        # Check critical values first
        if 'critical' in ref:
            if value <= ref.get('critical_low', -float('inf')):
                return 'CRITICAL_LOW'
            if value >= ref.get('critical', float('inf')):
                return 'CRITICAL_HIGH'
        
        # Check normal range
        if normal_range[0] <= value <= normal_range[1]:
            return 'NORMAL'
        
        if value < normal_range[0]:
            return 'LOW'
        else:
            return 'HIGH'


class IndianDiseaseSpecificFeatures:
    """Features for diseases common in Indian hospitals"""
    
    def __init__(self):
        self.disease_features = {
            'dengue': {
                'name': 'Dengue Fever',
                'prevalence_india': 'High (seasonal)',
                'key_markers': ['platelet_count', 'hematocrit', 'ast', 'alt', 'albumin'],
                'warning_signs': [
                    'Platelet drop >50%/day',
                    'Hematocrit rise >20%',
                    'Plasma leakage (ascites)',
                    'GI bleeding',
                    'Lethargy'
                ],
                'typical_treatment': 'Supportive - IV fluids, platelet transfusion',
                'critical_threshold': 'DHF Grade III/IV'
            },
            'tb_with_complications': {
                'name': 'Tuberculosis with complications',
                'prevalence_india': 'Very High',
                'key_markers': ['albumin', 'lymphocytes', 'chest_xray_findings'],
                'warning_signs': [
                    'Miliary TB (disseminated)',
                    'TB meningitis',
                    'TB pericarditis',
                    'Multi-drug resistant TB'
                ],
                'typical_treatment': 'Long-term anti-TB therapy',
                'critical_threshold': 'Extrapulmonary TB'
            },
            'malaria_severe': {
                'name': 'Severe Malaria',
                'prevalence_india': 'Moderate (seasonal)',
                'key_markers': ['hemoglobin', 'parasitemia', 'creatinine', 'bilirubin'],
                'warning_signs': [
                    'Cerebral malaria (seizures)',
                    'Severe anemia',
                    'Acute kidney injury',
                    'Acidosis',
                    'Pulmonary edema'
                ],
                'typical_treatment': 'IV Artesunate (WHO preferred)',
                'critical_threshold': 'Cerebral malaria or organ failure'
            },
            'snakebite_envenomation': {
                'name': 'Snake Bite Envenomation',
                'prevalence_india': 'High (rural areas)',
                'key_markers': ['coagulation_profile', 'creatinine', 'myoglobin'],
                'warning_signs': [
                    'Coagulopathy',
                    'Neurotoxicity',
                    'Rhabdomyolysis',
                    'Acute kidney injury'
                ],
                'typical_treatment': 'Anti-venom therapy',
                'critical_threshold': 'DIC or acute kidney injury'
            },
            'hepatitis_b_c': {
                'name': 'Hepatitis B/C',
                'prevalence_india': 'Moderate-High',
                'key_markers': ['bilirubin', 'ast', 'alt', 'albumin', 'INR'],
                'warning_signs': [
                    'Fulminant hepatic failure',
                    'Encephalopathy',
                    'Esophageal varices',
                    'Ascites'
                ],
                'typical_treatment': 'Supportive + antivirals',
                'critical_threshold': 'Hepatic encephalopathy'
            },
            'typhoid_enteric_fever': {
                'name': 'Typhoid/Enteric Fever',
                'prevalence_india': 'High (poor sanitation areas)',
                'key_markers': ['wbc', 'blood_culture', 'liver_enzymes'],
                'warning_signs': [
                    'Intestinal perforation',
                    'Septic shock',
                    'Encephalopathy',
                    'Myocarditis'
                ],
                'typical_treatment': 'Antibiotics (ceftriaxone first-line)',
                'critical_threshold': 'Perforation or septic shock'
            },
            'dengue_hemorrhagic': {
                'name': 'Dengue Hemorrhagic Fever',
                'prevalence_india': 'High (monsoon)',
                'key_markers': ['platelets', 'hematocrit', 'albumin', 'ast', 'alt'],
                'warning_signs': [
                    'Platelet <100k',
                    'Hematocrit rise',
                    'Plasma leakage',
                    'GI bleed'
                ],
                'typical_treatment': 'Fluid management, platelet transfusion',
                'critical_threshold': 'DSS (dengue shock syndrome)'
            }
        }
    
    def get_disease_info(self, disease_name: str) -> Dict:
        """Get features for a specific disease"""
        return self.disease_features.get(disease_name.lower(), {})
    
    def get_all_diseases(self) -> List[str]:
        """List all disease-specific features available"""
        return list(self.disease_features.keys())


class ResourceConstraintAdapter:
    """Adapt recommendations for resource-constrained Indian hospitals"""
    
    def __init__(self):
        self.adaptation_rules = {
            'dialysis_availability': {
                'not_available': [
                    'Recommend early transfer to facility with dialysis',
                    'Focus on fluid management and medication adjustment',
                    'Monitor closely - renal failure progresses quickly'
                ],
                'available': [
                    'Initiate dialysis early if creatinine >3',
                    'Daily monitoring of electrolytes',
                    'Optimize fluid balance'
                ]
            },
            'icu_beds': {
                'limited': [
                    'Prioritize ICU for critical patients',
                    'Early discharge/ward transfer when stable',
                    'High-dependency unit for intermediate care',
                    'Telehealth monitoring for discharged patients'
                ],
                'available': [
                    'Standard ICU monitoring protocols',
                    'Flexible transfer decisions based on merit'
                ]
            },
            'medication_availability': {
                'limited_options': [
                    'Use available antibiotics optimally',
                    'Consider combination therapy',
                    'Longer treatment duration may be needed',
                    'Monitor for treatment failure early'
                ],
                'full_options': [
                    'Standard treatment protocols',
                    'Optimize for individual patient'
                ]
            },
            'blood_products': {
                'limited': [
                    'Use blood products judiciously',
                    'Focus on prevention (prevent bleeding)',
                    'Target Hb >7 in stable patients',
                    'Platelet transfusion only if <20k or active bleed'
                ],
                'available': [
                    'Standard transfusion protocols',
                    'Target Hb >8-10 depending on condition'
                ]
            }
        }
    
    def get_recommendations(self, constraint: str, scenario: str) -> List[str]:
        """Get recommendations based on resource constraints"""
        
        if constraint not in self.adaptation_rules:
            return []
        
        rules = self.adaptation_rules[constraint]
        return rules.get(scenario, [])


class IndianCostAwarenessModule:
    """Optimize cost while maintaining quality"""
    
    def __init__(self):
        self.cost_levels = {
            'budget_conscious': {
                'focus': 'Essential care only',
                'recommendations': [
                    'Use generic medications where available',
                    'Optimize length of stay',
                    'Avoid unnecessary tests',
                    'Group tests to reduce redundancy'
                ]
            },
            'standard': {
                'focus': 'Good quality with reasonable cost',
                'recommendations': [
                    'Use standard treatment protocols',
                    'Regular monitoring',
                    'Balance safety and cost'
                ]
            },
            'comprehensive': {
                'focus': 'Premium care',
                'recommendations': [
                    'Advanced monitoring',
                    'Preventive interventions',
                    'Specialist consultations',
                    'Advanced diagnostic tests'
                ]
            }
        }
    
    def estimate_monthly_cost(self, patient_data: Dict) -> Dict:
        """Estimate monthly ICU cost for patient"""
        
        # Approximate costs for Indian ICUs (2026)
        base_cost = 15000  # Base ICU bed per day
        
        costs = {
            'base_icu_care': base_cost * 30,
            'medications': patient_data.get('medicine_count', 3) * 500 * 30,
            'monitoring_equipment': 5000,
            'nursing_care': 10000,
            'consultations': 3000,
            'diagnostics': patient_data.get('test_frequency', 3) * 1500,
            'contingency': 10000
        }
        
        total = sum(costs.values())
        
        return {
            'breakdown': costs,
            'total_monthly': total,
            'daily_cost': total / 30,
            'estimated_hospitalization': patient_data.get('estimated_days', 7),
            'total_estimated': total * (patient_data.get('estimated_days', 7) / 30)
        }


class IndianHospitalAdapter:
    """Main adapter for Indian hospital settings"""
    
    def __init__(self):
        self.lab_refs = IndianLabReferences()
        self.disease_features = IndianDiseaseSpecificFeatures()
        self.resource_adapter = ResourceConstraintAdapter()
        self.cost_module = IndianCostAwarenessModule()
    
    def analyze_patient(self, patient_data: Dict) -> Dict:
        """Comprehensive analysis for Indian hospital"""
        
        analysis = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'lab_classification': self._classify_labs(patient_data),
            'disease_mapping': self._identify_diseases(patient_data),
            'resource_recommendations': self._get_resource_recommendations(patient_data),
            'cost_estimate': self._estimate_costs(patient_data),
            'india_specific_alerts': self._get_india_specific_alerts(patient_data)
        }
        
        return analysis
    
    def _classify_labs(self, patient_data: Dict) -> Dict:
        """Classify labs using Indian ranges"""
        
        classification = {}
        for lab_name, value in patient_data.get('lab_values', {}).items():
            gender = patient_data.get('gender', 'M')
            classification[lab_name] = {
                'value': value,
                'classification': self.lab_refs.classify_value(lab_name, value, gender)
            }
        
        return classification
    
    def _identify_diseases(self, patient_data: Dict) -> List[Dict]:
        """Identify India-specific disease patterns"""
        
        identified = []
        
        # Simple pattern matching based on lab values and symptoms
        if patient_data.get('lab_values', {}).get('platelets', 1000000) < 100000:
            identified.append(self.disease_features.get_disease_info('dengue'))
        
        if patient_data.get('lab_values', {}).get('bilirubin', 0) > 2.0:
            identified.append(self.disease_features.get_disease_info('hepatitis_b_c'))
        
        return identified
    
    def _get_resource_recommendations(self, patient_data: Dict) -> List[str]:
        """Get recommendations based on resource availability"""
        
        # Example: if limited dialysis, recommend accordingly
        recommendations = []
        
        if patient_data.get('creatinine', 0) > 2.5:
            recommendations.extend(
                self.resource_adapter.get_recommendations('dialysis_availability', 'not_available')
            )
        
        return recommendations
    
    def _estimate_costs(self, patient_data: Dict) -> Dict:
        """Estimate costs"""
        return self.cost_module.estimate_monthly_cost(patient_data)
    
    def _get_india_specific_alerts(self, patient_data: Dict) -> List[str]:
        """Get India-specific clinical alerts"""
        
        alerts = []
        
        # Malaria risk in endemic areas
        if patient_data.get('location', '') in ['endemic_area', 'rural']:
            alerts.append('⚠️ In endemic area - consider malaria testing')
        
        # High dengue risk during monsoon
        if patient_data.get('month', '') in [6, 7, 8, 9]:  # June-Sept
            alerts.append('⚠️ Monsoon season - dengue risk high')
        
        # TB endemic
        alerts.append('⚠️ India TB endemic - consider TB screening if respiratory symptoms')
        
        # Snake bite in rural areas
        if patient_data.get('location', '') == 'rural':
            alerts.append('⚠️ Rural area - snake bite possible if bitten')
        
        return alerts


def main():
    """Demo of India-specific adaptation"""
    
    print("="*80)
    print("INDIA-SPECIFIC FEATURE EXTRACTOR - DEMO")
    print("="*80)
    
    # Initialize adapter
    adapter = IndianHospitalAdapter()
    
    # Example patient
    patient = {
        'patient_id': 'PATIENT_001',
        'gender': 'M',
        'location': 'urban',
        'month': 7,  # July - monsoon
        'lab_values': {
            'platelets': 95000,
            'hemoglobin': 10.5,
            'bilirubin': 1.8,
            'creatinine': 1.5,
            'albumin': 3.2,
            'ast': 85,
            'alt': 92
        },
        'medicine_count': 4,
        'test_frequency': 2,
        'estimated_days': 10
    }
    
    print("\n[STEP 1] Lab Value Classification (using Indian ranges)")
    print("-" * 80)
    
    analysis = adapter.analyze_patient(patient)
    
    for lab, result in analysis['lab_classification'].items():
        print(f"{lab.upper()}: {result['value']} → {result['classification']}")
    
    print("\n[STEP 2] Disease Identification")
    print("-" * 80)
    
    if analysis['disease_mapping']:
        for disease in analysis['disease_mapping']:
            print(f"\n✓ Detected: {disease.get('name', 'Unknown disease')}")
            print(f"  Prevalence in India: {disease.get('prevalence_india', 'Unknown')}")
            print(f"  Key markers: {', '.join(disease.get('key_markers', []))}")
    else:
        print("No specific India disease patterns detected")
    
    print("\n[STEP 3] India-Specific Alerts")
    print("-" * 80)
    
    for alert in analysis['india_specific_alerts']:
        print(f"  {alert}")
    
    print("\n[STEP 4] Cost Estimation")
    print("-" * 80)
    
    cost = analysis['cost_estimate']
    print(f"\nEstimated Monthly Cost: ₹{cost['total_monthly']:,.0f}")
    print(f"Daily Cost: ₹{cost['daily_cost']:,.0f}")
    print(f"Estimated Total (for {patient['estimated_days']} days): ₹{cost['total_estimated']:,.0f}")
    
    print("\nCost Breakdown:")
    for item, amount in cost['breakdown'].items():
        print(f"  • {item.replace('_', ' ').title()}: ₹{amount:,.0f}")
    
    print("\n[STEP 5] Resource Recommendations")
    print("-" * 80)
    
    if analysis['resource_recommendations']:
        for rec in analysis['resource_recommendations']:
            print(f"  • {rec}")
    else:
        print("  Standard care recommendations")
    
    return adapter, analysis


if __name__ == '__main__':
    adapter, analysis = main()
    print("\n✨ India-specific feature extraction COMPLETE!")
