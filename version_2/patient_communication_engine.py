"""
PATIENT COMMUNICATION ENGINE

Converts ML predictions into simple, non-technical explanations
suitable for patient families in Indian hospital settings.

Focuses on:
- Plain language explanations
- Visual risk indicators
- Daily progress summaries
- Actionable recommendations
- Emotional sensitivity
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd

class RiskCommunicator:
    """Converts risk predictions to patient-family-friendly messages"""
    
    def __init__(self):
        # Simple language risk level definitions
        self.risk_levels = {
            'green': {
                'name': 'Low Risk',
                'emoji': '✅',
                'family_message': 'Your loved one is stable and progressing well.',
                'activity_level': 'Standard hospital care',
                'monitoring': 'Regular checks',
                'what_to_expect': 'Routine recovery expected'
            },
            'yellow': {
                'name': 'Moderate Risk',
                'emoji': '⚠️',
                'family_message': 'Your loved one needs extra attention but is responding to treatment.',
                'activity_level': 'Careful monitoring',
                'monitoring': 'Frequent checks (every 2-4 hours)',
                'what_to_expect': 'Close watching for any changes'
            },
            'orange': {
                'name': 'High Risk',
                'emoji': '🏥',
                'family_message': 'Your loved one needs intensive care. Doctors are actively treating.',
                'activity_level': 'Intensive monitoring',
                'monitoring': 'Continuous monitoring',
                'what_to_expect': 'Daily updates from doctors, possible interventions'
            },
            'red': {
                'name': 'Critical Risk',
                'emoji': '🚨',
                'family_message': 'Your loved one is in critical condition. Senior doctors and specialists are involved.',
                'activity_level': 'Critical care',
                'monitoring': '24/7 specialist attention',
                'what_to_expect': 'Frequent communication with doctors'
            }
        }
    
    def get_risk_color(self, mortality_probability: float) -> str:
        """Convert probability to color risk level"""
        if mortality_probability < 0.10:
            return 'green'
        elif mortality_probability < 0.20:
            return 'yellow'
        elif mortality_probability < 0.35:
            return 'orange'
        else:
            return 'red'
    
    def get_family_message(self, mortality_probability: float, 
                          condition_name: str = 'health') -> Dict:
        """Generate family-friendly communication"""
        
        color = self.get_risk_color(mortality_probability)
        risk_info = self.risk_levels[color]
        
        # Create detailed message
        message = {
            'risk_level': risk_info['name'],
            'color': color,
            'emoji': risk_info['emoji'],
            'main_message': risk_info['family_message'],
            'what_to_expect': risk_info['what_to_expect'],
            'monitoring_level': risk_info['monitoring'],
            'activity_level': risk_info['activity_level'],
            'probability_percent': f"{mortality_probability*100:.1f}%",
            'timestamp': datetime.now().isoformat(),
            'condition': condition_name
        }
        
        return message
    
    def create_daily_summary(self, patient_data: Dict) -> str:
        """Create simple daily summary for family"""
        
        summary = f"""
╔════════════════════════════════════════════════════════════════╗
║              PATIENT DAILY SUMMARY FOR FAMILY                  ║
║                    Date: {datetime.now().strftime('%d-%b-%Y')}                        ║
╚════════════════════════════════════════════════════════════════╝

👤 PATIENT: {patient_data.get('name', 'Patient')}
🏥 CONDITION: {patient_data.get('condition', 'Under medical care')}

{self._get_status_visualization(patient_data.get('mortality_probability', 0))}

📊 TODAY'S STATUS:
   • Vital Signs: {patient_data.get('vital_status', 'Stable')}
   • Feeding/Nutrition: {patient_data.get('nutrition_status', 'As per doctor')}
   • Pain Level: {patient_data.get('pain_level', 'Controlled')}
   • Medicines: {patient_data.get('medicine_count', '0')} medicines

🏥 CARE PLAN:
   • Current Focus: {patient_data.get('care_focus', 'Recovery and monitoring')}
   • Next Steps: {patient_data.get('next_steps', 'Continue current treatment')}

💊 IMPORTANT MEDICATIONS:
   {self._format_medicines(patient_data.get('medicines', []))}

📈 RECOVERY TREND:
   {self._get_trend_description(patient_data.get('trend', 'stable'))}

❓ WHEN TO ASK DOCTOR:
   • If you notice changes in breathing
   • If fever appears or increases
   • If your loved one seems uncomfortable
   • Any other concerning changes

📞 CONTACT: Call nurse station anytime if concerned

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚡ REMEMBER: We are doing everything possible to help your loved one recover.
            Your presence and support matter greatly!

"""
        return summary
    
    def _get_status_visualization(self, mortality_prob: float) -> str:
        """Create visual representation of status"""
        
        color = self.get_risk_color(mortality_prob)
        risk_info = self.risk_levels[color]
        
        # Create progress bar
        bar_length = 20
        filled = int(bar_length * mortality_prob)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        return f"""
   STATUS: {risk_info['emoji']} {risk_info['name']}
   
   Risk Level: [{bar}] {mortality_prob*100:.0f}%
   
   Meaning: {risk_info['family_message']}
"""
    
    def _format_medicines(self, medicines: List[Dict]) -> str:
        """Format medicines in simple way"""
        if not medicines or len(medicines) == 0:
            return "   As per doctor's prescription"
        
        lines = []
        for med in medicines[:3]:  # Show top 3
            lines.append(f"   • {med.get('name', 'Medicine')}: {med.get('dose', '')} {med.get('frequency', '')}")
        
        if len(medicines) > 3:
            lines.append(f"   ... and {len(medicines) - 3} more medicines")
        
        return '\n'.join(lines)
    
    def _get_trend_description(self, trend: str) -> str:
        """Convert trend to simple description"""
        
        descriptions = {
            'improving': '✅ Getting better - Keep supporting treatment',
            'stable': '➡️ Holding steady - Good, we are maintaining progress',
            'declining': '⚠️ Needs attention - Doctor will increase care',
            'unknown': '❓ Being monitored - Need more time to assess'
        }
        
        return descriptions.get(trend, descriptions['unknown'])


class ProgressTracker:
    """Track and communicate patient progress"""
    
    def __init__(self, patient_id: str):
        self.patient_id = patient_id
        self.daily_records = []
    
    def log_daily_progress(self, date: str, mortality_prob: float,
                          vital_status: str, notes: str = ''):
        """Log daily progress"""
        
        record = {
            'date': date,
            'mortality_probability': mortality_prob,
            'vital_status': vital_status,
            'notes': notes,
            'timestamp': datetime.now().isoformat()
        }
        
        self.daily_records.append(record)
        return record
    
    def get_weekly_summary(self) -> Dict:
        """Get simplified weekly summary"""
        
        if not self.daily_records:
            return {'status': 'No data yet'}
        
        recent_records = self.daily_records[-7:]  # Last 7 days
        
        probs = [r['mortality_probability'] for r in recent_records]
        trend = 'improving' if probs[-1] < probs[0] else 'declining' if probs[-1] > probs[0] else 'stable'
        
        summary = {
            'period': 'Last 7 days',
            'trend': trend,
            'starting_risk': f"{probs[0]*100:.1f}%",
            'current_risk': f"{probs[-1]*100:.1f}%",
            'best_day': f"{min(probs)*100:.1f}%",
            'worst_day': f"{max(probs)*100:.1f}%",
            'message': self._get_trend_message(trend, probs[0], probs[-1])
        }
        
        return summary
    
    def _get_trend_message(self, trend: str, start_prob: float, end_prob: float) -> str:
        """Create trend message"""
        
        if trend == 'improving':
            return f"Great news! Risk has improved from {start_prob*100:.0f}% to {end_prob*100:.0f}%"
        elif trend == 'stable':
            return f"Good - Risk is holding steady around {end_prob*100:.0f}%"
        else:
            return f"Caution needed - Risk increased from {start_prob*100:.0f}% to {end_prob*100:.0f}%"


class GuidelinesCommunicator:
    """Provide simple health guidelines for patient care"""
    
    def __init__(self):
        self.guidelines = {
            'visiting': [
                'Visit during designated hours to avoid disturbing treatment',
                'Wash hands before and after visiting',
                'Avoid loud voices - keep environment calm',
                'Limit visitors to immediate family',
                'Children visits - check with doctors first'
            ],
            'support': [
                'Your presence helps recovery',
                'Talk gently to your loved one - they may hear you',
                'Bring comfort items from home (with doctor approval)',
                'Take care of yourself too - get rest and eat well',
                'Ask doctors any questions anytime'
            ],
            'physical': [
                'Help with gentle exercises if allowed',
                'Assist with cleaning/personal care as permitted',
                'Help patient change position to avoid bed sores',
                'Ensure patient sips water when allowed',
                'Report any pain or discomfort to nurse'
            ],
            'emotional': [
                'Encourage patient - positive psychology helps recovery',
                'Play soft music if doctor permits',
                'Tell stories from happier times',
                'Express your love and support',
                'Be patient - recovery takes time'
            ]
        }
    
    def get_visiting_guidelines(self) -> str:
        """Simple visiting guidelines"""
        
        msg = """
╔═══════════════════════════════════════════════╗
║         HOSPITAL VISITING GUIDELINES          ║
╚═══════════════════════════════════════════════╝

✅ DO:
"""
        for guideline in self.guidelines['visiting']:
            msg += f"   • {guideline}\n"
        
        msg += """
❌ DON'T:
   • Bring large groups
   • Use mobile phones loudly
   • Bring outside food without permission
   • Stay too long during recovery phases
   • Discuss bad news/stress in front of patient

"""
        return msg
    
    def get_support_tips(self) -> str:
        """Tips for supporting patient"""
        
        msg = "💪 HOW YOU CAN HELP:\n\n"
        
        for i, tip in enumerate(self.guidelines['support'], 1):
            msg += f"{i}. {tip}\n"
        
        return msg


def main():
    """Demo of patient communication engine"""
    
    print("="*80)
    print("PATIENT COMMUNICATION ENGINE - DEMO")
    print("="*80)
    
    # Create communicator
    communicator = RiskCommunicator()
    
    # Example 1: Low risk patient
    print("\n[EXAMPLE 1] Low Risk Patient")
    print("-" * 80)
    msg = communicator.get_family_message(0.08, 'Pneumonia')
    print(f"Risk Level: {msg['emoji']} {msg['risk_level']}")
    print(f"Message: {msg['main_message']}")
    
    # Example 2: High risk patient
    print("\n[EXAMPLE 2] High Risk Patient")
    print("-" * 80)
    msg = communicator.get_family_message(0.25, 'Sepsis')
    print(f"Risk Level: {msg['emoji']} {msg['risk_level']}")
    print(f"Message: {msg['main_message']}")
    
    # Example 3: Daily summary
    print("\n[EXAMPLE 3] Daily Summary for Family")
    print("-" * 80)
    
    patient_data = {
        'name': 'Rajesh Kumar',
        'condition': 'Pneumonia (Respiratory infection)',
        'mortality_probability': 0.12,
        'vital_status': 'Stable',
        'nutrition_status': 'Eating soft diet',
        'pain_level': 'Mild',
        'medicine_count': 5,
        'care_focus': 'Improving lung function and nutrition',
        'next_steps': 'Continue antibiotics for 5 more days',
        'trend': 'improving',
        'medicines': [
            {'name': 'Ceftriaxone', 'dose': '2g', 'frequency': 'Every 12 hours'},
            {'name': 'Oxygen support', 'dose': '2L/min', 'frequency': 'Continuous'},
            {'name': 'Vitamin supplements', 'dose': 'As prescribed', 'frequency': 'Daily'}
        ]
    }
    
    summary = communicator.create_daily_summary(patient_data)
    print(summary)
    
    # Example 4: Progress tracking
    print("\n[EXAMPLE 4] Weekly Progress Summary")
    print("-" * 80)
    
    tracker = ProgressTracker('PATIENT_001')
    tracker.log_daily_progress('2026-04-03', 0.25, 'Critical')
    tracker.log_daily_progress('2026-04-04', 0.22, 'Critical')
    tracker.log_daily_progress('2026-04-05', 0.18, 'Serious')
    tracker.log_daily_progress('2026-04-06', 0.15, 'Serious')
    tracker.log_daily_progress('2026-04-07', 0.12, 'Moderate')
    tracker.log_daily_progress('2026-04-08', 0.10, 'Stable')
    tracker.log_daily_progress('2026-04-09', 0.08, 'Stable')
    
    weekly = tracker.get_weekly_summary()
    print(f"\n📈 WEEK SUMMARY:")
    print(f"   Trend: {weekly['trend'].upper()}")
    print(f"   Started at: {weekly['starting_risk']}")
    print(f"   Now at: {weekly['current_risk']}")
    print(f"   Message: {weekly['message']}")
    
    # Example 5: Guidelines
    print("\n[EXAMPLE 5] Support Guidelines")
    print("-" * 80)
    
    guidelines = GuidelinesCommunicator()
    print(guidelines.get_support_tips())
    
    return communicator, tracker


if __name__ == '__main__':
    communicator, tracker = main()
    print("\n✨ Patient communication engine COMPLETE!")
