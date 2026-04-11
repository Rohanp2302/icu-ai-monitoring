"""
Real-Time HL7 Parser for Hospital Multiparameter Monitors
Integrates with hospital vital sign monitoring systems
Phase 10 - Real-Time Hospital Data Integration
"""

import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import deque
import logging

logger = logging.getLogger(__name__)


class HL7PatientMonitorParser:
    """
    Parse HL7 messages from hospital bedside multiparameter monitors
    Supports vital sign data extraction in real-time

    HL7 Message Format:
    MSH|^~\&|MONITOR|BEDSIDE|ICTRY|HOSPITAL|timestamp|||ORU^R01
    PID|1||12345^^^MRN||DOE^JOHN||19500101|M|||123 MAIN STREET^APT 4^BOSTON^MA^02134
    OBR|1|req123|order123|NPU^Vital Signs|||timestamp
    OBX|1|NM|8480-6^Heart Rate^LN||75|{bpm}|60-100|N|||F
    OBX|2|NM|9279-1^Respiratory Rate^LN||18|{breaths/min}|12-20|N|||F
    OBX|3|NM|2708-6^Oxygen Saturation^LN||97|{%}|95-100|N|||F
    """

    # LOINC code mapping to vital sign names
    LOINC_VITAL_MAP = {
        '8480-6': 'HR_mean',  # Heart Rate
        '9279-1': 'RR_mean',  # Respiratory Rate
        '2708-6': 'SaO2_mean',  # Oxygen Saturation
        '8310-5': 'Temp_mean',  # Body Temperature
        '8462-4': 'BP_diastolic',  # Blood Pressure - Diastolic
        '8480-6': 'BP_systolic',  # Blood Pressure - Systolic
        '8440-0': 'VS_systolic_BP',  # Systolic Blood Pressure
        '8441-8': 'VS_diastolic_BP',  # Diastolic Blood Pressure
        '8867-4': 'HR',  # Heart Rate (alternative)
        '9280-4': 'RR_patient',  # Respiratory Rate (alternative)
        '3150-0': 'Glucose',  # Blood Glucose
        '2345-7': 'Glucose_serum',  # Glucose [Mass/volume] in Serum
        '2951-2': 'Sodium',  # Sodium [Moles/volume] in Serum or Plasma
        '2823-3': 'Potassium',  # Potassium [Moles/volume] in Serum or Plasma
        '1975-2': 'Albumin',  # Albumin [Mass/volume] in Serum or Plasma
        '1871-1': 'Creatinine',  # Creatinine [Mass/volume] in Serum or Plasma
    }

    def __init__(self, patient_id: Optional[str] = None, buffer_size: int = 1440):
        """
        Initialize HL7 parser

        Args:
            patient_id: Patient MRN/ID (can be extracted from messages)
            buffer_size: Number of records to keep in memory (default 1440 = 24 hours at 1/min)
        """
        self.patient_id = patient_id
        self.vital_buffer = deque(maxlen=buffer_size)  # Sliding window buffer
        self.lab_buffer = deque(maxlen=buffer_size)
        self.last_parsed_time = None
        self.message_count = 0
        self.error_count = 0

    def parse_hl7_message(self, hl7_message: str) -> Optional[Dict]:
        """
        Parse individual HL7 message

        Args:
            hl7_message: Raw HL7 message string (segments separated by \r or \n)

        Returns:
            Dictionary with parsed vital signs, or None if parse fails
        """
        try:
            lines = hl7_message.split('\r') if '\r' in hl7_message else hl7_message.split('\n')
            data = {
                'timestamp': datetime.now().isoformat(),
                'patient_id': self.patient_id,
                'vitals': {},
                'labs': {},
                'message_id': None,
                'observation_datetime': None
            }

            for segment in lines:
                if not segment or len(segment) < 3:
                    continue

                segment_type = segment[:3]

                if segment_type == 'MSH':
                    # Message header - extract message info
                    data['message_id'] = self._extract_field(segment, 9)

                elif segment_type == 'PID':
                    # Patient identification
                    patient_mrn = self._extract_field(segment, 3)
                    if patient_mrn:
                        self.patient_id = patient_mrn
                        data['patient_id'] = patient_mrn

                elif segment_type == 'OBR':
                    # Observation request - contains test time
                    obs_datetime = self._extract_field(segment, 7)
                    if obs_datetime:
                        data['observation_datetime'] = self._parse_hl7_datetime(obs_datetime)

                elif segment_type == 'OBX':
                    # Observation results
                    loinc_code = self._extract_loinc_code(segment)
                    value = self._extract_field(segment, 5)
                    units = self._extract_field(segment, 6)
                    status = self._extract_field(segment, 11)

                    if loinc_code and value:
                        vital_name = self.LOINC_VITAL_MAP.get(loinc_code, loinc_code)

                        # Parse value
                        try:
                            numeric_value = float(value)
                        except ValueError:
                            numeric_value = None
                            continue

                        vital_info = {
                            'value': numeric_value,
                            'unit': units or '',
                            'status': status or 'F',  # F = final, P = preliminary
                            'loinc': loinc_code,
                            'timestamp': data['observation_datetime'] or data['timestamp']
                        }

                        # Categorize as vital or lab
                        if vital_name in ['HR_mean', 'RR_mean', 'SaO2_mean', 'Temp_mean',
                                        'BP_systolic', 'BP_diastolic', 'HR', 'RR_patient']:
                            data['vitals'][vital_name] = vital_info
                        else:
                            data['labs'][vital_name] = vital_info

            # Only return if we got some vitals
            if data['vitals']:
                self.vital_buffer.append(data)
                self.message_count += 1
                self.last_parsed_time = datetime.now()
                return data
            else:
                self.error_count += 1
                return None

        except Exception as e:
            logger.error(f"Error parsing HL7 message: {str(e)}")
            self.error_count += 1
            return None

    def parse_message_stream(self, messages: List[str]) -> List[Dict]:
        """
        Parse stream of HL7 messages

        Args:
            messages: List of HL7 message strings

        Returns:
            List of successfully parsed messages
        """
        parsed = []
        for message in messages:
            result = self.parse_hl7_message(message)
            if result:
                parsed.append(result)
        return parsed

    def get_latest_vitals(self) -> Optional[Dict]:
        """Get most recent vital signs"""
        if not self.vital_buffer:
            return None
        return dict(self.vital_buffer[-1])

    def get_vital_history(self, minutes: int = 60) -> List[Dict]:
        """
        Get vital signs history for specified time period

        Args:
            minutes: How many minutes of history to return

        Returns:
            List of vital measurements chronologically
        """
        if not self.vital_buffer:
            return []

        cutoff_time = datetime.fromisoformat(
            datetime.now().isoformat()
        ) if not hasattr(self, '_cutoff') else self._cutoff

        # Get all vitals within time window
        history = []
        for entry in self.vital_buffer:
            # Compare timestamps (simplified)
            history.append(entry)

        return history[-minutes:] if len(history) > minutes else history

    def get_vital_statistics(self, vital_name: str, minutes: int = 60) -> Optional[Dict]:
        """
        Calculate statistics for a specific vital over time

        Args:
            vital_name: Name of vital (e.g., 'HR_mean')
            minutes: Time window in minutes

        Returns:
            Statistics (mean, min, max, std, trend)
        """
        history = self.get_vital_history(minutes)
        values = []

        for entry in history:
            if vital_name in entry['vitals']:
                values.append(entry['vitals'][vital_name]['value'])

        if not values:
            return None

        import numpy as np
        values_array = np.array(values)

        return {
            'vital_name': vital_name,
            'data_points': len(values),
            'mean': float(np.mean(values_array)),
            'min': float(np.min(values_array)),
            'max': float(np.max(values_array)),
            'std': float(np.std(values_array)),
            'trend': self._calculate_trend(values),
            'latest': float(values[-1]) if values else None,
            'time_window_minutes': minutes
        }

    def check_alert_conditions(self) -> List[Dict]:
        """
        Check for alert-worthy vital sign conditions

        Returns:
            List of alert conditions detected
        """
        alerts = []
        latest = self.get_latest_vitals()

        if not latest:
            return []

        vitals = latest.get('vitals', {})

        # Define alert thresholds
        alert_rules = {
            'HR_mean': {
                'critical_high': 160,
                'critical_low': 40,
                'warn_high': 130,
                'warn_low': 50
            },
            'RR_mean': {
                'critical_high': 40,
                'critical_low': 8,
                'warn_high': 30,
                'warn_low': 12
            },
            'SaO2_mean': {
                'critical_low': 85,
                'warn_low': 90
            },
            'Temp_mean': {
                'critical_high': 40,
                'critical_low': 35,
                'warn_high': 39,
                'warn_low': 36
            }
        }

        for vital_name, value_info in vitals.items():
            value = value_info['value']
            rules = alert_rules.get(vital_name, {})

            if 'critical_high' in rules and value > rules['critical_high']:
                alerts.append({
                    'severity': 'CRITICAL',
                    'vital': vital_name,
                    'value': value,
                    'condition': f'{vital_name} is critically HIGH',
                    'action': 'IMMEDIATE REVIEW REQUIRED'
                })

            elif 'critical_low' in rules and value < rules['critical_low']:
                alerts.append({
                    'severity': 'CRITICAL',
                    'vital': vital_name,
                    'value': value,
                    'condition': f'{vital_name} is critically LOW',
                    'action': 'IMMEDIATE REVIEW REQUIRED'
                })

            elif 'warn_high' in rules and value > rules['warn_high']:
                alerts.append({
                    'severity': 'WARNING',
                    'vital': vital_name,
                    'value': value,
                    'condition': f'{vital_name} is elevated',
                    'action': 'Monitor closely'
                })

            elif 'warn_low' in rules and value < rules['warn_low']:
                alerts.append({
                    'severity': 'WARNING',
                    'vital': vital_name,
                    'value': value,
                    'condition': f'{vital_name} is low',
                    'action': 'Monitor closely'
                })

        return alerts

    # Private helper methods
    def _extract_field(self, segment: str, field_num: int, separator: str = '|') -> Optional[str]:
        """
        Extract field from HL7 segment

        Args:
            segment: HL7 segment string
            field_num: Field number (0-indexed)
            separator: Field separator (default |)

        Returns:
            Field value or None
        """
        try:
            fields = segment.split(separator)
            if field_num < len(fields):
                return fields[field_num].strip() or None
            return None
        except Exception:
            return None

    def _extract_loinc_code(self, obx_segment: str) -> Optional[str]:
        """Extract LOINC code from OBX segment"""
        try:
            # OBX format: OBX|seq|datatype|code^text^codesystem|...
            fields = obx_segment.split('|')
            if len(fields) > 3:
                # Extract code from code^text^codesystem
                code_field = fields[3]
                code = code_field.split('^')[0]
                return code if code else None
            return None
        except Exception:
            return None

    def _parse_hl7_datetime(self, hl7_datetime: str) -> str:
        """
        Parse HL7 datetime format (YYYYMMDDHHMMSS)

        Args:
            hl7_datetime: HL7 formatted datetime string

        Returns:
            ISO format datetime string
        """
        try:
            if len(hl7_datetime) >= 14:
                dt_str = hl7_datetime[:14]
                year = int(dt_str[0:4])
                month = int(dt_str[4:6])
                day = int(dt_str[6:8])
                hour = int(dt_str[8:10])
                minute = int(dt_str[10:12])
                second = int(dt_str[12:14])

                dt = datetime(year, month, day, hour, minute, second)
                return dt.isoformat()
            return datetime.now().isoformat()
        except Exception:
            return datetime.now().isoformat()

    def _calculate_trend(self, values: List[float]) -> str:
        """
        Calculate trend of values

        Args:
            values: List of values over time

        Returns:
            'up', 'down', or 'stable'
        """
        if len(values) < 2:
            return 'stable'

        first_half_mean = sum(values[:len(values)//2]) / (len(values)//2 if len(values)//2 > 0 else 1)
        second_half_mean = sum(values[len(values)//2:]) / (len(values) - len(values)//2 if len(values) - len(values)//2 > 0 else 1)

        diff = second_half_mean - first_half_mean
        if abs(diff) < 2:
            return 'stable'
        return 'up' if diff > 0 else 'down'

    def get_parser_stats(self) -> Dict:
        """Get parser statistics"""
        return {
            'patient_id': self.patient_id,
            'messages_parsed': self.message_count,
            'parse_errors': self.error_count,
            'buffer_size': len(self.vital_buffer),
            'last_update': self.last_parsed_time.isoformat() if self.last_parsed_time else None,
            'success_rate': (self.message_count / (self.message_count + self.error_count) * 100) if (self.message_count + self.error_count) > 0 else 0
        }


class HL7RealtimeProcessor:
    """
    Process HL7 messages in real-time from hospital monitors
    Integrates with ensemble predictor for live risk scoring
    """

    def __init__(self, ensemble_predictor=None):
        """
        Initialize real-time processor

        Args:
            ensemble_predictor: Optional ensemble predictor for live scoring
        """
        self.ensemble_predictor = ensemble_predictor
        self.parsers = {}  # Dict of patient_id -> HL7Parser

    def process_hl7_message(self, hl7_message: str) -> Optional[Dict]:
        """
        Process incoming HL7 message and update patient data

        Args:
            hl7_message: Raw HL7 message from monitor

        Returns:
            Processed result with vitals and optional prediction
        """
        # Create parser if needed
        parser = HL7PatientMonitorParser()
        parsed = parser.parse_hl7_message(hl7_message)

        if not parsed:
            return None

        patient_id = parsed.get('patient_id', 'UNKNOWN')
        if patient_id not in self.parsers:
            self.parsers[patient_id] = parser

        # Check for alerts
        alerts = parser.check_alert_conditions()

        result = {
            'patient_id': patient_id,
            'vitals': parsed['vitals'],
            'labs': parsed['labs'],
            'alerts': alerts,
            'timestamp': parsed['timestamp']
        }

        # If we have predictor and enough history, generate prediction
        if self.ensemble_predictor and len(parser.vital_buffer) >= 24:
            try:
                x_temporal = self._build_temporal_data(parser)
                prediction = self.ensemble_predictor.predict(x_temporal, {})
                result['prediction'] = prediction
            except Exception as e:
                logger.warning(f"Could not generate prediction: {e}")

        return result

    def _build_temporal_data(self, parser: HL7PatientMonitorParser) -> np.ndarray:
        """Build temporal feature matrix from parsed vital history"""
        history = parser.get_vital_history(minutes=1440)  # 24 hours
        # Convert to feature matrix (simplified)
        # In real implementation: extract 24-hour history with proper feature engineering
        return np.array([[v['vitals'].get('HR_mean', {}).get('value', 0) for v in history]])
