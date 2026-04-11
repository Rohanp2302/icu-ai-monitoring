"""
Multi-Language Translation Support for Family Explanations
Supports: English (en), Hindi (hi), Tamil (ta), Telugu (te), Kannada (kn), Marathi (mr)
"""

from typing import Dict, Optional, List
from enum import Enum


class SupportedLanguages(Enum):
    """Supported languages for family explanations"""
    ENGLISH = 'en'
    HINDI = 'hi'
    TAMIL = 'ta'
    TELUGU = 'te'
    KANNADA = 'kn'
    MARATHI = 'mr'


class MultiLanguageTranslator:
    """Translate family explanations to Indian languages"""

    def __init__(self, language: str = 'en'):
        """
        Initialize translator

        Args:
            language: Language code ('en', 'hi', 'ta', 'te', 'kn', 'mr')
        """
        self.language = language.lower()
        if self.language not in [lang.value for lang in SupportedLanguages]:
            print(f"Warning: Language {language} not supported, defaulting to English")
            self.language = 'en'

    def translate_risk_message(self, risk_class: str) -> str:
        """Translate main risk message"""

        translations = {
            'en': {
                'LOW': "Your loved one's condition is stable right now",
                'MEDIUM': "Your loved one needs close attention from the doctors",
                'HIGH': "Your loved one's condition is serious and needs immediate care",
                'CRITICAL': "Your loved one's condition is critical - the ICU team is doing everything possible"
            },
            'hi': {
                'LOW': "आपके प्रियजन की स्थिति अभी स्थिर है",
                'MEDIUM': "आपके प्रियजन को डॉक्टरों का करीबी ध्यान लग रहा है",
                'HIGH': "आपके प्रियजन की स्थिति गंभीर है और तत्काल देखभाल की जरूरत है",
                'CRITICAL': "आपके प्रियजन की स्थिति गंभीर है - ICU टीम सब कुछ संभव कर रही है"
            },
            'ta': {
                'LOW': "உங்கள் அன்புக்குரியவரின் நிலை இப்போது நிலையாக உள்ளது",
                'MEDIUM': "உங்கள் அன்புக்குரியவர் டாக்டரின் நெருக்கமான கவனம் தேவை",
                'HIGH': "உங்கள் அன்புக்குரியவரின் நிலை கடுமையாக உள்ளது மற்றும் உடனடி கவனம் தேவை",
                'CRITICAL': "உங்கள் அன்புக்குரியவரின் நிலை முக்கியமானது - ICU குழு எல்லாம் செய்து கொண்டிருக்கிறது"
            },
            'te': {
                'LOW': "మీ ప్రియమైనవారి పరిస్థితి ఇప్పుడు స్థిరంగా ఉంది",
                'MEDIUM': "మీ ప్రియమైనవారికి డాక్టరుల సన్నిహిత శ్రద్ధ అవసరం",
                'HIGH': "మీ ప్రియమైనవారి పరిస్థితి తీవ్రమైనది మరియు తక్షణ సంరక్షణ అవసరం",
                'CRITICAL': "మీ ప్రియమైనవారి స్థితి సంక్లిష్టమైనది - ICU గాని సమస్త సాధ్యమైన చర్యలు చేస్తున్నారు"
            },
            'kn': {
                'LOW': "ನಿಮ್ಮ ಪ್ರಿಯವರ ಸ್ಥಿತಿ ಈ ಸಮಯದಲ್ಲಿ ಸ್ಥಿರವಾಗಿದೆ",
                'MEDIUM': "ನಿಮ್ಮ ಪ್ರಿಯವರಿಗೆ ವೈದ್ಯರ ಆಸ್ಥೆಯ ಅಗತ್ಯವಿದೆ",
                'HIGH': "ನಿಮ್ಮ ಪ್ರಿಯವರ ಸ್ಥಿತಿ ಗಂಭೀರವಾಗಿದೆ ಮತ್ತು ತಕ್ಷಣ ಕಾಳಜಿ ಅಗತ್ಯವಿದೆ",
                'CRITICAL': "ನಿಮ್ಮ ಪ್ರಿಯವರ ಸ್ಥಿತಿ ತುಂಬಾ ಗಂಭೀರವಾಗಿದೆ - ICU ಗಾಂಪ ಸಕಲ ಸಂಭವನೀಯ ಕೆಲಸ ಮಾಡುತ್ತಿದೆ"
            },
            'mr': {
                'LOW': "आपल्या प्रियजनाची स्थिती सध्या स्थिर आहे",
                'MEDIUM': "आपल्या प्रियजनালाई डॉक्टरांचे जवळचे लक्ष्य हवेय",
                'HIGH': "आपल्या प्रियजनाची स्थिती गंभीर आहे आणि तात्काळ काळजी आवश्यक आहे",
                'CRITICAL': "आपल्या प्रियजनाची स्थिती गंभीर आहे - ICU संघ सर्वकाही संभव करत आहे"
            }
        }

        if self.language in translations and risk_class in translations[self.language]:
            return translations[self.language][risk_class]
        return translations['en'].get(risk_class, "Your loved one is being monitored")

    def translate_vital_name(self, vital_english: str) -> str:
        """Translate vital sign names"""

        translations = {
            'en': {
                'Heart_Rate': 'Heart Rate',
                'Respiration_Rate': 'Breathing Rate',
                'Oxygen_Saturation': 'Oxygen Levels',
                'Temperature': 'Fever',
                'Blood_Pressure': 'Blood Pressure'
            },
            'hi': {
                'Heart_Rate': 'दिल की धड़कन',
                'Respiration_Rate': 'सांस की दर',
                'Oxygen_Saturation': 'ऑक्सीजन स्तर',
                'Temperature': 'बुखार',
                'Blood_Pressure': 'रक्तचाप'
            },
            'ta': {
                'Heart_Rate': 'இதய துடிப்பு',
                'Respiration_Rate': 'சுவாசக் கதை',
                'Oxygen_Saturation': 'ஆக்ஸிஜன் அளவு',
                'Temperature': 'காய்ச்சல்',
                'Blood_Pressure': 'இரத்த அழுத்தம்'
            },
            'te': {
                'Heart_Rate': 'గుండె స్పందన',
                'Respiration_Rate': 'శ్వాస రేటు',
                'Oxygen_Saturation': 'ఆక్సిజన్ స్థాయిలు',
                'Temperature': 'జ్వరం',
                'Blood_Pressure': 'రక్త పీడనం'
            },
            'kn': {
                'Heart_Rate': 'ಹೃದಯ ಬೀಟ್',
                'Respiration_Rate': 'ಶ್ವಾಸೋಚ್ಛ್ವಾಸ ದರ',
                'Oxygen_Saturation': 'ಆಕ್ಸಿಜನ್ ಮಟ್ಟ',
                'Temperature': 'ಜ್ವರ',
                'Blood_Pressure': 'ರಕ್ತ ಒತ್ತಡ'
            },
            'mr': {
                'Heart_Rate': 'हृदय गती',
                'Respiration_Rate': 'श्वसन दर',
                'Oxygen_Saturation': 'ऑक्सिजन स्तर',
                'Temperature': 'ताप',
                'Blood_Pressure': 'रक्तदाब'
            }
        }

        lang_dict = translations.get(self.language, translations['en'])
        return lang_dict.get(vital_english, vital_english)

    def translate_simple_explanation(self, english_text: str, context: str = 'general') -> str:
        """Translate family-friendly explanations"""

        # This is a key-based translation system
        simple_translations = {
            'en': {
                'normal': 'This is good.',
                'high': 'This is higher than normal',
                'low': 'This is lower than normal',
                'fever_detected': 'Your loved one has a fever',
                'low_oxygen': 'Oxygen levels are low',
                'high_heart_rate': 'Heart is beating faster than normal',
                'fast_breathing': 'Breathing is faster than normal'
            },
            'hi': {
                'normal': 'यह अच्छा है।',
                'high': 'यह सामान्य से अधिक है',
                'low': 'यह सामान्य से कम है',
                'fever_detected': 'आपके प्रियजन को बुखार है',
                'low_oxygen': 'ऑक्सीजन का स्तर कम है',
                'high_heart_rate': 'दिल सामान्य से तेज धड़क रहा है',
                'fast_breathing': 'सांस सामान्य से तेज है'
            },
            'ta': {
                'normal': 'இது நல்லது.',
                'high': 'இது சாதாரணத்தை விட அதிகமாக உள்ளது',
                'low': 'இது சாதாரணத்தை விட குறைவாக உள்ளது',
                'fever_detected': 'உங்கள் அன்புக்குரியவருக்கு காய்ச்சல் உள்ளது',
                'low_oxygen': 'ஆக்சிஜன் அளவு குறைவுப்பட்டுள்ளது',
                'high_heart_rate': 'இதயம் சாதாரணத்தை விட வேகமாக துடிக்கிறது',
                'fast_breathing': 'சுவாசம் சாதாரணத்தை விட வேகமாக உள்ளது'
            },
            'te': {
                'normal': 'ఇది బాగుంది.',
                'high': 'ఇది సాధారణం కంటే ఎక్కువ',
                'low': 'ఇది సాధారణం కంటే తక్కువ',
                'fever_detected': 'మీ ప్రియమైనవారికి జ్వరం ఉంది',
                'low_oxygen': 'ఆక్సిజన్ స్థాయిలు తక్కువ',
                'high_heart_rate': 'గుండె సాధారణం కంటే వేగంగా పీసుకుంటుంది',
                'fast_breathing': 'శ్వాస సాధారణం కంటే వేగంగా ఉంది'
            },
            'kn': {
                'normal': 'ಇದು ಚೆನ್ನಾಗಿದೆ.',
                'high': 'ಇದು ಸಾಮಾನ್ಯಕ್ಕಿಂತ ಹೆಚ್ಚಾಗಿದೆ',
                'low': 'ಇದು ಸಾಮಾನ್ಯಕ್ಕಿಂತ ಕಡಿಮೆ',
                'fever_detected': 'ನಿಮ್ಮ ಪ್ರಿಯವರಿಗೆ ಜ್ವರ ಇದೆ',
                'low_oxygen': 'ಆಕ್ಸಿಜನ್ ಮಟ್ಟ ಕಡಿಮೆ',
                'high_heart_rate': 'ಹೃದಯ ಸಾಮಾನ್ಯಕ್ಕಿಂತ ವೇಗವಾಗಿ ಸ್ಪಂದಿಸುತ್ತಿದೆ',
                'fast_breathing': 'ಶ್ವಾಸೋಚ್ಛ್ವಾಸ ಸಾಮಾನ್ಯಕ್ಕಿಂತ ವೇಗವಾಗಿದೆ'
            },
            'mr': {
                'normal': 'हे चांगले आहे.',
                'high': 'हे सामान्यपेक्षा अधिक आहे',
                'low': 'हे सामान्यपेक्षा कमी आहे',
                'fever_detected': 'आपल्या प्रियजनाला ताप आहे',
                'low_oxygen': 'ऑक्सिजन स्तर कमी आहे',
                'high_heart_rate': 'हृदय सामान्यपेक्षा वेगाने धडधडत आहे',
                'fast_breathing': 'श्वसन सामान्यपेक्षा वेगवान आहे'
            }
        }

        lang_dict = simple_translations.get(self.language, simple_translations['en'])
        return lang_dict.get(english_text, english_text)

    def get_suggested_questions(self) -> List[str]:
        """Get suggested questions for doctors (translated)"""

        questions = {
            'en': [
                "What are the main things you're watching right now?",
                "When will we know if the treatment is working?",
                "What else can the family do to help?",
                "How often can we get updates?",
                "What does this risk number mean in real terms?",
                "What happens next if things get worse or better?"
            ],
            'hi': [
                "आप अभी क्या मुख्य चीजें देख रहे हैं?",
                "हम कब जान सकेंगे कि इलाज काम कर रहा है?",
                "परिवार और क्या मदद कर सकता है?",
                "हम कितनी बार अपडेट पा सकते हैं?",
                "यह जोखिम संख्या वास्तव में क्या मायने रखती है?",
                "अगर चीजें बदतर या बेहतर हों तो क्या होता है?"
            ],
            'ta': [
                "நீங்கள் இப்போது எந்த முக்கிய விஷயங்களை பார்த்துக்கொண்டிருக்கிறீர்கள்?",
                "சிகிச்சை வேலை செய்கிறது என்பது நாம் எப்போது அறிந்து கொள்ளுவோம்?",
                "குடும்பம் மேலும் என்ன உதவலாம்?",
                "நாம் எவ்வளவு முறை புதுப்பிப்புகளைப் பெறலாம்?",
                "இந்த ஆபத்து எண் உண்மையில் என்ன அர்த்தம்?",
                "விஷயங்கள் மோசமாக அல்லது சிறப்பாக இருந்தால் அடுத்து என்ன ஆகிறது?"
            ],
            'te': [
                "మీరు ఇప్పుడు ఏ ప్రధాన విషయాలను చూస్తున్నారు?",
                "చికిత్స పనిచేస్తున్నట్లు మనకు ఎప్పుడు తెలుస్తుంది?",
                "కుటుంబం ఇంకా ఏమి సహాయం చేయగలదు?",
                "మనకు ఎంత తరచుగా అప్‌డేట్‌లు వచ్చతాయి?",
                "ఈ ఆపద సంఖ్య వాస్తవానికి ఏ అర్థం?",
                "విషయాలు తీవ్రమైనవి లేదా బాగుపడితే తరువాత ఏమి జరుగుతుంది?"
            ],
            'kn': [
                "ನೀವು ಇದೀಗ ಮುಖ್ಯವಾದ ಯಾವ ವಿಷಯಗಳನ್ನು ನೋಡುತ್ತಿರುವಿರಿ?",
                "ಚಿಕಿತ್ಸೆ ಕೆಲಸ ಮಾಡುತ್ತಿದೆ ಎಂಬುದನ್ನು ನಾವು ಯಾವಾಗ ತಿಳಿಯುವೆವು?",
                "ಕುಟುಂಬ ಇನ್ನು ಏನು ಸಹಾಯ ಮಾಡಬಹುದು?",
                "ನಾವು ಎಷ್ಟು ಬಾರಿ ನವೀಕರಣಗಳನ್ನು ಪಡೆಯಬಹುದು?",
                "ಈ ಅಪಾಯ ಸಂಖ್ಯೆ ವಾಸ್ತವವಾಗಿ ಏನು ಅರ್ಥ?",
                "ವಿಷಯಗಳು ಕೆಟ್ಟವಾಗುತ್ತಿದ್ದರೆ ಅಥವಾ ಸುಧಾರಿತವಾಗಿದ್ದರೆ ಮುಂದೆ ಏನಾಗುತ್ತದೆ?"
            ],
            'mr': [
                "आप सध्या कोणत्या मुख्य गोष्टी पाहत आहात?",
                "उपचार कार्य करत असल्याची आपल्याला कधी माहिती होईल?",
                "कुटुंब आणखी काय मदत करू शकते?",
                "आपल्याला किती वेळा अपडेट मिळू शकतात?",
                "हा धोका क्रमांक खरेतर काय अर्थ राखतो?",
                "गोष्टी वाईट व्हायल्या किंवा सुधारल्या तर पुढे काय होते?"
            ]
        }

        lang_questions = questions.get(self.language, questions['en'])
        return lang_questions

    def get_language_name(self) -> str:
        """Get the name of the current language"""

        lang_names = {
            'en': 'English',
            'hi': 'हिंदी (Hindi)',
            'ta': 'தமிழ் (Tamil)',
            'te': 'తెలుగు (Telugu)',
            'kn': 'ಕನ್ನಡ (Kannada)',
            'mr': 'मराठी (Marathi)'
        }

        return lang_names.get(self.language, 'English')

    def translate_dict(self, data: Dict) -> Dict:
        """
        Recursively translate dictionary values if they have translations available

        Args:
            data: Dictionary with English text values

        Returns:
            Dictionary with translated values
        """

        translated = {}
        for key, value in data.items():
            if isinstance(value, str):
                # Try to translate if it's a known phrase
                translated[key] = self.translate_simple_explanation(value)
            elif isinstance(value, dict):
                translated[key] = self.translate_dict(value)
            elif isinstance(value, list):
                translated[key] = [
                    self.translate_dict(item) if isinstance(item, dict) else
                    self.translate_simple_explanation(item) if isinstance(item, str) else
                    item
                    for item in value
                ]
            else:
                translated[key] = value

        return translated


# Convenience functions for common translations
def translate_to_language(text: str, language: str) -> str:
    """Quick translation function"""
    translator = MultiLanguageTranslator(language)
    return translator.translate_simple_explanation(text)


def get_vital_name_in_language(vital_english: str, language: str) -> str:
    """Quick vital name translation"""
    translator = MultiLanguageTranslator(language)
    return translator.translate_vital_name(vital_english)


def get_supported_languages() -> List[Dict]:
    """Get list of supported languages"""
    return [
        {'code': 'en', 'name': 'English'},
        {'code': 'hi', 'name': 'हिंदी (Hindi)'},
        {'code': 'ta', 'name': 'தமிழ் (Tamil)'},
        {'code': 'te', 'name': 'తెలుగు (Telugu)'},
        {'code': 'kn', 'name': 'ಕನ್ನಡ (Kannada)'},
        {'code': 'mr', 'name': 'मराठी (Marathi)'}
    ]
