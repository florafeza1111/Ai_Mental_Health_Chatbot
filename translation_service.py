"""
Translation service for AIMHSA chatbot
Supports language detection and translation to Kinyarwanda, French, and Kiswahili
"""

import re
from typing import Dict, List, Optional, Tuple
from langdetect import detect, detect_langs, DetectorFactory
from deep_translator import GoogleTranslator

# Set seed for consistent language detection
DetectorFactory.seed = 0

class TranslationService:
    def __init__(self):
        self.translator = GoogleTranslator()
        
        # Language mappings
        self.language_codes = {
            'kinyarwanda': 'rw',
            'french': 'fr', 
            'kiswahili': 'sw',
            'english': 'en'
        }
        
        # Common Kinyarwanda words/phrases for better detection (deduplicated)
        self.kinyarwanda_indicators = list({
            'muraho', 'murakoze', 'mwiriwe', 'murabeho', 'murakoze cyane',
            'ndabizi', 'sindabizi', 'ndabashaka', 'umunsi', 'ijoro', 'umunsi mwiza'
        })
        
        # Common Kiswahili words/phrases
        self.kiswahili_indicators = list({
            'hujambo', 'sijambo', 'asante', 'karibu', 'pole',
            'samahani', 'hapana', 'ndiyo', 'sawa', 'mzuri',
            'habari', 'salama', 'asante sana', 'karibu sana'
        })
        
        # Common French words/phrases
        self.french_indicators = list({
            'bonjour', 'bonsoir', 'merci', 's\'il vous plaît', 'excusez-moi',
            'pardon', 'oui', 'non', 'comment', 'ça va', 'très bien',
            'mal', 'bien', 'merci beaucoup', 'de rien', 'au revoir'
        })

    def detect_language(self, text: str) -> str:
        """
        Detect the language of the input text
        Returns: 'kinyarwanda', 'french', 'kiswahili', or 'english'
        """
        if not text or len(text.strip()) < 2:
            return 'english'
        
        text_lower = text.lower().strip()
        
        # Heuristic indicators (strong signal if at least 2 matches)
        kinyarwanda_score = sum(1 for indicator in self.kinyarwanda_indicators if indicator in text_lower)
        kiswahili_score = sum(1 for indicator in self.kiswahili_indicators if indicator in text_lower)
        french_score = sum(1 for indicator in self.french_indicators if indicator in text_lower)

        if kinyarwanda_score >= 2:
            return 'kinyarwanda'
        if kiswahili_score >= 2:
            return 'kiswahili'
        if french_score >= 2:
            return 'french'

        # Otherwise, use probability-based langdetect and require confidence
        try:
            langs = detect_langs(text)  # e.g., [en:0.99]
            if not langs:
                return 'english'

            top = max(langs, key=lambda l: l.prob)
            code = top.lang
            prob = getattr(top, 'prob', 0.0)

            # Accept only with sufficient confidence
            if prob < 0.70:
                # Low confidence → fall back to English
                return 'english'

            if code in ('rw', 'kin'):
                return 'kinyarwanda'
            if code in ('fr', 'fra'):
                return 'french'
            if code in ('sw', 'swa'):
                return 'kiswahili'
            return 'english'
                
        except Exception:
            return 'english'

    def translate_text(self, text: str, target_language: str) -> str:
        """
        Translate text to target language
        """
        if not text or target_language == 'english':
            return text
            
        try:
            target_code = self.language_codes.get(target_language, 'en')
            translator = GoogleTranslator(source='en', target=target_code)
            result = translator.translate(text)
            return result
        except Exception as e:
            print(f"Translation error: {e}")
            return text

    def get_multilingual_response(self, english_response: str, user_language: str) -> Dict[str, str]:
        """
        Get response in multiple languages
        Returns: Dict with 'english', 'kinyarwanda', 'french', 'kiswahili' keys
        """
        translations = {
            'english': english_response,
            'kinyarwanda': '',
            'french': '',
            'kiswahili': ''
        }
        
        # Translate to each language
        for lang in ['kinyarwanda', 'french', 'kiswahili']:
            try:
                translations[lang] = self.translate_text(english_response, lang)
            except Exception as e:
                print(f"Error translating to {lang}: {e}")
                translations[lang] = english_response
        
        return translations

    def get_appropriate_response(self, english_response: str, user_language: str) -> str:
        """
        Get response in the user's detected language
        """
        if user_language == 'english':
            return english_response
        
        try:
            return self.translate_text(english_response, user_language)
        except Exception:
            return english_response

    def get_language_name(self, lang_code: str) -> str:
        """
        Get full language name from code
        """
        names = {
            'kinyarwanda': 'Ikinyarwanda',
            'french': 'Français', 
            'kiswahili': 'Kiswahili',
            'english': 'English'
        }
        return names.get(lang_code, 'English')

# Global translation service instance
translation_service = TranslationService()
