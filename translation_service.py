"""
Professional Multilingual Chatbot Translation Service
Supports English, French, Kiswahili, and Kinyarwanda

Features:
- Automatic language detection from user input
- Exclusively responds in the detected language
- Uses GoogleTranslator from deep_translator for accurate translation
- Maintains natural tone, accuracy, and clarity in all supported languages
"""

import re
from typing import Dict, List, Optional, Tuple
from langdetect import detect, detect_langs, DetectorFactory
from deep_translator import GoogleTranslator

# Optional, higher-quality detectors/translators
try:
    import langid  # Lightweight, fast language id
except Exception:  # pragma: no cover
    langid = None

try:
    import pycld3  # Google Compact Language Detector v3
except Exception:  # pragma: no cover
    pycld3 = None

# Set seed for consistent language detection
DetectorFactory.seed = 0

class TranslationService:
    def __init__(self):
        # Initialize GoogleTranslator for all translations
        try:
            self.translator = GoogleTranslator()
        except Exception as e:
            print(f"Warning: Failed to initialize GoogleTranslator: {e}")
            self.translator = None
        
        # Language mappings for supported languages
        self.language_codes = {
            'kinyarwanda': 'rw',
            'french': 'fr', 
            'kiswahili': 'sw',
            'english': 'en'
        }
        
        # Supported language codes for detection
        self.supported_languages = ['en', 'fr', 'sw', 'rw']

        # Domain glossary for consistent Kinyarwanda phrasing
        # Maps common English/French mental health phrases to preferred Kinyarwanda
        self.rw_glossary = [
            # Emergency and helplines
            (r"(?i)mental health hotline\s*:?\s*105", "Umurongo wa telefone w'ubufasha mu by'ubuzima bwo mu mutwe: 105"),
            (r"(?i)ligne d'assistance en santé mentale\s*:?\s*105", "Umurongo wa telefone w'ubufasha mu by'ubuzima bwo mu mutwe: 105"),
            (r"(?i)call\s*112", "Hamagara 112 mu gihe cy'ibyago byihutirwa"),
            (r"(?i)emergency", "ibyago byihutirwa"),
            # Organizations
            (r"(?i)caraes\s*ndera\s*hospital", "CARAES Ndera"),
            (r"(?i)hdi\s*rwanda\s*counseling", "HDI Rwanda (Inama n'Ubujyanama)"),
            (r"(?i)arct\s*ruhuka", "ARCT Ruhuka"),
            # General MH terminology
            (r"(?i)mental health", "ubuzima bwo mu mutwe"),
            (r"(?i)anxiety", "impungenge"),
            (r"(?i)depression", "agahinda kenshi"),
            (r"(?i)stress", "umunaniro w'ubwonko"),
            (r"(?i)coping strategies", "uburyo bwo kwifasha"),
            # Common bad fragments we observed
            (r"(?i)ku bihano[,\s]*", ""),
            (r"(?i)komeza amajwi make ariko akunze", ""),
        ]
        

    def detect_language(self, text: str) -> str:
        """
        Professional language detection for multilingual chatbot.
        Detects language from user input and returns one of: 'en', 'fr', 'sw', 'rw'
        
        Uses ensemble method combining pattern matching, multiple detectors,
        and domain-specific knowledge for maximum accuracy.
        """
        if not text or not text.strip():
            return 'en'  # Default to English for empty text
        
        # Clean the text for better detection
        cleaned_text = re.sub(r'[^\w\s]', '', text.strip().lower())
        
        if len(cleaned_text) < 2:
            return 'en'  # Default to English for very short text
        
        try:
            # Initialize scores for all supported languages
            scores: Dict[str, float] = {lang: 0.0 for lang in self.supported_languages}

            # 1) Pattern-based detection (most reliable for our domain)
            pattern_lang = self._detect_by_patterns(text)
            if pattern_lang and pattern_lang in scores:
                scores[pattern_lang] += 2.0  # High confidence for pattern matches
                print(f"Pattern detection: {pattern_lang}")

            # 2) pycld3 (Google's language detector)
            if pycld3 is not None:
                try:
                    pred = pycld3.get_language(text)
                    if pred and pred.probability and pred.probability > 0.5:
                        mapped = self._map_code(pred.language)
                        if mapped in scores:
                            scores[mapped] += float(pred.probability) * 1.5
                            print(f"CLD3: {pred.language} ({pred.probability:.2f}) -> {mapped}")
                except Exception:
                    pass

            # 3) langid (fast and lightweight)
            if langid is not None:
                try:
                    lid, conf = langid.classify(text)
                    mapped = self._map_code(lid)
                    if mapped in scores:
                        # Convert confidence to positive score
                        confidence_score = max(0, conf) if conf > 0 else 0.5
                        scores[mapped] += confidence_score
                        print(f"langid: {lid} (conf: {conf:.2f}) -> {mapped}")
                except Exception:
                    pass

            # 4) langdetect (probabilistic)
            try:
                detected_langs = detect_langs(text)
                for cand in detected_langs[:3]:
                    mapped = self._map_code(cand.lang)
                    if mapped in scores:
                        scores[mapped] += float(cand.prob) * 1.2
                if detected_langs:
                    print("LangDetect top:", [(c.lang, f"{c.prob:.2f}") for c in detected_langs[:3]])
            except Exception:
                pass

            # 5) Domain-specific bias for Kinyarwanda
            if self._has_strong_kinyarwanda_tokens(cleaned_text):
                scores['rw'] += 1.0
                print("Kinyarwanda bias applied")

            # 6) French bias for common French patterns
            if self._has_strong_french_tokens(cleaned_text):
                scores['fr'] += 0.8
                print("French bias applied")

            # 7) Kiswahili bias for common Swahili patterns
            if self._has_strong_kiswahili_tokens(cleaned_text):
                scores['sw'] += 0.8
                print("Kiswahili bias applied")

            # Choose the language with the highest score
            best_lang = max(scores.items(), key=lambda x: x[1])[0]
            best_score = scores[best_lang]
            
            # If all scores are very low, default to English
            if best_score < 0.3:
                print(f"Low confidence detection (best: {best_lang} with {best_score:.2f}), defaulting to English")
                return 'en'
            
            print(f"Language detected: {best_lang} (confidence: {best_score:.2f})")
            return best_lang

        except Exception as e:
            print(f"Language detection error: {e}")
            return 'en'
    
    def _detect_by_patterns(self, text: str) -> str:
        """
        Detect language using comprehensive pattern matching for better accuracy
        """
        text_lower = text.lower().strip()
        
        # Count matches for each language to determine the strongest signal
        language_scores = {'rw': 0, 'fr': 0, 'sw': 0, 'en': 0}
        
        # Kinyarwanda patterns - more comprehensive
        kinyarwanda_patterns = [
            # Greetings
            r'\b(muraho|murakaza|murabe|murakoze|mwiriwe|mwaramutse|murakaza neza|muraho rwose|muraho neza)\b',
            # Common words
            r'\b(ndabizi|ntabwo|ndabishaka|ndabishimira|ndabishimye|ndabishimye cyane|ndumva)\b',
            # Time/day related
            r'\b(umunsi|umunsi mwiza|umunsi mubi|ejo|ejo hazaza|ejo hashize|uyu munsi)\b',
            # Peace/well-being
            r'\b(amahoro|amahoro yose|amahoro yanyu|amahoro yanjye)\b',
            # Mental health related
            r'\b(ubwoba|ubwoba bubabaje|ubwoba bunyuma|ubwoba bwinshi|umutwe|umereye|nabi)\b',
            r'\b(umutima|umutima wanjye|umutima wanyu|umutima wanjye)\b',
            r'\b(ubuzima|ubuzima bwiza|ubuzima bubi|ubuzima bwinshi)\b',
            # Common verbs
            r'\b(nshaka|ntabwo|ndabizi|ndabishimira|ndabishimye|ndumva|ndabishimye)\b',
            # Pronouns
            r'\b(jewe|wewe|we|jewe|twebwe|mwebwe|bo)\b',
            # Common phrases
            r'\b(murakoze|murakoze cyane|murakoze cane|murakoze rwose)\b',
            r"\b(ntabwo|ntabwo bimeze|ntabwo bimeze nk'uko)\b",
            # Additional Kinyarwanda words
            r'\b(umutwe|umereye|nabi|ndumva|cyane|rwose|neza)\b'
        ]
        
        # French patterns - more comprehensive
        french_patterns = [
            # Greetings
            r'\b(bonjour|bonsoir|salut|bonne journée|bonne soirée)\b',
            r'\b(merci|merci beaucoup|merci bien|de rien)\b',
            # Common phrases
            r'\b(comment allez-vous|comment ça va|ça va bien|ça va mal)\b',
            r'\b(je suis|je vais|je peux|je veux|je dois|je fais)\b',
            # Feelings/emotions
            r'\b(très bien|très mal|pas mal|comme ci comme ça|ça va)\b',
            r'\b(anxieux|anxieuse|déprimé|déprimée|stressé|stressée)\b',
            # Time related
            r"\b(depuis|pendant|maintenant|hier|demain|aujourd'hui)\b",
            # Mental health
            r'\b(problème|difficulté|souci|inquiétude|santé mentale)\b',
            r'\b(santé|mental|psychologique|émotionnel|psychologue)\b',
            # Common words
            r'\b(avec|sans|pour|dans|sur|sous|entre|parmi)\b',
            r'\b(et|ou|mais|donc|car|ni|puis)\b'
        ]
        
        # Kiswahili patterns - more comprehensive
        kiswahili_patterns = [
            # Greetings
            r'\b(hujambo|hamjambo|habari|habari yako|habari za asubuhi|habari za mchana)\b',
            r'\b(asante|asante sana|karibu|pole|pole sana|pole kwa ajili)\b',
            # Common responses
            r'\b(sijambo|hajambo|hatujambo|hamjambo|hawajambo)\b',
            # Pronouns
            r'\b(mimi|wewe|yeye|sisi|nyinyi|wao)\b',
            # Verbs
            r'\b(nina|una|ana|tuna|mna|wana|niko|uko|ako|tuko|mko|wako)\b',
            # Mental health related
            r'\b(shida|matatizo|huzuni|furaha|wasiwasi|msongo wa mawazo)\b',
            r'\b(afya ya akili|moyo|roho|hisia|mawazo)\b',
            # Common words
            r'\b(rafiki|mpenzi|mama|baba|mtoto|mzee|mke|mume)\b',
            r'\b(leo|jana|kesho|sasa|zamani|baadaye)\b',
            # Common phrases
            r'\b(naomba|tafadhali|samahani|pole|pole sana)\b'
        ]
        
        # English patterns - to distinguish from other languages
        english_patterns = [
            r'\b(hello|hi|hey|good morning|good afternoon|good evening)\b',
            r'\b(thank you|thanks|please|sorry|excuse me)\b',
            r"\b(i am|i'm|i have|i can|i will|i would)\b",
            r'\b(help|support|assistance|mental health|anxiety|depression)\b',
            r'\b(how are you|how do you|what is|where is|when is)\b'
        ]
        
        # Count pattern matches
        for pattern in kinyarwanda_patterns:
            if re.search(pattern, text_lower):
                language_scores['rw'] += 1
        
        for pattern in french_patterns:
            if re.search(pattern, text_lower):
                language_scores['fr'] += 1
        
        for pattern in kiswahili_patterns:
            if re.search(pattern, text_lower):
                language_scores['sw'] += 1
                
        for pattern in english_patterns:
            if re.search(pattern, text_lower):
                language_scores['en'] += 1
        
        # Return the language with the highest score
        if max(language_scores.values()) > 0:
            best_language = max(language_scores.items(), key=lambda x: x[1])[0]
            print(f"Pattern detection scores: {language_scores}, best: {best_language}")
            return best_language
        
        return None

    def _map_code(self, code: str) -> str:
        """Map various detector codes into our set {en, fr, sw, rw}."""
        mapping = {
            'en': 'en', 'eng': 'en',
            'fr': 'fr', 'fra': 'fr', 'fre': 'fr',
            'sw': 'sw', 'swa': 'sw', 'swc': 'sw',
            'rw': 'rw', 'kin': 'rw',
        }
        return mapping.get(code, 'en')

    def _has_strong_kinyarwanda_tokens(self, text_lower: str) -> bool:
        """Check for strong Kinyarwanda indicators"""
        tokens = [
            'muraho', 'mwiriwe', 'mwaramutse', 'murakoze', 'ndumva',
            'ubwoba', 'umutwe', 'umereye', 'nabi', 'amahoro', 'ubuzima',
            'ndabizi', 'ntabwo', 'ndabishaka', 'ndabishimira', 'cyane', 'rwose'
        ]
        return any(t in text_lower for t in tokens)
    
    def _has_strong_french_tokens(self, text_lower: str) -> bool:
        """Check for strong French indicators"""
        tokens = [
            'bonjour', 'bonsoir', 'merci', 'comment', 'allez-vous', 'ça va',
            'je suis', 'je vais', 'je peux', 'très bien', 'très mal',
            'anxieux', 'déprimé', 'stressé', 'santé mentale', 'problème'
        ]
        return any(t in text_lower for t in tokens)
    
    def _has_strong_kiswahili_tokens(self, text_lower: str) -> bool:
        """Check for strong Kiswahili indicators"""
        tokens = [
            'hujambo', 'hamjambo', 'habari', 'asante', 'karibu', 'pole',
            'sijambo', 'hajambo', 'mimi', 'wewe', 'yeye', 'sisi', 'nyinyi',
            'nina', 'una', 'ana', 'tuna', 'mna', 'wana', 'shida', 'matatizo'
        ]
        return any(t in text_lower for t in tokens)
    
    def _is_common_greeting(self, text: str) -> bool:
        """Check if text is a common greeting that should default to English"""
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
        return text.lower().strip() in greetings

    def translate_text(self, text: str, target_language: str) -> str:
        """
        Professional translation using GoogleTranslator exclusively.
        Translates text to target language with high accuracy and natural tone.
        
        Args:
            text: Text to translate
            target_language: Target language code ('en', 'fr', 'sw', 'rw')
            
        Returns:
            Translated text in target language
        """
        if not text or not text.strip():
            return text
            
        if target_language == 'en':
            return text
            
        try:
            # Normalize target language code
            target_code = self._normalize_language_code(target_language)
            
            if not self.translator:
                print("GoogleTranslator not available, returning original text")
                return text

            # Create translator instance for this specific translation
            translator = GoogleTranslator(source='auto', target=target_code)
            
            # Perform translation
            result = translator.translate(text)
            
            if not result or not result.strip():
                print(f"Translation returned empty for: {text[:50]}...")
                return text
                
            # Check if translation is identical to original (likely failed)
            if result.strip().lower() == text.strip().lower():
                print(f"Translation identical to original for: {text[:50]}...")
                return text
            
            # Apply language-specific post-processing
            if target_code == 'rw':
                result = self.normalize_kinyarwanda(result)
            elif target_code == 'fr':
                result = self.normalize_french(result)
            elif target_code == 'sw':
                result = self.normalize_kiswahili(result)
            
            print(f"Successfully translated to {target_language}: {text[:30]}... -> {result[:30]}...")
            return result
            
        except Exception as e:
            print(f"Translation error for '{text[:50]}...' to {target_language}: {e}")
            return text
    
    def _normalize_language_code(self, lang: str) -> str:
        """Normalize language code to GoogleTranslator format"""
        mapping = {
            'en': 'en', 'english': 'en',
            'fr': 'fr', 'french': 'fr', 'français': 'fr',
            'sw': 'sw', 'kiswahili': 'sw', 'swahili': 'sw',
            'rw': 'rw', 'kinyarwanda': 'rw', 'kin': 'rw', 'ikinyarwanda': 'rw'
        }
        return mapping.get(lang.lower(), 'en')

    def normalize_kinyarwanda(self, text: str) -> str:
        """
        Post-process Kinyarwanda to remove mixed-language fragments and enforce
        consistent, professional terminology using a small domain glossary.
        """
        if not text:
            return text
        normalized = text
        # Remove common French connective phrases that sometimes leak in
        french_leak_patterns = [
            r"(?i)ligne d'assistance en santé mentale",
            r"(?i)pour|avec|sans|dans|sur|entre|car|donc|mais|ou",
        ]
        for pat in french_leak_patterns:
            normalized = re.sub(pat, '', normalized).strip()

        # Apply glossary replacements
        for pat, repl in self.rw_glossary:
            normalized = re.sub(pat, repl, normalized)

        # Trim repetitive spaces and stray punctuation
        normalized = re.sub(r"\s+", " ", normalized).strip()
        normalized = re.sub(r"\s+,", ",", normalized)
        normalized = re.sub(r"\s+\.", ".", normalized)
        return normalized
    
    def normalize_french(self, text: str) -> str:
        """
        Post-process French text to ensure natural, professional tone
        """
        if not text:
            return text
            
        normalized = text
        
        # Fix common translation artifacts
        french_fixes = [
            (r'\bje suis\s+je suis\b', 'je suis'),  # Remove duplicate "je suis"
            (r'\btrès\s+très\b', 'très'),  # Remove duplicate "très"
            (r'\bde\s+de\b', 'de'),  # Remove duplicate "de"
            (r'\bdu\s+du\b', 'du'),  # Remove duplicate "du"
            (r'\bdes\s+des\b', 'des'),  # Remove duplicate "des"
        ]
        
        for pattern, replacement in french_fixes:
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        
        # Clean up spacing and punctuation
        normalized = re.sub(r"\s+", " ", normalized).strip()
        normalized = re.sub(r"\s+,", ",", normalized)
        normalized = re.sub(r"\s+\.", ".", normalized)
        
        return normalized
    
    def normalize_kiswahili(self, text: str) -> str:
        """
        Post-process Kiswahili text to ensure natural, professional tone
        """
        if not text:
            return text
            
        normalized = text
        
        # Fix common translation artifacts
        kiswahili_fixes = [
            (r'\bmimi\s+mimi\b', 'mimi'),  # Remove duplicate "mimi"
            (r'\bwewe\s+wewe\b', 'wewe'),  # Remove duplicate "wewe"
            (r'\byeye\s+yeye\b', 'yeye'),  # Remove duplicate "yeye"
            (r'\bsisi\s+sisi\b', 'sisi'),  # Remove duplicate "sisi"
            (r'\bnyinyi\s+nyinyi\b', 'nyinyi'),  # Remove duplicate "nyinyi"
            (r'\bwao\s+wao\b', 'wao'),  # Remove duplicate "wao"
        ]
        
        for pattern, replacement in kiswahili_fixes:
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        
        # Clean up spacing and punctuation
        normalized = re.sub(r"\s+", " ", normalized).strip()
        normalized = re.sub(r"\s+,", ",", normalized)
        normalized = re.sub(r"\s+\.", ".", normalized)
        
        return normalized

    def get_appropriate_response(self, english_response: str, user_language: str) -> str:
        """
        Get response in the user's detected language with improved reliability.
        This is the main method for ensuring single-language responses.
        """
        if user_language == 'en' or not user_language:
            return english_response
        
        try:
            translated = self.translate_text(english_response, user_language)
            # Verify the translation is different from original
            if translated != english_response:
                return translated
            else:
                print(f"Translation verification failed, returning English")
                return english_response
        except Exception as e:
            print(f"Error in get_appropriate_response: {e}")
            return english_response
    
    def process_user_message(self, user_message: str, english_response: str) -> str:
        """
        Main method for professional multilingual chatbot.
        
        Automatically detects the user's language from their message and responds
        exclusively in that same language. This is the primary interface method.
        
        Args:
            user_message: The user's input message
            english_response: The AI-generated response in English
            
        Returns:
            Response translated to the user's detected language
        """
        if not user_message or not english_response:
            return english_response
        
        # Detect language from user's message
        detected_language = self.detect_language(user_message)
        
        print(f"User message language detected: {detected_language}")
        print(f"User message: {user_message[:100]}...")
        
        # If English detected, return original response
        if detected_language == 'en':
            return english_response
        
        # Translate response to detected language
        try:
            translated_response = self.translate_text(english_response, detected_language)
            
            # Verify translation quality
            if translated_response and translated_response != english_response:
                print(f"Successfully translated response to {detected_language}")
                return translated_response
            else:
                print(f"Translation failed, returning English response")
                return english_response
                
        except Exception as e:
            print(f"Error in process_user_message: {e}")
            return english_response

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


    def get_language_name(self, lang_code: str) -> str:
        """
        Get full language name from code
        """
        # Map language codes to full names
        code_to_name = {
            'en': 'English',
            'fr': 'Français',
            'sw': 'Kiswahili',
            'rw': 'Ikinyarwanda',
            'kin': 'Ikinyarwanda'
        }
        return code_to_name.get(lang_code, 'English')
    
    def is_supported_language(self, lang_code: str) -> bool:
        """
        Check if a language code is supported
        """
        return lang_code.lower() in self.supported_languages
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported language codes
        """
        return self.supported_languages.copy()

# Global translation service instance
translation_service = TranslationService()

# Convenience function for easy integration
def translate_chatbot_response(user_message: str, english_response: str) -> str:
    """
    Convenience function for translating chatbot responses.
    
    This is the main function to use for integrating the multilingual
    chatbot functionality into your application.
    
    Args:
        user_message: The user's input message
        english_response: The AI-generated response in English
        
    Returns:
        Response translated to the user's detected language
    """
    return translation_service.process_user_message(user_message, english_response)
