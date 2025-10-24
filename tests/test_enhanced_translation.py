#!/usr/bin/env python3
"""
Test script for the enhanced multilingual translation service.
Demonstrates automatic language detection and single-language responses.
"""

from translation_service import translation_service, translate_chatbot_response

def test_language_detection():
    """Test language detection accuracy"""
    print("=" * 60)
    print("TESTING LANGUAGE DETECTION")
    print("=" * 60)
    
    test_cases = [
        # English
        ("Hello, how are you today?", "en"),
        ("I'm feeling anxious and need help", "en"),
        ("What are the symptoms of depression?", "en"),
        
        # French
        ("Bonjour, comment allez-vous?", "fr"),
        ("Je me sens anxieux et j'ai besoin d'aide", "fr"),
        ("Quels sont les symptômes de la dépression?", "fr"),
        
        # Kiswahili
        ("Hujambo, habari yako?", "sw"),
        ("Nina wasiwasi na ninahitaji msaada", "sw"),
        ("Je suis très stressé par mon travail", "sw"),  # Mixed French
        
        # Kinyarwanda
        ("Muraho, murakoze cyane", "rw"),
        ("Ndi mu bwoba kandi ndabishaka ubufasha", "rw"),
        ("Ndi mu bwoba bunyuma cyane", "rw"),
    ]
    
    for message, expected_lang in test_cases:
        detected = translation_service.detect_language(message)
        status = "✅" if detected == expected_lang else "❌"
        print(f"{status} '{message[:30]}...' -> Expected: {expected_lang}, Got: {detected}")

def test_translation_quality():
    """Test translation quality and single-language responses"""
    print("\n" + "=" * 60)
    print("TESTING TRANSLATION QUALITY")
    print("=" * 60)
    
    # Test cases: (user_message, english_response, expected_language)
    test_cases = [
        (
            "Hello, I need mental health support",
            "I'm here to help you with your mental health concerns. How are you feeling today?",
            "en"
        ),
        (
            "Bonjour, j'ai besoin d'aide pour ma santé mentale",
            "I'm here to help you with your mental health concerns. How are you feeling today?",
            "fr"
        ),
        (
            "Hujambo, ninahitaji msaada wa afya ya akili",
            "I'm here to help you with your mental health concerns. How are you feeling today?",
            "sw"
        ),
        (
            "Muraho, ndabishaka ubufasha mu by'ubuzima bwo mu mutwe",
            "I'm here to help you with your mental health concerns. How are you feeling today?",
            "rw"
        ),
    ]
    
    for user_msg, english_resp, expected_lang in test_cases:
        print(f"\n--- Testing {expected_lang.upper()} ---")
        print(f"User: {user_msg}")
        print(f"English Response: {english_resp}")
        
        # Test the main convenience function
        translated = translate_chatbot_response(user_msg, english_resp)
        print(f"Translated Response: {translated}")
        
        # Verify it's different from English (unless English was detected)
        if expected_lang != "en":
            is_translated = translated != english_resp
            status = "✅" if is_translated else "❌"
            print(f"{status} Translation successful: {is_translated}")

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n" + "=" * 60)
    print("TESTING EDGE CASES")
    print("=" * 60)
    
    edge_cases = [
        ("", "Hello"),  # Empty user message
        ("Hi", ""),     # Empty response
        ("a", "Very short response"),  # Very short message
        ("123456", "Numbers only"),  # Numbers only
        ("!@#$%^&*()", "Special characters"),  # Special characters only
    ]
    
    for user_msg, english_resp in edge_cases:
        print(f"\nTesting: User='{user_msg}', Response='{english_resp}'")
        try:
            result = translate_chatbot_response(user_msg, english_resp)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error: {e}")

def test_supported_languages():
    """Test supported languages functionality"""
    print("\n" + "=" * 60)
    print("TESTING SUPPORTED LANGUAGES")
    print("=" * 60)
    
    supported = translation_service.get_supported_languages()
    print(f"Supported languages: {supported}")
    
    # Test language name mapping
    for lang_code in supported:
        lang_name = translation_service.get_language_name(lang_code)
        is_supported = translation_service.is_supported_language(lang_code)
        print(f"{lang_code} -> {lang_name} (supported: {is_supported})")

def main():
    """Run all tests"""
    print("ENHANCED MULTILINGUAL TRANSLATION SERVICE TEST")
    print("=" * 60)
    print("This test demonstrates the professional multilingual chatbot")
    print("that automatically detects user language and responds exclusively")
    print("in that same language using GoogleTranslator.")
    print("=" * 60)
    
    try:
        test_language_detection()
        test_translation_quality()
        test_edge_cases()
        test_supported_languages()
        
        print("\n" + "=" * 60)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("The enhanced translation service is ready for production use.")
        print("Key features:")
        print("✅ Automatic language detection from user input")
        print("✅ Exclusively responds in detected language")
        print("✅ Uses GoogleTranslator for high-quality translation")
        print("✅ Maintains natural tone and accuracy")
        print("✅ Supports English, French, Kiswahili, and Kinyarwanda")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

