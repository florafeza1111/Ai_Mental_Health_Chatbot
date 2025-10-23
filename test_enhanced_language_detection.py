#!/usr/bin/env python3
"""
Enhanced test script for improved language detection accuracy
"""

import requests
import json
import time

def test_language_detection_accuracy():
    """Test the chatbot with various language inputs to verify accuracy"""
    
    base_url = "http://localhost:5057"
    
    # Comprehensive test cases with different languages and scenarios
    test_cases = [
        # English tests
        {
            "language": "English",
            "code": "en", 
            "query": "Hello, I'm feeling anxious today. Can you help me?",
            "expected_keywords": ["hello", "help", "support", "anxiety"]
        },
        {
            "language": "English",
            "code": "en", 
            "query": "I need help with my mental health",
            "expected_keywords": ["help", "mental", "health"]
        },
        
        # French tests
        {
            "language": "French",
            "code": "fr",
            "query": "Bonjour, je me sens anxieux aujourd'hui. Pouvez-vous m'aider?",
            "expected_keywords": ["bonjour", "anxieux", "aider", "vous"]
        },
        {
            "language": "French",
            "code": "fr",
            "query": "J'ai des problèmes de santé mentale",
            "expected_keywords": ["problèmes", "santé", "mentale"]
        },
        {
            "language": "French",
            "code": "fr",
            "query": "Merci beaucoup pour votre aide",
            "expected_keywords": ["merci", "aide", "votre"]
        },
        
        # Kinyarwanda tests
        {
            "language": "Kinyarwanda", 
            "code": "rw",
            "query": "Muraho, ndabishimye uyu munsi. Murakoze mufasha?",
            "expected_keywords": ["muraho", "ndabishimye", "murakoze", "mufasha"]
        },
        {
            "language": "Kinyarwanda", 
            "code": "rw",
            "query": "Nshaka ubufasha bw'ubuzima bw'ubwoba",
            "expected_keywords": ["nshaka", "ubufasha", "ubuzima", "ubwoba"]
        },
        {
            "language": "Kinyarwanda", 
            "code": "rw",
            "query": "Murakoze cyane, ndabishimye cane",
            "expected_keywords": ["murakoze", "cyane", "ndabishimye", "cane"]
        },
        
        # Kiswahili tests
        {
            "language": "Kiswahili",
            "code": "sw", 
            "query": "Hujambo, nina wasiwasi leo. Unaweza kunisaidia?",
            "expected_keywords": ["hujambo", "wasiwasi", "kunisaidia"]
        },
        {
            "language": "Kiswahili",
            "code": "sw", 
            "query": "Nina shida za afya ya akili",
            "expected_keywords": ["shida", "afya", "akili"]
        },
        {
            "language": "Kiswahili",
            "code": "sw", 
            "query": "Asante sana kwa msaada wako",
            "expected_keywords": ["asante", "msaada", "wako"]
        },
        
        # Mixed language tests (should detect dominant language)
        {
            "language": "Mixed (French dominant)",
            "code": "fr",
            "query": "Bonjour, I need help avec mon anxiety",
            "expected_keywords": ["bonjour", "avec"]
        },
        {
            "language": "Mixed (Kinyarwanda dominant)",
            "code": "rw", 
            "query": "Muraho, I am feeling ndabishimye today",
            "expected_keywords": ["muraho", "ndabishimye"]
        },
        
        # Short message tests
        {
            "language": "French",
            "code": "fr",
            "query": "Bonjour",
            "expected_keywords": ["bonjour"]
        },
        {
            "language": "Kinyarwanda",
            "code": "rw",
            "query": "Muraho",
            "expected_keywords": ["muraho"]
        },
        {
            "language": "Kiswahili",
            "code": "sw",
            "query": "Hujambo",
            "expected_keywords": ["hujambo"]
        }
    ]
    
    print("🧪 Testing Enhanced AIMHSA Language Detection Accuracy")
    print("=" * 70)
    
    correct_detections = 0
    total_tests = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing {test_case['language']} ({test_case['code']})")
        print(f"   Query: {test_case['query']}")
        
        try:
            # Make request to the chatbot
            response = requests.post(f"{base_url}/ask", json={
                "query": test_case['query'],
                "account": "test_user"
            }, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get('answer', 'No answer received')
                print(f"   Response: {answer[:80]}...")
                
                # Check if response contains expected language keywords
                answer_lower = answer.lower()
                found_keywords = []
                for keyword in test_case['expected_keywords']:
                    if keyword.lower() in answer_lower:
                        found_keywords.append(keyword)
                
                if found_keywords:
                    print(f"   ✅ Language keywords found: {found_keywords}")
                    correct_detections += 1
                else:
                    print(f"   ❌ Expected keywords not found: {test_case['expected_keywords']}")
                    
            else:
                print(f"   ❌ Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
        
        # Small delay to avoid overwhelming the server
        time.sleep(0.5)
    
    # Calculate accuracy
    accuracy = (correct_detections / total_tests) * 100
    print("\n" + "=" * 70)
    print(f"🎯 Test Results:")
    print(f"   Correct detections: {correct_detections}/{total_tests}")
    print(f"   Accuracy: {accuracy:.1f}%")
    
    if accuracy >= 80:
        print("   ✅ Language detection is working well!")
    elif accuracy >= 60:
        print("   ⚠️  Language detection needs improvement")
    else:
        print("   ❌ Language detection needs significant improvement")
    
    print("\n🔧 Key Improvements Made:")
    print("• Enhanced pattern matching with comprehensive word lists")
    print("• Scoring system for multiple language indicators")
    print("• Better confidence thresholds (0.8 for high confidence)")
    print("• Pattern override for medium confidence cases")
    print("• False positive detection for English")
    print("• Conversation history analysis for consistency")
    print("• Detailed logging for debugging")

if __name__ == "__main__":
    test_language_detection_accuracy()

