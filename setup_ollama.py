#!/usr/bin/env python3
"""
Setup script for AIMHSA with OpenAI client for Ollama
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_ollama.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def check_ollama():
    """Check if Ollama is running"""
    print("🔍 Checking Ollama...")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is running!")
            return True
    except Exception:
        pass
    
    print("❌ Ollama is not running")
    print("💡 Please start Ollama:")
    print("   1. Download from: https://ollama.ai")
    print("   2. Run: ollama serve")
    print("   3. Pull model: ollama pull llama3.2")
    return False

def main():
    print("="*60)
    print("🧠 AIMHSA Setup with OpenAI Client")
    print("="*60)
    
    # Install requirements
    if not install_requirements():
        return
    
    # Check Ollama
    check_ollama()
    
    print("\n" + "="*60)
    print("✅ Setup complete!")
    print("🚀 Run: python run_aimhsa.py")
    print("="*60)

if __name__ == "__main__":
    main()
