#!/usr/bin/env python3
"""
Test Email Configuration Script
This script will test your email configuration and send a test email.
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

def test_email_configuration():
    """Test the email configuration by sending a test email."""
    
    # Load environment variables
    load_dotenv()
    
    # Get email configuration
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_username = os.getenv("SMTP_USERNAME", "")
    smtp_password = os.getenv("SMTP_PASSWORD", "")
    from_email = os.getenv("FROM_EMAIL", "noreply@aimhsa.rw")
    
    print("🔧 Testing Email Configuration...")
    print(f"SMTP Server: {smtp_server}")
    print(f"SMTP Port: {smtp_port}")
    print(f"Username: {smtp_username}")
    print(f"Password: {'*' * len(smtp_password) if smtp_password else 'NOT SET'}")
    print(f"From Email: {from_email}")
    print()
    
    if not smtp_username or not smtp_password:
        print("❌ Email configuration is incomplete!")
        print("Please update the .env file with your Gmail credentials.")
        return False
    
    if smtp_password == "your-app-password-here":
        print("❌ Please replace 'your-app-password-here' with your actual Gmail App Password!")
        return False
    
    try:
        # Create test email
        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = smtp_username  # Send to yourself for testing
        msg['Subject'] = "AI Mental Health Chatbot - Email Test"
        
        body = """
        Hello!
        
        This is a test email from your AI Mental Health Chatbot system.
        
        If you receive this email, your email configuration is working correctly!
        
        Best regards,
        AI Mental Health Chatbot System
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Connect to SMTP server
        print("📧 Connecting to SMTP server...")
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_username, smtp_password)
        
        # Send email
        print("📤 Sending test email...")
        text = msg.as_string()
        server.sendmail(from_email, smtp_username, text)
        server.quit()
        
        print("✅ Email sent successfully!")
        print(f"📬 Check your inbox at {smtp_username}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        print("\n🔍 Troubleshooting tips:")
        print("1. Make sure you're using a Gmail App Password (not your regular password)")
        print("2. Ensure 2-Factor Authentication is enabled on your Gmail account")
        print("3. Check that the App Password is correct (16 characters, no spaces)")
        return False

if __name__ == "__main__":
    test_email_configuration()
