#!/usr/bin/env python3
"""
Test Email Configuration Script for AIMHSA
"""
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

def test_email_config():
    """Test the email configuration"""
    
    print("=" * 60)
    print("📧 AIMHSA - Email Configuration Test")
    print("=" * 60)
    
    # Load environment variables
    load_dotenv()
    
    # Get email configuration
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_username = os.getenv("SMTP_USERNAME", "")
    smtp_password = os.getenv("SMTP_PASSWORD", "")
    from_email = os.getenv("FROM_EMAIL", "noreply@aimhsa.rw")
    
    print(f"📧 SMTP Server: {smtp_server}:{smtp_port}")
    print(f"👤 Username: {smtp_username}")
    print(f"📨 From Email: {from_email}")
    print(f"🔑 Password: {'*' * len(smtp_password) if smtp_password else 'Not set'}")
    print()
    
    # Check if configuration is complete
    if not smtp_username or not smtp_password:
        print("❌ Email configuration is incomplete!")
        print("📋 Missing:")
        if not smtp_username:
            print("   - SMTP_USERNAME")
        if not smtp_password:
            print("   - SMTP_PASSWORD")
        print("\n💡 Run 'python setup_email.py' to configure email settings.")
        return False
    
    # Test email sending
    print("🧪 Testing email configuration...")
    
    try:
        # Create test message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = "AIMHSA - Email Configuration Test"
        msg['From'] = from_email
        msg['To'] = smtp_username  # Send test email to yourself
        
        # Create test content
        html_content = """
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <div style="text-align: center; margin-bottom: 30px;">
                    <h1 style="color: #2c5aa0;">AIMHSA</h1>
                    <p style="color: #666;">Mental Health Companion for Rwanda</p>
                </div>
                
                <div style="background-color: #f8f9fa; padding: 30px; border-radius: 8px; margin-bottom: 20px;">
                    <h2 style="color: #2c5aa0; margin-top: 0;">Email Configuration Test</h2>
                    <p>✅ Your email configuration is working correctly!</p>
                    <p>This is a test email to verify that the AIMHSA password reset system can send emails.</p>
                </div>
                
                <div style="text-align: center; color: #666; font-size: 12px;">
                    <p>© 2024 AIMHSA - Mental Health Companion for Rwanda</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        text_content = """
        AIMHSA - Email Configuration Test
        
        ✅ Your email configuration is working correctly!
        
        This is a test email to verify that the AIMHSA password reset system can send emails.
        
        © 2024 AIMHSA - Mental Health Companion for Rwanda
        """
        
        # Attach parts
        part1 = MIMEText(text_content, 'plain')
        part2 = MIMEText(html_content, 'html')
        
        msg.attach(part1)
        msg.attach(part2)
        
        # Send email
        print("📤 Connecting to SMTP server...")
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        
        print("🔐 Authenticating...")
        server.login(smtp_username, smtp_password)
        
        print("📧 Sending test email...")
        server.send_message(msg)
        server.quit()
        
        print("✅ Email configuration test successful!")
        print(f"📨 Test email sent to: {smtp_username}")
        print("\n🎉 Your forgot password system is now ready to send real emails!")
        
        return True
        
    except smtplib.SMTPAuthenticationError:
        print("❌ Authentication failed!")
        print("💡 Check your username and password.")
        print("   For Gmail: Make sure you're using an App Password, not your regular password.")
        return False
        
    except smtplib.SMTPConnectError:
        print("❌ Connection failed!")
        print("💡 Check your SMTP server and port settings.")
        return False
        
    except Exception as e:
        print(f"❌ Email test failed: {e}")
        return False

def check_env_file():
    """Check if .env file exists and show its contents (without passwords)"""
    
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        print("💡 Run 'python setup_email.py' to create email configuration.")
        return False
    
    print("📄 .env file found. Configuration:")
    print("-" * 40)
    
    with open('.env', 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                if 'PASSWORD' in line:
                    # Hide password
                    key, value = line.split('=', 1)
                    print(f"{key}=***")
                else:
                    print(line)
    
    print("-" * 40)
    return True

if __name__ == "__main__":
    print("Choose test option:")
    print("1. Check .env file")
    print("2. Test email configuration")
    print("3. Both")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        check_env_file()
    elif choice == "2":
        test_email_config()
    elif choice == "3":
        if check_env_file():
            print()
            test_email_config()
    else:
        print("Invalid choice. Running both tests...")
        if check_env_file():
            print()
            test_email_config()
