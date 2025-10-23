#!/usr/bin/env python3
"""
Email Configuration Setup Script for AIMHSA
"""
import os
import getpass

def setup_email_config():
    """Interactive setup for email configuration"""
    
    print("=" * 60)
    print("📧 AIMHSA - Email Configuration Setup")
    print("=" * 60)
    print()
    
    # Check if .env file already exists
    if os.path.exists('.env'):
        print("⚠️  .env file already exists!")
        choice = input("Do you want to overwrite it? (y/n): ").lower().strip()
        if choice not in ['y', 'yes']:
            print("❌ Setup cancelled.")
            return
    
    print("Choose your email provider:")
    print("1. Gmail (Recommended)")
    print("2. Outlook/Hotmail")
    print("3. Yahoo Mail")
    print("4. Custom SMTP Server")
    print()
    
    choice = input("Enter your choice (1-4): ").strip()
    
    if choice == "1":
        setup_gmail()
    elif choice == "2":
        setup_outlook()
    elif choice == "3":
        setup_yahoo()
    elif choice == "4":
        setup_custom()
    else:
        print("❌ Invalid choice. Setup cancelled.")
        return

def setup_gmail():
    """Setup Gmail configuration"""
    print("\n📧 Gmail Configuration")
    print("-" * 30)
    
    email = input("Enter your Gmail address: ").strip()
    if not email or '@gmail.com' not in email:
        print("❌ Invalid Gmail address.")
        return
    
    print("\n🔑 Gmail App Password Setup:")
    print("1. Go to your Google Account settings")
    print("2. Enable 2-Factor Authentication")
    print("3. Generate an 'App Password' for this application")
    print("4. Use the 16-character app password (not your regular password)")
    print()
    
    password = getpass.getpass("Enter your Gmail App Password: ")
    if not password:
        print("❌ Password is required.")
        return
    
    from_email = input("From email address (default: noreply@aimhsa.rw): ").strip()
    if not from_email:
        from_email = "noreply@aimhsa.rw"
    
    # Create .env content
    env_content = f"""# AIMHSA Email Configuration - Gmail
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME={email}
SMTP_PASSWORD={password}
FROM_EMAIL={from_email}

# Chat Model Configuration
CHAT_MODEL=llama3.2:3b
EMBED_MODEL=nomic-embed-text
SENT_EMBED_MODEL=nomic-embed-text
"""
    
    save_env_file(env_content, "Gmail")

def setup_outlook():
    """Setup Outlook configuration"""
    print("\n📧 Outlook/Hotmail Configuration")
    print("-" * 30)
    
    email = input("Enter your Outlook/Hotmail address: ").strip()
    if not email or ('@outlook.com' not in email and '@hotmail.com' not in email):
        print("❌ Invalid Outlook/Hotmail address.")
        return
    
    password = getpass.getpass("Enter your password: ")
    if not password:
        print("❌ Password is required.")
        return
    
    from_email = input("From email address (default: noreply@aimhsa.rw): ").strip()
    if not from_email:
        from_email = "noreply@aimhsa.rw"
    
    env_content = f"""# AIMHSA Email Configuration - Outlook
SMTP_SERVER=smtp-mail.outlook.com
SMTP_PORT=587
SMTP_USERNAME={email}
SMTP_PASSWORD={password}
FROM_EMAIL={from_email}

# Chat Model Configuration
CHAT_MODEL=llama3.2:3b
EMBED_MODEL=nomic-embed-text
SENT_EMBED_MODEL=nomic-embed-text
"""
    
    save_env_file(env_content, "Outlook")

def setup_yahoo():
    """Setup Yahoo configuration"""
    print("\n📧 Yahoo Mail Configuration")
    print("-" * 30)
    
    email = input("Enter your Yahoo Mail address: ").strip()
    if not email or '@yahoo.com' not in email:
        print("❌ Invalid Yahoo Mail address.")
        return
    
    print("\n🔑 Yahoo App Password Setup:")
    print("1. Go to your Yahoo Account settings")
    print("2. Enable 2-Factor Authentication")
    print("3. Generate an 'App Password' for this application")
    print()
    
    password = getpass.getpass("Enter your Yahoo App Password: ")
    if not password:
        print("❌ Password is required.")
        return
    
    from_email = input("From email address (default: noreply@aimhsa.rw): ").strip()
    if not from_email:
        from_email = "noreply@aimhsa.rw"
    
    env_content = f"""# AIMHSA Email Configuration - Yahoo
SMTP_SERVER=smtp.mail.yahoo.com
SMTP_PORT=587
SMTP_USERNAME={email}
SMTP_PASSWORD={password}
FROM_EMAIL={from_email}

# Chat Model Configuration
CHAT_MODEL=llama3.2:3b
EMBED_MODEL=nomic-embed-text
SENT_EMBED_MODEL=nomic-embed-text
"""
    
    save_env_file(env_content, "Yahoo")

def setup_custom():
    """Setup custom SMTP configuration"""
    print("\n📧 Custom SMTP Configuration")
    print("-" * 30)
    
    server = input("Enter SMTP server: ").strip()
    if not server:
        print("❌ SMTP server is required.")
        return
    
    port = input("Enter SMTP port (default: 587): ").strip()
    if not port:
        port = "587"
    
    username = input("Enter SMTP username: ").strip()
    if not username:
        print("❌ Username is required.")
        return
    
    password = getpass.getpass("Enter SMTP password: ")
    if not password:
        print("❌ Password is required.")
        return
    
    from_email = input("From email address (default: noreply@aimhsa.rw): ").strip()
    if not from_email:
        from_email = "noreply@aimhsa.rw"
    
    env_content = f"""# AIMHSA Email Configuration - Custom SMTP
SMTP_SERVER={server}
SMTP_PORT={port}
SMTP_USERNAME={username}
SMTP_PASSWORD={password}
FROM_EMAIL={from_email}

# Chat Model Configuration
CHAT_MODEL=llama3.2:3b
EMBED_MODEL=nomic-embed-text
SENT_EMBED_MODEL=nomic-embed-text
"""
    
    save_env_file(env_content, "Custom SMTP")

def save_env_file(content, provider):
    """Save .env file with configuration"""
    try:
        with open('.env', 'w') as f:
            f.write(content)
        
        print(f"\n✅ {provider} configuration saved to .env file!")
        print("\n📋 Next Steps:")
        print("1. Restart your Flask application")
        print("2. Test the forgot password functionality")
        print("3. Check the logs for email sending status")
        print("\n🔒 Security Note: Never commit .env files to version control!")
        
    except Exception as e:
        print(f"❌ Error saving .env file: {e}")

if __name__ == "__main__":
    setup_email_config()
