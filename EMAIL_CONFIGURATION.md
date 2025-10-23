# Email Configuration Guide for AIMHSA

## Current Status
The email service is not configured, which is why you're seeing the error:
```
ERROR in app: Failed to send email: Email service not configured
```

## How to Configure Email Service

### Option 1: Gmail (Recommended)

1. **Create a `.env` file** in your project root directory
2. **Add the following configuration:**

```env
# Gmail SMTP Configuration
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
FROM_EMAIL=noreply@aimhsa.rw
```

3. **Get Gmail App Password:**
   - Go to your Google Account settings
   - Enable 2-Factor Authentication
   - Generate an "App Password" for this application
   - Use the 16-character app password (not your regular password)

### Option 2: Outlook/Hotmail

```env
SMTP_SERVER=smtp-mail.outlook.com
SMTP_PORT=587
SMTP_USERNAME=your-email@outlook.com
SMTP_PASSWORD=your-password
FROM_EMAIL=noreply@aimhsa.rw
```

### Option 3: Yahoo Mail

```env
SMTP_SERVER=smtp.mail.yahoo.com
SMTP_PORT=587
SMTP_USERNAME=your-email@yahoo.com
SMTP_PASSWORD=your-app-password
FROM_EMAIL=noreply@aimhsa.rw
```

### Option 4: Custom SMTP Server

```env
SMTP_SERVER=your-smtp-server.com
SMTP_PORT=587
SMTP_USERNAME=your-username
SMTP_PASSWORD=your-password
FROM_EMAIL=noreply@aimhsa.rw
```

## Testing Email Configuration

After creating the `.env` file:

1. **Restart your Flask application**
2. **Test the forgot password functionality**
3. **Check the logs** for email sending status

## Current Behavior (Without Email Configuration)

- ✅ **Forgot password still works** - returns reset token in response
- ✅ **Password reset functionality works** - you can use the token manually
- ❌ **No actual emails sent** - token is displayed in the UI for testing

## Security Notes

- **Never commit `.env` files** to version control
- **Use app passwords** instead of regular passwords for Gmail
- **Keep email credentials secure**
- **Consider using environment variables** in production

## Troubleshooting

### Common Issues:

1. **"Authentication failed"** - Check username/password
2. **"Connection refused"** - Check SMTP server and port
3. **"App password required"** - Enable 2FA and generate app password
4. **"Less secure app access"** - Use app passwords instead

### Test Email Configuration:

```bash
# Test with curl
curl -X POST http://localhost:5057/forgot_password \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com"}'
```
