# SMS Integration for AIMHSA

This document describes the SMS integration with HDEV SMS Gateway for automated booking notifications.

## 🚀 Features

- **Automated SMS Notifications**: Users and professionals receive SMS when bookings are created
- **Real-time Alerts**: Immediate notifications for high-risk cases
- **Professional Notifications**: Mental health professionals get SMS alerts for new bookings
- **User Confirmations**: Users receive booking confirmation and details via SMS

## 📱 SMS Service Configuration

### HDEV SMS Gateway Setup

The system uses HDEV SMS Gateway with the following credentials:
- **API ID**: `HDEV-23fb1b59-aec0-4aef-a351-bfc1c3aa3c52-ID`
- **API Key**: `HDEV-6e36c286-19bb-4b45-838e-8b5cd0240857-KEY`

### Environment Variables

Add these to your `.env` file:

```bash
# SMS Configuration
HDEV_SMS_API_ID=HDEV-23fb1b59-aec0-4aef-a351-bfc1c3aa3c52-ID
HDEV_SMS_API_KEY=HDEV-6e36c286-19bb-4b45-838e-8b5cd0240857-KEY
```

## 🔧 Implementation Details

### Files Added/Modified

1. **`sms_service.py`** - New SMS service class
2. **`app.py`** - Updated with SMS integration
3. **`test_sms_integration.py`** - Test script for SMS functionality

### Key Functions

#### SMS Service Class (`sms_service.py`)
- `send_sms()` - Send basic SMS message
- `send_booking_notification()` - Send booking confirmation to user
- `send_professional_notification()` - Send alert to professional
- `_format_phone_number()` - Format phone numbers for Rwanda (+250XXXXXXXXX)

#### Updated Booking System (`app.py`)
- `create_automated_booking()` - Now includes SMS notifications
- `get_user_data()` - Retrieves user data for SMS
- New admin endpoints for SMS testing and management

## 📋 SMS Message Templates

### User Booking Notification
```
AIMHSA Mental Health Support

URGENT: Professional mental health support has been scheduled

Professional: Dr. Marie Mukamana
Specialization: Psychiatrist
Scheduled: 2024-01-15 14:30
Session Type: Emergency

You will be contacted shortly. If this is an emergency, call 112 or the Mental Health Hotline at 105.

Stay safe and take care.
AIMHSA Team
```

### Professional Notification
```
AIMHSA Professional Alert

New HIGH risk booking assigned to you.

Booking ID: 12345-abcde-67890
User: John Doe
Risk Level: HIGH
Scheduled: 2024-01-15 14:30

Please check your professional dashboard for details and accept/decline the booking.

AIMHSA System
```

## 🧪 Testing

### Run SMS Integration Tests

```bash
python test_sms_integration.py
```

This will test:
1. SMS service status
2. SMS sending functionality
3. User registration with phone numbers
4. Automated booking with SMS notifications

### Manual Testing via API

#### Test SMS Service Status
```bash
curl -X GET http://localhost:5057/admin/sms/status
```

#### Send Test SMS
```bash
curl -X POST http://localhost:5057/admin/sms/test \
  -H "Content-Type: application/json" \
  -d '{"phone": "+250788123456", "message": "Test message"}'
```

#### Send Booking Notification
```bash
curl -X POST http://localhost:5057/admin/sms/send-booking-notification \
  -H "Content-Type: application/json" \
  -d '{"booking_id": "your-booking-id"}'
```

## 📞 Phone Number Format

The system automatically formats phone numbers for Rwanda:
- Input: `0788123456` → Output: `+250788123456`
- Input: `250788123456` → Output: `+250788123456`
- Input: `+250788123456` → Output: `+250788123456`

## 🔄 Integration Flow

1. **User sends high-risk message** → Risk assessment triggered
2. **High/Critical risk detected** → Professional matching activated
3. **Booking created** → SMS sent to user and professional
4. **Professional receives SMS** → Can accept/decline via dashboard
5. **User receives confirmation** → Knows help is on the way

## 🛠️ Troubleshooting

### Common Issues

1. **SMS not sending**
   - Check API credentials in `.env` file
   - Verify phone number format (+250XXXXXXXXX)
   - Check HDEV SMS Gateway account balance

2. **Phone number not found**
   - Ensure users register with phone numbers
   - Check database for user telephone field
   - Verify professional phone numbers in database

3. **API errors**
   - Check network connectivity
   - Verify HDEV SMS Gateway is accessible
   - Check API rate limits

### Debug Mode

Enable debug logging to see SMS service activity:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📊 Monitoring

### SMS Status Endpoint
- **URL**: `GET /admin/sms/status`
- **Purpose**: Check SMS service configuration and connectivity

### Logs
SMS activities are logged with the following levels:
- **INFO**: Successful SMS sends
- **WARNING**: Failed SMS sends (retryable)
- **ERROR**: SMS service errors

## 🔒 Security Considerations

- API credentials are stored in environment variables
- Phone numbers are validated before sending
- SMS content is sanitized to prevent injection
- Rate limiting prevents SMS spam

## 📈 Future Enhancements

- SMS delivery status tracking
- Multi-language SMS support
- SMS templates customization
- Bulk SMS for announcements
- SMS-based appointment reminders

## 🤝 Support

For SMS integration issues:
1. Check the test script output
2. Review application logs
3. Verify HDEV SMS Gateway status
4. Contact HDEV support: info@hdevtech.cloud

---

**Note**: This integration requires an active HDEV SMS Gateway account with sufficient credits for sending SMS messages.

