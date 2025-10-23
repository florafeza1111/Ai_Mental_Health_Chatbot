# AIMHSA User Information Flow - Complete Implementation

## 🎯 **Overview**

This document describes the complete implementation of user information flow in AIMHSA, ensuring that when bookings are made, **all user information** (including phone number and location) is properly sent to the professional dashboard and SMS notifications are sent to the professional's phone number.

## 🔄 **Complete User Information Flow**

### **1. User Registration & Data Collection**

#### **User Registration Process**
- **Endpoint**: `POST /register`
- **Required Fields**:
  - `username` - Unique identifier
  - `email` - Contact email
  - `fullname` - Full name for professional reference
  - `telephone` - Phone number for SMS notifications
  - `province` - Location province
  - `district` - Location district
  - `password` - Account security

#### **Database Storage**
```sql
CREATE TABLE users (
    username TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    email TEXT,
    fullname TEXT,
    telephone TEXT,
    province TEXT,
    district TEXT,
    created_ts REAL NOT NULL
)
```

### **2. Risk Assessment & Booking Creation**

#### **Automated Booking Process**
When a user sends a high-risk message:

1. **Risk Assessment Triggered**
   - AI analyzes message content
   - Pattern matching for risk indicators
   - Risk score calculation (0-1 scale)
   - Risk level determination (low/medium/high/critical)

2. **Professional Matching**
   - AI finds best available professional
   - Considers specialization, location, availability
   - Selects optimal match for user's needs

3. **User Data Retrieval**
   ```python
   def get_user_data(username: str) -> Optional[Dict]:
       # Retrieves complete user information:
       # - username, email, fullname
       # - telephone, province, district
   ```

4. **Booking Creation**
   ```python
   def create_automated_booking(conv_id: str, risk_assessment: Dict, user_account: str = None):
       # Creates booking with complete user information
       # Includes user contact details in booking record
   ```

### **3. Professional Notification System**

#### **Dashboard Notifications**
- **Enhanced Notification Title**: Includes user name
- **Comprehensive Message**: Contains user contact information
- **Priority Levels**: Urgent for critical, High for high-risk

```python
# Notification includes:
user_contact_info = f"""
User Contact Information:
Name: {user_data.get('fullname', 'Not provided')}
Phone: {user_data.get('telephone', 'Not provided')}
Email: {user_data.get('email', 'Not provided')}
Location: {user_data.get('district', 'Unknown')}, {user_data.get('province', 'Unknown')}
"""
```

#### **SMS Notifications to Professionals**
Enhanced SMS messages include:

```
AIMHSA Professional Alert

New HIGH risk booking assigned to you.

Booking ID: 12345-abcde-67890
User: John Doe
Risk Level: HIGH
Scheduled: 2024-01-15 14:30

USER CONTACT INFORMATION:
Phone: +250788123456
Email: john.doe@example.com
Location: Gasabo, Kigali City

Please check your professional dashboard for details and accept/decline the booking.

AIMHSA System
```

### **4. Professional Dashboard Display**

#### **Session Cards Enhancement**
Each session card now displays:
- **Basic Session Info**: Type, time, risk level
- **User Contact Information**:
  - 📞 **Phone**: Clickable tel: link
  - 📧 **Email**: Clickable mailto: link
  - 📍 **Location**: District, Province

#### **Session Details Modal**
Comprehensive user information including:
- **Contact Details**: Name, phone, email, location
- **Session History**: Previous bookings and risk levels
- **Risk Assessment History**: Past assessments
- **Conversation History**: Recent conversations

#### **Booked Users Section**
Complete user profiles with:
- **Contact Information**: All user details
- **Session Statistics**: Total bookings, risk levels
- **Location Data**: Province and district
- **Risk History**: Assessment timeline

### **5. API Endpoints Enhanced**

#### **Professional Sessions API**
```python
@app.get("/professional/sessions")
def get_professional_sessions():
    # Enhanced to include user contact information:
    # - userPhone, userEmail, userLocation
    # - JOIN with users table for complete data
```

#### **Booked Users API**
```python
@app.get("/professional/booked-users")
def get_all_booked_users():
    # Comprehensive user information:
    # - Contact details, location, session history
    # - Risk assessments, conversation summaries
```

## 📱 **SMS Integration**

### **SMS Service Configuration**
- **Provider**: HDEV SMS Gateway
- **Format**: Rwanda phone numbers (+250XXXXXXXXX)
- **Sender ID**: 250788

### **SMS Notification Flow**

#### **User SMS Notification**
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

#### **Professional SMS Notification**
```
AIMHSA Professional Alert

New HIGH risk booking assigned to you.

Booking ID: 12345-abcde-67890
User: John Doe
Risk Level: HIGH
Scheduled: 2024-01-15 14:30

USER CONTACT INFORMATION:
Phone: +250788123456
Email: john.doe@example.com
Location: Gasabo, Kigali City

Please check your professional dashboard for details and accept/decline the booking.

AIMHSA System
```

### **SMS Capability Checks**
- **User Phone Validation**: Checks if user has phone number
- **Professional Phone Validation**: Checks if professional has phone number
- **Fallback Handling**: Continues booking even if SMS fails
- **Detailed Logging**: Tracks SMS success/failure

## 🎨 **UI/UX Enhancements**

### **Professional Dashboard Updates**

#### **Session Cards**
- **Contact Information Display**: Prominent phone, email, location
- **Clickable Links**: Direct tel: and mailto: functionality
- **Visual Indicators**: Contact info highlighted with icons
- **Responsive Design**: Mobile-friendly contact display

#### **CSS Enhancements**
```css
.contact-info {
    background: rgba(102, 126, 234, 0.1);
    border-radius: 8px;
    padding: 0.5rem;
    margin: 0.25rem 0;
    border-left: 3px solid #667eea;
}

.contact-link {
    color: #667eea;
    text-decoration: none;
    font-weight: 500;
    transition: color 0.3s ease;
}
```

### **User Experience Improvements**
- **Quick Contact Access**: One-click phone/email from session cards
- **Location Awareness**: Clear district/province display
- **Risk Context**: Contact info displayed with risk level
- **Professional Context**: Full user profile available

## 🔧 **Technical Implementation**

### **Database Schema Updates**
- **Users Table**: Enhanced with contact fields
- **Bookings Table**: Links to user information
- **Notifications Table**: Includes user contact details
- **Sessions Table**: Comprehensive user data

### **API Response Structure**
```json
{
  "bookingId": "12345-abcde-67890",
  "userName": "John Doe",
  "userPhone": "+250788123456",
  "userEmail": "john.doe@example.com",
  "userLocation": "Gasabo, Kigali City",
  "riskLevel": "high",
  "scheduledDatetime": 1705327800,
  "sessionType": "emergency"
}
```

### **Error Handling**
- **Missing Contact Info**: Graceful fallback to "Not provided"
- **SMS Failures**: Booking continues, error logged
- **Data Validation**: Phone number format validation
- **Location Handling**: Province/district fallback logic

## 🧪 **Testing & Validation**

### **Test Script Features**
- **User Registration**: Complete information validation
- **Risk Simulation**: High-risk conversation testing
- **Professional Dashboard**: Contact info display verification
- **SMS Capability**: Service availability testing
- **Data Completeness**: Information flow validation

### **Test Coverage**
- ✅ User registration with complete information
- ✅ High-risk conversation triggering
- ✅ Professional session data with contact info
- ✅ SMS notifications with user details
- ✅ Professional dashboard display
- ✅ Booked users comprehensive information

## 📊 **Monitoring & Analytics**

### **Information Completeness Tracking**
- **Contact Info Score**: Phone, email, location completeness
- **SMS Success Rate**: Notification delivery tracking
- **Professional Response**: Dashboard usage analytics
- **User Engagement**: Contact information updates

### **Quality Metrics**
- **Data Completeness**: Percentage of users with full contact info
- **SMS Delivery**: Success rate of notifications
- **Professional Response**: Time to accept/decline bookings
- **User Satisfaction**: Contact information accuracy

## 🚀 **Deployment Checklist**

### **Pre-Deployment**
- [ ] SMS service configured and tested
- [ ] User registration form includes all required fields
- [ ] Professional dashboard displays contact information
- [ ] SMS notifications include user contact details
- [ ] Database schema updated with contact fields
- [ ] API endpoints return complete user information

### **Post-Deployment**
- [ ] Monitor SMS delivery rates
- [ ] Track user information completeness
- [ ] Verify professional dashboard functionality
- [ ] Test booking creation with contact info
- [ ] Validate notification content
- [ ] Monitor error rates and fallbacks

## 🎯 **Key Benefits**

### **For Users**
- **Immediate Support**: Professionals have contact information
- **Faster Response**: Direct phone/email access
- **Location Awareness**: Professionals know user location
- **Comprehensive Care**: Full context available

### **For Professionals**
- **Complete Information**: All user details available
- **Direct Contact**: Phone and email readily accessible
- **Location Context**: Know user's district/province
- **Risk Context**: Contact info with risk assessment
- **SMS Alerts**: Immediate notifications with contact details

### **For System**
- **Data Completeness**: Comprehensive user profiles
- **Professional Efficiency**: Faster response times
- **Better Outcomes**: Complete information for care
- **System Reliability**: Robust error handling

## 🔮 **Future Enhancements**

### **Planned Improvements**
- **Location Services**: GPS coordinates for precise location
- **Emergency Contacts**: Additional contact persons
- **Language Preferences**: Communication language tracking
- **Accessibility**: Special needs information
- **Medical History**: Relevant health information
- **Insurance Information**: Coverage details

### **Advanced Features**
- **Real-time Location**: Live location sharing for emergencies
- **Multi-language SMS**: Notifications in user's preferred language
- **Voice Calls**: Direct calling from dashboard
- **Video Consultations**: Integrated video calling
- **Appointment Scheduling**: Calendar integration
- **Follow-up Tracking**: Automated follow-up reminders

---

## ✅ **Implementation Status**

- [x] **User Registration**: Complete information collection
- [x] **Database Schema**: Enhanced with contact fields
- [x] **Booking Creation**: Includes user contact information
- [x] **Professional Notifications**: Enhanced with contact details
- [x] **SMS Integration**: User contact info in notifications
- [x] **Professional Dashboard**: Contact info display
- [x] **API Endpoints**: Complete user information
- [x] **UI/UX**: Contact information prominently displayed
- [x] **Error Handling**: Graceful fallbacks
- [x] **Testing**: Comprehensive test coverage
- [x] **Documentation**: Complete implementation guide

**🎉 The complete user information flow is now fully implemented and ready for production use!**

