# AIMHSA - AI Mental Health Support Assistant with Automated Therapy Booking

## Overview

AIMHSA is an advanced AI-powered mental health support system specifically designed for Rwanda. The system combines conversational AI with automated therapy booking capabilities, providing immediate support and professional intervention when needed.

## 🚀 New Features: Automated Therapy Booking System

### Core Functionality

1. **Real-time Risk Assessment**: Every user message is analyzed for mental health risk indicators
2. **Automated Professional Matching**: AI matches users with appropriate mental health professionals
3. **Emergency Session Booking**: Automatic booking for high-risk cases
4. **Professional Dashboard**: Complete session management for mental health professionals
5. **Admin Dashboard**: System monitoring and professional management
6. **Notification System**: Real-time alerts for professionals and administrators

### Risk Detection Levels

- **Critical**: Immediate professional intervention required (suicidal ideation, self-harm)
- **High**: Urgent professional support needed (severe depression, crisis)
- **Medium**: Professional consultation recommended (anxiety, stress)
- **Low**: General support and monitoring (mild concerns)

## 🏗️ System Architecture

### Backend (Flask API)
- **Port**: 5057
- **Database**: SQLite with extended schema
- **AI Models**: Ollama (llama3.2:3b for chat, nomic-embed-text for embeddings)
- **Risk Assessment**: Multi-layered analysis (pattern matching + AI analysis)

### Frontend Components
- **Main Chat Interface**: Enhanced with risk assessment display
- **Admin Dashboard**: Professional and system management
- **Professional Dashboard**: Session management and notifications
- **Professional Login**: Secure access for mental health professionals

## 📊 Database Schema

### New Tables Added

1. **professionals**: Mental health professional profiles
2. **risk_assessments**: Real-time risk analysis records
3. **automated_bookings**: Emergency session bookings
4. **professional_notifications**: Alert system for professionals
5. **therapy_sessions**: Session records and notes
6. **admin_users**: Administrative access control

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- Ollama installed and running
- Required Python packages (see requirements.txt)

### Setup Steps

1. **Install Dependencies**
   ```bash
   pip install flask flask-cors ollama python-dotenv sqlite3 werkzeug numpy pytesseract
   ```

2. **Initialize Database**
   ```bash
   python app.py
   # This will create all necessary tables
   ```

3. **Create Sample Data**
   ```bash
   python create_sample_data.py
   # Creates sample professionals, users, and admin user
   ```

4. **Test Login System**
   ```bash
   python test_login.py
   # Verifies all login functionality
   ```

5. **Start Backend**
   ```bash
   python app.py
   # Runs on http://localhost:5057
   ```

5. **Start Frontend**
   ```bash
   python run_frontend.py
   # Runs on http://localhost:8000
   ```

## 👥 User Roles & Access

### 1. Regular Users
- **Access**: Main chat interface
- **Features**: 
  - AI mental health support
  - Risk assessment display
  - Emergency booking notifications
  - Conversation history

### 2. Mental Health Professionals
- **Access**: Professional dashboard
- **Login**: `/professional_login.html`
- **Features**:
  - View assigned sessions
  - Accept/decline bookings
  - Add session notes
  - Manage treatment plans
  - Receive notifications

### 3. Administrators
- **Access**: Admin dashboard
- **Login**: Use admin credentials
- **Features**:
  - Manage professionals
  - Monitor risk assessments
  - View all bookings
  - System analytics
  - Real-time monitoring

## 🔐 Default Credentials

### Sample Users
- **testuser** / `password123`
- **john_doe** / `password123`
- **jane_smith** / `password123`
- **rwanda_user** / `password123`

### Sample Professionals
- **Dr. Marie Mukamana** (Psychiatrist): `dr_mukamana` / `password123`
- **Jean Ntwari** (Counselor): `counselor_ntwari` / `password123`
- **Grace Umutoni** (Psychologist): `psychologist_umutoni` / `password123`
- **Claudine Nyiraneza** (Social Worker): `social_worker_nyiraneza` / `password123`

### Admin User
- **Username**: `admin`
- **Password**: `admin123`

## 🚨 Emergency Response Flow

1. **User sends message** → Risk assessment triggered
2. **High/Critical risk detected** → Professional matching algorithm activated
3. **Best professional selected** → Automated booking created
4. **Professional notified** → Real-time notification sent
5. **Session scheduled** → User receives confirmation
6. **Professional accepts** → Session confirmed
7. **Session conducted** → Notes and treatment plan recorded

## 📱 API Endpoints

### User Endpoints
- `POST /ask` - Enhanced with risk assessment
- `GET /session` - Session management
- `GET /history` - Conversation history

### Professional Endpoints
- `POST /professional/login` - Professional authentication
- `GET /professional/notifications` - Get notifications
- `PUT /professional/notifications/{id}/read` - Mark as read
- `GET /professional/sessions` - Get assigned sessions
- `PUT /professional/sessions/{id}/status` - Accept/decline sessions
- `POST /professional/sessions/{id}/notes` - Add session notes

### Admin Endpoints
- `POST /admin/professionals` - Create professional
- `GET /admin/professionals` - List professionals
- `GET /admin/bookings` - View all bookings
- `GET /admin/risk-assessments` - Risk assessment history

### Monitoring Endpoints
- `GET /monitor/risk-stats` - Real-time risk statistics
- `GET /monitor/recent-assessments` - Recent assessments

## 🎯 Key Features

### Risk Assessment Engine
- **Pattern Matching**: Regex-based detection of risk indicators
- **AI Analysis**: Ollama-powered sentiment and context analysis
- **Conversation Patterns**: Escalation detection across message history
- **Rwanda-Specific**: Specialized indicators for local context

### Professional Matching Algorithm
- **Specialization Mapping**: Matches risk types to professional expertise
- **Location Proximity**: Considers geographical distance
- **Availability**: Real-time availability checking
- **Experience Scoring**: Prioritizes experienced professionals

### Notification System
- **Real-time Alerts**: Immediate notifications for professionals
- **Priority Levels**: Urgent, high, normal priority classification
- **Multi-channel**: Dashboard notifications with visual indicators
- **Auto-refresh**: Live updates every 30 seconds

## 🔧 Configuration

### Environment Variables
```bash
CHAT_MODEL=llama3.2:3b
EMBED_MODEL=nomic-embed-text
SENT_EMBED_MODEL=nomic-embed-text
```

### Risk Assessment Thresholds
- **Critical**: ≥ 0.8 risk score
- **High**: ≥ 0.6 risk score
- **Medium**: ≥ 0.4 risk score
- **Low**: < 0.4 risk score

### Session Scheduling
- **Critical Risk**: 1 hour from detection
- **High Risk**: 24 hours from detection
- **Emergency Sessions**: Immediate professional contact

## 📈 Monitoring & Analytics

### Real-time Dashboards
- **Risk Statistics**: Live count by risk level
- **Professional Activity**: Session acceptance rates
- **System Health**: Response times and error rates
- **Geographic Distribution**: Risk patterns by location

### Reporting
- **Daily Risk Reports**: Summary of assessments
- **Professional Performance**: Session completion rates
- **System Usage**: User engagement metrics
- **Emergency Response**: Time-to-intervention tracking

## 🛡️ Security & Privacy

### Data Protection
- **Encrypted Storage**: Password hashing with Werkzeug
- **Session Management**: Secure session tokens
- **Access Control**: Role-based permissions
- **Audit Logging**: Complete activity tracking

### Privacy Compliance
- **Anonymized Data**: IP-based sessions for guests
- **Consent Management**: Clear data usage policies
- **Data Retention**: Configurable retention periods
- **Secure Communication**: HTTPS-ready architecture

## 🚀 Deployment

### Production Considerations
- **Database**: Consider PostgreSQL for production
- **Caching**: Redis for session management
- **Load Balancing**: Multiple API instances
- **Monitoring**: Application performance monitoring
- **Backup**: Automated database backups

### Scaling
- **Horizontal Scaling**: Multiple API servers
- **Database Sharding**: Partition by region/district
- **CDN**: Static asset delivery
- **Microservices**: Split into specialized services

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create feature branch
3. Implement changes
4. Add tests
5. Submit pull request

### Code Standards
- **Python**: PEP 8 compliance
- **JavaScript**: ES6+ standards
- **Documentation**: Comprehensive docstrings
- **Testing**: Unit and integration tests

## 📞 Support & Contact

### Technical Support
- **Issues**: GitHub Issues
- **Documentation**: This README
- **Community**: Development discussions

### Mental Health Resources
- **Rwanda Mental Health Hotline**: 105
- **CARAES Ndera Hospital**: +250 788 305 703
- **Emergency Services**: 112

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Rwanda Mental Health Community**: For guidance and requirements
- **Open Source Contributors**: For the amazing tools and libraries
- **Mental Health Professionals**: For their expertise and feedback

---

**⚠️ Important**: This system is designed to supplement, not replace, professional mental health care. Always encourage users to seek professional help when needed.
