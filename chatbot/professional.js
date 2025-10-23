(() => {
    // Compute API root: if served from 8000, use 5057; otherwise same-origin
    let apiRoot;
    try {
        const loc = window.location;
        if (loc.port === '8000') {
            apiRoot = `${loc.protocol}//${loc.hostname}:5057`;
        } else if (loc.port === '5057' || loc.port === '') {
            apiRoot = loc.origin;
        } else {
            apiRoot = 'http://localhost:5057';
        }
    } catch (_) {
        apiRoot = 'http://localhost:5057';
    }
    const API_ROOT = apiRoot;
    
    // Elements
    const professionalName = document.getElementById('professionalName');
    const notificationsList = document.getElementById('notificationsList');
    const upcomingSessions = document.getElementById('upcomingSessions');
    const sessionHistory = document.getElementById('sessionHistory');
    const markAllReadBtn = document.getElementById('markAllReadBtn');
    const refreshSessionsBtn = document.getElementById('refreshSessionsBtn');
    const refreshNotificationsBtn = document.getElementById('refreshNotificationsBtn');
    const logoutBtn = document.getElementById('logoutBtn');
    const sessionModal = document.getElementById('sessionModal');
    const notesModal = document.getElementById('notesModal');
    const reportsModal = document.getElementById('reportsModal');
    const emergencyModal = document.getElementById('emergencyModal');
    const notesForm = document.getElementById('notesForm');
    const followUpRequired = document.getElementById('followUpRequired');
    const followUpDateGroup = document.getElementById('followUpDateGroup');
    
    // New elements
    const totalSessions = document.getElementById('totalSessions');
    const unreadNotifications = document.getElementById('unreadNotifications');
    const upcomingToday = document.getElementById('upcomingToday');
    const highRiskSessions = document.getElementById('highRiskSessions');
    const sessionFilter = document.getElementById('sessionFilter');
    const historyFilter = document.getElementById('historyFilter');
    const viewAllSessionsBtn = document.getElementById('viewAllSessionsBtn');
    const viewBookedUsersBtn = document.getElementById('viewBookedUsersBtn');
    const addSessionNotesBtn = document.getElementById('addSessionNotesBtn');
    const viewReportsBtn = document.getElementById('viewReportsBtn');
    const emergencyContactsBtn = document.getElementById('emergencyContactsBtn');
    const generateReportBtn = document.getElementById('generateReportBtn');
    const reportContent = document.getElementById('reportContent');
    const openIntakeBtn = document.getElementById('openIntakeBtn');
    const intakeModal = document.getElementById('intakeModal');
    const intakeForm = document.getElementById('intakeForm');
    
    // Booked users elements
    const bookedUsersList = document.getElementById('bookedUsersList');
    const userFilter = document.getElementById('userFilter');
    const refreshUsersBtn = document.getElementById('refreshUsersBtn');
    const userProfileModal = document.getElementById('userProfileModal');
    const userProfileContent = document.getElementById('userProfileContent');
    
    // State
    let currentProfessional = null;
    let notifications = [];
    let sessions = [];
    let bookedUsers = [];
    let currentSessionId = null;
    
    // Initialize
    init();
    
    async function init() {
        // Check if professional is logged in
        const professionalData = localStorage.getItem('aimhsa_professional');
        if (!professionalData) {
            // Check if they're logged in as a different type of user
            const userData = localStorage.getItem('aimhsa_account');
            const adminData = localStorage.getItem('aimhsa_admin');
            
            if (userData && userData !== 'null') {
                alert('You are logged in as a regular user. Please logout and login as a professional.');
                window.location.href = '/';
                return;
            }
            
            if (adminData) {
                alert('You are logged in as an admin. Please logout and login as a professional.');
                window.location.href = '/admin_dashboard.html';
                return;
            }
            
            window.location.href = '/login';
            return;
        }
        
        currentProfessional = JSON.parse(professionalData);
        professionalName.textContent = currentProfessional.name;
        
        // Load initial data
        await loadDashboardData();
        await loadNotifications();
        await loadSessions();
        await loadBookedUsers();
        
        // Set up auto-refresh
        setInterval(loadDashboardData, 30000); // Every 30 seconds
        setInterval(loadNotifications, 30000); // Every 30 seconds
        setInterval(loadSessions, 60000); // Every minute
        
        // Set up event listeners
        setupEventListeners();
    }
    
    // API Helper
    async function api(path, opts = {}) {
        const url = API_ROOT + path;
        const headers = {
            'Content-Type': 'application/json',
            ...opts.headers
        };
        // Include professional id header if available
        if (currentProfessional?.professional_id) {
            headers['X-Professional-ID'] = String(currentProfessional.professional_id);
        }
        const res = await fetch(url, {
            headers,
            ...opts
        });
        
        if (!res.ok) {
            const txt = await res.text();
            throw new Error(txt || res.statusText);
        }
        
        return await res.json();
    }
    
    function setupEventListeners() {
        // Logout
        logoutBtn.addEventListener('click', logout);
        
        // Notifications
        markAllReadBtn.addEventListener('click', markAllNotificationsRead);
        refreshNotificationsBtn.addEventListener('click', loadNotifications);
        
        // Sessions
        refreshSessionsBtn.addEventListener('click', loadSessions);
        sessionFilter.addEventListener('change', filterSessions);
        historyFilter.addEventListener('change', filterSessionHistory);
        
        // Quick actions
        viewAllSessionsBtn.addEventListener('click', () => {
            sessionFilter.value = 'all';
            loadSessions();
        });
        
        viewBookedUsersBtn.addEventListener('click', () => {
            userFilter.value = 'all';
            loadBookedUsers();
        });
        
        addSessionNotesBtn.addEventListener('click', openNotesModal);
        viewReportsBtn.addEventListener('click', openReportsModal);
        emergencyContactsBtn.addEventListener('click', openEmergencyModal);
        openIntakeBtn.addEventListener('click', openIntakeModal);
        
        // Booked users
        refreshUsersBtn.addEventListener('click', loadBookedUsers);
        userFilter.addEventListener('change', filterBookedUsers);
        
        // Modals
        document.querySelectorAll('.close').forEach(closeBtn => {
            closeBtn.addEventListener('click', closeModals);
        });
        
        // Notes form
        notesForm.addEventListener('submit', saveSessionNotes);
        followUpRequired.addEventListener('change', toggleFollowUpDate);
        intakeForm.addEventListener('submit', submitIntakeForm);
        
        // Report generation
        generateReportBtn.addEventListener('click', generateReport);
        
        // Close modals when clicking outside
        window.addEventListener('click', (e) => {
            if (e.target.classList.contains('modal')) {
                closeModals();
            }
        });
    }
    
    async function loadDashboardData() {
        try {
            // Load dashboard stats
            const stats = await api('/professional/dashboard-stats');
            updateDashboardStats(stats);
        } catch (error) {
            console.error('Error loading dashboard data:', error);
        }
    }
    
    async function loadNotifications() {
        try {
            const data = await api('/professional/notifications');
            notifications = data;
            displayNotifications(notifications);
        } catch (error) {
            console.error('Error loading notifications:', error);
            notificationsList.innerHTML = '<p>Error loading notifications</p>';
        }
    }
    
    async function loadSessions() {
        try {
            const data = await api('/professional/sessions');
            sessions = data;
            displaySessions(sessions);
        } catch (error) {
            console.error('Error loading sessions:', error);
            upcomingSessions.innerHTML = '<p>Error loading sessions</p>';
        }
    }
    
    async function loadBookedUsers() {
        try {
            const data = await api('/professional/booked-users');
            bookedUsers = data.users || [];
            displayBookedUsers(bookedUsers);
        } catch (error) {
            console.error('Error loading booked users:', error);
            bookedUsersList.innerHTML = '<p>Error loading booked users</p>';
        }
    }
    
    function updateDashboardStats(stats) {
        totalSessions.textContent = stats.totalSessions || 0;
        unreadNotifications.textContent = stats.unreadNotifications || 0;
        upcomingToday.textContent = stats.upcomingToday || 0;
        highRiskSessions.textContent = stats.highRiskCases || 0;
    }
    
    function displayNotifications(notificationsData) {
        if (!notificationsData || notificationsData.length === 0) {
            notificationsList.innerHTML = '<p>No notifications</p>';
            return;
        }
        
        notificationsList.innerHTML = notificationsData.map(notification => `
            <div class="notification-item ${notification.isRead ? '' : 'unread'}" onclick="markNotificationRead('${notification.id}')">
                <div class="notification-icon">
                    <i class="fas ${getNotificationIcon(notification.type)}"></i>
                </div>
                <div class="notification-content">
                    <div class="notification-title">${notification.title}</div>
                    <div class="notification-message">${notification.message}</div>
                    <div class="notification-time">${formatDateTime(notification.createdAt)}</div>
                </div>
            </div>
        `).join('');
    }
    
    function displaySessions(sessionsData) {
        if (!sessionsData || sessionsData.length === 0) {
            upcomingSessions.innerHTML = '<p>No sessions found</p>';
            return;
        }
        
        upcomingSessions.innerHTML = sessionsData.map(session => `
            <div class="session-card ${session.riskLevel === 'high' ? 'high-risk' : ''}">
                <div class="session-header">
                    <div class="session-user">
                        <div class="user-avatar">${session.userName ? session.userName.charAt(0).toUpperCase() : 'U'}</div>
                        <div class="user-info">
                            <h4>${session.userName || 'Anonymous User'}</h4>
                            <p>${session.userAccount || 'Guest'}</p>
                        </div>
                    </div>
                    <div class="session-status status-${session.bookingStatus}">
                        ${session.bookingStatus}
                    </div>
                </div>
                
                <div class="session-details">
                    <div class="detail-row">
                        <span class="detail-label">Session Type:</span>
                        <span class="detail-value">${session.sessionType}</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Scheduled:</span>
                        <span class="detail-value">${formatDateTime(session.scheduledDatetime)}</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Risk Level:</span>
                        <span class="detail-value">${session.riskLevel}</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Risk Score:</span>
                        <span class="detail-value">${session.riskScore}/100</span>
                    </div>
                    ${session.userPhone ? `
                    <div class="detail-row contact-info">
                        <span class="detail-label">📞 Contact:</span>
                        <span class="detail-value">
                            <a href="tel:${session.userPhone}" class="contact-link">${session.userPhone}</a>
                        </span>
                    </div>
                    ` : ''}
                    ${session.userEmail ? `
                    <div class="detail-row contact-info">
                        <span class="detail-label">📧 Email:</span>
                        <span class="detail-value">
                            <a href="mailto:${session.userEmail}" class="contact-link">${session.userEmail}</a>
                        </span>
                    </div>
                    ` : ''}
                    ${session.userLocation ? `
                    <div class="detail-row contact-info">
                        <span class="detail-label">📍 Location:</span>
                        <span class="detail-value">${session.userLocation}</span>
                    </div>
                    ` : ''}
                </div>
                
                <div class="session-actions">
                    <button class="btn btn-primary btn-small" onclick="viewSessionDetails('${session.bookingId}')">
                        <i class="fas fa-eye"></i> View Details
                    </button>
                    ${session.bookingStatus === 'pending' ? `
                        <button class="btn btn-primary btn-small" onclick="acceptSession('${session.bookingId}')">
                            <i class="fas fa-check"></i> Accept
                        </button>
                        <button class="btn btn-secondary btn-small" onclick="declineSession('${session.bookingId}')">
                            <i class="fas fa-times"></i> Decline
                        </button>
                    ` : ''}
                </div>
            </div>
        `).join('');
    }
    
    function filterSessions() {
        const filter = sessionFilter.value;
        let filteredSessions = sessions;
        
        switch(filter) {
            case 'today':
                filteredSessions = sessions.filter(session => isToday(new Date(session.scheduledDatetime * 1000)));
                break;
            case 'this_week':
                filteredSessions = sessions.filter(session => isThisWeek(new Date(session.scheduledDatetime * 1000)));
                break;
            case 'high_risk':
                filteredSessions = sessions.filter(session => session.riskLevel === 'high' || session.riskLevel === 'critical');
                break;
        }
        
        displaySessions(filteredSessions);
    }
    
    function filterSessionHistory() {
        const filter = historyFilter.value;
        // Implementation for filtering session history
        console.log('Filtering session history by:', filter);
    }
    
    function displayBookedUsers(usersData) {
        if (!usersData || usersData.length === 0) {
            bookedUsersList.innerHTML = '<p>No booked users found</p>';
            return;
        }
        
        bookedUsersList.innerHTML = usersData.map(user => `
            <div class="user-card ${user.highestRiskLevel === 'high' || user.highestRiskLevel === 'critical' ? 'high-risk' : ''}">
                <div class="user-header">
                    <div class="user-avatar">${user.fullName.charAt(0).toUpperCase()}</div>
                    <div class="user-info">
                        <h4>${user.fullName}</h4>
                        <p>@${user.userAccount}</p>
                        <span class="user-status status-${user.highestRiskLevel}">${user.highestRiskLevel.toUpperCase()}</span>
                    </div>
                </div>
                
                <div class="user-details">
                    <div class="detail-row">
                        <span class="detail-label">Email:</span>
                        <span class="detail-value">${user.email}</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Phone:</span>
                        <span class="detail-value">${user.telephone}</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Location:</span>
                        <span class="detail-value">${user.district}, ${user.province}</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Total Bookings:</span>
                        <span class="detail-value">${user.totalBookings}</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Highest Risk Score:</span>
                        <span class="detail-value">${user.highestRiskScore}/100</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Last Booking:</span>
                        <span class="detail-value">${formatDateTime(user.lastBookingTime)}</span>
                    </div>
                </div>
                
                <div class="user-actions">
                    <button class="btn btn-primary btn-small" onclick="viewUserProfile('${user.userAccount}')">
                        <i class="fas fa-user"></i> View Profile
                    </button>
                    <button class="btn btn-secondary btn-small" onclick="viewUserSessions('${user.userAccount}')">
                        <i class="fas fa-calendar"></i> Sessions
                    </button>
                </div>
            </div>
        `).join('');
    }
    
    function filterBookedUsers() {
        const filter = userFilter.value;
        let filteredUsers = bookedUsers;
        
        switch(filter) {
            case 'high_risk':
                filteredUsers = bookedUsers.filter(user => 
                    user.highestRiskLevel === 'high' || user.highestRiskLevel === 'critical'
                );
                break;
            case 'recent':
                const oneWeekAgo = Date.now() - (7 * 24 * 60 * 60 * 1000);
                filteredUsers = bookedUsers.filter(user => 
                    user.lastBookingTime * 1000 > oneWeekAgo
                );
                break;
            case 'multiple_sessions':
                filteredUsers = bookedUsers.filter(user => user.totalBookings > 1);
                break;
        }
        
        displayBookedUsers(filteredUsers);
    }
    
    async function markAllNotificationsRead() {
        try {
            await api('/professional/notifications/mark-all-read', { method: 'POST' });
            await loadNotifications();
            await loadDashboardData();
        } catch (error) {
            console.error('Error marking notifications as read:', error);
        }
    }
    
    async function markNotificationRead(notificationId) {
        try {
            await api(`/professional/notifications/${notificationId}/read`, { method: 'POST' });
            await loadNotifications();
            await loadDashboardData();
        } catch (error) {
            console.error('Error marking notification as read:', error);
        }
    }
    
    async function acceptSession(bookingId) {
        try {
            await api(`/professional/sessions/${bookingId}/accept`, { method: 'POST' });
            await loadSessions();
            await loadDashboardData();
            alert('Session accepted successfully');
        } catch (error) {
            console.error('Error accepting session:', error);
            alert('Failed to accept session');
        }
    }

    async function declineSession(bookingId) {
        try {
            await api(`/professional/sessions/${bookingId}/decline`, { method: 'POST' });
            await loadSessions();
            await loadDashboardData();
            alert('Session declined');
        } catch (error) {
            console.error('Error declining session:', error);
            alert('Failed to decline session');
        }
    }
    
    async function viewSessionDetails(bookingId) {
        try {
            currentSessionId = bookingId;
            const sessionDetails = await api(`/professional/sessions/${bookingId}`);
            
            // Convert the detailed session data to the format expected by displaySessionDetailsModal
            const session = {
                bookingId: sessionDetails.bookingId,
                convId: sessionDetails.convId,
                userAccount: sessionDetails.userAccount,
                userName: sessionDetails.userName,
                userIp: sessionDetails.userIp,
                riskLevel: sessionDetails.riskLevel,
                riskScore: sessionDetails.riskScore,
                detectedIndicators: sessionDetails.detectedIndicators,
                conversationSummary: sessionDetails.conversationSummary,
                bookingStatus: sessionDetails.bookingStatus,
                scheduledDatetime: sessionDetails.scheduledDatetime,
                sessionType: sessionDetails.sessionType,
                createdTs: sessionDetails.createdTs,
                updatedTs: sessionDetails.updatedTs,
                userPhone: sessionDetails.userPhone,
                userEmail: sessionDetails.userEmail,
                userLocation: sessionDetails.userLocation
            };
            
            displaySessionDetailsModal(session, sessionDetails);
        } catch (error) {
            console.error('Error loading session details:', error);
            alert('Failed to load session details');
        }
    }
    
    async function displaySessionDetailsModal(session, sessionDetails = null) {
        const modal = document.getElementById('sessionModal');
        const content = document.getElementById('sessionDetails');
        
        let userInfo = null;
        
        // If sessionDetails is provided, extract user information from it
        if (sessionDetails) {
            userInfo = {
                userAccount: sessionDetails.userAccount,
                fullName: sessionDetails.userFullName,
                email: sessionDetails.userEmail,
                telephone: sessionDetails.userPhone,
                province: sessionDetails.userProvince,
                district: sessionDetails.userDistrict,
                userCreatedAt: sessionDetails.userCreatedAt,
                totalBookings: sessionDetails.sessions ? sessionDetails.sessions.length : 0,
                highestRiskLevel: sessionDetails.riskAssessments && sessionDetails.riskAssessments.length > 0 
                    ? sessionDetails.riskAssessments[0].riskLevel : 'unknown',
                highestRiskScore: sessionDetails.riskAssessments && sessionDetails.riskAssessments.length > 0 
                    ? Math.max(...sessionDetails.riskAssessments.map(r => r.riskScore)) : 0,
                firstBookingTime: sessionDetails.sessions && sessionDetails.sessions.length > 0 
                    ? Math.min(...sessionDetails.sessions.map(s => s.createdTs)) : null,
                lastBookingTime: sessionDetails.sessions && sessionDetails.sessions.length > 0 
                    ? Math.max(...sessionDetails.sessions.map(s => s.createdTs)) : null,
                sessions: sessionDetails.sessions || [],
                riskAssessments: sessionDetails.riskAssessments || [],
                conversations: sessionDetails.conversationHistory || []
            };
        } else {
            // Fallback: try to get user info from booked users or API
            try {
                const user = bookedUsers.find(u => u.userAccount === session.userAccount);
                if (user) {
                    userInfo = user;
                } else {
                    // Try to get from individual user API
                    userInfo = await api(`/professional/users/${session.userAccount}`);
                }
            } catch (fallbackError) {
                console.error('Error loading user info:', fallbackError);
            }
        }
        
        content.innerHTML = `
            <div class="session-details-modal">
                <!-- Session Header -->
                <div class="session-header-section">
                    <div class="session-header-info">
                        <div class="session-user-avatar">
                            ${session.userName ? session.userName.charAt(0).toUpperCase() : 'U'}
                        </div>
                        <div class="session-user-details">
                            <h2>${session.userName || 'Anonymous User'}</h2>
                            <p class="user-account">@${session.userAccount || 'Guest'}</p>
                            <div class="session-meta">
                                <span class="session-id">Booking ID: ${session.bookingId}</span>
                                <span class="session-date">${formatDateTime(session.scheduledDatetime)}</span>
                            </div>
                        </div>
                    </div>
                    <div class="session-status-badge status-${session.bookingStatus}">
                        ${session.bookingStatus.toUpperCase()}
                    </div>
                </div>
                
                <!-- Session Information Grid -->
                <div class="session-info-grid">
                    <div class="info-card session-basic-info">
                        <h3>📋 Session Details</h3>
                        <div class="info-item">
                            <span class="info-label">Session Type:</span>
                            <span class="info-value session-type-${session.sessionType.toLowerCase()}">${session.sessionType}</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Scheduled Time:</span>
                            <span class="info-value">${formatDateTime(session.scheduledDatetime)}</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Created:</span>
                            <span class="info-value">${formatDateTime(session.createdTs)}</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Last Updated:</span>
                            <span class="info-value">${formatDateTime(session.updatedTs)}</span>
                        </div>
                    </div>
                    
                    <div class="info-card risk-assessment">
                        <h3>⚠️ Risk Assessment</h3>
                        <div class="risk-level-display risk-${session.riskLevel}">
                            <div class="risk-level-badge">${session.riskLevel.toUpperCase()}</div>
                            <div class="risk-score-display">${session.riskScore}/100</div>
                        </div>
                        ${session.detectedIndicators ? `
                            <div class="risk-indicators-compact">
                                <h4>Detected Indicators:</h4>
                                <div class="indicators-tags">
                                    ${JSON.parse(session.detectedIndicators).slice(0, 5).map(indicator => `
                                        <span class="indicator-tag">${indicator}</span>
                                    `).join('')}
                                    ${JSON.parse(session.detectedIndicators).length > 5 ? `
                                        <span class="indicator-more">+${JSON.parse(session.detectedIndicators).length - 5} more</span>
                                    ` : ''}
                                </div>
                            </div>
                        ` : ''}
                    </div>
                    
                    <div class="info-card contact-info-card">
                        <h3>📞 Contact Information</h3>
                        ${session.userPhone ? `
                            <div class="contact-item">
                                <span class="contact-icon">📞</span>
                                <div class="contact-details">
                                    <span class="contact-label">Phone</span>
                                    <a href="tel:${session.userPhone}" class="contact-value">${session.userPhone}</a>
                                </div>
                            </div>
                        ` : ''}
                        ${session.userEmail ? `
                            <div class="contact-item">
                                <span class="contact-icon">📧</span>
                                <div class="contact-details">
                                    <span class="contact-label">Email</span>
                                    <a href="mailto:${session.userEmail}" class="contact-value">${session.userEmail}</a>
                                </div>
                            </div>
                        ` : ''}
                        ${session.userLocation ? `
                            <div class="contact-item">
                                <span class="contact-icon">📍</span>
                                <div class="contact-details">
                                    <span class="contact-label">Location</span>
                                    <span class="contact-value">${session.userLocation}</span>
                                </div>
                            </div>
                        ` : ''}
                    </div>
                </div>
                
                <!-- User Comprehensive Information -->
                ${userInfo ? `
                    <div class="user-comprehensive-info">
                        <h3>👤 Complete User Profile</h3>
                        <div class="user-info-grid">
                            <div class="info-section">
                                <h4>📋 Personal Information</h4>
                                <div class="info-item">
                                    <strong>Full Name:</strong> 
                                    <span class="info-value ${!userInfo.fullName ? 'missing-data' : ''}">${userInfo.fullName || 'Not provided'}</span>
                                </div>
                                <div class="info-item">
                                    <strong>Email Address:</strong> 
                                    <span class="info-value ${!userInfo.email ? 'missing-data' : ''}">${userInfo.email || 'Not provided'}</span>
                                </div>
                                <div class="info-item">
                                    <strong>Phone Number:</strong> 
                                    <span class="info-value ${!userInfo.telephone ? 'missing-data' : ''}">${userInfo.telephone || 'Not provided'}</span>
                                </div>
                                <div class="info-item">
                                    <strong>Location:</strong> 
                                    <span class="info-value ${!userInfo.district && !userInfo.province ? 'missing-data' : ''}">${formatLocation(userInfo.district, userInfo.province)}</span>
                                </div>
                                <div class="info-item">
                                    <strong>Account Created:</strong> 
                                    <span class="info-value ${!userInfo.userCreatedAt ? 'missing-data' : ''}">${userInfo.userCreatedAt ? formatDateTime(userInfo.userCreatedAt) : 'Not available'}</span>
                                </div>
                            </div>
                            
                            <div class="info-section">
                                <h4>📊 Session Statistics</h4>
                                <div class="info-item">
                                    <strong>Total Bookings:</strong> 
                                    <span class="info-value highlight">${userInfo.totalBookings || 0}</span>
                                </div>
                                <div class="info-item">
                                    <strong>Highest Risk Level:</strong> 
                                    <span class="info-value risk-badge risk-${userInfo.highestRiskLevel || 'unknown'}">${userInfo.highestRiskLevel || 'Unknown'}</span>
                                </div>
                                <div class="info-item">
                                    <strong>Highest Risk Score:</strong> 
                                    <span class="info-value highlight">${userInfo.highestRiskScore || 0}/100</span>
                                </div>
                                <div class="info-item">
                                    <strong>First Booking:</strong> 
                                    <span class="info-value ${!userInfo.firstBookingTime ? 'missing-data' : ''}">${userInfo.firstBookingTime ? formatDateTime(userInfo.firstBookingTime) : 'Not available'}</span>
                                </div>
                                <div class="info-item">
                                    <strong>Last Booking:</strong> 
                                    <span class="info-value ${!userInfo.lastBookingTime ? 'missing-data' : ''}">${userInfo.lastBookingTime ? formatDateTime(userInfo.lastBookingTime) : 'Not available'}</span>
                                </div>
                            </div>
                        </div>
                        
                        ${userInfo.sessions && userInfo.sessions.length > 0 ? `
                            <div class="recent-sessions-section">
                                <h4>📅 Recent Sessions (Last 5)</h4>
                                <div class="sessions-timeline">
                                    ${userInfo.sessions.slice(0, 5).map(s => `
                                        <div class="timeline-item ${s.bookingId === session.bookingId ? 'current-session' : ''}">
                                            <div class="timeline-marker"></div>
                                            <div class="timeline-content">
                                                <div class="session-header">
                                                    <strong>${s.sessionType}</strong>
                                                    <span class="session-status status-${s.bookingStatus}">${s.bookingStatus}</span>
                                                </div>
                                                <div class="session-details">
                                                    <span class="risk-info">Risk: ${s.riskLevel} (${s.riskScore}/100)</span>
                                                    <span class="date-info">${formatDateTime(s.scheduledDatetime)}</span>
                                                </div>
                                                ${s.bookingId === session.bookingId ? '<div class="current-indicator">Current Session</div>' : ''}
                                            </div>
                                        </div>
                                    `).join('')}
                                </div>
                            </div>
                        ` : ''}
                        
                        ${userInfo.riskAssessments && userInfo.riskAssessments.length > 0 ? `
                            <div class="risk-history-section">
                                <h4>📈 Risk Assessment History</h4>
                                <div class="risk-timeline">
                                    ${userInfo.riskAssessments.slice(0, 5).map(risk => `
                                        <div class="risk-timeline-item">
                                            <div class="risk-marker risk-${risk.riskLevel}"></div>
                                            <div class="risk-content">
                                                <div class="risk-info">
                                                    <span class="risk-level risk-${risk.riskLevel}">${risk.riskLevel}</span>
                                                    <span class="risk-score">${risk.riskScore}/100</span>
                                                </div>
                                                <div class="risk-date">${formatDateTime(risk.timestamp)}</div>
                                            </div>
                                        </div>
                                    `).join('')}
                                </div>
                            </div>
                        ` : ''}
                        
                        ${userInfo.conversations && userInfo.conversations.length > 0 ? `
                            <div class="conversation-history-section">
                                <h4>💬 Recent Conversations</h4>
                                <div class="conversations-timeline">
                                    ${userInfo.conversations.slice(0, 3).map(conv => `
                                        <div class="conversation-timeline-item">
                                            <div class="conv-marker"></div>
                                            <div class="conv-content">
                                                <div class="conv-preview">${conv.preview}</div>
                                                <div class="conv-date">${formatDateTime(conv.timestamp)}</div>
                                            </div>
                                        </div>
                                    `).join('')}
                                </div>
                            </div>
                        ` : ''}
                        
                        <div class="user-actions-section">
                            <button class="btn btn-primary" onclick="viewUserProfile('${session.userAccount}')">
                                <i class="fas fa-user"></i> View Complete Profile
                            </button>
                            <button class="btn btn-secondary" onclick="viewUserSessions('${session.userAccount}')">
                                <i class="fas fa-calendar"></i> View All Sessions
                            </button>
                            <button class="btn btn-success" onclick="openNotesModal()">
                                <i class="fas fa-notes-medical"></i> Add Session Notes
                            </button>
                        </div>
                    </div>
                ` : ''}
                
                <!-- Conversation Summary -->
                ${session.conversationSummary ? `
                    <div class="conversation-summary-section">
                        <h3>💭 Conversation Summary</h3>
                        <div class="summary-content">
                            <p>${session.conversationSummary}</p>
                        </div>
                    </div>
                ` : ''}
                
                <!-- Additional Session Details -->
                ${sessionDetails ? `
                    <div class="additional-session-details">
                        <h3>📋 Additional Session Information</h3>
                        <div class="details-grid">
                            ${sessionDetails.conversationHistory && sessionDetails.conversationHistory.length > 0 ? `
                                <div class="detail-section">
                                    <h4>💬 Full Conversation</h4>
                                    <div class="conversation-full">
                                        ${sessionDetails.conversationHistory.map(msg => `
                                            <div class="message-item ${msg.sender === 'user' ? 'user-message' : 'bot-message'}">
                                                <div class="message-content">${msg.content}</div>
                                                <div class="message-time">${formatDateTime(msg.timestamp)}</div>
                                            </div>
                                        `).join('')}
                                    </div>
                                </div>
                            ` : ''}
                            
                            ${sessionDetails.sessionNotes && sessionDetails.sessionNotes.notes ? `
                                <div class="detail-section">
                                    <h4>📝 Session Notes</h4>
                                    <div class="notes-content">
                                        <p>${sessionDetails.sessionNotes.notes}</p>
                                    </div>
                                </div>
                            ` : ''}
                            
                            ${sessionDetails.sessionNotes && sessionDetails.sessionNotes.treatmentPlan ? `
                                <div class="detail-section">
                                    <h4>🎯 Treatment Plan</h4>
                                    <div class="treatment-content">
                                        <p>${sessionDetails.sessionNotes.treatmentPlan}</p>
                                    </div>
                                </div>
                            ` : ''}
                        </div>
                    </div>
                ` : ''}
            </div>
        `;
        
        modal.style.display = 'block';
    }
    
    function openNotesModal() {
        notesModal.style.display = 'block';
    }
    
    function openReportsModal() {
        reportsModal.style.display = 'block';
    }
    
    function openEmergencyModal() {
        emergencyModal.style.display = 'block';
    }
    
    async function viewUserProfile(userAccount) {
        try {
            // Try to get user from booked users first
            let user = bookedUsers.find(u => u.userAccount === userAccount);
            
            if (!user) {
                // If not found, get from API
                user = await api(`/professional/users/${userAccount}`);
            }
            
            if (!user) {
                alert('User not found');
                return;
            }
            
            displayUserProfileModal(user);
        } catch (error) {
            console.error('Error loading user profile:', error);
            alert('Failed to load user profile');
        }
    }
    
    function displayUserProfileModal(user) {
        userProfileContent.innerHTML = `
            <div class="user-profile-details">
                <div class="profile-header">
                    <div class="profile-avatar">${user.fullName.charAt(0).toUpperCase()}</div>
                    <div class="profile-info">
                        <h3>${user.fullName}</h3>
                        <p>@${user.userAccount}</p>
                        <span class="risk-badge risk-${user.highestRiskLevel}">${user.highestRiskLevel.toUpperCase()}</span>
                    </div>
                </div>
                
                <div class="profile-sections">
                    <div class="profile-section">
                        <h4>Contact Information</h4>
                        <div class="profile-details">
                            <div class="detail-item">
                                <strong>Email:</strong> ${user.email}
                            </div>
                            <div class="detail-item">
                                <strong>Phone:</strong> ${user.telephone}
                            </div>
                            <div class="detail-item">
                                <strong>Location:</strong> ${user.district}, ${user.province}
                            </div>
                            <div class="detail-item">
                                <strong>User Since:</strong> ${formatDateTime(user.userCreatedAt)}
                            </div>
                        </div>
                    </div>
                    
                    <div class="profile-section">
                        <h4>Session Statistics</h4>
                        <div class="profile-details">
                            <div class="detail-item">
                                <strong>Total Bookings:</strong> ${user.totalBookings}
                            </div>
                            <div class="detail-item">
                                <strong>Highest Risk Level:</strong> ${user.highestRiskLevel}
                            </div>
                            <div class="detail-item">
                                <strong>Highest Risk Score:</strong> ${user.highestRiskScore}/100
                            </div>
                            <div class="detail-item">
                                <strong>First Booking:</strong> ${formatDateTime(user.firstBookingTime)}
                            </div>
                            <div class="detail-item">
                                <strong>Last Booking:</strong> ${formatDateTime(user.lastBookingTime)}
                            </div>
                        </div>
                    </div>
                    
                    <div class="profile-section">
                        <h4>Recent Sessions</h4>
                        <div class="sessions-list">
                            ${user.sessions.slice(0, 5).map(session => `
                                <div class="session-item">
                                    <div class="session-info">
                                        <strong>${session.sessionType}</strong>
                                        <span class="session-status status-${session.bookingStatus}">${session.bookingStatus}</span>
                                    </div>
                                    <div class="session-details">
                                        <span>Risk: ${session.riskLevel} (${session.riskScore}/100)</span>
                                        <span>Date: ${formatDateTime(session.scheduledDatetime)}</span>
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                    
                    <div class="profile-section">
                        <h4>Risk Assessment History</h4>
                        <div class="risk-history">
                            ${user.riskAssessments.slice(0, 5).map(risk => `
                                <div class="risk-item">
                                    <div class="risk-info">
                                        <span class="risk-level risk-${risk.riskLevel}">${risk.riskLevel}</span>
                                        <span class="risk-score">${risk.riskScore}/100</span>
                                    </div>
                                    <div class="risk-date">${formatDateTime(risk.timestamp)}</div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                    
                    <div class="profile-section">
                        <h4>Conversation History</h4>
                        <div class="conversations-list">
                            ${user.conversations.map(conv => `
                                <div class="conversation-item">
                                    <div class="conv-preview">${conv.preview}</div>
                                    <div class="conv-date">${formatDateTime(conv.timestamp)}</div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        userProfileModal.style.display = 'block';
    }
    
    function viewUserSessions(userAccount) {
        // Filter sessions to show only this user's sessions
        const userSessions = sessions.filter(session => session.userAccount === userAccount);
        displaySessions(userSessions);
        // Scroll to sessions section
        document.querySelector('.sessions-section').scrollIntoView({ behavior: 'smooth' });
    }
    
    function closeModals() {
        document.querySelectorAll('.modal').forEach(modal => {
            modal.style.display = 'none';
        });
    }

    function openIntakeModal() {
        // Prefill from selected session when available
        const selected = sessions.find(s => s.bookingId === currentSessionId);
        if (selected) {
            document.getElementById('intakeUsername').value = selected.userAccount || '';
            document.getElementById('intakeEmail').value = '';
            document.getElementById('intakeFullName').value = selected.userName || '';
            document.getElementById('intakePhone').value = '';
            document.getElementById('intakeProvince').value = '';
            document.getElementById('intakeDistrict').value = '';
        } else {
            intakeForm.reset();
        }
        intakeModal.style.display = 'block';
    }

    async function submitIntakeForm(e) {
        e.preventDefault();
        const payload = {
            username: document.getElementById('intakeUsername').value.trim(),
            email: document.getElementById('intakeEmail').value.trim(),
            full_name: document.getElementById('intakeFullName').value.trim(),
            phone: document.getElementById('intakePhone').value.trim(),
            province: document.getElementById('intakeProvince').value.trim(),
            district: document.getElementById('intakeDistrict').value.trim(),
            password: document.getElementById('intakePassword').value,
            confirm_password: document.getElementById('intakeConfirmPassword').value
        };
        try {
            await api('/professional/users/intake', {
                method: 'POST',
                body: JSON.stringify(payload)
            });
            alert('User profile saved');
            closeModals();
        } catch (err) {
            console.error('Intake save failed:', err);
            alert('Failed to save user');
        }
    }
    
    function toggleFollowUpDate() {
        followUpDateGroup.style.display = followUpRequired.checked ? 'block' : 'none';
    }
    
    async function saveSessionNotes(e) {
        e.preventDefault();
        try {
            if (!currentSessionId) {
                alert('Open a session to add notes');
                return;
            }
            const payload = {
                notes: document.getElementById('sessionNotes').value,
                treatmentPlan: document.getElementById('treatmentPlan').value,
                followUpRequired: followUpRequired.checked,
                followUpDate: document.getElementById('followUpDate').value || null,
                professional_id: currentProfessional?.professional_id
            };
            await api(`/professional/sessions/${currentSessionId}/notes`, {
                method: 'POST',
                body: JSON.stringify(payload)
            });
            alert('Session notes saved successfully');
            closeModals();
        } catch (err) {
            console.error('Error saving notes:', err);
            alert('Failed to save notes');
        }
    }
    
    async function generateReport() {
        try {
            const report = await api('/professional/reports/generate', {
                method: 'POST',
                body: JSON.stringify({
                    period: document.getElementById('reportPeriod').value,
                    type: document.getElementById('reportType').value
                })
            });
            
            displayReport(report);
        } catch (error) {
            console.error('Error generating report:', error);
            alert('Failed to generate report');
        }
    }
    
    function displayReport(report) {
        reportContent.innerHTML = `
            <div class="report-summary">
                <h3>Report Summary</h3>
                <div class="report-stats">
                    <div class="stat-item">
                        <span class="stat-label">Total Sessions:</span>
                        <span class="stat-value">${report.totalSessions}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Unique Users:</span>
                        <span class="stat-value">${report.uniqueUsers}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">High Risk Cases:</span>
                        <span class="stat-value">${report.highRiskCases}</span>
                    </div>
                </div>
            </div>
        `;
    }
    
    function logout() {
        localStorage.removeItem('aimhsa_professional');
        window.location.href = '/login';
    }
    
    // Utility functions
    function formatDateTime(timestamp) {
        if (!timestamp) return 'N/A';
        const date = new Date(timestamp * 1000);
        return date.toLocaleString();
    }
    
    function isToday(date) {
        const today = new Date();
        return date.toDateString() === today.toDateString();
    }
    
    function isThisWeek(date) {
        const today = new Date();
        const weekAgo = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000);
        return date >= weekAgo && date <= today;
    }
    
    function getNotificationIcon(type) {
        const icons = {
            'session': 'fa-calendar-check',
            'risk': 'fa-exclamation-triangle',
            'user': 'fa-user',
            'system': 'fa-cog',
            'emergency': 'fa-bell'
        };
        return icons[type] || 'fa-bell';
    }
    
    // Helper functions
    function formatLocation(district, province) {
        if (!district && !province) return 'Not provided';
        if (district && province) return `${district}, ${province}`;
        if (district) return district;
        if (province) return province;
        return 'Not provided';
    }
    
    function getRiskBadgeClass(riskLevel) {
        const riskClasses = {
            'low': 'risk-low',
            'medium': 'risk-medium', 
            'high': 'risk-high',
            'critical': 'risk-critical',
            'unknown': 'risk-unknown'
        };
        return riskClasses[riskLevel] || 'risk-unknown';
    }
    
    function formatMissingData(value, fallback = 'Not provided') {
        return value || fallback;
    }
    
    // Global functions for onclick handlers
    window.markNotificationRead = markNotificationRead;
    window.acceptSession = acceptSession;
    window.declineSession = declineSession;
    window.viewSessionDetails = viewSessionDetails;
    window.viewUserProfile = viewUserProfile;
    window.viewUserSessions = viewUserSessions;
    
    // AdminLTE 4 Enhancements
    document.addEventListener('DOMContentLoaded', function() {
        // Initialize AdminLTE components
        if (typeof $ !== 'undefined' && $.fn.DataTable) {
            // Initialize DataTables if available
            initializeDataTables();
        }
        
        // Initialize mobile menu toggle
        initializeMobileMenu();
        
        // Initialize tooltips
        initializeTooltips();
        
        // Initialize loading states
        initializeLoadingStates();
        
        // Initialize animations
        initializeAnimations();
        
        // Initialize enhanced notifications
        initializeEnhancedNotifications();
    });
    
    function initializeDataTables() {
        // Enhanced table functionality with AdminLTE styling
        const tables = document.querySelectorAll('table');
        tables.forEach(table => {
            if (typeof $ !== 'undefined' && $.fn.DataTable) {
                $(table).DataTable({
                    responsive: true,
                    pageLength: 10,
                    order: [[0, 'desc']], // Sort by first column descending
                    columnDefs: [
                        { targets: [-1], orderable: false } // Actions column
                    ],
                    language: {
                        search: "Search:",
                        lengthMenu: "Show _MENU_ entries per page",
                        info: "Showing _START_ to _END_ of _TOTAL_ entries",
                        paginate: {
                            first: "First",
                            last: "Last",
                            next: "Next",
                            previous: "Previous"
                        }
                    }
                });
            }
        });
    }
    
    function initializeMobileMenu() {
        const mobileToggle = document.getElementById('mobileMenuToggle');
        const professionalHeader = document.querySelector('.professional-header');
        
        if (mobileToggle && professionalHeader) {
            mobileToggle.addEventListener('click', function() {
                professionalHeader.classList.toggle('mobile-open');
            });
            
            // Close mobile menu when clicking outside
            document.addEventListener('click', function(e) {
                if (!professionalHeader.contains(e.target) && !mobileToggle.contains(e.target)) {
                    professionalHeader.classList.remove('mobile-open');
                }
            });
        }
    }
    
    function initializeTooltips() {
        // Initialize Bootstrap tooltips if available
        if (typeof $ !== 'undefined' && $.fn.tooltip) {
            $('[data-toggle="tooltip"]').tooltip();
        }
    }
    
    function initializeLoadingStates() {
        // Add loading states to buttons and forms
        const forms = document.querySelectorAll('form');
        forms.forEach(form => {
            form.addEventListener('submit', function() {
                const submitBtn = form.querySelector('button[type="submit"]');
                if (submitBtn) {
                    submitBtn.classList.add('loading');
                    submitBtn.disabled = true;
                    
                    // Remove loading state after 3 seconds (adjust as needed)
                    setTimeout(() => {
                        submitBtn.classList.remove('loading');
                        submitBtn.disabled = false;
                    }, 3000);
                }
            });
        });
        
        // Add loading states to refresh buttons
        const refreshButtons = document.querySelectorAll('[id$="Btn"]');
        refreshButtons.forEach(btn => {
            if (btn.id.includes('refresh') || btn.id.includes('Refresh')) {
                btn.addEventListener('click', function() {
                    btn.classList.add('loading');
                    btn.disabled = true;
                    
                    setTimeout(() => {
                        btn.classList.remove('loading');
                        btn.disabled = false;
                    }, 2000);
                });
            }
        });
    }
    
    function initializeAnimations() {
        // Add fade-in animation to cards
        const cards = document.querySelectorAll('.stat-card, .action-btn, .user-card, .session-card, .notification-item');
        cards.forEach((card, index) => {
            card.classList.add('fade-in');
            card.style.animationDelay = `${index * 0.1}s`;
        });
        
        // Add bounce-in animation to modals
        const modals = document.querySelectorAll('.modal');
        modals.forEach(modal => {
            modal.addEventListener('show.bs.modal', function() {
                const modalContent = modal.querySelector('.modal-content');
                if (modalContent) {
                    modalContent.classList.add('bounce-in');
                }
            });
        });
    }
    
    function initializeEnhancedNotifications() {
        // Enhanced notification system using AdminLTE toast
        window.showProfessionalMessage = function(text, type = 'info') {
            if (typeof $ !== 'undefined' && $.fn.toast) {
                // Use AdminLTE toast if available
                const toastHtml = `
                    <div class="toast" role="alert" aria-live="assertive" aria-atomic="true">
                        <div class="toast-header">
                            <i class="fas fa-${getToastIcon(type)} mr-2"></i>
                            <strong class="mr-auto">AIMHSA Professional</strong>
                            <button type="button" class="ml-2 mb-1 close" data-dismiss="toast">
                                <span aria-hidden="true">&times;</span>
                            </button>
                        </div>
                        <div class="toast-body">
                            ${text}
                        </div>
                    </div>
                `;
                
                // Create toast container if it doesn't exist
                let toastContainer = document.querySelector('.toast-container');
                if (!toastContainer) {
                    toastContainer = document.createElement('div');
                    toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
                    toastContainer.style.zIndex = '9999';
                    document.body.appendChild(toastContainer);
                }
                
                // Add toast to container
                toastContainer.insertAdjacentHTML('beforeend', toastHtml);
                
                // Initialize and show toast
                const toastElement = toastContainer.lastElementChild;
                $(toastElement).toast({
                    autohide: true,
                    delay: 4000
                });
                $(toastElement).toast('show');
                
                // Remove toast after it's hidden
                $(toastElement).on('hidden.bs.toast', function() {
                    $(this).remove();
                });
            } else {
                // Fallback to alert
                alert(text);
            }
        };
        
        function getToastIcon(type) {
            const icons = {
                'success': 'check-circle',
                'error': 'exclamation-triangle',
                'warning': 'exclamation-circle',
                'info': 'info-circle',
                'loading': 'spinner'
            };
            return icons[type] || 'info-circle';
        }
    }
    
    // Enhanced refresh functionality
    function refreshAllData() {
        const refreshButtons = document.querySelectorAll('[id$="Btn"]');
        refreshButtons.forEach(btn => {
            if (btn.id.includes('refresh') || btn.id.includes('Refresh')) {
                btn.classList.add('loading');
                btn.disabled = true;
            }
        });
        
        // Refresh all data
        Promise.all([
            loadNotifications(),
            loadUpcomingSessions(),
            loadSessionHistory(),
            loadBookedUsers(),
            loadDashboardStats()
        ]).finally(() => {
            refreshButtons.forEach(btn => {
                if (btn.id.includes('refresh') || btn.id.includes('Refresh')) {
                    btn.classList.remove('loading');
                    btn.disabled = false;
                }
            });
            if (window.showProfessionalMessage) {
                window.showProfessionalMessage('All data refreshed successfully', 'success');
            }
        });
    }
    
    // Add refresh functionality to all refresh buttons
    document.addEventListener('DOMContentLoaded', function() {
        const refreshButtons = document.querySelectorAll('[id$="Btn"]');
        refreshButtons.forEach(btn => {
            if (btn.id.includes('refresh') || btn.id.includes('Refresh')) {
                btn.addEventListener('click', function() {
                    refreshAllData();
                });
            }
        });
    });
    
    // Enhanced modal functionality
    function initializeEnhancedModals() {
        const modals = document.querySelectorAll('.modal');
        modals.forEach(modal => {
            // Add AdminLTE modal functionality
            if (typeof $ !== 'undefined' && $.fn.modal) {
                $(modal).on('show.bs.modal', function() {
                    const modalContent = $(this).find('.modal-content');
                    modalContent.addClass('bounce-in');
                });
                
                $(modal).on('hidden.bs.modal', function() {
                    const modalContent = $(this).find('.modal-content');
                    modalContent.removeClass('bounce-in');
                });
            }
        });
    }
    
    // Initialize enhanced modals
    document.addEventListener('DOMContentLoaded', function() {
        initializeEnhancedModals();
    });
    
})();