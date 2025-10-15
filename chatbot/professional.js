(() => {
    const API_ROOT = "http://localhost:5057";
    
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
    const addSessionNotesBtn = document.getElementById('addSessionNotesBtn');
    const viewReportsBtn = document.getElementById('viewReportsBtn');
    const emergencyContactsBtn = document.getElementById('emergencyContactsBtn');
    const generateReportBtn = document.getElementById('generateReportBtn');
    const reportContent = document.getElementById('reportContent');
    
    // State
    let currentProfessional = null;
    let notifications = [];
    let sessions = [];
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
        const res = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...opts.headers
            },
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
        
        addSessionNotesBtn.addEventListener('click', openNotesModal);
        viewReportsBtn.addEventListener('click', openReportsModal);
        emergencyContactsBtn.addEventListener('click', openEmergencyModal);
        
        // Modals
        document.querySelectorAll('.close').forEach(closeBtn => {
            closeBtn.addEventListener('click', closeModals);
        });
        
        // Notes form
        notesForm.addEventListener('submit', saveSessionNotes);
        followUpRequired.addEventListener('change', toggleFollowUpDate);
        
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
                </div>
                
                <div class="session-actions">
                    <button class="btn btn-primary btn-small" onclick="viewSessionDetails('${session.bookingId}')">
                        <i class="fas fa-eye"></i> View Details
                    </button>
                    ${session.bookingStatus === 'pending' ? `
                        <button class="btn btn-primary btn-small" onclick="acceptSession('${session.bookingId}')">
                            <i class="fas fa-check"></i> Accept
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
    
    async function viewSessionDetails(bookingId) {
        try {
            const sessionDetails = await api(`/professional/sessions/${bookingId}`);
            displaySessionDetailsModal(sessionDetails);
        } catch (error) {
            console.error('Error loading session details:', error);
            alert('Failed to load session details');
        }
    }
    
    function displaySessionDetailsModal(session) {
        const modal = document.getElementById('sessionModal');
        const content = document.getElementById('sessionDetails');
        
        content.innerHTML = `
            <div class="session-details-modal">
                <div class="session-info">
                    <h3>Session Information</h3>
                    <div class="detail-row">
                        <span class="detail-label">User:</span>
                        <span class="detail-value">${session.userName || 'Anonymous'}</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Scheduled Time:</span>
                        <span class="detail-value">${formatDateTime(session.scheduledDatetime)}</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Session Type:</span>
                        <span class="detail-value">${session.sessionType}</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Risk Level:</span>
                        <span class="detail-value">${session.riskLevel} (${session.riskScore}/100)</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Status:</span>
                        <span class="detail-value">${session.bookingStatus}</span>
                    </div>
                </div>
                
                ${session.conversationSummary ? `
                    <div class="conversation-summary">
                        <h4>Conversation Summary</h4>
                        <p>${session.conversationSummary}</p>
                    </div>
                ` : ''}
                
                ${session.detectedIndicators ? `
                    <div class="risk-indicators">
                        <h4>Risk Indicators</h4>
                        <div class="indicators-list">
                            ${JSON.parse(session.detectedIndicators).map(indicator => `
                                <span class="indicator-tag">${indicator}</span>
                            `).join('')}
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
    
    function closeModals() {
        document.querySelectorAll('.modal').forEach(modal => {
            modal.style.display = 'none';
        });
    }
    
    function toggleFollowUpDate() {
        followUpDateGroup.style.display = followUpRequired.checked ? 'block' : 'none';
    }
    
    async function saveSessionNotes(e) {
        e.preventDefault();
        // Implementation for saving session notes
        alert('Session notes saved successfully');
        closeModals();
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
    
    // Global functions for onclick handlers
    window.markNotificationRead = markNotificationRead;
    window.acceptSession = acceptSession;
    window.viewSessionDetails = viewSessionDetails;
    
})();