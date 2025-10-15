(() => {
    const API_ROOT = "http://localhost:5057";
    
    // Check admin authentication
    const adminData = localStorage.getItem('aimhsa_admin');
    if (!adminData) {
        // Check if they're logged in as a different type of user
        const userData = localStorage.getItem('aimhsa_account');
        const professionalData = localStorage.getItem('aimhsa_professional');
        
        if (userData && userData !== 'null') {
            alert('You are logged in as a regular user. Please logout and login as an admin.');
            window.location.href = '/';
            return;
        }
        
        if (professionalData) {
            alert('You are logged in as a professional. Please logout and login as an admin.');
            window.location.href = '/professional_dashboard.html';
            return;
        }
        
        window.location.href = '/login';
        return;
    }
    
    // Elements
    const navLinks = document.querySelectorAll('.nav-link');
    const adminSections = document.querySelectorAll('.admin-section');
    const addProfessionalBtn = document.getElementById('addProfessionalBtn');
    const professionalModal = document.getElementById('professionalModal');
    const professionalForm = document.getElementById('professionalForm');
    const professionalsGrid = document.getElementById('professionalsGrid');
    const bookingsTableBody = document.getElementById('bookingsTableBody');
    const statusFilter = document.getElementById('statusFilter');
    const riskLevelFilter = document.getElementById('riskLevelFilter');
    const refreshStatsBtn = document.getElementById('refreshStatsBtn');
    const recentAssessments = document.getElementById('recentAssessments');
    const logoutBtn = document.getElementById('logoutBtn');
    
    // State
    let professionals = [];
    let bookings = [];
    let assessments = [];
    
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
        
        return res.json();
    }
    
    // Navigation
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const target = link.getAttribute('href').substring(1);
            
            // Update active nav link
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');
            
            // Show target section
            adminSections.forEach(section => {
                section.classList.remove('active');
                if (section.id === target) {
                    section.classList.add('active');
                    
                    // Load section data
                    switch(target) {
                        case 'professionals':
                            loadProfessionals();
                            break;
                        case 'bookings':
                            loadBookings();
                            break;
                        case 'risk-monitor':
                            loadRiskStats();
                            loadRecentAssessments();
                            break;
                        case 'analytics':
                            loadAnalytics();
                            break;
                    }
                }
            });
        });
    });
    
    // Load Professionals
    async function loadProfessionals() {
        try {
            const response = await api('/admin/professionals');
            professionals = response.professionals || [];
            renderProfessionals();
        } catch (error) {
            console.error('Failed to load professionals:', error);
            showMessage('Failed to load professionals', 'error');
        }
    }
    
    function renderProfessionals() {
        professionalsGrid.innerHTML = '';
        
        if (professionals.length === 0) {
            professionalsGrid.innerHTML = '<p class="no-data">No professionals found</p>';
            return;
        }
        
        professionals.forEach(prof => {
            const card = document.createElement('div');
            card.className = 'professional-card';
            
            const expertiseAreas = prof.expertise_areas.join(', ');
            const languages = prof.languages.join(', ');
            
            card.innerHTML = `
                <div class="professional-header">
                    <div>
                        <div class="professional-name">${prof.first_name} ${prof.last_name}</div>
                        <div class="professional-specialization">${prof.specialization}</div>
                    </div>
                    <div class="professional-status ${prof.is_active ? 'status-active' : 'status-inactive'}">
                        ${prof.is_active ? 'Active' : 'Inactive'}
                    </div>
                </div>
                
                <div class="professional-details">
                    <div class="professional-detail">
                        <span>Email:</span>
                        <span>${prof.email}</span>
                    </div>
                    <div class="professional-detail">
                        <span>Phone:</span>
                        <span>${prof.phone || 'Not provided'}</span>
                    </div>
                    <div class="professional-detail">
                        <span>Experience:</span>
                        <span>${prof.experience_years} years</span>
                    </div>
                    <div class="professional-detail">
                        <span>District:</span>
                        <span>${prof.district || 'Not specified'}</span>
                    </div>
                    <div class="professional-detail">
                        <span>Expertise:</span>
                        <span>${expertiseAreas || 'Not specified'}</span>
                    </div>
                    <div class="professional-detail">
                        <span>Languages:</span>
                        <span>${languages}</span>
                    </div>
                </div>
                
                <div class="professional-actions">
                    <button class="btn-small btn-secondary" onclick="editProfessional(${prof.id})">Edit</button>
                    <button class="btn-small btn-secondary" onclick="toggleProfessionalStatus(${prof.id}, ${prof.is_active})">
                        ${prof.is_active ? 'Deactivate' : 'Activate'}
                    </button>
                </div>
            `;
            
            professionalsGrid.appendChild(card);
        });
    }
    
    // Load Bookings
    async function loadBookings() {
        try {
            const params = new URLSearchParams();
            if (statusFilter.value) params.append('status', statusFilter.value);
            if (riskLevelFilter.value) params.append('risk_level', riskLevelFilter.value);
            
            const response = await api(`/admin/bookings?${params}`);
            bookings = response.bookings || [];
            renderBookings();
        } catch (error) {
            console.error('Failed to load bookings:', error);
            showMessage('Failed to load bookings', 'error');
        }
    }
    
    function renderBookings() {
        bookingsTableBody.innerHTML = '';
        
        if (bookings.length === 0) {
            bookingsTableBody.innerHTML = '<tr><td colspan="7" class="no-data">No bookings found</td></tr>';
            return;
        }
        
        bookings.forEach(booking => {
            const row = document.createElement('tr');
            
            const scheduledTime = new Date(booking.scheduled_datetime * 1000).toLocaleString();
            const userInfo = booking.user_account || `IP: ${booking.user_ip}`;
            
            row.innerHTML = `
                <td>${booking.booking_id.substring(0, 8)}...</td>
                <td>${userInfo}</td>
                <td>${booking.first_name} ${booking.last_name}</td>
                <td><span class="risk-badge risk-${booking.risk_level}">${booking.risk_level.toUpperCase()}</span></td>
                <td>${scheduledTime}</td>
                <td><span class="status-badge status-${booking.booking_status}">${booking.booking_status.toUpperCase()}</span></td>
                <td>
                    <button class="btn-small btn-secondary" onclick="viewBookingDetails('${booking.booking_id}')">View</button>
                </td>
            `;
            
            bookingsTableBody.appendChild(row);
        });
    }
    
    // Load Risk Stats
    async function loadRiskStats() {
        try {
            const response = await api('/monitor/risk-stats');
            const stats = response.risk_stats || {};
            
            document.getElementById('criticalCount').textContent = stats.critical || 0;
            document.getElementById('highCount').textContent = stats.high || 0;
            document.getElementById('mediumCount').textContent = stats.medium || 0;
            document.getElementById('lowCount').textContent = stats.low || 0;
        } catch (error) {
            console.error('Failed to load risk stats:', error);
        }
    }
    
    // Load Recent Assessments
    async function loadRecentAssessments() {
        try {
            const response = await api('/monitor/recent-assessments?limit=10');
            assessments = response.recent_assessments || [];
            renderRecentAssessments();
        } catch (error) {
            console.error('Failed to load recent assessments:', error);
        }
    }
    
    function renderRecentAssessments() {
        recentAssessments.innerHTML = '';
        
        if (assessments.length === 0) {
            recentAssessments.innerHTML = '<p class="no-data">No recent assessments</p>';
            return;
        }
        
        assessments.forEach(assessment => {
            const item = document.createElement('div');
            item.className = 'assessment-item';
            
            const time = new Date(assessment.assessment_timestamp * 1000).toLocaleString();
            const query = assessment.user_query.length > 60 ? 
                assessment.user_query.substring(0, 60) + '...' : 
                assessment.user_query;
            
            item.innerHTML = `
                <div class="assessment-info">
                    <div class="assessment-query">${query}</div>
                    <div class="assessment-time">${time}</div>
                </div>
                <div>
                    <span class="risk-badge risk-${assessment.risk_level}">${assessment.risk_level.toUpperCase()}</span>
                </div>
            `;
            
            recentAssessments.appendChild(item);
        });
    }
    
    // Load Analytics
    async function loadAnalytics() {
        try {
            // Load professionals count
            const profResponse = await api('/admin/professionals');
            document.getElementById('totalProfessionals').textContent = profResponse.professionals?.length || 0;
            
            // Load active bookings count
            const bookingsResponse = await api('/admin/bookings');
            const activeBookings = bookingsResponse.bookings?.filter(b => 
                ['pending', 'confirmed'].includes(b.booking_status)
            ).length || 0;
            document.getElementById('activeBookings').textContent = activeBookings;
            
            // Load completed sessions count
            const completedSessions = bookingsResponse.bookings?.filter(b => 
                b.booking_status === 'completed'
            ).length || 0;
            document.getElementById('completedSessions').textContent = completedSessions;
            
            // Load assessments today count
            const assessmentsResponse = await api('/admin/risk-assessments?limit=1000');
            const today = new Date().toDateString();
            const assessmentsToday = assessmentsResponse.assessments?.filter(a => 
                new Date(a.assessment_timestamp * 1000).toDateString() === today
            ).length || 0;
            document.getElementById('assessmentsToday').textContent = assessmentsToday;
            
        } catch (error) {
            console.error('Failed to load analytics:', error);
        }
    }
    
    // Professional Management
    addProfessionalBtn.addEventListener('click', () => {
        openProfessionalModal();
    });
    
    function openProfessionalModal(professional = null) {
        const modal = document.getElementById('professionalModal');
        const form = document.getElementById('professionalForm');
        const title = document.getElementById('modalTitle');
        
        if (professional) {
            title.textContent = 'Edit Professional';
            populateForm(professional);
        } else {
            title.textContent = 'Add New Professional';
            form.reset();
        }
        
        modal.style.display = 'block';
    }
    
    function populateForm(professional) {
        document.getElementById('username').value = professional.username;
        document.getElementById('first_name').value = professional.first_name;
        document.getElementById('last_name').value = professional.last_name;
        document.getElementById('email').value = professional.email;
        document.getElementById('phone').value = professional.phone || '';
        document.getElementById('specialization').value = professional.specialization;
        document.getElementById('experience_years').value = professional.experience_years || 0;
        document.getElementById('district').value = professional.district || '';
        document.getElementById('consultation_fee').value = professional.consultation_fee || '';
        document.getElementById('bio').value = professional.bio || '';
        
        // Check expertise areas
        const expertiseCheckboxes = document.querySelectorAll('input[name="expertise"]');
        expertiseCheckboxes.forEach(checkbox => {
            checkbox.checked = professional.expertise_areas.includes(checkbox.value);
        });
    }
    
    professionalForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const formData = new FormData(professionalForm);
        const data = Object.fromEntries(formData.entries());
        
        // Get expertise areas
        const expertiseAreas = Array.from(document.querySelectorAll('input[name="expertise"]:checked'))
            .map(cb => cb.value);
        
        const professionalData = {
            ...data,
            expertise_areas: expertiseAreas,
            languages: ['english'], // Default for now
            qualifications: [], // Default for now
            availability_schedule: {} // Default for now
        };
        
        try {
            await api('/admin/professionals', {
                method: 'POST',
                body: JSON.stringify(professionalData)
            });
            
            showMessage('Professional created successfully', 'success');
            closeModal();
            loadProfessionals();
        } catch (error) {
            console.error('Failed to create professional:', error);
            showMessage('Failed to create professional', 'error');
        }
    });
    
    // Modal Management
    function closeModal() {
        document.querySelectorAll('.modal').forEach(modal => {
            modal.style.display = 'none';
        });
    }
    
    document.querySelectorAll('.close').forEach(closeBtn => {
        closeBtn.addEventListener('click', closeModal);
    });
    
    document.querySelectorAll('.modal').forEach(modal => {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                closeModal();
            }
        });
    });
    
    // Event Listeners
    statusFilter.addEventListener('change', loadBookings);
    riskLevelFilter.addEventListener('change', loadBookings);
    refreshStatsBtn.addEventListener('click', () => {
        loadRiskStats();
        loadRecentAssessments();
    });
    
    logoutBtn.addEventListener('click', () => {
        if (confirm('Are you sure you want to logout?')) {
            localStorage.removeItem('aimhsa_admin');
            localStorage.removeItem('aimhsa_account');
            localStorage.removeItem('aimhsa_professional');
            window.location.href = '/login';
        }
    });
    
    refreshStatsBtn.addEventListener('click', () => {
        loadRiskStats();
        loadRecentAssessments();
    });
    
    // Global Functions (for onclick handlers)
    window.editProfessional = (id) => {
        const professional = professionals.find(p => p.id === id);
        if (professional) {
            openProfessionalModal(professional);
        }
    };
    
    window.toggleProfessionalStatus = async (id, currentStatus) => {
        try {
            // This would require a new API endpoint
            showMessage('Status toggle not implemented yet', 'error');
        } catch (error) {
            console.error('Failed to toggle status:', error);
            showMessage('Failed to toggle status', 'error');
        }
    };
    
    window.viewBookingDetails = (bookingId) => {
        const booking = bookings.find(b => b.booking_id === bookingId);
        if (booking) {
            const modal = document.getElementById('bookingModal');
            const details = document.getElementById('bookingDetails');
            
            const scheduledTime = new Date(booking.scheduled_datetime * 1000).toLocaleString();
            const userInfo = booking.user_account || `IP: ${booking.user_ip}`;
            const indicators = booking.detected_indicators.join(', ');
            
            details.innerHTML = `
                <div class="booking-detail">
                    <h3>Booking Information</h3>
                    <p><strong>Booking ID:</strong> ${booking.booking_id}</p>
                    <p><strong>User:</strong> ${userInfo}</p>
                    <p><strong>Professional:</strong> ${booking.first_name} ${booking.last_name}</p>
                    <p><strong>Specialization:</strong> ${booking.specialization}</p>
                    <p><strong>Risk Level:</strong> <span class="risk-badge risk-${booking.risk_level}">${booking.risk_level.toUpperCase()}</span></p>
                    <p><strong>Risk Score:</strong> ${(booking.risk_score * 100).toFixed(1)}%</p>
                    <p><strong>Scheduled Time:</strong> ${scheduledTime}</p>
                    <p><strong>Session Type:</strong> ${booking.session_type}</p>
                    <p><strong>Status:</strong> <span class="status-badge status-${booking.booking_status}">${booking.booking_status.toUpperCase()}</span></p>
                </div>
                
                <div class="booking-detail">
                    <h3>Risk Indicators</h3>
                    <p>${indicators}</p>
                </div>
                
                <div class="booking-detail">
                    <h3>Conversation Summary</h3>
                    <p>${booking.conversation_summary || 'No summary available'}</p>
                </div>
            `;
            
            modal.style.display = 'block';
        }
    };
    
    // Initialize
    loadProfessionals();
    
    // Auto-refresh risk stats every 30 seconds
    setInterval(() => {
        if (document.querySelector('#risk-monitor').classList.contains('active')) {
            loadRiskStats();
            loadRecentAssessments();
        }
    }, 30000);
    
})();
