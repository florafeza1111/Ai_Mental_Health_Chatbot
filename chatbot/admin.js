(() => {
    // Pick API base intelligently: if UI runs on 8000, target 5057; otherwise same-origin
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
    
    // Simple inline message helper
    function showMessage(text, type = 'error') {
        const existing = document.querySelector('.inline-message');
        if (existing) existing.remove();
        const message = document.createElement('div');
        message.className = `inline-message ${type}`;
        message.textContent = text;
        message.style.cssText = `
            position: fixed; left: 50%; transform: translateX(-50%);
            top: 12px; z-index: 10000; padding: 10px 14px; border-radius: 8px;
            font-size: 14px; font-weight: 500;
            ${type === 'success'
                ? 'background: rgba(16,185,129,0.12); color:#047857; border:1px solid rgba(16,185,129,0.3);'
                : 'background: rgba(239,68,68,0.12); color:#991b1b; border:1px solid rgba(239,68,68,0.3);'}
        `;
        document.body.appendChild(message);
        setTimeout(() => message.remove(), 3500);
    }

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
    let filteredProfessionals = [];
    let professionalsPage = 1;
    const PRO_PAGE_SIZE = 8;

    let bookings = [];
    let filteredBookings = [];
    let bookingsPage = 1;
    const BOOK_PAGE_SIZE = 10;

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

    // Toolbar actions (global search and refresh)
    const globalSearch = document.getElementById('globalSearch');
    const timeRange = document.getElementById('timeRange');
    const refreshAllBtn = document.getElementById('refreshAllBtn');
    if (globalSearch) {
        globalSearch.addEventListener('input', debounce(() => {
            const q = globalSearch.value.toLowerCase().trim();
            if (document.getElementById('professionals').classList.contains('active')) {
                filteredProfessionals = professionals.filter(p => (
                    `${p.first_name} ${p.last_name}`.toLowerCase().includes(q) ||
                    (p.specialization || '').toLowerCase().includes(q) ||
                    (p.email || '').toLowerCase().includes(q)
                ));
                professionalsPage = 1; renderProfessionals();
            } else if (document.getElementById('bookings').classList.contains('active')) {
                filteredBookings = bookings.filter(b => (
                    (b.user_account || '').toLowerCase().includes(q) ||
                    `${b.first_name} ${b.last_name}`.toLowerCase().includes(q) ||
                    (b.risk_level || '').toLowerCase().includes(q)
                ));
                bookingsPage = 1; renderBookings();
            }
        }, 250));
    }
    if (refreshAllBtn) {
        refreshAllBtn.addEventListener('click', () => {
            loadProfessionals();
            loadBookings();
            loadRiskStats();
            loadRecentAssessments();
            loadAnalytics();
            showToast('Data refreshed', 'success');
        });
    }
    
    // Utility: Toast notifications
    function showToast(message, type = 'info') {
        const host = document.body;
        let container = document.getElementById('toastContainer');
        if (!container) {
            container = document.createElement('div');
            container.id = 'toastContainer';
            container.style.cssText = 'position:fixed;right:16px;bottom:16px;display:flex;flex-direction:column;gap:10px;z-index:9999';
            host.appendChild(container);
        }
        const el = document.createElement('div');
        el.className = `toast toast-${type}`;
        el.textContent = message;
        container.appendChild(el);
        setTimeout(() => { el.style.opacity = '0'; el.style.transform = 'translateY(10px)'; }, 10);
        setTimeout(() => container.removeChild(el), 3000);
    }

    // Debounce helper
    function debounce(fn, wait = 300) {
        let t;
        return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), wait); };
    }

    // Load Professionals
    async function loadProfessionals() {
        try {
            const response = await api('/admin/professionals');
            professionals = response.professionals || [];
            filteredProfessionals = professionals.slice();
            professionalsPage = 1;
            renderProfessionals();
        } catch (error) {
            console.error('Failed to load professionals:', error);
            showMessage('Failed to load professionals', 'error');
            showToast('Failed to load professionals', 'error');
        }
    }
    
    function renderProfessionals() {
        professionalsGrid.innerHTML = '';
        
        if (filteredProfessionals.length === 0) {
            professionalsGrid.innerHTML = '<p class="no-data">No professionals found</p>';
            return;
        }
        
        // Paging
        const start = (professionalsPage - 1) * PRO_PAGE_SIZE;
        const end = start + PRO_PAGE_SIZE;
        const pageItems = filteredProfessionals.slice(start, end);

        pageItems.forEach(prof => {
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
                    <button class="btn-small btn-danger" onclick="deleteProfessional(${prof.id}, '${prof.first_name} ${prof.last_name}')">Delete</button>
                </div>
            `;
            
            professionalsGrid.appendChild(card);
        });

        // Pagination controls
        renderProPagination();
    }

    function renderProPagination() {
        const totalPages = Math.max(1, Math.ceil(filteredProfessionals.length / PRO_PAGE_SIZE));
        let pager = document.getElementById('proPager');
        if (!pager) {
            pager = document.createElement('div');
            pager.id = 'proPager';
            pager.className = 'pager';
            professionalsGrid.parentElement.appendChild(pager);
        }
        pager.innerHTML = '';
        const prev = document.createElement('button'); prev.textContent = 'Prev'; prev.className = 'pager-btn'; prev.disabled = professionalsPage <= 1;
        const next = document.createElement('button'); next.textContent = 'Next'; next.className = 'pager-btn'; next.disabled = professionalsPage >= totalPages;
        const info = document.createElement('span'); info.className = 'pager-info'; info.textContent = `Page ${professionalsPage} of ${totalPages}`;
        prev.onclick = () => { professionalsPage = Math.max(1, professionalsPage - 1); renderProfessionals(); };
        next.onclick = () => { professionalsPage = Math.min(totalPages, professionalsPage + 1); renderProfessionals(); };
        pager.appendChild(prev); pager.appendChild(info); pager.appendChild(next);
    }
    
    // Load Bookings
    async function loadBookings() {
        try {
            const params = new URLSearchParams();
            if (statusFilter.value) params.append('status', statusFilter.value);
            if (riskLevelFilter.value) params.append('risk_level', riskLevelFilter.value);
            
            const response = await api(`/admin/bookings?${params}`);
            bookings = response.bookings || [];
            filteredBookings = bookings.slice();
            bookingsPage = 1;
            renderBookings();
        } catch (error) {
            console.error('Failed to load bookings:', error);
            showMessage('Failed to load bookings', 'error');
            showToast('Failed to load bookings', 'error');
        }
    }
    
    function renderBookings() {
        bookingsTableBody.innerHTML = '';
        
        if (filteredBookings.length === 0) {
            bookingsTableBody.innerHTML = '<tr><td colspan="7" class="no-data">No bookings found</td></tr>';
            return;
        }
        
        const start = (bookingsPage - 1) * BOOK_PAGE_SIZE;
        const end = start + BOOK_PAGE_SIZE;
        const pageItems = filteredBookings.slice(start, end);

        pageItems.forEach(booking => {
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

        renderBookingsPagination();
        ensureExportButton();
    }

    function renderBookingsPagination() {
        const totalPages = Math.max(1, Math.ceil(filteredBookings.length / BOOK_PAGE_SIZE));
        let pager = document.getElementById('bookPager');
        if (!pager) {
            pager = document.createElement('div');
            pager.id = 'bookPager';
            pager.className = 'pager';
            const table = document.getElementById('bookingsTable');
            table.parentElement.appendChild(pager);
        }
        pager.innerHTML = '';
        const prev = document.createElement('button'); prev.textContent = 'Prev'; prev.className = 'pager-btn'; prev.disabled = bookingsPage <= 1;
        const next = document.createElement('button'); next.textContent = 'Next'; next.className = 'pager-btn'; next.disabled = bookingsPage >= totalPages;
        const info = document.createElement('span'); info.className = 'pager-info'; info.textContent = `Page ${bookingsPage} of ${totalPages}`;
        prev.onclick = () => { bookingsPage = Math.max(1, bookingsPage - 1); renderBookings(); };
        next.onclick = () => { bookingsPage = Math.min(totalPages, bookingsPage + 1); renderBookings(); };
        pager.appendChild(prev); pager.appendChild(info); pager.appendChild(next);
    }

    // Export CSV for current bookings view
    function ensureExportButton() {
        let btn = document.getElementById('exportBookingsBtn');
        if (!btn) {
            btn = document.createElement('button');
            btn.id = 'exportBookingsBtn';
            btn.className = 'fab';
            btn.title = 'Export current bookings to CSV';
            btn.textContent = '⇩ CSV';
            document.body.appendChild(btn);
            btn.addEventListener('click', exportBookingsCsv);
        }
    }

    function exportBookingsCsv() {
        const headers = ['Booking ID','User','Professional','Risk Level','Scheduled Time','Status'];
        const rows = filteredBookings.map(b => [
            b.booking_id,
            b.user_account || `IP: ${b.user_ip}`,
            `${b.first_name} ${b.last_name}`,
            b.risk_level,
            new Date(b.scheduled_datetime * 1000).toISOString(),
            b.booking_status
        ]);
        const csv = [headers, ...rows].map(r => r.map(x => `"${String(x).replace(/"/g,'""')}"`).join(',')).join('\n');
        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = `bookings_${Date.now()}.csv`; a.click();
        URL.revokeObjectURL(url);
        showToast('Bookings exported');
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

            // Update KPI header cards
            const activeBookingsKpi = activeBookings;
            const riskStats = await api('/monitor/risk-stats');
            document.getElementById('kpiActiveBookings')?.replaceChildren(document.createTextNode(activeBookingsKpi));
            document.getElementById('kpiCritical')?.replaceChildren(document.createTextNode(riskStats.risk_stats?.critical || 0));
            document.getElementById('kpiProfessionals')?.replaceChildren(document.createTextNode(profResponse.professionals?.length || 0));
            document.getElementById('kpiAssessments')?.replaceChildren(document.createTextNode(assessmentsToday));

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
        const passwordField = document.getElementById('password');
        const passwordRequired = document.getElementById('passwordRequired');
        const passwordHelp = document.getElementById('passwordHelp');
        
        // Clear any previous validation states
        clearFormValidation();
        
        if (professional) {
            title.textContent = 'Edit Professional';
            form.dataset.professionalId = professional.id;
            
            // Make password optional for edit mode
            passwordField.required = false;
            passwordRequired.textContent = '';
            passwordHelp.style.display = 'block';
        } else {
            title.textContent = 'Add New Professional';
            delete form.dataset.professionalId;
            
            // Make password required for create mode
            passwordField.required = true;
            passwordRequired.textContent = '*';
            passwordHelp.style.display = 'none';
        }
        
        // Use Bootstrap modal show method
        $(modal).modal('show');
        
        // Handle modal events properly
        $(modal).off('shown.bs.modal').on('shown.bs.modal', function() {
            // Reset form first
            form.reset();
            
            // Ensure all inputs are working
            ensureInputsWorking();
            
            if (professional) {
                // Populate form for edit mode
                populateForm(professional);
            } else {
                // Set default values for add mode
                document.getElementById('experience_years').value = '0';
                document.getElementById('consultation_fee').value = '';
            }
            
            // Re-setup form event listeners
            setupFormEventListeners();
            
            // Focus on first input
            setTimeout(() => {
                const firstInput = modal.querySelector('input[required]');
                if (firstInput) {
                    firstInput.focus();
                    firstInput.select();
                }
            }, 300);
        });
    }
    
    function populateForm(professional) {
        // Clear all fields first
        const form = document.getElementById('professionalForm');
        form.reset();
        
        // Populate text fields
        document.getElementById('username').value = professional.username || '';
        document.getElementById('first_name').value = professional.first_name || '';
        document.getElementById('last_name').value = professional.last_name || '';
        document.getElementById('email').value = professional.email || '';
        document.getElementById('phone').value = professional.phone || '';
        document.getElementById('specialization').value = professional.specialization || '';
        document.getElementById('experience_years').value = professional.experience_years || 0;
        document.getElementById('district').value = professional.district || '';
        document.getElementById('consultation_fee').value = professional.consultation_fee || '';
        document.getElementById('bio').value = professional.bio || '';
        
        // Clear and check expertise areas
        const expertiseCheckboxes = document.querySelectorAll('input[name="expertise"]');
        expertiseCheckboxes.forEach(checkbox => {
            checkbox.checked = false;
        });
        
        if (professional.expertise_areas && Array.isArray(professional.expertise_areas)) {
            professional.expertise_areas.forEach(area => {
                const checkbox = document.querySelector(`input[name="expertise"][value="${area}"]`);
                if (checkbox) {
                    checkbox.checked = true;
                }
            });
        }
        
        // Trigger validation
        validateForm();
    }
    
    function clearFormValidation() {
        const form = document.getElementById('professionalForm');
        const inputs = form.querySelectorAll('.form-control');
        inputs.forEach(input => {
            input.classList.remove('is-valid', 'is-invalid');
            // Ensure input is enabled and working
            input.disabled = false;
            input.readOnly = false;
        });
    }
    
    function ensureInputsWorking() {
        const form = document.getElementById('professionalForm');
        const inputs = form.querySelectorAll('input, select, textarea');
        
        inputs.forEach(input => {
            // Ensure all inputs are enabled
            input.disabled = false;
            input.readOnly = false;
            
            // Add click handler to ensure focus
            input.addEventListener('click', function() {
                this.focus();
            });
            
            // Add keydown handler to ensure typing works
            input.addEventListener('keydown', function(e) {
                // Allow all normal typing
                if (e.key.length === 1 || e.key === 'Backspace' || e.key === 'Delete' || e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
                    return true;
                }
            });
        });
    }
    
    function validateForm() {
        const form = document.getElementById('professionalForm');
        const requiredFields = form.querySelectorAll('[required]');
        let isValid = true;
        
        requiredFields.forEach(field => {
            const value = field.value.trim();
            if (!value) {
                field.classList.add('is-invalid');
                field.classList.remove('is-valid');
                isValid = false;
            } else {
                field.classList.remove('is-invalid');
                field.classList.add('is-valid');
            }
        });
        
        // Check if at least one expertise area is selected
        const expertiseCheckboxes = form.querySelectorAll('input[name="expertise"]');
        const expertiseSelected = Array.from(expertiseCheckboxes).some(cb => cb.checked);
        
        if (!expertiseSelected) {
            // Add visual indicator for expertise requirement
            const expertiseContainer = form.querySelector('label[for="expertise_areas"]');
            if (expertiseContainer) {
                expertiseContainer.style.color = '#dc3545';
            }
            isValid = false;
        } else {
            const expertiseContainer = form.querySelector('label[for="expertise_areas"]');
            if (expertiseContainer) {
                expertiseContainer.style.color = '';
            }
        }
        
        // Enable/disable submit button
        const submitBtn = form.querySelector('button[type="submit"]');
        if (submitBtn) {
            submitBtn.disabled = !isValid;
        }
        
        return isValid;
    }
    
    professionalForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Validate form before submission
        if (!validateForm()) {
            showMessage('Please fill in all required fields', 'error');
            return;
        }
        
        // Show loading state
        const submitBtn = professionalForm.querySelector('button[type="submit"]');
        const originalText = submitBtn.textContent;
        submitBtn.disabled = true;
        submitBtn.textContent = 'Saving...';
        
        try {
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
            
            // Remove empty password for updates
            if (professionalForm.dataset.professionalId && !professionalData.password) {
                delete professionalData.password;
            }
            
            const professionalId = professionalForm.dataset.professionalId;
            let response;
            
            if (professionalId) {
                // Update existing professional
                response = await api(`/admin/professionals/${professionalId}`, {
                    method: 'PUT',
                    body: JSON.stringify(professionalData)
                });
                
                if (response && response.success) {
                    showMessage('Professional updated successfully', 'success');
                    showToast('Professional updated', 'success');
                    closeModal();
                    loadProfessionals();
                } else {
                    showMessage(response?.error || 'Failed to update professional', 'error');
                }
            } else {
                // Create new professional
                response = await api('/admin/professionals', {
                    method: 'POST',
                    body: JSON.stringify(professionalData)
                });
                
                if (response && response.success) {
                    showMessage('Professional created successfully', 'success');
                    showToast('Professional created', 'success');
                    closeModal();
                    loadProfessionals();
                } else {
                    showMessage(response?.error || 'Failed to create professional', 'error');
                }
            }
        } catch (error) {
            console.error('Failed to save professional:', error);
            const errorMessage = professionalForm.dataset.professionalId ? 
                'Failed to update professional' : 'Failed to create professional';
            showMessage(errorMessage, 'error');
            showToast(errorMessage, 'error');
        } finally {
            // Reset button state
            submitBtn.disabled = false;
            submitBtn.textContent = originalText;
        }
    });
    
    // Modal Management
    function closeModal() {
        // Close all Bootstrap modals properly
        $('.modal').modal('hide');
        
        // Clear form validation states
        setTimeout(() => {
            clearFormValidation();
        }, 300);
    }
    
    // Enhanced modal event handlers
    document.querySelectorAll('.close').forEach(closeBtn => {
        closeBtn.addEventListener('click', (e) => {
            e.preventDefault();
            closeModal();
        });
    });
    
    // Handle cancel button
    document.querySelectorAll('[data-dismiss="modal"]').forEach(cancelBtn => {
        cancelBtn.addEventListener('click', (e) => {
            e.preventDefault();
            closeModal();
        });
    });
    
    // Handle modal backdrop clicks
    document.querySelectorAll('.modal').forEach(modal => {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                closeModal();
            }
        });
    });
    
    // Handle modal close on escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            const openModal = document.querySelector('.modal.show');
            if (openModal) {
                closeModal();
            }
        }
    });
    
    // Handle modal hidden event to reset form
    document.getElementById('professionalModal').addEventListener('hidden.bs.modal', function() {
        const form = document.getElementById('professionalForm');
        form.reset();
        clearFormValidation();
        delete form.dataset.professionalId;
    });
    
    // Add comprehensive form event listeners
    function setupFormEventListeners() {
        const form = document.getElementById('professionalForm');
        
        // Remove existing listeners to prevent duplicates
        form.removeEventListener('input', handleFormInput);
        form.removeEventListener('blur', handleFormBlur);
        form.removeEventListener('change', handleFormChange);
        
        // Add new listeners
        form.addEventListener('input', handleFormInput);
        form.addEventListener('blur', handleFormBlur, true);
        form.addEventListener('change', handleFormChange);
    }
    
    function handleFormInput(e) {
        // Real-time validation on input
        validateForm();
        
        // Ensure input is working
        if (e.target.type === 'text' || e.target.type === 'email' || e.target.type === 'tel' || e.target.type === 'password') {
            e.target.classList.remove('is-invalid');
            if (e.target.value.trim()) {
                e.target.classList.add('is-valid');
            }
        }
    }
    
    function handleFormBlur(e) {
        // Validation on blur for better UX
        if (e.target.classList.contains('form-control')) {
            validateForm();
        }
    }
    
    function handleFormChange(e) {
        // Handle expertise area changes
        if (e.target.name === 'expertise') {
            validateForm();
        }
        
        // Handle specialization changes
        if (e.target.name === 'specialization') {
            validateForm();
        }
    }
    
    // Initialize form event listeners
    setupFormEventListeners();
    
    // Debug function to check input functionality
    function debugInputs() {
        const form = document.getElementById('professionalForm');
        const inputs = form.querySelectorAll('input, select, textarea');
        
        console.log('Debugging form inputs:');
        inputs.forEach((input, index) => {
            console.log(`Input ${index}:`, {
                type: input.type,
                name: input.name,
                id: input.id,
                disabled: input.disabled,
                readOnly: input.readOnly,
                value: input.value,
                style: input.style.cssText
            });
        });
    }
    
    // Add global debug function
    window.debugFormInputs = debugInputs;
    
    // Event Listeners
    statusFilter.addEventListener('change', loadBookings);
    riskLevelFilter.addEventListener('change', loadBookings);
    // Professional search with debounce
    const professionalSearch = document.getElementById('professionalSearch');
    if (professionalSearch) {
        professionalSearch.addEventListener('input', debounce(() => {
            const q = professionalSearch.value.toLowerCase().trim();
            filteredProfessionals = professionals.filter(p => (
                `${p.first_name} ${p.last_name}`.toLowerCase().includes(q) ||
                (p.specialization || '').toLowerCase().includes(q) ||
                (p.email || '').toLowerCase().includes(q)
            ));
            professionalsPage = 1;
            renderProfessionals();
        }, 250));
    }
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
            const res = await api(`/admin/professionals/${id}/status`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ is_active: !currentStatus })
            });
            
            if (res && res.success) {
                showToast(`Professional ${!currentStatus ? 'activated' : 'deactivated'}`, 'success');
                await loadProfessionals();
            } else {
                showMessage(res?.error || 'Could not update status', 'error');
            }
        } catch (error) {
            console.error('Failed to toggle status:', error);
            showMessage('Failed to toggle status', 'error');
        }
    };
    
    window.deleteProfessional = async (id, name) => {
        if (!confirm(`Are you sure you want to delete "${name}"? This action cannot be undone.`)) {
            return;
        }
        
        try {
            const res = await api(`/admin/professionals/${id}`, {
                method: 'DELETE'
            });
            
            if (res && res.success) {
                showToast(`Professional "${name}" deleted successfully`, 'success');
                await loadProfessionals();
            } else {
                showMessage(res?.error || 'Failed to delete professional', 'error');
            }
        } catch (error) {
            console.error('Failed to delete professional:', error);
            const errorText = await error.text?.() || error.message || 'Failed to delete professional';
            showMessage(errorText, 'error');
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
    });
    
    function initializeDataTables() {
        // Enhanced table functionality with AdminLTE styling
        const tables = document.querySelectorAll('table');
        tables.forEach(table => {
            if (table.id === 'bookingsTable') {
                // Add DataTables to bookings table if jQuery and DataTables are available
                if (typeof $ !== 'undefined' && $.fn.DataTable) {
                    $(table).DataTable({
                        responsive: true,
                        pageLength: 25,
                        order: [[4, 'desc']], // Sort by scheduled time
                        columnDefs: [
                            { targets: [6], orderable: false } // Actions column
                        ],
                        language: {
                            search: "Search bookings:",
                            lengthMenu: "Show _MENU_ bookings per page",
                            info: "Showing _START_ to _END_ of _TOTAL_ bookings",
                            paginate: {
                                first: "First",
                                last: "Last",
                                next: "Next",
                                previous: "Previous"
                            }
                        }
                    });
                }
            }
        });
    }
    
    function initializeMobileMenu() {
        const mobileToggle = document.getElementById('mobileMenuToggle');
        const adminNav = document.querySelector('.admin-nav');
        
        if (mobileToggle && adminNav) {
            mobileToggle.addEventListener('click', function() {
                adminNav.classList.toggle('mobile-open');
            });
            
            // Close mobile menu when clicking outside
            document.addEventListener('click', function(e) {
                if (!adminNav.contains(e.target) && !mobileToggle.contains(e.target)) {
                    adminNav.classList.remove('mobile-open');
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
    }
    
    function initializeAnimations() {
        // Add fade-in animation to cards
        const cards = document.querySelectorAll('.kpi-card, .stat-card, .analytics-card, .rag-card');
        cards.forEach((card, index) => {
            card.classList.add('fade-in');
            card.style.animationDelay = `${index * 0.1}s`;
        });
        
        // Add slide-in animation to sections
        const sections = document.querySelectorAll('.admin-section');
        sections.forEach((section, index) => {
            section.classList.add('slide-in');
            section.style.animationDelay = `${index * 0.2}s`;
        });
    }
    
    // Enhanced notification system using AdminLTE toast
    function showAdminLTEMessage(text, type = 'error') {
        if (typeof $ !== 'undefined' && $.fn.toast) {
            // Use AdminLTE toast if available
            const toastHtml = `
                <div class="toast" role="alert" aria-live="assertive" aria-atomic="true">
                    <div class="toast-header">
                        <i class="fas fa-${type === 'success' ? 'check-circle' : 'exclamation-triangle'} mr-2"></i>
                        <strong class="mr-auto">AIMHSA Admin</strong>
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
                delay: 5000
            });
            $(toastElement).toast('show');
            
            // Remove toast after it's hidden
            $(toastElement).on('hidden.bs.toast', function() {
                $(this).remove();
            });
        } else {
            // Fallback to original message system
            showMessage(text, type);
        }
    }
    
    // Override the original showMessage function with AdminLTE version
    window.showMessage = showAdminLTEMessage;
    
    // Enhanced refresh functionality
    function refreshAllData() {
        const refreshBtn = document.getElementById('refreshAllBtn');
        if (refreshBtn) {
            refreshBtn.classList.add('loading');
            refreshBtn.disabled = true;
            
            // Refresh all data
            Promise.all([
                loadProfessionals(),
                loadBookings(),
                loadRiskStats(),
                loadRecentAssessments()
            ]).finally(() => {
                refreshBtn.classList.remove('loading');
                refreshBtn.disabled = false;
                showAdminLTEMessage('All data refreshed successfully', 'success');
            });
        }
    }
    
    // Add refresh button event listener
    document.addEventListener('DOMContentLoaded', function() {
        const refreshBtn = document.getElementById('refreshAllBtn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', refreshAllData);
        }
    });
    
})();
