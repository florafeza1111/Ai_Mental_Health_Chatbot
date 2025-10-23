/**
 * AIMHSA Professional Dashboard JavaScript
 * Simplified version without AdminLTE dependencies
 */

(() => {
    'use strict';

    // API Configuration
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

    // Global variables
    let currentSection = 'dashboard';
    let charts = {};
    let dataTables = {};
    let currentUser = null;

    // Initialize when DOM is ready
    $(document).ready(function() {
        console.log('Professional Dashboard Initializing...');
        
        // Initialize components (simplified)
        initializeNavigation();
        initializeDataTables();
        initializeCharts();
        initializeEventHandlers();
        
        // Get current user and load data
        getCurrentUser();
        
        // Start auto-refresh
        startAutoRefresh();
        
        // Force initial data load after a short delay
        setTimeout(() => {
            console.log('Force loading dashboard data...');
            loadDashboardData();
        }, 1000);
        
        // Add refresh button functionality
        $(document).on('click', '#refreshDashboardBtn', function() {
            console.log('Manual dashboard refresh triggered');
            loadDashboardData();
        });
    });

    /**
     * Initialize basic components (no AdminLTE dependencies)
     */
    function initializeBasicComponents() {
        // Initialize tooltips if Bootstrap is available
        if (typeof $ !== 'undefined' && $.fn.tooltip) {
        $('[data-toggle="tooltip"]').tooltip();
        }
        
        // Initialize popovers if Bootstrap is available
        if (typeof $ !== 'undefined' && $.fn.popover) {
        $('[data-toggle="popover"]').popover();
        }
    }

    /**
     * Initialize navigation system
     */
    function initializeNavigation() {
        // Handle sidebar navigation
        $('.nav-sidebar .nav-link').on('click', function(e) {
            e.preventDefault();
            
            const section = $(this).data('section');
            if (section) {
                showSection(section);
                updateActiveNavItem($(this));
                updateBreadcrumb(section);
            }
        });

        // Handle breadcrumb navigation
        $('.breadcrumb a').on('click', function(e) {
            e.preventDefault();
            const section = $(this).attr('href').substring(1);
            showSection(section);
        });
    }

    /**
     * Show specific section and hide others
     */
    function showSection(sectionName) {
        // Hide all sections
        $('.content-section').hide();
        
        // Show target section
        $(`#${sectionName}-section`).show();
        
        // Update current section
        currentSection = sectionName;
        
        // Load section-specific data
        loadSectionData(sectionName);
    }

    /**
     * Update active navigation item
     */
    function updateActiveNavItem(activeItem) {
        $('.nav-sidebar .nav-link').removeClass('active');
        activeItem.addClass('active');
    }

    /**
     * Update breadcrumb
     */
    function updateBreadcrumb(section) {
        const sectionNames = {
            'dashboard': 'Dashboard',
            'sessions': 'My Sessions',
            'notifications': 'Notifications',
            'reports': 'Reports',
            'profile': 'Profile',
            'settings': 'Settings'
        };
        
        $('#pageTitle').text(sectionNames[section] || 'Dashboard');
        $('#breadcrumbActive').text(sectionNames[section] || 'Dashboard');
    }

    /**
     * Initialize DataTables (if available)
     */
    function initializeDataTables() {
        // Sessions table
        if ($('#sessionsTable').length && typeof $ !== 'undefined' && $.fn.DataTable) {
            dataTables.sessions = $('#sessionsTable').DataTable({
                responsive: true,
                processing: true,
                serverSide: false,
                pageLength: 25,
                order: [[0, 'desc']],
                columnDefs: [
                    { targets: [-1], orderable: false }
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

    }

    /**
     * Initialize charts (if Chart.js is available)
     */
    function initializeCharts() {
        // Only initialize if Chart.js is available
        if (typeof Chart === 'undefined') {
            console.log('Chart.js not available, skipping chart initialization');
            return;
        }
        
        // Session trend chart
        if ($('#sessionTrendChart').length) {
            const ctx = document.getElementById('sessionTrendChart').getContext('2d');
            charts.sessionTrend = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                    datasets: [{
                        label: 'Completed Sessions',
                        data: [12, 15, 18, 14, 16, 19],
                        borderColor: '#28a745',
                        backgroundColor: 'rgba(40, 167, 69, 0.1)',
                        tension: 0.4
                    }, {
                        label: 'Scheduled Sessions',
                        data: [8, 12, 10, 15, 13, 17],
                        borderColor: '#007bff',
                        backgroundColor: 'rgba(0, 123, 255, 0.1)',
                        tension: 0.4
                    }, {
                        label: 'Cancelled Sessions',
                        data: [2, 3, 1, 4, 2, 3],
                        borderColor: '#dc3545',
                        backgroundColor: 'rgba(220, 53, 69, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
        }

        // Risk distribution chart
        if ($('#riskDistributionChart').length) {
            const ctx = document.getElementById('riskDistributionChart').getContext('2d');
            charts.riskDistribution = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: ['Low Risk', 'Medium Risk', 'High Risk', 'Critical'],
                    datasets: [{
                        data: [45, 35, 15, 5],
                        backgroundColor: ['#28a745', '#17a2b8', '#ffc107', '#dc3545'],
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom'
                        }
                    }
                }
            });
        }
    }

    /**
     * Initialize Select2 (if available)
     */
    function initializeSelect2() {
        if (typeof $ !== 'undefined' && $.fn.select2) {
        $('.select2').select2({
            theme: 'bootstrap4',
            width: '100%'
        });
        }
    }

    /**
     * Initialize event handlers
     */
    function initializeEventHandlers() {
        // Session management
        $('#addNewSessionBtn').on('click', function() {
            if (typeof $ !== 'undefined' && $.fn.modal) {
            $('#sessionNotesModal').modal('show');
            } else {
                // Fallback for when Bootstrap modal is not available
                document.getElementById('sessionNotesModal').style.display = 'block';
            }
        });

        $('#saveNotesBtn').on('click', function() {
            saveSessionNotes();
        });

        // Follow-up checkbox handler
        $('#followUpRequired').on('change', function() {
            if ($(this).is(':checked')) {
                $('#followUpDateGroup').show();
            } else {
                $('#followUpDateGroup').hide();
            }
        });

        // Session notes form handler
        $('#saveNotesBtn').on('click', function() {
            const bookingId = $('#sessionId').val();
            const notes = $('#sessionNotes').val();
            const followUpRequired = $('#followUpRequired').is(':checked');
            const followUpDate = $('#followUpDate').val();

            if (!notes.trim()) {
                if (typeof Swal !== 'undefined') {
                    Swal.fire('Error', 'Please enter session notes', 'error');
                } else {
                    alert('Please enter session notes');
                }
                return;
            }

            // For now, just show success message
            if (typeof Swal !== 'undefined') {
                Swal.fire('Success', 'Session notes saved successfully', 'success');
                $('#sessionNotesModal').modal('hide');
            } else {
                alert('Session notes saved successfully');
                document.getElementById('sessionNotesModal').style.display = 'none';
            }
            loadSessions(); // Refresh sessions
        });

        // Refresh buttons
        $('[id$="RefreshBtn"], [id$="refreshBtn"]').on('click', function() {
            const section = $(this).closest('.content-section').attr('id').replace('-section', '');
            loadSectionData(section);
        });


        // Filter functionality
        $('#sessionStatusFilter, #riskLevelFilter').on('change', function() {
            applyFilters();
        });

        // Logout
        $('#logoutBtn').on('click', function() {
            if (typeof Swal !== 'undefined') {
            Swal.fire({
                title: 'Logout?',
                text: 'Are you sure you want to logout?',
                icon: 'question',
                showCancelButton: true,
                confirmButtonColor: '#3085d6',
                cancelButtonColor: '#d33',
                confirmButtonText: 'Yes, logout!'
            }).then((result) => {
                if (result.isConfirmed) {
                        // Clear all session data
                        localStorage.removeItem('aimhsa_professional');
                        localStorage.removeItem('aimhsa_account');
                        localStorage.removeItem('aimhsa_admin');
                        localStorage.removeItem('currentUser');
                        
                        // Redirect to login page
                        window.location.href = '/login.html';
                    }
                });
            } else {
                // Fallback for when SweetAlert is not available
                if (confirm('Are you sure you want to logout?')) {
                    // Clear all session data
                    localStorage.removeItem('aimhsa_professional');
                    localStorage.removeItem('aimhsa_account');
                    localStorage.removeItem('aimhsa_admin');
                    localStorage.removeItem('currentUser');
                    
                    // Redirect to login page
                    window.location.href = '/login.html';
                }
            }
        });
    }

    /**
     * Get current user information
     */
    function getCurrentUser() {
        console.log('Getting current user...');
        
        // Check for professional login session
        const professionalSession = localStorage.getItem('aimhsa_professional');
        if (professionalSession) {
            try {
                currentUser = JSON.parse(professionalSession);
                console.log('Professional user authenticated:', currentUser.name);
                updateUserInfo();
                loadDashboardData();
                return;
            } catch (error) {
                console.warn('Invalid professional session:', error);
            }
        }
        
        // Check for other user types and redirect if needed
        const userSession = localStorage.getItem('aimhsa_account');
        const adminSession = localStorage.getItem('aimhsa_admin');
        
        if (userSession) {
            console.warn('User is logged in as regular user, redirecting to login');
            alert('You are logged in as a regular user. Please logout and login as a professional.');
            window.location.href = '/login.html';
            return;
        }
        
        if (adminSession) {
            console.warn('User is logged in as admin, redirecting to login');
            alert('You are logged in as an admin. Please logout and login as a professional.');
            window.location.href = '/login.html';
            return;
        }
        
        // No valid session found - create a test professional session for demo purposes
        console.log('No professional session found, creating test session for demo');
        createTestProfessionalSession();
    }
    
    /**
     * Create a test professional session for demo purposes
     */
    function createTestProfessionalSession() {
        // Create a test professional user
        currentUser = {
            professional_id: 18,
            name: 'Dashboard Test Professional',
            username: 'dashboard_test_prof',
            email: 'dashboard.test@example.com',
            fullname: 'Dashboard Test Professional',
            specialization: 'Psychologist'
        };
        
        // Store in localStorage
        localStorage.setItem('aimhsa_professional', JSON.stringify(currentUser));
        
        console.log('Test professional session created:', currentUser);
        updateUserInfo();
        loadDashboardData();
    }

    /**
     * Update user information display
     */
    function updateUserInfo() {
        if (currentUser) {
            console.log('Updating UI with user info:', currentUser);
            
            // Update user name in sidebar
            $('#professionalNameSidebar').text(currentUser.name || currentUser.fullname || 'Professional');
            $('#professionalName').text(currentUser.name || currentUser.fullname || 'Professional');
            
            // Update user role/specialization
            $('#professionalRole').text(currentUser.specialization || 'Mental Health Professional');
            
            // Update user email if available
            if (currentUser.email) {
                $('.user-panel .info span').text(currentUser.email);
            }
            
            // Update user profile image with initials
            updateUserProfileImage(currentUser.name || currentUser.fullname || 'Professional');
            
            console.log('User info updated:', currentUser.name);
        }
    }
    
    /**
     * Update user profile image with initials
     */
    function updateUserProfileImage(fullName) {
        if (!fullName) return;
        
        // Extract initials from full name
        const initials = fullName.split(' ')
            .map(name => name.charAt(0).toUpperCase())
            .join('')
            .substring(0, 2);
        
        // Create SVG with initials
        const svgData = `data:image/svg+xml;base64,${btoa(`
            <svg width="160" height="160" viewBox="0 0 160 160" fill="none" xmlns="http://www.w3.org/2000/svg">
                <rect width="160" height="160" rx="80" fill="#28a745"/>
                <text x="80" y="95" font-family="Arial, sans-serif" font-size="64" font-weight="bold" fill="white" text-anchor="middle">${initials}</text>
            </svg>
        `)}`;
        
        // Update the profile image
        $('#userProfileImage').attr('src', svgData);
        $('#profileUserImage').attr('src', svgData);
    }

    /**
     * Load dashboard data from database
     */
    function loadDashboardData() {
        console.log('🔄 Loading professional dashboard data...');
        
        if (!currentUser) {
            console.log('⚠️ No current user, waiting...');
            setTimeout(loadDashboardData, 1000);
            return;
        }
        
        console.log('✅ Current user found:', currentUser.name);
        
        // Update user info in UI
        updateUserInfo();
        
        // Load dashboard statistics from API
        loadDashboardStats();
        
        // Load today's schedule
        loadTodaySchedule();
        
        // Load recent notifications
        loadRecentNotifications();
        
        // Load dashboard charts
        loadDashboardCharts();
        
        console.log('✅ Dashboard data loading completed');
    }

    /**
     * Update user information in the UI
     */
    function updateUserInfo() {
        if (currentUser) {
            console.log(' Updating UI with user info:', currentUser);
            $('#professionalName').text(currentUser.fullname || currentUser.username);
            $('#professionalNameSidebar').text(currentUser.fullname || currentUser.username);
            $('#professionalRole').text(currentUser.specialization || 'Mental Health Professional');
            $('#profileName').text(currentUser.fullname || currentUser.username);
            $('#profileRole').text(currentUser.specialization || 'Mental Health Professional');
        }
    }

    /**
     * Load dashboard statistics from API
     */
    function loadDashboardStats() {
        console.log('Loading dashboard statistics...');
        
        if (!currentUser) {
            console.log('No current user for stats');
            return;
        }
        
        console.log('Fetching data from:', `${API_ROOT}/admin/bookings`);
        
        // Show loading state
        $('#totalSessions').html('<i class="fas fa-spinner fa-spin"></i>');
        $('#unreadNotifications').html('<i class="fas fa-spinner fa-spin"></i>');
        $('#upcomingToday').html('<i class="fas fa-spinner fa-spin"></i>');
        $('#highRiskSessions').html('<i class="fas fa-spinner fa-spin"></i>');
        
        // Get user's booking statistics
        fetch(`${API_ROOT}/admin/bookings`)
            .then(response => {
                console.log('API response status:', response.status);
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                console.log('Dashboard stats received:', data);
                
                // Calculate statistics
                const totalSessions = data.total || 0;
                const confirmedSessions = data.confirmed || 0;
                const pendingSessions = data.pending || 0;
                const criticalSessions = data.critical || 0;
                
                // Calculate today's sessions
                const today = new Date().toDateString();
                const todaySessions = data.bookings ? data.bookings.filter(booking => {
                    const bookingDate = new Date(booking.scheduled_datetime * 1000).toDateString();
                    return bookingDate === today;
                }).length : 0;
                
                console.log('Updating UI with:', { 
                    totalSessions, 
                    confirmedSessions, 
                    pendingSessions, 
                    criticalSessions, 
                    todaySessions 
                });
                
                // Update KPI cards with real data
                $('#totalSessions').text(totalSessions);
                $('#unreadNotifications').text(pendingSessions + criticalSessions); // Show urgent notifications
                $('#upcomingToday').text(todaySessions);
                $('#highRiskSessions').text(criticalSessions);
                
                // Update notification badge in navbar
                $('#notificationBadge').text(pendingSessions + criticalSessions);
                $('#notificationCount').text(pendingSessions + criticalSessions);
                
                // Update notification dropdown
                updateNotificationDropdown(data.bookings || []);
                
                console.log('Dashboard statistics updated successfully with real data');
            })
            .catch(error => {
                console.error('Error loading dashboard stats:', error);
                
                // Show error state
                $('#totalSessions').html('<i class="fas fa-exclamation-triangle text-warning"></i>');
                $('#unreadNotifications').html('<i class="fas fa-exclamation-triangle text-warning"></i>');
                $('#upcomingToday').html('<i class="fas fa-exclamation-triangle text-warning"></i>');
                $('#highRiskSessions').html('<i class="fas fa-exclamation-triangle text-warning"></i>');
                
                // Show error message
                if (typeof Swal !== 'undefined') {
                    Swal.fire({
                        title: 'Data Loading Error',
                        text: 'Unable to load dashboard data. Please check your connection and try again.',
                        icon: 'warning',
                        timer: 5000,
                        showConfirmButton: false
                    });
                } else {
                    console.error('Unable to load dashboard data. Please check your connection and try again.');
                }
            });
    }
    
    /**
     * Update notification dropdown with real data
     */
    function updateNotificationDropdown(bookings) {
        const notificationMenu = $('.dropdown-menu.dropdown-menu-lg');
        const urgentBookings = bookings.filter(booking => 
            booking.risk_level === 'critical' || booking.booking_status === 'pending'
        ).slice(0, 5); // Show top 5 urgent items
        
        let notificationHtml = `<span class="dropdown-item dropdown-header">${urgentBookings.length} Urgent Items</span>`;
        
        if (urgentBookings.length === 0) {
            notificationHtml += `
                <div class="dropdown-divider"></div>
                <a href="#" class="dropdown-item">
                    <i class="fas fa-check-circle mr-2 text-success"></i> All caught up!
                    <span class="float-right text-muted text-sm">No urgent items</span>
                </a>
            `;
        } else {
            urgentBookings.forEach(booking => {
                const timeAgo = getTimeAgo(booking.created_ts);
                const icon = booking.risk_level === 'critical' ? 'fa-exclamation-triangle' : 'fa-calendar';
                const color = booking.risk_level === 'critical' ? 'text-danger' : 'text-warning';
                
                notificationHtml += `
                    <div class="dropdown-divider"></div>
                    <a href="#" class="dropdown-item" onclick="viewBookingDetails('${booking.booking_id}')">
                        <i class="fas ${icon} mr-2 ${color}"></i> 
                        ${booking.user_fullname || 'Patient'} - ${booking.risk_level || 'New booking'}
                        <span class="float-right text-muted text-sm">${timeAgo}</span>
                    </a>
                `;
            });
        }
        
        notificationHtml += `
            <div class="dropdown-divider"></div>
            <a href="#notifications" class="dropdown-item dropdown-footer" data-section="notifications">See All Notifications</a>
        `;
        
        notificationMenu.html(notificationHtml);
    }
    
    /**
     * Get time ago string from timestamp
     */
    function getTimeAgo(timestamp) {
        if (!timestamp) return 'Unknown';
        
        const now = Date.now() / 1000;
        const diff = now - timestamp;
        
        if (diff < 60) return 'Just now';
        if (diff < 3600) return `${Math.floor(diff / 60)} mins ago`;
        if (diff < 86400) return `${Math.floor(diff / 3600)} hours ago`;
        return `${Math.floor(diff / 86400)} days ago`;
    }

    /**
     * View booking details modal
     */
    window.viewBookingDetails = function(bookingId) {
        console.log('Viewing booking details for:', bookingId);
        
        // Find the booking in the current data
        fetch(`${API_ROOT}/admin/bookings`)
            .then(response => response.json())
            .then(data => {
                const booking = data.bookings.find(b => b.booking_id === bookingId);
                if (booking) {
                    showBookingDetailsModal(booking);
                } else {
                    Swal.fire('Error', 'Booking not found', 'error');
                }
            })
            .catch(error => {
                console.error('Error fetching booking details:', error);
                Swal.fire('Error', 'Failed to load booking details', 'error');
            });
    };
    
    /**
     * Show booking details modal
     */
    function showBookingDetailsModal(booking) {
        const scheduledTime = new Date(booking.scheduled_datetime * 1000).toLocaleString();
        const riskBadgeClass = getRiskBadgeClass(booking.risk_level);
        const statusBadgeClass = getStatusBadgeClass(booking.booking_status);
        
        const modalHtml = `
            <div class="modal fade" id="bookingDetailsModal" tabindex="-1" role="dialog">
                <div class="modal-dialog modal-lg" role="document">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">
                                <i class="fas fa-calendar-check text-primary"></i>
                                Booking Details
                            </h5>
                            <button type="button" class="close" data-dismiss="modal">
                                <span>&times;</span>
                            </button>
                        </div>
                        <div class="modal-body">
                            <div class="row">
                                <div class="col-md-6">
                                    <h6><i class="fas fa-user text-info"></i> Patient Information</h6>
                                    <p><strong>Name:</strong> ${booking.user_fullname || 'Unknown'}</p>
                                    <p><strong>Email:</strong> ${booking.user_email || 'N/A'}</p>
                                    <p><strong>Phone:</strong> ${booking.user_phone || 'N/A'}</p>
                                    <p><strong>Location:</strong> ${booking.user_location || 'N/A'}</p>
                                </div>
                                <div class="col-md-6">
                                    <h6><i class="fas fa-calendar text-success"></i> Session Details</h6>
                                    <p><strong>Scheduled:</strong> ${scheduledTime}</p>
                                    <p><strong>Status:</strong> <span class="badge badge-${statusBadgeClass}">${booking.booking_status}</span></p>
                                    <p><strong>Risk Level:</strong> <span class="badge badge-${riskBadgeClass}">${booking.risk_level}</span></p>
                                    <p><strong>Session Type:</strong> ${booking.session_type || 'Routine'}</p>
                                </div>
                            </div>
                            ${booking.detected_indicators && booking.detected_indicators.length > 0 ? `
                                <div class="row mt-3">
                                    <div class="col-12">
                                        <h6><i class="fas fa-exclamation-triangle text-warning"></i> Detected Indicators</h6>
                                        <div class="alert alert-warning">
                                            ${booking.detected_indicators.map(indicator => `<span class="badge badge-warning mr-1">${indicator}</span>`).join('')}
                                        </div>
                                    </div>
                                </div>
                            ` : ''}
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-dismiss="modal">Close</button>
                            <button type="button" class="btn btn-primary" onclick="startSession('${booking.booking_id}')">
                                <i class="fas fa-play"></i> Start Session
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Remove existing modal if any
        $('#bookingDetailsModal').remove();
        
        // Add modal to body
        $('body').append(modalHtml);
        
        // Show modal
        if (typeof $ !== 'undefined' && $.fn.modal) {
            $('#bookingDetailsModal').modal('show');
        } else {
            // Fallback for when Bootstrap modal is not available
            document.getElementById('bookingDetailsModal').style.display = 'block';
        }
    };
    
    /**
     * Get risk badge class
     */
    function getRiskBadgeClass(riskLevel) {
        switch (riskLevel) {
            case 'critical': return 'danger';
            case 'high': return 'warning';
            case 'medium': return 'info';
            case 'low': return 'success';
            default: return 'secondary';
        }
    }
    
    /**
     * Get status badge class
     */
    function getStatusBadgeClass(status) {
        switch (status) {
            case 'confirmed': return 'success';
            case 'pending': return 'warning';
            case 'completed': return 'info';
            case 'cancelled': return 'danger';
            default: return 'secondary';
        }
    }
    
    /**
     * Start session function
     */
    window.startSession = function(bookingId) {
        if (typeof Swal !== 'undefined') {
            Swal.fire({
                title: 'Start Session',
                text: 'Are you ready to start this session?',
                icon: 'question',
                showCancelButton: true,
                confirmButtonText: 'Yes, start session',
                cancelButtonText: 'Cancel'
            }).then((result) => {
                if (result.isConfirmed) {
                    // Here you would implement the actual session start logic
                    Swal.fire('Session Started', 'The session has been started successfully!', 'success');
                    if (typeof $ !== 'undefined' && $.fn.modal) {
                        $('#bookingDetailsModal').modal('hide');
                    } else {
                        document.getElementById('bookingDetailsModal').style.display = 'none';
                    }
                    
                    // Refresh the dashboard data
                    loadDashboardData();
                }
            });
        } else {
            // Fallback for when SweetAlert is not available
            if (confirm('Are you ready to start this session?')) {
                alert('Session Started! The session has been started successfully!');
                if (typeof $ !== 'undefined' && $.fn.modal) {
                    $('#bookingDetailsModal').modal('hide');
                } else {
                    document.getElementById('bookingDetailsModal').style.display = 'none';
                }
                
                // Refresh the dashboard data
                loadDashboardData();
            }
        }
    };
    
    /**
     * View session function
     */
    window.viewSession = function(bookingId) {
        console.log('Viewing session:', bookingId);
        viewBookingDetails(bookingId);
    };
    
    /**
     * Add session notes function
     */
    window.addSessionNotes = function(bookingId) {
        if (typeof Swal !== 'undefined') {
            Swal.fire({
                title: 'Add Session Notes',
                input: 'textarea',
                inputLabel: 'Session Notes',
                inputPlaceholder: 'Enter your session notes here...',
                inputAttributes: {
                    'aria-label': 'Type your session notes here'
                },
                showCancelButton: true,
                confirmButtonText: 'Save Notes',
                cancelButtonText: 'Cancel'
            }).then((result) => {
                if (result.isConfirmed) {
                    // Here you would save the notes to the database
                    Swal.fire('Notes Saved', 'Session notes have been saved successfully!', 'success');
                    console.log('Session notes for', bookingId, ':', result.value);
                }
            });
        } else {
            // Fallback for when SweetAlert is not available
            const notes = prompt('Enter your session notes here:');
            if (notes !== null && notes.trim() !== '') {
                alert('Notes Saved! Session notes have been saved successfully!');
                console.log('Session notes for', bookingId, ':', notes);
            }
        }
    };
    
    /**
     * Complete session function
     */
    window.completeSession = function(bookingId) {
        if (typeof Swal !== 'undefined') {
            Swal.fire({
                title: 'Complete Session',
                text: 'Mark this session as completed?',
                icon: 'question',
                showCancelButton: true,
                confirmButtonText: 'Yes, complete',
                cancelButtonText: 'Cancel'
            }).then((result) => {
                if (result.isConfirmed) {
                    // Here you would update the session status in the database
                    Swal.fire('Session Completed', 'The session has been marked as completed!', 'success');
                    console.log('Session completed:', bookingId);
                    
                    // Refresh the dashboard data
                    loadDashboardData();
                }
            });
        } else {
            // Fallback for when SweetAlert is not available
            if (confirm('Mark this session as completed?')) {
                alert('Session Completed! The session has been marked as completed!');
                console.log('Session completed:', bookingId);
                
                // Refresh the dashboard data
                loadDashboardData();
            }
        }
    };

    /**
     * Load section-specific data
     */
    function loadSectionData(section) {
        switch (section) {
            case 'sessions':
                loadSessions();
                break;
            case 'notifications':
                loadNotifications();
                break;
            case 'reports':
                loadReports();
                break;
            case 'profile':
                loadProfile();
                break;
            case 'settings':
                loadSettings();
                break;
            case 'dashboard':
                loadDashboardCharts();
                break;
        }
    }
    
    /**
     * Load dashboard charts
     */
    function loadDashboardCharts() {
        console.log('Loading dashboard charts...');
        
        // Load session trends chart
        loadSessionTrendsChart();
        
        // Load risk distribution chart
        loadRiskDistributionChart();
    }
    
    /**
     * Load session trends chart
     */
    function loadSessionTrendsChart() {
        // Only proceed if Chart.js is available
        if (typeof Chart === 'undefined') {
            console.log('Chart.js not available, skipping session trends chart');
            return;
        }
        
        fetch(`${API_ROOT}/admin/bookings`)
            .then(response => response.json())
            .then(data => {
                const bookings = data.bookings || [];
                
                // Group bookings by date for the last 7 days
                const last7Days = [];
                for (let i = 6; i >= 0; i--) {
                    const date = new Date();
                    date.setDate(date.getDate() - i);
                    last7Days.push(date.toISOString().split('T')[0]);
                }
                
                const sessionData = last7Days.map(date => {
                    return bookings.filter(booking => {
                        const bookingDate = new Date(booking.scheduled_datetime * 1000).toISOString().split('T')[0];
                        return bookingDate === date;
                    }).length;
                });
                
                // Create or update chart
                const ctx = document.getElementById('sessionTrendsChart');
                if (ctx) {
                    if (charts.sessionTrends) {
                        charts.sessionTrends.destroy();
                    }
                    
                    charts.sessionTrends = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: last7Days.map(date => new Date(date).toLocaleDateString('en-US', { weekday: 'short' })),
                            datasets: [{
                                label: 'Sessions',
                                data: sessionData,
                                borderColor: 'rgb(75, 192, 192)',
                                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                                tension: 0.1
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                y: {
                                    beginAtZero: true,
                                    ticks: {
                                        stepSize: 1
                                    }
                                }
                            }
                        }
                    });
                }
            })
            .catch(error => {
                console.error('Error loading session trends chart:', error);
            });
    }
    
    /**
     * Load risk distribution chart
     */
    function loadRiskDistributionChart() {
        // Only proceed if Chart.js is available
        if (typeof Chart === 'undefined') {
            console.log('Chart.js not available, skipping risk distribution chart');
            return;
        }
        
        fetch(`${API_ROOT}/admin/bookings`)
            .then(response => response.json())
            .then(data => {
                const bookings = data.bookings || [];
                
                // Count bookings by risk level
                const riskCounts = {
                    low: 0,
                    medium: 0,
                    high: 0,
                    critical: 0
                };
                
                bookings.forEach(booking => {
                    const riskLevel = booking.risk_level || 'low';
                    if (riskCounts.hasOwnProperty(riskLevel)) {
                        riskCounts[riskLevel]++;
                    }
                });
                
                // Create or update chart
                const ctx = document.getElementById('riskDistributionChart');
                if (ctx) {
                    if (charts.riskDistribution) {
                        charts.riskDistribution.destroy();
                    }
                    
                    charts.riskDistribution = new Chart(ctx, {
                        type: 'doughnut',
                        data: {
                            labels: ['Low Risk', 'Medium Risk', 'High Risk', 'Critical'],
                            datasets: [{
                                data: [riskCounts.low, riskCounts.medium, riskCounts.high, riskCounts.critical],
                                backgroundColor: [
                                    'rgba(40, 167, 69, 0.8)',
                                    'rgba(23, 162, 184, 0.8)',
                                    'rgba(255, 193, 7, 0.8)',
                                    'rgba(220, 53, 69, 0.8)'
                                ],
                                borderColor: [
                                    'rgba(40, 167, 69, 1)',
                                    'rgba(23, 162, 184, 1)',
                                    'rgba(255, 193, 7, 1)',
                                    'rgba(220, 53, 69, 1)'
                                ],
                                borderWidth: 2
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {
                                legend: {
                                    position: 'bottom'
                                }
                            }
                        }
                    });
                }
            })
            .catch(error => {
                console.error('Error loading risk distribution chart:', error);
            });
    }

    /**
     * Load KPI data
     */
    function loadKPIData() {
        // Simulate API call
        setTimeout(() => {
            $('#totalSessions').text('15');
            $('#unreadNotifications').text('8');
            $('#upcomingToday').text('3');
            $('#highRiskSessions').text('2');
        }, 500);
    }

    /**
     * Load today's schedule
     */
    function loadTodaySchedule() {
        console.log('Loading today\'s schedule...');
        
        // Load real data from bookings API
        fetch(`${API_ROOT}/admin/bookings`)
            .then(response => response.json())
            .then(data => {
                console.log('Schedule data received:', data);

        const tbody = $('#todayScheduleTable');
        tbody.empty();

                if (data.bookings && data.bookings.length > 0) {
                    // Show first 3 bookings as today's schedule
                    data.bookings.slice(0, 3).forEach(booking => {
                        const time = new Date(booking.scheduled_datetime * 1000).toLocaleTimeString('en-US', { 
                            hour: '2-digit', 
                            minute: '2-digit' 
                        });
                        const patientName = booking.user_fullname || booking.user_account || 'Unknown Patient';
                        const sessionType = booking.session_type || 'Routine';
                        const status = booking.booking_status;
                        
            const row = `
                <tr>
                                <td>${time}</td>
                                <td>${patientName}</td>
                                <td>${sessionType}</td>
                                <td><span class="badge badge-${getStatusBadgeClass(status)}">${status}</span></td>
                </tr>
            `;
            tbody.append(row);
                    });
                } else {
                    tbody.html('<tr><td colspan="4" class="text-center text-muted">No sessions scheduled for today</td></tr>');
                }
                
                console.log('Today\'s schedule loaded');
            })
            .catch(error => {
                console.error('Error loading schedule:', error);
                const tbody = $('#todayScheduleTable');
                tbody.html('<tr><td colspan="4" class="text-center text-danger">Error loading schedule</td></tr>');
        });
    }

    /**
     * Load recent notifications
     */
    function loadRecentNotifications() {
        console.log('Loading recent notifications...');
        
        // For now, show sample notifications
        const notifications = [
            {
                id: 1,
                title: 'New Session Booking',
                message: 'A new session has been booked for tomorrow at 10:00 AM',
                time: new Date().toLocaleString(),
                type: 'info',
                read: false
            },
            {
                id: 2,
                title: 'High Risk Alert',
                message: 'A patient has been flagged as high risk and needs immediate attention',
                time: new Date(Date.now() - 3600000).toLocaleString(),
                type: 'warning',
                read: false
            },
            {
                id: 3,
                title: 'System Update',
                message: 'The system has been updated with new features',
                time: new Date(Date.now() - 7200000).toLocaleString(),
                type: 'success',
                read: true
            }
        ];
        
        console.log('Recent notifications loaded');
    }

    /**
     * Load sessions data
     */
    function loadSessions() {
        console.log('Loading sessions...');

        const tbody = $('#sessionsTableBody');
        tbody.html('<tr><td colspan="7" class="text-center"><i class="fas fa-spinner fa-spin"></i> Loading sessions...</td></tr>');
        
        console.log('Fetching from:', `${API_ROOT}/admin/bookings`);
        
        fetch(`${API_ROOT}/admin/bookings`)
            .then(response => {
                console.log('API response status:', response.status);
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                console.log('Sessions data received:', data);
                console.log('Bookings count:', data.bookings ? data.bookings.length : 0);
        tbody.empty();

                if (data.bookings && data.bookings.length > 0) {
                    console.log('Processing bookings...');
                    data.bookings.forEach((booking, index) => {
                        console.log(`Processing booking ${index + 1}:`, booking.user_fullname || booking.user_account);
                        
                        const scheduledTime = new Date(booking.scheduled_datetime * 1000).toLocaleString();
                        const patientName = booking.user_fullname || booking.user_account || 'Unknown Patient';
                        
            const row = `
                <tr>
                                <td>${booking.booking_id.substring(0, 8)}...</td>
                                <td>
                                    <div class="user-info">
                                        <strong>${patientName}</strong>
                                        <br><small class="text-muted">${booking.user_email || 'No email'}</small>
                                    </div>
                                </td>
                                <td>${scheduledTime}</td>
                                <td>${booking.session_type || 'Routine'}</td>
                                <td><span class="badge badge-${getRiskBadgeClass(booking.risk_level)}">${booking.risk_level}</span></td>
                                <td><span class="badge badge-${getStatusBadgeClass(booking.booking_status)}">${booking.booking_status}</span></td>
                                <td>
                                    <button class="btn btn-sm btn-info" onclick="viewSession('${booking.booking_id}')" title="View Details">
                            <i class="fas fa-eye"></i>
                        </button>
                                    <button class="btn btn-sm btn-warning" onclick="addSessionNotes('${booking.booking_id}')" title="Add Notes">
                                        <i class="fas fa-sticky-note"></i>
                        </button>
                    </td>
                </tr>
            `;
            tbody.append(row);
        });
                    console.log('All bookings processed successfully');
                } else {
                    console.log('No bookings found in data');
                    tbody.html(`
                        <tr>
                            <td colspan="7" class="text-center text-muted">
                                <i class="fas fa-calendar-times"></i>
                                <br>No sessions found
                            </td>
                        </tr>
                    `);
                }
                
                // Update DataTable if available
                if (dataTables.sessions && typeof $ !== 'undefined' && $.fn.DataTable) {
                    console.log('Updating DataTable...');
            dataTables.sessions.clear().rows.add($(tbody).find('tr')).draw();
        }
                
                console.log('Sessions loaded successfully');
            })
            .catch(error => {
                console.error('Error loading sessions:', error);
                tbody.html(`
                    <tr>
                        <td colspan="7" class="text-center text-danger">
                            <i class="fas fa-exclamation-triangle"></i>
                            <br>Error loading sessions: ${error.message}
                        </td>
                    </tr>
                `);
            });
    }


    /**
     * Load notifications
     */
    function loadNotifications() {
        console.log('🔔 Loading notifications...');
        
        const container = $('#notificationsList');
        container.html('<div class="text-center"><i class="fas fa-spinner fa-spin"></i> Loading notifications...</div>');
        
        // For now, show sample notifications
        const notifications = [
            {
                id: 1,
                title: 'New Session Booking',
                message: 'A new session has been booked for tomorrow at 10:00 AM',
                time: new Date().toLocaleString(),
                type: 'info',
                read: false
            },
            {
                id: 2,
                title: 'High Risk Alert',
                message: 'A patient has been flagged as high risk and needs immediate attention',
                time: new Date(Date.now() - 3600000).toLocaleString(),
                type: 'warning',
                read: false
            },
            {
                id: 3,
                title: 'System Update',
                message: 'The system has been updated with new features',
                time: new Date(Date.now() - 7200000).toLocaleString(),
                type: 'success',
                read: true
            }
        ];

        container.empty();

        if (notifications.length > 0) {
        notifications.forEach(notification => {
            const notificationHtml = `
                    <div class="alert alert-${notification.type} alert-dismissible ${notification.read ? '' : 'alert-unread'}" data-notification-id="${notification.id}">
                        <button type="button" class="close" data-dismiss="alert" aria-hidden="true" onclick="markNotificationRead(${notification.id})">&times;</button>
                    <h5><i class="icon fas fa-${getNotificationIcon(notification.type)}"></i> ${notification.title}</h5>
                    <p>${notification.message}</p>
                    <small><i class="fas fa-clock"></i> ${notification.time}</small>
                        ${!notification.read ? `
                            <button class="btn btn-sm btn-outline-primary float-right" onclick="markNotificationRead(${notification.id})">
                                Mark as Read
                            </button>
                        ` : ''}
                </div>
            `;
            container.append(notificationHtml);
        });
        } else {
            container.html(`
                <div class="text-center text-muted">
                    <i class="fas fa-bell-slash"></i>
                    <br>No notifications found
                </div>
            `);
        }
        
        console.log(' Notifications loaded successfully');
    }

    /**
     * Load reports
     */
    function loadReports() {
        // Reports are already set in HTML
        // This function can be used to load dynamic report data
    }

    /**
     * Load profile
     */
    function loadProfile() {
        // Profile data is already set in HTML
        // This function can be used to load dynamic profile data
    }

    /**
     * Load settings
     */
    function loadSettings() {
        // Settings are already set in HTML
        // This function can be used to load dynamic settings
    }

    /**
     * Save session notes
     */
    function saveSessionNotes() {
        const formData = {
            sessionId: $('#sessionId').val(),
            patientName: $('#patientName').val(),
            sessionNotes: $('#sessionNotes').val(),
            followUpRequired: $('#followUpRequired').is(':checked'),
            followUpDate: $('#followUpDate').val()
        };

        // Simulate API call
        if (typeof Swal !== 'undefined') {
        Swal.fire({
            title: 'Success!',
            text: 'Session notes saved successfully.',
            icon: 'success',
            timer: 2000
        }).then(() => {
                if (typeof $ !== 'undefined' && $.fn.modal) {
            $('#sessionNotesModal').modal('hide');
                } else {
                    document.getElementById('sessionNotesModal').style.display = 'none';
                }
            loadSessions();
        });
        } else {
            alert('Session notes saved successfully.');
            if (typeof $ !== 'undefined' && $.fn.modal) {
                $('#sessionNotesModal').modal('hide');
            } else {
                document.getElementById('sessionNotesModal').style.display = 'none';
            }
            loadSessions();
        }
    }

    /**
     * Apply filters
     */
    function applyFilters() {
        const status = $('#sessionStatusFilter').val();
        const riskLevel = $('#riskLevelFilter').val();

        // Apply filters to DataTables if available
        if (dataTables.sessions && typeof $ !== 'undefined' && $.fn.DataTable) {
            dataTables.sessions.column(5).search(status).draw();
        }
    }

    /**
     * Get risk badge class
     */
    function getRiskBadgeClass(riskLevel) {
        const classes = {
            'critical': 'danger',
            'high': 'warning',
            'medium': 'info',
            'low': 'success'
        };
        return classes[riskLevel.toLowerCase()] || 'secondary';
    }

    /**
     * Get status badge class
     */
    function getStatusBadgeClass(status) {
        const classes = {
            'scheduled': 'info',
            'completed': 'success',
            'cancelled': 'danger',
            'pending': 'warning',
            'confirmed': 'success',
            'high risk': 'danger'
        };
        return classes[status.toLowerCase()] || 'secondary';
    }

    /**
     * Get notification icon
     */
    function getNotificationIcon(type) {
        const icons = {
            'info': 'info-circle',
            'warning': 'exclamation-triangle',
            'success': 'check-circle',
            'danger': 'times-circle'
        };
        return icons[type] || 'info-circle';
    }

    /**
     * Start auto-refresh
     */
    function startAutoRefresh() {
        setInterval(() => {
            if (currentSection === 'dashboard') {
                loadDashboardData();
            } else {
                loadSectionData(currentSection);
            }
        }, 30000); // Refresh every 30 seconds
    }

    /**
     * Session management functions
     */
    window.acceptSession = function(bookingId) {
        console.log(' Accepting session:', bookingId);
        
        if (typeof Swal !== 'undefined') {
            Swal.fire({
                title: 'Session Accepted!',
                text: 'The session has been accepted successfully.',
                icon: 'success',
                timer: 2000
            });
        } else {
            alert('Session Accepted! The session has been accepted successfully.');
        }
        loadSessions(); // Refresh sessions
    };

    window.declineSession = function(bookingId) {
        console.log(' Declining session:', bookingId);
        
        if (typeof Swal !== 'undefined') {
            Swal.fire({
                title: 'Decline Session?',
                text: 'Are you sure you want to decline this session?',
                icon: 'warning',
                showCancelButton: true,
                confirmButtonText: 'Yes, decline',
                cancelButtonText: 'Cancel'
            }).then((result) => {
                if (result.isConfirmed) {
                    Swal.fire('Session Declined', 'The session has been declined.', 'success');
                    loadSessions(); // Refresh sessions
                }
            });
        } else {
            if (confirm('Are you sure you want to decline this session?')) {
                alert('Session Declined! The session has been declined.');
                loadSessions(); // Refresh sessions
            }
        }
    };

    window.addSessionNotes = function(bookingId) {
        console.log('📝 Adding notes for session:', bookingId);
        
        // Populate modal with session details
        $('#sessionId').val(bookingId);
        $('#patientName').val('Patient Name');
        $('#sessionNotes').val('');
        $('#followUpRequired').prop('checked', false);
        $('#followUpDateGroup').hide();
        
        // Show modal
        $('#sessionNotesModal').modal('show');
    };

    window.markNotificationRead = function(notificationId) {
        console.log('📖 Marking notification as read:', notificationId);
        
        // Remove the notification from the UI
        $(`[data-notification-id="${notificationId}"]`).fadeOut();
        loadDashboardStats(); // Refresh dashboard stats
    };

    /**
     * Global functions for onclick handlers
     */
    window.addNewSession = function() {
        $('#sessionNotesModal').modal('show');
    };

    window.refreshSessions = function() {
        loadSessions();
        Swal.fire('Refreshed!', 'Sessions data has been refreshed.', 'success');
    };

    window.markAllAsRead = function() {
        Swal.fire('Success!', 'All notifications marked as read.', 'success');
        $('#notificationBadge').text('0');
        $('#notificationCount').text('0');
    };

    window.viewSession = function(id) {
        Swal.fire('View Session', `View session with ID: ${id}`, 'info');
    };

    window.editSession = function(id) {
        Swal.fire('Edit Session', `Edit session with ID: ${id}`, 'info');
    };


    // Show dashboard by default
    showSection('dashboard');

})();



