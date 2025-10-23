/**
 * AIMHSA Advanced Admin Dashboard JavaScript
 * Enhanced functionality with AdminLTE 4, DataTables, Charts, and more
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
    let currentProfessionalId = null;
    let currentUser = null;
    let userRole = 'admin'; // Default role

    // Initialize when DOM is ready
    $(document).ready(function() {
        console.log('Admin Dashboard Initializing...');
        
        try {
            // Check if jQuery is loaded
            if (typeof $ === 'undefined') {
                console.error('jQuery not loaded');
                showErrorMessage('jQuery library not loaded. Please refresh the page.');
                return;
            }
            
            console.log('jQuery loaded');
            
            // Check user authentication and role
            checkUserAuthentication();
            
            // Initialize components
        initializeAdminLTE();
        initializeNavigation();
        initializeDataTables();
        initializeCharts();
        initializeSelect2();
        initializeExpertiseAreas();
        initializeEventHandlers();
            
            // Load dashboard data with error handling
        loadDashboardData();
            
            // Start auto-refresh
        startAutoRefresh();
            
            console.log(' Admin Dashboard initialized successfully');
            
            // Show success message
            setTimeout(() => {
                if (typeof Swal !== 'undefined') {
                    Swal.fire({
                        title: 'Dashboard Ready!',
                        text: `Welcome ${currentUser?.username || 'Admin'}! Dashboard loaded successfully.`,
                        icon: 'success',
                        timer: 2000,
                        toast: true,
                        position: 'top-end'
                    });
                }
            }, 1000);
            
        } catch (error) {
            console.error(' Error initializing admin dashboard:', error);
            showErrorMessage('Dashboard initialization failed: ' + error.message);
        }
    });
    
    /**
     * Check user authentication and role
     */
    function checkUserAuthentication() {
        console.log('🔐 Checking user authentication...');
        
        // Check localStorage for user session
        const adminSession = localStorage.getItem('aimhsa_admin');
        const professionalSession = localStorage.getItem('aimhsa_professional');
        const userSession = localStorage.getItem('aimhsa_account');
        
        if (adminSession) {
            try {
                currentUser = JSON.parse(adminSession);
                userRole = 'admin';
                console.log(' Admin user authenticated:', currentUser.username);
                updateUserInterface();
            } catch (error) {
                console.warn(' Invalid admin session, using default');
                setDefaultUser();
            }
        } else if (professionalSession) {
            try {
                currentUser = JSON.parse(professionalSession);
                userRole = 'professional';
                console.log(' Professional user authenticated:', currentUser.username);
                updateUserInterface();
            } catch (error) {
                console.warn(' Invalid professional session, using default');
                setDefaultUser();
            }
        } else if (userSession) {
            try {
                currentUser = JSON.parse(userSession);
                userRole = 'user';
                console.log(' Regular user authenticated:', currentUser.username);
                updateUserInterface();
            } catch (error) {
                console.warn(' Invalid user session, using default');
                setDefaultUser();
            }
        } else {
            console.warn(' No user session found, using default admin');
            setDefaultUser();
        }
    }
    
    /**
     * Set default user when no session is found
     */
    function setDefaultUser() {
        currentUser = {
            username: 'admin',
            email: 'admin@aimhsa.rw',
            fullname: 'System Administrator',
            role: 'admin'
        };
        userRole = 'admin';
        updateUserInterface();
    }
    
    /**
     * Update user interface based on current user
     */
    function updateUserInterface() {
        console.log('👤 Updating user interface for:', currentUser.username, 'Role:', userRole);
        
        // Update sidebar user info
        $('.user-panel .info a').text(currentUser.fullname || currentUser.username);
        $('.user-panel .info small').text(getRoleDisplayName(userRole));
        
        // Update navbar user info
        $('.navbar-nav .nav-item:last-child .nav-link span').text(currentUser.fullname || currentUser.username);
        
        // Update page title based on role
        if (userRole === 'professional') {
            $('#pageTitle').text('Professional Dashboard');
            $('.brand-text').text('AIMHSA Professional');
        } else if (userRole === 'user') {
            $('#pageTitle').text('User Dashboard');
            $('.brand-text').text('AIMHSA User');
        } else {
            $('#pageTitle').text('Admin Dashboard');
            $('.brand-text').text('AIMHSA Admin');
        }
        
        // Show/hide sections based on role
        updateNavigationForRole();
    }
    
    /**
     * Get display name for user role
     */
    function getRoleDisplayName(role) {
        const roleNames = {
            'admin': 'System Administrator',
            'professional': 'Mental Health Professional',
            'user': 'User Account'
        };
        return roleNames[role] || 'User';
    }
    
    /**
     * Update navigation based on user role
     */
    function updateNavigationForRole() {
        if (userRole === 'professional') {
            // Hide admin-only sections
            $('.nav-item[data-section="professionals"]').hide();
            $('.nav-item[data-section="reports"]').hide();
            $('.nav-item[data-section="settings"]').hide();
            
            // Show professional-specific sections
            $('.nav-item[data-section="bookings"]').show();
            $('.nav-item[data-section="risk-monitor"]').show();
            $('.nav-item[data-section="analytics"]').show();
        } else if (userRole === 'user') {
            // Hide admin and professional sections
            $('.nav-item[data-section="professionals"]').hide();
            $('.nav-item[data-section="reports"]').hide();
            $('.nav-item[data-section="settings"]').hide();
            $('.nav-item[data-section="bookings"]').hide();
            
            // Show user-specific sections
            $('.nav-item[data-section="risk-monitor"]').show();
            $('.nav-item[data-section="analytics"]').show();
        } else {
            // Admin - show all sections
            $('.nav-item').show();
        }
    }
    
    /**
     * Show error message to user
     */
    function showErrorMessage(message) {
        const errorHtml = `
            <div class="alert alert-danger alert-dismissible fade show" role="alert">
                <strong>Dashboard Error:</strong> ${message}
                <button type="button" class="close" data-dismiss="alert">
                    <span>&times;</span>
                </button>
            </div>
        `;
        $('.content-wrapper').prepend(errorHtml);
    }

    /**
     * Initialize AdminLTE components
     */
    function initializeAdminLTE() {
        console.log('🔧 Initializing AdminLTE components...');
        
        try {
            // Initialize push menu with fallback
            if (typeof $.fn.PushMenu !== 'undefined') {
        $('[data-widget="pushmenu"]').PushMenu('toggle');
            } else {
                // Fallback for push menu
                $('[data-widget="pushmenu"]').on('click', function(e) {
                    e.preventDefault();
                    $('body').toggleClass('sidebar-collapse');
                });
            }
        
        // Initialize tooltips
            if (typeof $.fn.tooltip !== 'undefined') {
        $('[data-toggle="tooltip"]').tooltip();
            }
        
        // Initialize popovers
            if (typeof $.fn.popover !== 'undefined') {
        $('[data-toggle="popover"]').popover();
            }
        
            // Initialize card widgets with fallback
            if (typeof $.fn.cardWidget !== 'undefined') {
        $('.card').cardWidget();
            } else {
                // Fallback for card widgets
                $('[data-card-widget="collapse"]').on('click', function(e) {
                    e.preventDefault();
                    const card = $(this).closest('.card');
                    card.toggleClass('collapsed-card');
                });
                
                $('[data-card-widget="remove"]').on('click', function(e) {
                    e.preventDefault();
                    const card = $(this).closest('.card');
                    card.fadeOut(300, function() {
                        $(this).remove();
                    });
                });
            }
            
            // Initialize direct chat with fallback
            if (typeof $.fn.DirectChat !== 'undefined') {
        $('[data-widget="chat-pane-toggle"]').DirectChat('toggle');
            } else {
                $('[data-widget="chat-pane-toggle"]').on('click', function(e) {
                    e.preventDefault();
                    $(this).closest('.direct-chat').toggleClass('direct-chat-contacts-open');
                });
            }
            
            console.log(' AdminLTE components initialized');
            
        } catch (error) {
            console.warn(' Some AdminLTE components failed to initialize:', error);
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
        // Hide all sections with fade effect
        $('.content-section').fadeOut(200, function() {
            // Show target section
            $(`#${sectionName}-section`).fadeIn(200);
        });
        
        // Update current section
        console.log('🔄 Switching to section:', sectionName);
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
            'professionals': 'Professionals',
            'bookings': 'Bookings',
            'risk-monitor': 'Risk Monitor',
            'analytics': 'Analytics',
            'rag-status': 'RAG Status',
            'reports': 'Reports',
            'settings': 'Settings'
        };
        
        $('#pageTitle').text(sectionNames[section] || 'Dashboard');
        $('#breadcrumbActive').text(sectionNames[section] || 'Dashboard');
    }

    /**
     * Initialize DataTables
     */
    function initializeDataTables() {
        // Professionals table
        if ($('#professionalsTable').length) {
            dataTables.professionals = $('#professionalsTable').DataTable({
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

        // Bookings table
        if ($('#bookingsTable').length) {
            dataTables.bookings = $('#bookingsTable').DataTable({
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
     * Initialize charts
     */
    function initializeCharts() {
        // Risk trend chart - will be updated with real data
        if ($('#riskTrendChart').length) {
            const ctx = document.getElementById('riskTrendChart').getContext('2d');
            charts.riskTrend = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Critical',
                        data: [],
                        borderColor: '#dc3545',
                        backgroundColor: 'rgba(220, 53, 69, 0.1)',
                        tension: 0.4
                    }, {
                        label: 'High',
                        data: [],
                        borderColor: '#ffc107',
                        backgroundColor: 'rgba(255, 193, 7, 0.1)',
                        tension: 0.4
                    }, {
                        label: 'Medium',
                        data: [],
                        borderColor: '#17a2b8',
                        backgroundColor: 'rgba(23, 162, 184, 0.1)',
                        tension: 0.4
                    }, {
                        label: 'Low',
                        data: [],
                        borderColor: '#28a745',
                        backgroundColor: 'rgba(40, 167, 69, 0.1)',
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
            
            // Load real data for risk trend chart after a short delay
            setTimeout(() => {
                loadRiskTrendData();
            }, 100);
        }

        // Risk distribution chart - will be updated with real data
        if ($('#riskDistributionChart').length) {
            const ctx = document.getElementById('riskDistributionChart').getContext('2d');
            charts.riskDistribution = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: ['Critical', 'High', 'Medium', 'Low'],
                    datasets: [{
                        data: [0, 0, 0, 0],
                        backgroundColor: ['#dc3545', '#ffc107', '#17a2b8', '#28a745'],
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
            
            // Load real data for risk distribution chart after a short delay
            setTimeout(() => {
                loadRiskDistributionData();
            }, 100);
        }

        // Monthly trends chart
        if ($('#monthlyTrendsChart').length) {
            const ctx = document.getElementById('monthlyTrendsChart').getContext('2d');
            charts.monthlyTrends = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                    datasets: [{
                        label: 'Bookings',
                        data: [45, 52, 38, 61, 55, 67],
                        backgroundColor: '#007bff'
                    }, {
                        label: 'Completed',
                        data: [42, 48, 35, 58, 52, 63],
                        backgroundColor: '#28a745'
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

        // Professional performance chart
        if ($('#professionalPerformanceChart').length) {
            const ctx = document.getElementById('professionalPerformanceChart').getContext('2d');
            charts.professionalPerformance = new Chart(ctx, {
                type: 'radar',
                data: {
                    labels: ['Sessions', 'Satisfaction', 'Response Time', 'Availability', 'Quality'],
                    datasets: [{
                        label: 'Dr. Marie',
                        data: [85, 92, 88, 90, 87],
                        borderColor: '#007bff',
                        backgroundColor: 'rgba(0, 123, 255, 0.2)'
                    }, {
                        label: 'Dr. John',
                        data: [78, 85, 82, 88, 84],
                        borderColor: '#28a745',
                        backgroundColor: 'rgba(40, 167, 69, 0.2)'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        r: {
                            beginAtZero: true,
                            max: 100
                        }
                    }
                }
            });
        }
    }

    /**
     * Load risk trend data for chart
     */
    function loadRiskTrendData() {
        if (!charts.riskTrend) {
            console.warn('Risk trend chart not initialized yet');
            return;
        }
        
        fetch(`${API_ROOT}/admin/risk-assessments?limit=100`)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                if (data.assessments && charts.riskTrend) {
                    // Group assessments by day for the last 7 days
                    const last7Days = [];
                    for (let i = 6; i >= 0; i--) {
                        const date = new Date();
                        date.setDate(date.getDate() - i);
                        last7Days.push(date.toISOString().split('T')[0]);
                    }
                    
                    const riskCounts = {
                        critical: new Array(7).fill(0),
                        high: new Array(7).fill(0),
                        medium: new Array(7).fill(0),
                        low: new Array(7).fill(0)
                    };
                    
                    data.assessments.forEach(assessment => {
                        const assessmentDate = new Date(assessment.assessment_timestamp * 1000).toISOString().split('T')[0];
                        const dayIndex = last7Days.indexOf(assessmentDate);
                        if (dayIndex !== -1) {
                            const riskLevel = assessment.risk_level.toLowerCase();
                            if (riskCounts[riskLevel]) {
                                riskCounts[riskLevel][dayIndex]++;
                            }
                        }
                    });
                    
                    // Update chart data
                    charts.riskTrend.data.labels = last7Days.map(date => {
                        const d = new Date(date);
                        return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
                    });
                    charts.riskTrend.data.datasets[0].data = riskCounts.critical;
                    charts.riskTrend.data.datasets[1].data = riskCounts.high;
                    charts.riskTrend.data.datasets[2].data = riskCounts.medium;
                    charts.riskTrend.data.datasets[3].data = riskCounts.low;
                    charts.riskTrend.update();
                }
            })
            .catch(error => {
                console.error('Error loading risk trend data:', error);
            });
    }

    /**
     * Load risk distribution data for chart
     */
    function loadRiskDistributionData() {
        if (!charts.riskDistribution) {
            console.warn('Risk distribution chart not initialized yet');
            return;
        }
        
        fetch(`${API_ROOT}/admin/risk-assessments?limit=100`)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                if (data.assessments && charts.riskDistribution) {
                    const riskCounts = {
                        critical: 0,
                        high: 0,
                        medium: 0,
                        low: 0
                    };
                    
                    data.assessments.forEach(assessment => {
                        const riskLevel = assessment.risk_level.toLowerCase();
                        if (riskCounts.hasOwnProperty(riskLevel)) {
                            riskCounts[riskLevel]++;
                        }
                    });
                    
                    // Update chart data
                    charts.riskDistribution.data.datasets[0].data = [
                        riskCounts.critical,
                        riskCounts.high,
                        riskCounts.medium,
                        riskCounts.low
                    ];
                    charts.riskDistribution.update();
                }
            })
            .catch(error => {
                console.error('Error loading risk distribution data:', error);
            });
    }

    /**
     * Initialize Select2
     */
    function initializeSelect2() {
        $('.select2').select2({
            theme: 'bootstrap4',
            width: '100%'
        });
    }

    /**
     * Initialize event handlers
     */
    function initializeEventHandlers() {
        // Professional management
        $('#addProfessionalBtn').on('click', function() {
            console.log('➕ Opening Add Professional modal...');
            resetProfessionalForm();
            $('#modalTitle').text('Add New Professional');
            $('#passwordRequired').text('*');
            $('#passwordHelp').hide();
            $('#professionalModal').modal('show');
            
            // Ensure inputs work properly
            setTimeout(() => {
                ensureInputsWorking();
                forceInputFunctionality();
                debugFormInputs();
                const firstInput = $('#professionalModal input[required]').first();
                if (firstInput.length) {
                    firstInput.focus();
                    console.log(' Focused on first input:', firstInput.attr('name'));
                }
            }, 300);
        });

        // Handle form submission
        $('#professionalForm').on('submit', function(e) {
            e.preventDefault();
            saveProfessional();
        });
        
        // Also handle button click as backup
        $('#saveProfessionalBtn').on('click', function() {
            $('#professionalForm').submit();
        });

        // Refresh buttons
        $('[id$="RefreshBtn"], [id$="refreshBtn"]').on('click', function() {
            const section = $(this).closest('.content-section').attr('id').replace('-section', '');
            loadSectionData(section);
        });

        // Global refresh button
        $('#refreshAllBtn').on('click', function() {
            refreshAllData();
        });

        // Expertise areas change handler
        $(document).on('change', 'input[name="expertise"]', function() {
            updateExpertiseValidation();
        });

        // Export buttons
        $('#exportBookingsBtn').on('click', function() {
            exportTableToCSV('bookingsTable', 'bookings.csv');
        });
        
        // Initialize bookings filtering
        initializeBookingsFiltering();

        // Search functionality
        $('#professionalSearch').on('keyup', function() {
            if (dataTables.professionals) {
                dataTables.professionals.search(this.value).draw();
            }
        });

        // Filter functionality
        $('#statusFilter, #riskLevelFilter, #specializationFilter').on('change', function() {
            applyFilters();
        });

        // Professional search functionality
        $('#professionalSearch').on('input', function() {
            const searchTerm = $(this).val().toLowerCase();
            filterProfessionals(searchTerm);
        });

        // Professional specialization filter
        $('#professionalSpecializationFilter').on('change', function() {
            const specialization = $(this).val();
            filterProfessionalsBySpecialization(specialization);
        });

        // Logout
        $('#logoutBtn').on('click', function() {
            Swal.fire({
                title: 'Logout?',
                text: `Are you sure you want to logout, ${currentUser?.username || 'User'}?`,
                icon: 'question',
                showCancelButton: true,
                confirmButtonColor: '#3085d6',
                cancelButtonColor: '#d33',
                confirmButtonText: 'Yes, logout!'
            }).then((result) => {
                if (result.isConfirmed) {
                    logoutUser();
                }
            });
        });
    }

    /**
     * Check API health
     */
    function checkAPIHealth() {
        const endpoints = [
            '/admin/bookings',
            '/admin/professionals', 
            '/admin/risk-assessments',
            '/monitor/risk-stats',
            '/monitor/recent-assessments'
        ];
        
        Promise.all(endpoints.map(endpoint => 
            fetch(`${API_ROOT}${endpoint}`)
                .then(response => ({ endpoint, status: response.ok, statusCode: response.status }))
                .catch(error => ({ endpoint, status: false, error: error.message }))
        )).then(results => {
            const failedEndpoints = results.filter(r => !r.status);
            if (failedEndpoints.length > 0) {
                console.warn('Some API endpoints are not responding:', failedEndpoints);
                // Show a warning to the user
                if (failedEndpoints.length === endpoints.length) {
                    Swal.fire({
                        title: 'API Connection Error',
                        text: 'Unable to connect to the backend API. Please check if the server is running.',
                        icon: 'error',
                        timer: 5000
                    });
                }
            }
        });
    }

    /**
     * Load dashboard data
     */
    function loadDashboardData() {
        console.log(' Loading dashboard data for role:', userRole);
        
        // Show loading state
        showLoadingState();
        
        // Check API health first
        checkAPIHealth();
        
        // Load role-specific data
        if (userRole === 'admin') {
            loadAdminDashboardData();
        } else if (userRole === 'professional') {
            loadProfessionalDashboardData();
        } else if (userRole === 'user') {
            loadUserDashboardData();
        } else {
            loadAdminDashboardData(); // Default to admin
        }
        
        // Hide loading state after a delay
        setTimeout(() => {
            hideLoadingState();
        }, 2000);
    }
    
    /**
     * Load admin dashboard data
     */
    function loadAdminDashboardData() {
        console.log('👑 Loading admin dashboard data...');
        loadKPIData();
        loadRecentBookings();
        loadSystemStatus();
        loadNotifications();
    }
    
    /**
     * Load professional dashboard data
     */
    function loadProfessionalDashboardData() {
        console.log(' Loading professional dashboard data...');
        loadProfessionalKPIData();
        loadProfessionalBookings();
        loadProfessionalNotifications();
        loadNotifications();
    }
    
    /**
     * Load user dashboard data
     */
    function loadUserDashboardData() {
        console.log('👤 Loading user dashboard data...');
        loadUserKPIData();
        loadUserRiskHistory();
        loadUserBookings();
        loadNotifications();
    }
    
    /**
     * Show loading state
     */
    function showLoadingState() {
        $('#loadingOverlay').show();
        // Add loading class to KPI cards
        $('.small-box .inner h3').html('<i class="fas fa-spinner fa-spin"></i>');
    }
    
    /**
     * Hide loading state
     */
    function hideLoadingState() {
        $('#loadingOverlay').hide();
    }

    /**
     * Load section-specific data
     */
    function loadSectionData(section) {
        console.log('🔄 Loading section data for:', section);
        try {
            switch (section) {
                case 'professionals':
                    console.log('📋 Loading professionals...');
                    loadProfessionals();
                    break;
                case 'bookings':
                    console.log('📋 Loading bookings...');
                    loadBookings();
                    break;
                case 'risk-monitor':
                    loadRiskData();
                    break;
                case 'analytics':
                    loadAnalyticsData();
                    break;
                case 'rag-status':
                    loadRAGStatus();
                    break;
                default:
                    console.warn(`Unknown section: ${section}`);
            }
        } catch (error) {
            console.error(`Error loading section data for ${section}:`, error);
        }
    }

    /**
     * Load KPI data from database
     */
    function loadKPIData() {
        console.log(' Loading KPI data...');
        
        // Show loading state
        $('#kpiActiveBookings, #kpiCritical, #kpiProfessionals, #kpiAssessments').html('<i class="fas fa-spinner fa-spin"></i>');
        
        // Try to load data from API endpoints with timeout
        const timeout = 5000; // 5 second timeout
        
        const apiPromises = [
            fetch(`${API_ROOT}/admin/bookings`, { signal: AbortSignal.timeout(timeout) })
                .then(res => res.ok ? res.json() : null)
                .catch(err => {
                    console.warn('Bookings API failed:', err);
                    return null;
                }),
            fetch(`${API_ROOT}/admin/professionals`, { signal: AbortSignal.timeout(timeout) })
                .then(res => res.ok ? res.json() : null)
                .catch(err => {
                    console.warn('Professionals API failed:', err);
                    return null;
                }),
            fetch(`${API_ROOT}/admin/risk-assessments?limit=100`, { signal: AbortSignal.timeout(timeout) })
                .then(res => res.ok ? res.json() : null)
                .catch(err => {
                    console.warn('Risk assessments API failed:', err);
                    return null;
                }),
            fetch(`${API_ROOT}/monitor/risk-stats`, { signal: AbortSignal.timeout(timeout) })
                .then(res => res.ok ? res.json() : null)
                .catch(err => {
                    console.warn('Risk stats API failed:', err);
                    return null;
                })
        ];
        
        Promise.all(apiPromises).then(([bookingsData, professionalsData, riskData, riskStats]) => {
            console.log(' API Data received:', { bookingsData, professionalsData, riskData, riskStats });
            
            let hasRealData = false;
            
            // Active bookings (pending + confirmed)
            const activeBookings = bookingsData?.bookings ? 
                bookingsData.bookings.filter(b => ['pending', 'confirmed'].includes(b.booking_status)).length : 0;
            $('#kpiActiveBookings').text(activeBookings);
            if (activeBookings > 0) hasRealData = true;
            
            // Critical risks
            const criticalRisks = riskStats?.critical || 
                (riskData?.assessments ? riskData.assessments.filter(r => r.risk_level === 'critical').length : 0);
            $('#kpiCritical').text(criticalRisks);
            if (criticalRisks > 0) hasRealData = true;
            
            // Total professionals
            const totalProfessionals = professionalsData?.professionals ? professionalsData.professionals.length : 0;
            $('#kpiProfessionals').text(totalProfessionals);
            if (totalProfessionals > 0) hasRealData = true;
            
            // Assessments today
            const today = new Date().toISOString().split('T')[0];
            const assessmentsToday = riskData?.assessments ? 
                riskData.assessments.filter(r => {
                    const assessmentDate = new Date(r.assessment_timestamp * 1000).toISOString().split('T')[0];
                    return assessmentDate === today;
                }).length : 0;
            $('#kpiAssessments').text(assessmentsToday);
            if (assessmentsToday > 0) hasRealData = true;
            
            if (!hasRealData) {
                console.log(' No real data found, showing demo data');
                showDemoData();
            } else {
                console.log(' KPI data loaded successfully');
            }
            
        }).catch(error => {
            console.warn(' API not available, using demo data:', error);
            showDemoData();
        });
    }
    
    /**
     * Show demo data when API is not available
     */
    function showDemoData() {
        // Show demo data when API is not available
        $('#kpiActiveBookings').html('<span class="text-success" title="Demo data">12</span>');
        $('#kpiCritical').html('<span class="text-danger" title="Demo data">3</span>');
        $('#kpiProfessionals').html('<span class="text-info" title="Demo data">8</span>');
        $('#kpiAssessments').text('25');
        
        // Show demo mode notification
        if (typeof Swal !== 'undefined') {
            Swal.fire({
                title: 'Demo Mode',
                text: 'Backend API not available. Showing demo data.',
                icon: 'info',
                timer: 3000,
                toast: true,
                position: 'top-end'
            });
        }
    }

    /**
     * Load recent bookings from database
     */
    function loadRecentBookings() {
        console.log('📋 Loading recent bookings...');
        
        const tbody = $('#recentBookingsTable');
        tbody.html('<tr><td colspan="4" class="text-center"><i class="fas fa-spinner fa-spin"></i> Loading...</td></tr>');
        
        // Try to load real data with timeout
        fetch(`${API_ROOT}/admin/bookings?limit=5`, { signal: AbortSignal.timeout(5000) })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                return response.json();
            })
            .then(data => {
                console.log('📋 Bookings data received:', data);
        tbody.empty();

                if (data.bookings && data.bookings.length > 0) {
                    data.bookings.forEach(booking => {
            const row = `
                <tr>
                                <td>${booking.booking_id || 'N/A'}</td>
                                <td>${booking.user_account || 'Guest'}</td>
                                <td><span class="badge badge-${getRiskBadgeClass(booking.risk_level)}">${booking.risk_level || 'Unknown'}</span></td>
                                <td><span class="badge badge-${getStatusBadgeClass(booking.booking_status)}">${booking.booking_status || 'Unknown'}</span></td>
                </tr>
            `;
            tbody.append(row);
                    });
                    console.log(' Recent bookings loaded successfully');
                } else {
                    showDemoBookings();
                }
            })
            .catch(error => {
                console.warn(' Error loading recent bookings, showing demo data:', error);
                showDemoBookings();
            });
    }
    
    /**
     * Show demo bookings data
     */
    function showDemoBookings() {
        const tbody = $('#recentBookingsTable');
        tbody.html(`
            <tr>
                <td>BK001</td>
                <td>Demo User</td>
                <td><span class="badge badge-danger">Critical</span></td>
                <td><span class="badge badge-warning">Pending</span></td>
            </tr>
            <tr>
                <td>BK002</td>
                <td>Test User</td>
                <td><span class="badge badge-warning">High</span></td>
                <td><span class="badge badge-info">Confirmed</span></td>
            </tr>
            <tr>
                <td>BK003</td>
                <td>Sample User</td>
                <td><span class="badge badge-info">Medium</span></td>
                <td><span class="badge badge-success">Completed</span></td>
            </tr>
            <tr>
                <td colspan="4" class="text-center text-info">
                    <i class="fas fa-info-circle"></i> Demo data - API not available
                </td>
            </tr>
        `);
    }
    
    /**
     * Load professional KPI data
     */
    function loadProfessionalKPIData() {
        console.log(' Loading professional KPI data...');
        
        // Update KPI labels for professional
        $('#kpiActiveBookings').parent().find('p').text('My Sessions');
        $('#kpiCritical').parent().find('p').text('High Risk Cases');
        $('#kpiProfessionals').parent().find('p').text('Total Patients');
        $('#kpiAssessments').parent().find('p').text('Today\'s Assessments');
        
        // Try to load professional-specific data
        fetch(`${API_ROOT}/professional/dashboard-stats`, { signal: AbortSignal.timeout(5000) })
            .then(response => response.json())
            .then(data => {
                console.log(' Professional data received:', data);
                
                $('#kpiActiveBookings').text(data.totalSessions || 0);
                $('#kpiCritical').text(data.highRiskCases || 0);
                $('#kpiProfessionals').text(data.activeUsers || 0);
                $('#kpiAssessments').text(data.unreadNotifications || 0);
                
                console.log(' Professional KPI data loaded successfully');
            })
            .catch(error => {
                console.warn(' Professional API not available, using demo data:', error);
                showProfessionalDemoData();
            });
    }
    
    /**
     * Show professional demo data
     */
    function showProfessionalDemoData() {
        $('#kpiActiveBookings').html('<span class="text-success" title="Demo data">15</span>');
        $('#kpiCritical').html('<span class="text-danger" title="Demo data">2</span>');
        $('#kpiProfessionals').html('<span class="text-info" title="Demo data">8</span>');
        $('#kpiAssessments').text('5');
    }
    
    /**
     * Load user KPI data
     */
    function loadUserKPIData() {
        console.log('👤 Loading user KPI data...');
        
        // Update KPI labels for user
        $('#kpiActiveBookings').parent().find('p').text('My Bookings');
        $('#kpiCritical').parent().find('p').text('Risk Level');
        $('#kpiProfessionals').parent().find('p').text('Sessions Completed');
        $('#kpiAssessments').parent().find('p').text('Assessments Done');
        
        // Show user-specific demo data
        $('#kpiActiveBookings').html('<span class="text-info" title="Demo data">3</span>');
        $('#kpiCritical').html('<span class="text-warning" title="Demo data">Medium</span>');
        $('#kpiProfessionals').html('<span class="text-success" title="Demo data">2</span>');
        $('#kpiAssessments').text('12');
    }
    
    /**
     * Load professional bookings
     */
    function loadProfessionalBookings() {
        console.log(' Loading professional bookings...');
        // This would load bookings specific to the professional
        loadRecentBookings(); // Use existing function for now
    }
    
    /**
     * Load professional notifications
     */
    function loadProfessionalNotifications() {
        console.log(' Loading professional notifications...');
        // This would load notifications for the professional
        // For now, just show demo data
    }
    
    /**
     * Load user risk history
     */
    function loadUserRiskHistory() {
        console.log('👤 Loading user risk history...');
        // This would load the user's risk assessment history
    }
    
    /**
     * Load user bookings
     */
    function loadUserBookings() {
        console.log('👤 Loading user bookings...');
        // This would load bookings specific to the user
        loadRecentBookings(); // Use existing function for now
    }
    
    /**
     * Logout user
     */
    function logoutUser() {
        console.log('🚪 Logging out user:', currentUser?.username);
        
        // Clear all session data
        localStorage.removeItem('aimhsa_admin');
        localStorage.removeItem('aimhsa_professional');
        localStorage.removeItem('aimhsa_account');
        
        // Show logout message
        if (typeof Swal !== 'undefined') {
            Swal.fire({
                title: 'Logged Out',
                text: 'You have been successfully logged out.',
                icon: 'success',
                timer: 2000
            }).then(() => {
                // Redirect to login page
                window.location.href = 'login.html';
            });
        } else {
            // Fallback redirect
            window.location.href = 'login.html';
        }
    }

    /**
     * Load system status
     */
    function loadSystemStatus() {
        // System status is already set in HTML
        // This function can be used to update real-time status
    }
    
    /**
     * Load notifications from database
     */
    function loadNotifications() {
        console.log('🔔 Loading notifications from database...');
        
        fetch(`${API_ROOT}/notifications`, { signal: AbortSignal.timeout(5000) })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                return response.json();
            })
            .then(data => {
                console.log('🔔 Notifications data received:', data);
                updateNotificationUI(data);
            })
            .catch(error => {
                console.warn(' Error loading notifications, using demo data:', error);
                showDemoNotifications();
            });
    }
    
    /**
     * Update notification UI with real data
     */
    function updateNotificationUI(data) {
        // Update notification badge
        const totalNotifications = data.totalNotifications || 0;
        $('.nav-link .badge').text(totalNotifications);
        
        // Update notification dropdown content
        const notificationsHtml = generateNotificationsHTML(data);
        $('.dropdown-menu.notifications-menu').html(notificationsHtml);
        
        console.log(' Notifications UI updated with real data');
    }
    
    /**
     * Generate notifications HTML from database data
     */
    function generateNotificationsHTML(data) {
        const notifications = data.notifications || [];
        const totalNotifications = data.totalNotifications || 0;
        
        let html = `
            <span class="dropdown-item dropdown-header">${totalNotifications} Notifications</span>
            <div class="dropdown-divider"></div>
        `;
        
        if (notifications.length === 0) {
            html += `
                <a href="#" class="dropdown-item">
                    <i class="fas fa-info-circle text-info mr-2"></i>
                    No new notifications
                </a>
            `;
        } else {
            notifications.slice(0, 5).forEach(notification => {
                const iconClass = getNotificationIcon(notification.type);
                const timeAgo = notification.timeAgo || 'Unknown';
                const isRead = notification.isRead ? '' : 'font-weight-bold';
                
                html += `
                    <a href="#" class="dropdown-item ${isRead}">
                        <i class="${iconClass} mr-2"></i>
                        ${notification.title}
                        <span class="float-right text-muted text-sm">${timeAgo}</span>
                    </a>
                `;
            });
        }
        
        html += `
            <div class="dropdown-divider"></div>
            <a href="#" class="dropdown-item dropdown-footer">See All Notifications</a>
        `;
        
        return html;
    }
    
    /**
     * Get icon class for notification type
     */
    function getNotificationIcon(type) {
        const iconMap = {
            'booking': 'fas fa-calendar-check text-success',
            'risk': 'fas fa-exclamation-triangle text-danger',
            'message': 'fas fa-envelope text-info',
            'user': 'fas fa-user text-primary',
            'system': 'fas fa-cog text-secondary',
            'default': 'fas fa-bell text-warning'
        };
        return iconMap[type] || iconMap['default'];
    }
    
    /**
     * Show demo notifications when API is not available
     */
    function showDemoNotifications() {
        const demoData = {
            totalNotifications: 15,
            notifications: [
                {
                    id: 1,
                    title: '4 new messages',
                    type: 'message',
                    timeAgo: '3 mins',
                    isRead: false
                },
                {
                    id: 2,
                    title: '8 friend requests',
                    type: 'user',
                    timeAgo: '12 hours',
                    isRead: false
                },
                {
                    id: 3,
                    title: '3 new reports',
                    type: 'system',
                    timeAgo: '2 days',
                    isRead: true
                }
            ]
        };
        
        updateNotificationUI(demoData);
        console.log('📱 Demo notifications displayed');
    }

    /**
     * Load professionals data from database
     */
    function loadProfessionals() {
        console.log('👥 Loading professionals...');
        const tbody = $('#professionalsTableBody');
        tbody.html('<tr><td colspan="8" class="text-center"><i class="fas fa-spinner fa-spin"></i> Loading...</td></tr>');
        
        fetch(`${API_ROOT}/admin/professionals`)
            .then(response => {
                console.log('📡 Professionals API response status:', response.status);
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                return response.json();
            })
            .then(data => {
                console.log(' Professionals data received:', data);
        tbody.empty();

                if (data.professionals && data.professionals.length > 0) {
                    data.professionals.forEach(prof => {
                        const fullName = `${prof.first_name || ''} ${prof.last_name || ''}`.trim();
                        const statusClass = prof.is_active ? 'success' : 'secondary';
                        const statusText = prof.is_active ? 'Active' : 'Inactive';
                        
            const row = `
                <tr>
                    <td>${prof.id}</td>
                                <td>${fullName || 'N/A'}</td>
                                <td>${prof.specialization || 'N/A'}</td>
                                <td>${prof.email || 'N/A'}</td>
                                <td>${prof.phone || 'N/A'}</td>
                                <td>${prof.experience_years || 0} years</td>
                                <td><span class="badge badge-${statusClass}">${statusText}</span></td>
                                <td>
                                    <button class="btn btn-sm btn-primary" onclick="editProfessional(${prof.id})" title="Edit">
                            <i class="fas fa-edit"></i>
                        </button>
                                    <button class="btn btn-sm btn-${prof.is_active ? 'warning' : 'success'}" 
                                            onclick="toggleProfessionalStatus(${prof.id})" 
                                            title="${prof.is_active ? 'Deactivate' : 'Activate'}">
                                        <i class="fas fa-${prof.is_active ? 'pause' : 'play'}"></i>
                                    </button>
                                    <button class="btn btn-sm btn-danger" onclick="deleteProfessional(${prof.id})" title="Delete">
                            <i class="fas fa-trash"></i>
                        </button>
                    </td>
                </tr>
            `;
            tbody.append(row);
        });
                    console.log(' Professionals loaded successfully');
                } else {
                    tbody.html('<tr><td colspan="8" class="text-center text-muted">No professionals found</td></tr>');
                }

                // Update DataTable if it exists
        if (dataTables.professionals) {
            dataTables.professionals.clear().rows.add($(tbody).find('tr')).draw();
        }
            })
            .catch(error => {
                console.error(' Error loading professionals:', error);
                tbody.html(`
                    <tr>
                        <td colspan="8" class="text-center text-danger">
                            <i class="fas fa-exclamation-triangle"></i> Error loading professionals
                            <br><small>${error.message}</small>
                        </td>
                    </tr>
                `);
            });
    }

    /**
     * Load bookings data from database with full user details
     */
    function loadBookings() {
        console.log(' Starting loadBookings function...');
        const tbody = $('#bookingsTableBody');
        console.log('Table body element:', tbody);
        console.log('Table body length:', tbody.length);
        console.log('Table is visible:', tbody.is(':visible'));
        
        tbody.html('<tr><td colspan="7" class="text-center bookings-loading"><i class="fas fa-spinner fa-spin"></i> Loading bookings...</td></tr>');
        
        // Show loading state in stats
        updateBookingStats({ total: 0, confirmed: 0, pending: 0, critical: 0 });
        
        fetch(`${API_ROOT}/admin/bookings`)
            .then(response => response.json())
            .then(data => {
        tbody.empty();

                if (data.bookings && data.bookings.length > 0) {
                    // Update stats
                    updateBookingStats(data);
                    
                    // Log the data for debugging
                    console.log(' Bookings data received:', {
                        total: data.total,
                        confirmed: data.confirmed,
                        pending: data.pending,
                        critical: data.critical,
                        bookingsCount: data.bookings ? data.bookings.length : 0
                    });
                    
                    console.log('🔄 Processing bookings data...');
                    data.bookings.forEach((booking, index) => {
                        console.log(`📋 Processing booking ${index + 1}:`, {
                            id: booking.booking_id,
                            user: booking.user_fullname,
                            professional: booking.professional_name,
                            status: booking.booking_status,
                            risk: booking.risk_level
                        });
                        
                        const scheduledTime = new Date(booking.scheduled_datetime * 1000).toLocaleString();
                        const professionalName = booking.professional_name || 'Unassigned';
                        const professionalSpecialization = booking.professional_specialization || '';
                        
                        // Get user initials for avatar
                        const userInitials = getUserInitials(booking.user_fullname || booking.user_account || 'Guest');
                        
                        // Create user details HTML
                        const userDetails = createUserDetailsHTML(booking);
                        
                        // Create professional details HTML
                        const professionalDetails = createProfessionalDetailsHTML(booking);
                        
                        // Create risk badge HTML
                        const riskBadge = createRiskBadgeHTML(booking.risk_level);
                        
                        // Create status badge HTML
                        const statusBadge = createStatusBadgeHTML(booking.booking_status);
                        
                        // Create action buttons HTML
                        const actionButtons = createActionButtonsHTML(booking.booking_id, booking.booking_status);
                        
            const row = `
                <tr>
                                <td>
                                    <div class="booking-id">
                                        <code>${booking.booking_id.substring(0, 8)}...</code>
                                        <small class="text-muted d-block">${booking.booking_id}</small>
                                    </div>
                    </td>
                                <td>${userDetails}</td>
                                <td>${professionalDetails}</td>
                                <td>${riskBadge}</td>
                                <td>
                                    <div class="scheduled-time">
                                        <i class="fas fa-calendar-alt text-primary"></i>
                                        <span class="ml-1">${scheduledTime}</span>
                                    </div>
                                </td>
                                <td>${statusBadge}</td>
                                <td>${actionButtons}</td>
                </tr>
            `;
            tbody.append(row);
            console.log(` Added row ${index + 1} to table`);
        });
        
        console.log(' Total rows in tbody:', tbody.find('tr').length);
                } else {
                    tbody.html(`
                        <tr>
                            <td colspan="7" class="text-center bookings-empty">
                                <i class="fas fa-calendar-times"></i>
                                <h4>No Bookings Found</h4>
                                <p>There are currently no bookings in the system.</p>
                            </td>
                        </tr>
                    `);
                    updateBookingStats({ total: 0, confirmed: 0, pending: 0, critical: 0 });
                }

                // Update DataTable
        console.log(' Checking DataTable status...');
        console.log('DataTable object:', dataTables.bookings);
        console.log('DataTable exists:', !!dataTables.bookings);
        
        // First, let's try to show the data without DataTable
        console.log(' Table body HTML after adding rows:');
        console.log(tbody.html());

        if (dataTables.bookings) {
            console.log('🔄 Updating DataTable with', tbody.find('tr').length, 'rows');
            try {
            dataTables.bookings.clear().rows.add($(tbody).find('tr')).draw();
                console.log(' DataTable updated successfully');
            } catch (error) {
                console.error(' Error updating DataTable:', error);
                console.log('🔄 Trying to destroy and recreate DataTable...');
                try {
                    dataTables.bookings.destroy();
                    dataTables.bookings = $('#bookingsTable').DataTable({
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
                    console.log(' DataTable recreated successfully');
                } catch (recreateError) {
                    console.error(' Error recreating DataTable:', recreateError);
                }
            }
        } else {
            console.log(' DataTable not initialized - trying to reinitialize...');
            // Try to reinitialize the DataTable
            if ($('#bookingsTable').length) {
                dataTables.bookings = $('#bookingsTable').DataTable({
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
                console.log(' DataTable reinitialized');
            }
        }
                
                console.log(' Bookings loaded successfully:', data.bookings?.length || 0, 'bookings');
            })
            .catch(error => {
                console.error('Error loading bookings:', error);
                tbody.html(`
                    <tr>
                        <td colspan="7" class="text-center text-danger">
                            <i class="fas fa-exclamation-triangle"></i>
                            Error loading bookings data
                        </td>
                    </tr>
                `);
            });
    }
    
    /**
     * Create user details HTML with full information
     */
    function createUserDetailsHTML(booking) {
        const userFullName = booking.user_fullname || booking.user_account || 'Guest User';
        const userEmail = booking.user_email || 'No email provided';
        const userPhone = booking.user_phone || 'No phone provided';
        const userLocation = booking.user_location || 'Location not specified';
        const userInitials = getUserInitials(userFullName);
        
        return `
            <div class="user-details">
                <div class="user-avatar">
                    ${userInitials}
                </div>
                <div class="user-info">
                    <h6 class="user-name">${userFullName}</h6>
                    <div class="user-contact">
                        <a href="mailto:${userEmail}" title="Email user">
                            <i class="fas fa-envelope"></i> ${userEmail}
                        </a>
                        <a href="tel:${userPhone}" title="Call user">
                            <i class="fas fa-phone"></i> ${userPhone}
                        </a>
                    </div>
                    <div class="user-location">
                        <i class="fas fa-map-marker-alt"></i>
                        <span>${userLocation}</span>
                    </div>
                </div>
            </div>
        `;
    }
    
    /**
     * Create professional details HTML
     */
    function createProfessionalDetailsHTML(booking) {
        const professionalName = booking.professional_name || 'Unassigned';
        const professionalSpecialization = booking.professional_specialization || '';
        const professionalInitials = getUserInitials(professionalName);
        
        if (professionalName === 'Unassigned') {
            return `
                <div class="professional-info">
                    <div class="professional-avatar" style="background: linear-gradient(135deg, #6b7280 0%, #9ca3af 100%);">
                        <i class="fas fa-user-slash"></i>
                    </div>
                    <div>
                        <div class="professional-name">Unassigned</div>
                        <div class="professional-specialization">Awaiting assignment</div>
                    </div>
                </div>
            `;
        }
        
        return `
            <div class="professional-info">
                <div class="professional-avatar">
                    ${professionalInitials}
                </div>
                <div>
                    <div class="professional-name">${professionalName}</div>
                    <div class="professional-specialization">${professionalSpecialization}</div>
                </div>
            </div>
        `;
    }
    
    /**
     * Create risk badge HTML
     */
    function createRiskBadgeHTML(riskLevel) {
        const riskIcons = {
            critical: 'fas fa-exclamation-triangle',
            high: 'fas fa-exclamation-circle',
            medium: 'fas fa-info-circle',
            low: 'fas fa-check-circle'
        };
        
        const icon = riskIcons[riskLevel.toLowerCase()] || 'fas fa-question-circle';
        
        return `
            <span class="risk-badge ${riskLevel.toLowerCase()}">
                <i class="${icon}"></i>
                ${riskLevel.toUpperCase()}
            </span>
        `;
    }
    
    /**
     * Create status badge HTML
     */
    function createStatusBadgeHTML(status) {
        const statusIcons = {
            pending: 'fas fa-clock',
            confirmed: 'fas fa-check-circle',
            completed: 'fas fa-check-double',
            declined: 'fas fa-times-circle',
            cancelled: 'fas fa-ban'
        };
        
        const icon = statusIcons[status.toLowerCase()] || 'fas fa-question-circle';
        
        return `
            <span class="status-badge ${status.toLowerCase()}">
                <i class="${icon}"></i>
                ${status.toUpperCase()}
            </span>
        `;
    }
    
    /**
     * Create action buttons HTML
     */
    function createActionButtonsHTML(bookingId, status) {
        const canEdit = status === 'pending' || status === 'confirmed';
        const canComplete = status === 'confirmed';
        const canCancel = status === 'pending' || status === 'confirmed';
        
        return `
            <div class="action-buttons">
                <button class="action-btn btn-info" onclick="viewBooking('${bookingId}')" title="View Details">
                    <i class="fas fa-eye"></i>
                </button>
                ${canEdit ? `
                    <button class="action-btn btn-warning" onclick="editBooking('${bookingId}')" title="Edit Booking">
                        <i class="fas fa-edit"></i>
                    </button>
                ` : ''}
                ${canComplete ? `
                    <button class="action-btn btn-success" onclick="completeBooking('${bookingId}')" title="Mark as Completed">
                        <i class="fas fa-check"></i>
                    </button>
                ` : ''}
                ${canCancel ? `
                    <button class="action-btn btn-danger" onclick="cancelBooking('${bookingId}')" title="Cancel Booking">
                        <i class="fas fa-times"></i>
                    </button>
                ` : ''}
            </div>
        `;
    }
    
    /**
     * Get user initials for avatar
     */
    function getUserInitials(name) {
        if (!name || name === 'Guest') return 'G';
        
        const words = name.trim().split(' ');
        if (words.length >= 2) {
            return (words[0][0] + words[1][0]).toUpperCase();
        }
        return name[0].toUpperCase();
    }
    
    /**
     * Update booking statistics
     */
    function updateBookingStats(data) {
        const stats = {
            total: data.total || 0,
            confirmed: data.confirmed || 0,
            pending: data.pending || 0,
            critical: data.critical || 0
        };
        
        $('#totalBookings').text(stats.total);
        $('#confirmedBookings').text(stats.confirmed);
        $('#pendingBookings').text(stats.pending);
        $('#criticalBookings').text(stats.critical);
    }

    /**
     * Load risk data from database
     */
    function loadRiskData() {
        // Show loading state
        $('#criticalCount, #highCount, #mediumCount, #lowCount').text('...');
        
        fetch(`${API_ROOT}/admin/risk-assessments?limit=100`)
            .then(response => response.json())
            .then(data => {
                if (data.assessments) {
                    const riskCounts = {
                        critical: 0,
                        high: 0,
                        medium: 0,
                        low: 0
                    };
                    
                    data.assessments.forEach(assessment => {
                        const riskLevel = assessment.risk_level.toLowerCase();
                        if (riskCounts.hasOwnProperty(riskLevel)) {
                            riskCounts[riskLevel]++;
                        }
                    });
                    
                    $('#criticalCount').text(riskCounts.critical);
                    $('#highCount').text(riskCounts.high);
                    $('#mediumCount').text(riskCounts.medium);
                    $('#lowCount').text(riskCounts.low);
                    
                    // Update recent assessments
                    updateRecentAssessments(data.assessments.slice(0, 10));
                } else {
                    $('#criticalCount, #highCount, #mediumCount, #lowCount').text('0');
                }
            })
            .catch(error => {
                console.error('Error loading risk data:', error);
                $('#criticalCount, #highCount, #mediumCount, #lowCount').text('0');
            });
    }
    
    /**
     * Update recent assessments display
     */
    function updateRecentAssessments(assessments) {
        const container = $('#recentAssessments');
        container.empty();
        
        if (assessments.length === 0) {
            container.html('<p class="text-muted">No recent assessments</p>');
            return;
        }
        
        assessments.forEach(assessment => {
            const timeAgo = getTimeAgo(assessment.assessment_timestamp);
            const riskClass = getRiskBadgeClass(assessment.risk_level);
            
            const assessmentItem = `
                <div class="timeline-item">
                    <span class="time"><i class="fas fa-clock"></i> ${timeAgo}</span>
                    <h3 class="timeline-header">
                        <span class="badge badge-${riskClass}">${assessment.risk_level.toUpperCase()}</span>
                        Risk Assessment
                    </h3>
                    <div class="timeline-body">
                        <p><strong>User:</strong> ${assessment.user_account || 'Guest'}</p>
                        <p><strong>Score:</strong> ${(assessment.risk_score * 100).toFixed(1)}%</p>
                        <p><strong>Indicators:</strong> ${assessment.detected_indicators || 'None detected'}</p>
                    </div>
                </div>
            `;
            container.append(assessmentItem);
        });
    }
    
    /**
     * Get time ago string
     */
    function getTimeAgo(timestamp) {
        const now = Date.now() / 1000;
        const diff = now - timestamp;
        
        if (diff < 60) return 'Just now';
        if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
        if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
        return `${Math.floor(diff / 86400)}d ago`;
    }

    /**
     * Load analytics data from database
     */
    function loadAnalyticsData() {
        // Show loading state
        $('#totalProfessionals, #activeBookings, #completedSessions, #assessmentsToday').text('...');
        
        Promise.all([
            fetch(`${API_ROOT}/admin/professionals`).then(res => res.json()),
            fetch(`${API_ROOT}/admin/bookings`).then(res => res.json()),
            fetch(`${API_ROOT}/admin/risk-assessments?limit=100`).then(res => res.json())
        ]).then(([professionalsData, bookingsData, riskData]) => {
            // Total professionals
            const totalProfessionals = professionalsData.professionals ? professionalsData.professionals.length : 0;
            $('#totalProfessionals').text(totalProfessionals);
            
            // Active bookings
            const activeBookings = bookingsData.bookings ? 
                bookingsData.bookings.filter(b => ['pending', 'confirmed'].includes(b.booking_status)).length : 0;
            $('#activeBookings').text(activeBookings);
            
            // Completed sessions
            const completedSessions = bookingsData.bookings ? 
                bookingsData.bookings.filter(b => b.booking_status === 'completed').length : 0;
            $('#completedSessions').text(completedSessions);
            
            // Assessments today
            const today = new Date().toISOString().split('T')[0];
            const assessmentsToday = riskData.assessments ? 
                riskData.assessments.filter(r => {
                    const assessmentDate = new Date(r.assessment_timestamp * 1000).toISOString().split('T')[0];
                    return assessmentDate === today;
                }).length : 0;
            $('#assessmentsToday').text(assessmentsToday);
            
        }).catch(error => {
            console.error('Error loading analytics data:', error);
            $('#totalProfessionals, #activeBookings, #completedSessions, #assessmentsToday').text('0');
        });
    }

    /**
     * Load RAG status
     */
    function loadRAGStatus() {
        // RAG status is already set in HTML
        // This function can be used to update real-time status
    }


    /**
     * Save professional
     */
    function saveProfessional() {
        console.log('Saving professional...');
        
        // Validate form
        if (!validateProfessionalForm()) {
            console.log('Form validation failed');
            return;
        }

        // Get form data
        const form = $('#professionalForm');
        const formData = new FormData(form[0]);
        const data = Object.fromEntries(formData.entries());
        
        console.log('Form data collected:', data);
        
        // Get expertise areas
        const expertiseAreas = $('input[name="expertise"]:checked').map(function() {
            return this.value;
        }).get();
        
        console.log('Expertise areas selected:', expertiseAreas);
        
        const professionalData = {
            username: data.username,
            first_name: data.first_name,
            last_name: data.last_name,
            email: data.email,
            phone: data.phone || '',
            specialization: data.specialization,
            experience_years: parseInt(data.experience_years) || 0,
            expertise_areas: expertiseAreas,
            district: data.district || '',
            consultation_fee: parseFloat(data.consultation_fee) || 0,
            bio: data.bio || '',
            languages: ['english'], // Default languages
            qualifications: [], // Default qualifications
            availability_schedule: {} // Default schedule
        };

        // Add password only for new professionals or if provided in edit mode
        const isEditMode = $('#modalTitle').text().includes('Edit');
        if (!isEditMode) {
            professionalData.password = data.password;
        } else {
            // For edit mode, only include password if provided
            const password = data.password;
            if (password && password.trim()) {
                professionalData.password = password;
            }
        }

        console.log('Professional data to send:', professionalData);

        // Show loading state
        $('#saveProfessionalBtn').prop('disabled', true).html('<i class="fas fa-spinner fa-spin"></i> Saving...');

        const url = isEditMode ? 
            `${API_ROOT}/admin/professionals/${currentProfessionalId}` : 
            `${API_ROOT}/admin/professionals`;
        const method = isEditMode ? 'PUT' : 'POST';
        
        console.log('Sending request to:', url, 'Method:', method);

        fetch(url, {
            method: method,
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(professionalData)
        })
        .then(response => response.json())
        .then(data => {
            console.log(' Response data:', data);
            if (data.success) {
        Swal.fire({
            title: 'Success!',
                    text: isEditMode ? 'Professional updated successfully!' : 'Professional added successfully!',
            icon: 'success',
            timer: 2000
        }).then(() => {
            $('#professionalModal').modal('hide');
            loadProfessionals();
                    resetProfessionalForm();
                });
            } else {
                Swal.fire({
                    title: 'Error!',
                    text: data.error || 'Failed to save professional.',
                    icon: 'error'
                });
            }
        })
        .catch(error => {
            console.error('Error saving professional:', error);
            Swal.fire({
                title: 'Error!',
                text: 'Failed to save professional. Please try again.',
                icon: 'error'
            });
        })
        .finally(() => {
            $('#saveProfessionalBtn').prop('disabled', false).html('Save Professional');
        });
    }
    
    /**
     * Validate professional form
     */
    function validateProfessionalForm() {
        console.log('Validating professional form...');
        
        const requiredFields = ['username', 'first_name', 'last_name', 'email', 'specialization'];
        const isEditMode = $('#modalTitle').text().includes('Edit');
        
        // Check if password is required (only for new professionals)
        if (!isEditMode) {
            requiredFields.push('password');
        } else {
            // For edit mode, make password optional
            $('#password').prop('required', false);
        }
        
        let isValid = true;
        let errorMessage = '';
        
        // Clear previous validation errors
        $('.is-invalid').removeClass('is-invalid');
        
        // Check required fields
        requiredFields.forEach(field => {
            const value = $(`#${field}`).val().trim();
            if (!value) {
                $(`#${field}`).addClass('is-invalid');
                isValid = false;
                const fieldName = field.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
                errorMessage += `${fieldName} is required.\n`;
            } else {
                $(`#${field}`).removeClass('is-invalid');
            }
        });
        
        // Validate email format
        const email = $('#email').val().trim();
        if (email && !isValidEmail(email)) {
            $('#email').addClass('is-invalid');
            isValid = false;
            errorMessage += 'Please enter a valid email address.\n';
        }
        
        // Validate phone format (if provided)
        const phone = $('#phone').val().trim();
        if (phone && !isValidPhone(phone)) {
            $('#phone').addClass('is-invalid');
            isValid = false;
            errorMessage += 'Please enter a valid phone number.\n';
        }
        
        // Check if at least one expertise area is selected
        if (!validateExpertiseAreas()) {
            isValid = false;
            errorMessage += 'Please select at least one expertise area.\n';
        }
        
        // Validate experience years
        const experienceYears = parseInt($('#experience_years').val()) || 0;
        if (experienceYears < 0) {
            $('#experience_years').addClass('is-invalid');
            isValid = false;
            errorMessage += 'Experience years cannot be negative.\n';
        }
        
        // Validate consultation fee
        const consultationFee = parseFloat($('#consultation_fee').val()) || 0;
        if (consultationFee < 0) {
            $('#consultation_fee').addClass('is-invalid');
            isValid = false;
            errorMessage += 'Consultation fee cannot be negative.\n';
        }
        
        if (!isValid) {
            console.error(' Form validation failed:', errorMessage);
            Swal.fire({
                title: 'Validation Error',
                text: errorMessage.trim(),
                icon: 'error',
                confirmButtonText: 'Fix Issues'
            });
        } else {
            console.log(' Form validation passed');
        }
        
        return isValid;
    }
    
    /**
     * Validate email format
     */
    function isValidEmail(email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    }
    
    /**
     * Validate phone format
     */
    function isValidPhone(phone) {
        const phoneRegex = /^[\+]?[0-9\s\-\(\)]{10,}$/;
        return phoneRegex.test(phone);
    }
    

    /**
     * Reset professional form
     */
    function resetProfessionalForm() {
        $('#professionalForm')[0].reset();
        $('input[name="expertise"]').prop('checked', false);
        $('.is-invalid').removeClass('is-invalid');
        currentProfessionalId = null;
        $('#password').prop('required', true);
    }
    
    /**
     * Ensure all inputs are working properly
     */
    function ensureInputsWorking() {
        const form = $('#professionalForm');
        const inputs = form.find('input, select, textarea');
        
        console.log('🔧 Ensuring input functionality for', inputs.length, 'inputs');
        
        inputs.each(function() {
            const input = $(this);
            
            // Ensure all inputs are enabled
            input.prop('disabled', false);
            input.prop('readonly', false);
            
            // Force CSS properties
            input.css({
                'background-color': '#fff !important',
                'color': '#495057 !important',
                'pointer-events': 'auto !important',
                'user-select': 'text !important',
                'cursor': 'text !important'
            });
            
            // Add click handler to ensure focus
            input.off('click.ensureFocus').on('click.ensureFocus', function() {
                $(this).focus();
                console.log(' Input clicked:', $(this).attr('name'));
            });
            
            // Add keydown handler to ensure typing works
            input.off('keydown.ensureTyping').on('keydown.ensureTyping', function(e) {
                console.log(' Key pressed:', e.key, 'in', $(this).attr('name'));
                // Allow all normal typing
                if (e.key.length === 1 || e.key === 'Backspace' || e.key === 'Delete' || 
                    e.key === 'ArrowLeft' || e.key === 'ArrowRight' || e.key === 'Tab') {
                    return true;
                }
            });
            
            // Add input handler for real-time validation
            input.off('input.validate').on('input.validate', function() {
                console.log(' Input changed:', $(this).attr('name'), '=', $(this).val());
                validateInput($(this));
            });
        });
        
        console.log(' Input functionality ensured for', inputs.length, 'inputs');
    }
    
    /**
     * Validate individual input
     */
    function validateInput(input) {
        const value = input.val().trim();
        const isRequired = input.prop('required');
        
        if (isRequired && !value) {
            input.removeClass('is-valid').addClass('is-invalid');
        } else if (value) {
            input.removeClass('is-invalid').addClass('is-valid');
        } else {
            input.removeClass('is-invalid is-valid');
        }
        
        // Check form validity
        checkFormValidity();
    }
    
    /**
     * Check overall form validity
     */
    function checkFormValidity() {
        const form = $('#professionalForm');
        const requiredFields = form.find('[required]');
        let isValid = true;
        
        requiredFields.each(function() {
            const field = $(this);
            const value = field.val().trim();
            if (!value) {
                isValid = false;
                return false;
            }
        });
        
        // Check expertise areas
        if (!validateExpertiseAreas()) {
            isValid = false;
        }
        
        // Enable/disable submit button
        const submitBtn = form.find('button[type="submit"]');
        if (submitBtn.length) {
            submitBtn.prop('disabled', !isValid);
        }
        
        return isValid;
    }
    
    /**
     * Debug form inputs
     */
    function debugFormInputs() {
        const form = $('#professionalForm');
        const inputs = form.find('input, select, textarea');
        
        console.log(' Debugging form inputs:');
        inputs.each(function(index) {
            const input = $(this);
            const isFocusable = function() {
                try {
                    input.focus();
                    return document.activeElement === input[0];
                } catch (e) {
                    return false;
                }
            }();
            
            console.log(`Input ${index}:`, {
                type: input.attr('type') || input.prop('tagName').toLowerCase(),
                name: input.attr('name'),
                id: input.attr('id'),
                value: input.val(),
                disabled: input.prop('disabled'),
                readonly: input.prop('readonly'),
                focusable: isFocusable,
                style: input.attr('style')
            });
        });
    }
    
    /**
     * Force input functionality
     */
    function forceInputFunctionality() {
        const form = $('#professionalForm');
        const inputs = form.find('input, select, textarea');
        
        inputs.each(function() {
            const input = $(this);
            
            // Force enable inputs
            input.prop('disabled', false);
            input.prop('readonly', false);
            
            // Force CSS properties
            input.css({
                'background-color': '#fff',
                'color': '#495057',
                'pointer-events': 'auto',
                'user-select': 'text',
                'cursor': 'text'
            });
            
            // Add event listeners
            input.off('click.force').on('click.force', function() {
                $(this).focus();
                console.log(' Input clicked:', $(this).attr('name'));
            });
            
            input.off('keydown.force').on('keydown.force', function(e) {
                console.log(' Key pressed:', e.key, 'in', $(this).attr('name'));
            });
            
            input.off('input.force').on('input.force', function() {
                console.log(' Input changed:', $(this).attr('name'), '=', $(this).val());
            });
        });
        
        console.log('🔧 Forced input functionality for', inputs.length, 'inputs');
    }

    /**
     * Filter professionals by search term
     */
    function filterProfessionals(searchTerm) {
        console.log(' Filtering professionals by:', searchTerm);
        
        if (dataTables.professionals) {
            dataTables.professionals.search(searchTerm).draw();
        } else {
            // Fallback: filter table rows manually
            $('#professionalsTableBody tr').each(function() {
                const row = $(this);
                const text = row.text().toLowerCase();
                if (text.includes(searchTerm)) {
                    row.show();
                } else {
                    row.hide();
                }
            });
        }
    }

    /**
     * Filter professionals by specialization
     */
    function filterProfessionalsBySpecialization(specialization) {
        console.log(' Filtering professionals by specialization:', specialization);
        
        if (dataTables.professionals) {
            if (specialization === '') {
                dataTables.professionals.column(2).search('').draw();
            } else {
                dataTables.professionals.column(2).search(specialization).draw();
            }
        } else {
            // Fallback: filter table rows manually
            $('#professionalsTableBody tr').each(function() {
                const row = $(this);
                const specializationCell = row.find('td:eq(2)').text().toLowerCase();
                if (specialization === '' || specializationCell.includes(specialization.toLowerCase())) {
                    row.show();
                } else {
                    row.hide();
                }
            });
        }
    }

    /**
     * Apply filters
     */
    function applyFilters() {
        const status = $('#statusFilter').val();
        const riskLevel = $('#riskLevelFilter').val();
        const specialization = $('#specializationFilter').val();

        // Apply filters to DataTables
        if (dataTables.bookings) {
            dataTables.bookings.column(5).search(status).draw();
        }
        if (dataTables.professionals) {
            dataTables.professionals.column(2).search(specialization).draw();
        }
    }

    /**
     * Export table to CSV
     */
    function exportTableToCSV(tableId, filename) {
        const table = document.getElementById(tableId);
        const rows = table.querySelectorAll('tr');
        let csv = [];

        for (let i = 0; i < rows.length; i++) {
            const row = [];
            const cols = rows[i].querySelectorAll('td, th');
            
            for (let j = 0; j < cols.length; j++) {
                row.push(cols[j].innerText);
            }
            csv.push(row.join(','));
        }

        const csvContent = csv.join('\n');
        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        window.URL.revokeObjectURL(url);
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
            'pending': 'warning',
            'confirmed': 'info',
            'completed': 'success',
            'declined': 'danger',
            'active': 'success',
            'inactive': 'secondary'
        };
        return classes[status.toLowerCase()] || 'secondary';
    }

    /**
     * Get current professional ID for editing
     */
    function getCurrentProfessionalId() {
        return currentProfessionalId;
    }

    /**
     * Handle API errors gracefully
     */
    function handleAPIError(error, context = 'API call') {
        console.error(`Error in ${context}:`, error);
        
        // Show user-friendly error message
        Swal.fire({
            title: 'Connection Error',
            text: 'Unable to connect to the server. Please check your internet connection and try again.',
            icon: 'error',
            timer: 5000
        });
    }

    /**
     * Refresh all data
     */
    function refreshAllData() {
        // Show loading state
        const refreshBtn = $('#refreshAllBtn');
        const originalText = refreshBtn.html();
        refreshBtn.prop('disabled', true).html('<i class="fas fa-spinner fa-spin"></i> Refreshing...');
        
        // Refresh current section data
        if (currentSection === 'dashboard') {
            loadDashboardData();
        } else {
            loadSectionData(currentSection);
        }
        
        // Reset button after a delay
        setTimeout(() => {
            refreshBtn.prop('disabled', false).html(originalText);
        }, 2000);
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
     * Toggle professional status
     */
    function toggleProfessionalStatus(profId) {
        // Get current status from the button
        const button = $(`button[onclick="toggleProfessionalStatus(${profId})"]`);
        const isCurrentlyActive = button.hasClass('btn-warning'); // warning = active, success = inactive
        
        fetch(`${API_ROOT}/admin/professionals/${profId}/status`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                is_active: !isCurrentlyActive
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                Swal.fire({
                    title: 'Success!',
                    text: data.message || 'Professional status updated.',
                    icon: 'success',
                    timer: 2000
                }).then(() => {
                    loadProfessionals();
                });
            } else {
                Swal.fire({
                    title: 'Error!',
                    text: data.error || 'Failed to update professional status.',
                    icon: 'error'
                });
            }
        })
        .catch(error => {
            console.error('Error toggling professional status:', error);
            Swal.fire({
                title: 'Error!',
                text: 'Failed to update professional status.',
                icon: 'error'
            });
        });
    }

    /**
     * Global functions for onclick handlers
     */
    window.editProfessional = function(id) {
        console.log(' Editing professional with ID:', id);
        
        // Show loading state
        Swal.fire({
            title: 'Loading...',
            text: 'Loading professional data...',
            allowOutsideClick: false,
            didOpen: () => {
                Swal.showLoading();
            }
        });
        
        // Load professional data and populate form
        fetch(`${API_ROOT}/admin/professionals`)
            .then(response => {
                console.log('📡 Edit professional API response status:', response.status);
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                return response.json();
            })
            .then(data => {
                console.log(' Professional data for editing:', data);
                const professional = data.professionals.find(p => p.id === id);
                
                if (professional) {
                    console.log(' Professional found:', professional);
                    
                    // Store current professional ID for editing
                    currentProfessionalId = id;
                    
                    // Populate form with professional data
                    $('#username').val(professional.username || '');
                    $('#first_name').val(professional.first_name || '');
                    $('#last_name').val(professional.last_name || '');
                    $('#email').val(professional.email || '');
                    $('#phone').val(professional.phone || '');
                    $('#specialization').val(professional.specialization || '');
                    $('#experience_years').val(professional.experience_years || 0);
                    $('#district').val(professional.district || '');
                    $('#consultation_fee').val(professional.consultation_fee || 0);
                    $('#bio').val(professional.bio || '');
                    
                    // Set expertise checkboxes
                    if (professional.expertise_areas) {
                        let expertiseAreas = [];
                        if (Array.isArray(professional.expertise_areas)) {
                            expertiseAreas = professional.expertise_areas;
                        } else if (typeof professional.expertise_areas === 'string') {
                            expertiseAreas = professional.expertise_areas.split(',').map(area => area.trim());
                        }
                        
                    $('input[name="expertise"]').prop('checked', false);
                    expertiseAreas.forEach(area => {
                            const trimmedArea = area.trim();
                            if (trimmedArea) {
                                $(`#expertise_${trimmedArea}`).prop('checked', true);
                            }
                    });
                    } else {
                        $('input[name="expertise"]').prop('checked', false);
                    }
                    
                    // Update modal for edit mode
                    $('#modalTitle').text('Edit Professional');
                    $('#passwordRequired').text('');
                    $('#passwordHelp').show();
                    $('#password').prop('required', false).val('');
                    
                    // Close loading dialog and show modal
                    Swal.close();
                    $('#professionalModal').modal('show');
                    
                    // Ensure inputs work properly after modal is shown
                    setTimeout(() => {
                        ensureInputsWorking();
                        forceInputFunctionality();
                        debugFormInputs();
                        const firstInput = $('#professionalModal input[required]').first();
                        if (firstInput.length) {
                            firstInput.focus();
                            console.log(' Focused on first input:', firstInput.attr('name'));
                        }
                    }, 300);
                    
                    console.log(' Professional form populated successfully');
                } else {
                    console.error(' Professional not found with ID:', id);
                    Swal.fire('Error', 'Professional not found.', 'error');
                }
            })
            .catch(error => {
                console.error(' Error loading professional:', error);
                Swal.fire('Error', `Failed to load professional data: ${error.message}`, 'error');
            });
    };

    window.deleteProfessional = function(id) {
        Swal.fire({
            title: 'Delete Professional?',
            text: 'This action cannot be undone!',
            icon: 'warning',
            showCancelButton: true,
            confirmButtonColor: '#d33',
            cancelButtonColor: '#3085d6',
            confirmButtonText: 'Yes, delete it!'
        }).then((result) => {
            if (result.isConfirmed) {
                fetch(`${API_ROOT}/admin/professionals/${id}`, {
                    method: 'DELETE'
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                Swal.fire('Deleted!', 'Professional has been deleted.', 'success');
                loadProfessionals();
                    } else {
                        Swal.fire('Error!', data.error || 'Failed to delete professional.', 'error');
                    }
                })
                .catch(error => {
                    console.error('Error deleting professional:', error);
                    Swal.fire('Error!', 'Failed to delete professional.', 'error');
                });
            }
        });
    };

    window.toggleProfessionalStatus = toggleProfessionalStatus;

    window.viewBooking = function(id) {
        // Show loading state
        $('#bookingDetails').html('<div class="text-center"><i class="fas fa-spinner fa-spin"></i> Loading booking details...</div>');
        $('#bookingModal').modal('show');
        
        // Load booking details
        fetch(`${API_ROOT}/admin/bookings`)
            .then(response => response.json())
            .then(data => {
                const booking = data.bookings.find(b => b.booking_id === id);
                if (booking) {
                    const scheduledTime = new Date(booking.scheduled_datetime * 1000).toLocaleString();
                    const createdTime = new Date(booking.created_ts * 1000).toLocaleString();
                    const userInitials = getUserInitials(booking.user_fullname || booking.user_account || 'Guest');
                    const professionalInitials = getUserInitials(booking.professional_name || 'Unassigned');
                    
                    const bookingDetails = `
                        <!-- Booking Header -->
                        <div class="booking-header mb-4">
                            <div class="row">
                                <div class="col-md-8">
                                    <h4 class="mb-2">
                                        <i class="fas fa-calendar-check text-primary"></i>
                                        Booking Details
                                    </h4>
                                    <p class="text-muted mb-0">Booking ID: <code>${booking.booking_id}</code></p>
                                </div>
                                <div class="col-md-4 text-right">
                                    <span class="status-badge ${booking.booking_status.toLowerCase()}">
                                        <i class="fas fa-${getStatusIcon(booking.booking_status)}"></i>
                                        ${booking.booking_status.toUpperCase()}
                                    </span>
                                </div>
                            </div>
                        </div>

                        <!-- User Information -->
                        <div class="card mb-4">
                            <div class="card-header bg-primary text-white">
                                <h5 class="mb-0">
                                    <i class="fas fa-user"></i> User Information
                                </h5>
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-md-2">
                                        <div class="user-avatar-large">
                                            ${userInitials}
                                        </div>
                                    </div>
                                    <div class="col-md-10">
                        <div class="row">
                            <div class="col-md-6">
                                                <h6 class="text-primary">Personal Details</h6>
                                                <p><strong>Full Name:</strong> ${booking.user_fullname || booking.user_account || 'Guest User'}</p>
                                                <p><strong>Username:</strong> ${booking.user_account || 'N/A'}</p>
                                                <p><strong>Email:</strong> 
                                                    <a href="mailto:${booking.user_email || ''}" class="text-primary">
                                                        <i class="fas fa-envelope"></i> ${booking.user_email || 'No email provided'}
                                                    </a>
                                                </p>
                                                <p><strong>Phone:</strong> 
                                                    <a href="tel:${booking.user_phone || ''}" class="text-primary">
                                                        <i class="fas fa-phone"></i> ${booking.user_phone || 'No phone provided'}
                                                    </a>
                                                </p>
                            </div>
                            <div class="col-md-6">
                                                <h6 class="text-primary">Location Information</h6>
                                                <p><strong>Province:</strong> ${booking.user_province || 'Not specified'}</p>
                                                <p><strong>District:</strong> ${booking.user_district || 'Not specified'}</p>
                                                <p><strong>Full Location:</strong> ${booking.user_location || 'Location not specified'}</p>
                                                <p><strong>IP Address:</strong> ${booking.user_ip || 'N/A'}</p>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Professional Information -->
                        <div class="card mb-4">
                            <div class="card-header bg-success text-white">
                                <h5 class="mb-0">
                                    <i class="fas fa-user-md"></i> Professional Information
                                </h5>
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-md-2">
                                        <div class="professional-avatar-large">
                                            ${professionalInitials}
                                        </div>
                                    </div>
                                    <div class="col-md-10">
                                        <div class="row">
                                            <div class="col-md-6">
                                                <h6 class="text-success">Professional Details</h6>
                                                <p><strong>Name:</strong> ${booking.professional_name || 'Unassigned'}</p>
                                                <p><strong>Specialization:</strong> ${booking.professional_specialization || 'Not specified'}</p>
                                                <p><strong>Email:</strong> 
                                                    <a href="mailto:${booking.professional_email || ''}" class="text-success">
                                                        <i class="fas fa-envelope"></i> ${booking.professional_email || 'No email provided'}
                                                    </a>
                                                </p>
                                                <p><strong>Phone:</strong> 
                                                    <a href="tel:${booking.professional_phone || ''}" class="text-success">
                                                        <i class="fas fa-phone"></i> ${booking.professional_phone || 'No phone provided'}
                                                    </a>
                                                </p>
                                            </div>
                                            <div class="col-md-6">
                                                <h6 class="text-success">Assignment Status</h6>
                                                <p><strong>Status:</strong> ${booking.professional_name ? 'Assigned' : 'Unassigned'}</p>
                                                <p><strong>Assignment Date:</strong> ${booking.professional_name ? createdTime : 'Pending'}</p>
                                                <p><strong>Professional ID:</strong> ${booking.professional_id || 'N/A'}</p>
                                                <p><strong>Experience:</strong> ${booking.professional_experience || 'N/A'} years</p>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Booking Details -->
                        <div class="card mb-4">
                            <div class="card-header bg-info text-white">
                                <h5 class="mb-0">
                                    <i class="fas fa-calendar-alt"></i> Booking Details
                                </h5>
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-md-6">
                                        <h6 class="text-info">Schedule Information</h6>
                                <p><strong>Scheduled Time:</strong> ${scheduledTime}</p>
                                <p><strong>Created:</strong> ${createdTime}</p>
                                        <p><strong>Session Type:</strong> ${booking.session_type || 'Emergency'}</p>
                                        <p><strong>Duration:</strong> ${booking.session_duration || '60 minutes'}</p>
                                    </div>
                                    <div class="col-md-6">
                                        <h6 class="text-info">Risk Assessment</h6>
                                        <p><strong>Risk Level:</strong> 
                                            <span class="risk-badge ${booking.risk_level.toLowerCase()}">
                                                <i class="fas fa-${getRiskIcon(booking.risk_level)}"></i>
                                                ${booking.risk_level.toUpperCase()}
                                            </span>
                                        </p>
                                <p><strong>Risk Score:</strong> ${(booking.risk_score * 100).toFixed(1)}%</p>
                                        <p><strong>Detected Indicators:</strong> ${booking.detected_indicators || 'None detected'}</p>
                                        <p><strong>Assessment Time:</strong> ${new Date(booking.assessment_timestamp * 1000).toLocaleString()}</p>
                            </div>
                        </div>
                            </div>
                        </div>

                        <!-- Additional Information -->
                        <div class="card">
                            <div class="card-header bg-warning text-dark">
                                <h5 class="mb-0">
                                    <i class="fas fa-info-circle"></i> Additional Information
                                </h5>
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-md-6">
                                        <h6 class="text-warning">Session Details</h6>
                                        <p><strong>Location Preference:</strong> ${booking.location_preference || 'Not specified'}</p>
                                        <p><strong>Session Notes:</strong> ${booking.session_notes || 'No notes available'}</p>
                                        <p><strong>Treatment Plan:</strong> ${booking.treatment_plan || 'Not available'}</p>
                                    </div>
                                    <div class="col-md-6">
                                        <h6 class="text-warning">System Information</h6>
                                        <p><strong>Conversation ID:</strong> ${booking.conv_id || 'N/A'}</p>
                                        <p><strong>Booking Source:</strong> ${booking.booking_source || 'Automated'}</p>
                                        <p><strong>Last Updated:</strong> ${new Date(booking.updated_ts * 1000).toLocaleString()}</p>
                                        <p><strong>System Notes:</strong> ${booking.notes || 'No additional notes'}</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    `;
                    $('#bookingDetails').html(bookingDetails);
                } else {
                    $('#bookingDetails').html('<div class="alert alert-warning">Booking not found.</div>');
                }
            })
            .catch(error => {
                console.error('Error loading booking details:', error);
                $('#bookingDetails').html('<div class="alert alert-danger">Error loading booking details.</div>');
            });
    };
    
    /**
     * Get status icon for display
     */
    function getStatusIcon(status) {
        const statusIcons = {
            pending: 'clock',
            confirmed: 'check-circle',
            completed: 'check-double',
            declined: 'times-circle',
            cancelled: 'ban'
        };
        return statusIcons[status.toLowerCase()] || 'question-circle';
    }
    
    /**
     * Get risk icon for display
     */
    function getRiskIcon(riskLevel) {
        const riskIcons = {
            critical: 'exclamation-triangle',
            high: 'exclamation-circle',
            medium: 'info-circle',
            low: 'check-circle'
        };
        return riskIcons[riskLevel.toLowerCase()] || 'question-circle';
    }

    window.editBooking = function(id) {
        Swal.fire({
            title: 'Edit Booking',
            text: `Edit booking with ID: ${id}`,
            icon: 'info',
            showCancelButton: true,
            confirmButtonText: 'View Details',
            cancelButtonText: 'Cancel'
        }).then((result) => {
            if (result.isConfirmed) {
                viewBooking(id);
            }
        });
    };



    // Global debug functions
    window.debugFormInputs = debugFormInputs;
    window.forceInputFunctionality = forceInputFunctionality;
    window.ensureInputsWorking = ensureInputsWorking;
    window.testInputs = function() {
        console.log('🧪 Testing input functionality...');
        const form = $('#professionalForm');
        if (form.length === 0) {
            console.log(' Form not found');
            return;
        }
        
        const inputs = form.find('input, select, textarea');
        console.log('📝 Found', inputs.length, 'inputs');
        
        inputs.each(function(index) {
            const input = $(this);
            console.log(`Input ${index}:`, {
                name: input.attr('name'),
                type: input.attr('type'),
                value: input.val(),
                disabled: input.prop('disabled'),
                readonly: input.prop('readonly')
            });
        });
        
        // Test focus
        const firstInput = inputs.first();
        if (firstInput.length) {
            firstInput.focus();
            console.log(' Focused on first input:', firstInput.attr('name'));
        }
    };

    // Show dashboard by default
    showSection('dashboard');

    /**
     * Initialize expertise areas functionality
     */
    function initializeExpertiseAreas() {
        console.log('🧠 Initializing expertise areas functionality...');
        
        // Select All button
        $('#selectAllExpertise').on('click', function() {
            $('input[name="expertise"]').prop('checked', true);
            updateExpertiseCount();
            updateExpertiseValidation();
            console.log(' All expertise areas selected');
        });
        
        // Clear All button
        $('#clearAllExpertise').on('click', function() {
            $('input[name="expertise"]').prop('checked', false);
            updateExpertiseCount();
            updateExpertiseValidation();
            console.log(' All expertise areas cleared');
        });
        
        // Individual checkbox change
        $('input[name="expertise"]').on('change', function() {
            updateExpertiseCount();
            updateExpertiseValidation();
            
            // Add visual feedback
            const label = $(this).next('.expertise-label');
            if ($(this).is(':checked')) {
                label.addClass('selected');
                console.log(' Expertise selected:', $(this).val());
            } else {
                label.removeClass('selected');
                console.log(' Expertise deselected:', $(this).val());
            }
        });
        
        // Initialize count
        updateExpertiseCount();
        updateExpertiseValidation();
        
        console.log(' Expertise areas functionality initialized');
    }
    
    /**
     * Update expertise selection count
     */
    function updateExpertiseCount() {
        const selectedCount = $('input[name="expertise"]:checked').length;
        const totalCount = $('input[name="expertise"]').length;
        
        $('#selectedCount').text(selectedCount);
        
        // Update select all button state
        const selectAllBtn = $('#selectAllExpertise');
        const clearAllBtn = $('#clearAllExpertise');
        
        if (selectedCount === totalCount) {
            selectAllBtn.addClass('btn-primary').removeClass('btn-outline-primary');
            selectAllBtn.html('<i class="fas fa-check-square"></i> All Selected');
        } else {
            selectAllBtn.addClass('btn-outline-primary').removeClass('btn-primary');
            selectAllBtn.html('<i class="fas fa-check-square"></i> Select All');
        }
        
        if (selectedCount === 0) {
            clearAllBtn.addClass('btn-secondary').removeClass('btn-outline-secondary');
            clearAllBtn.html('<i class="fas fa-square"></i> All Cleared');
        } else {
            clearAllBtn.addClass('btn-outline-secondary').removeClass('btn-secondary');
            clearAllBtn.html('<i class="fas fa-square"></i> Clear All');
        }
    }
    
    /**
     * Update expertise validation state
     */
    function updateExpertiseValidation() {
        const selectedCount = $('input[name="expertise"]:checked').length;
        const expertiseContainer = $('.form-group:has(input[name="expertise"])');
        
        if (selectedCount === 0) {
            expertiseContainer.addClass('is-invalid').removeClass('is-valid');
            $('input[name="expertise"]').addClass('is-invalid');
        } else {
            expertiseContainer.removeClass('is-invalid').addClass('is-valid');
            $('input[name="expertise"]').removeClass('is-invalid');
        }
    }
    
    /**
     * Validate expertise areas selection
     */
    function validateExpertiseAreas() {
        const selectedCount = $('input[name="expertise"]:checked').length;
        return selectedCount > 0;
    }
    
    /**
     * Initialize bookings filtering functionality
     */
    function initializeBookingsFiltering() {
        console.log(' Initializing bookings filtering...');
        
        // Filter change events
        $('#statusFilter, #riskLevelFilter, #professionalFilter, #fromDateFilter, #toDateFilter').on('change', function() {
            applyBookingsFilters();
        });
        
        // Search input
        $('#bookingSearch').on('keyup', function() {
            clearTimeout(this.searchTimeout);
            this.searchTimeout = setTimeout(() => {
                applyBookingsFilters();
            }, 300);
        });
        
        // Clear filters button
        $('#clearFiltersBtn').on('click', function() {
            clearBookingsFilters();
        });
        
        // Apply filters button
        $('#applyFiltersBtn').on('click', function() {
            applyBookingsFilters();
        });
        
        console.log(' Bookings filtering initialized');
    }
    
    /**
     * Apply bookings filters
     */
    function applyBookingsFilters() {
        const status = $('#statusFilter').val();
        const riskLevel = $('#riskLevelFilter').val();
        const professional = $('#professionalFilter').val();
        const fromDate = $('#fromDateFilter').val();
        const toDate = $('#toDateFilter').val();
        const search = $('#bookingSearch').val().toLowerCase();
        
        console.log(' Applying filters:', { status, riskLevel, professional, fromDate, toDate, search });
        
        if (dataTables.bookings) {
            dataTables.bookings.column(5).search(status); // Status column
            dataTables.bookings.column(3).search(riskLevel); // Risk level column
            dataTables.bookings.column(2).search(professional); // Professional column
            dataTables.bookings.search(search).draw();
        }
        
        // Update filter button states
        updateFilterButtonStates();
    }
    
    /**
     * Clear all bookings filters
     */
    function clearBookingsFilters() {
        $('#statusFilter, #riskLevelFilter, #professionalFilter').val('');
        $('#fromDateFilter, #toDateFilter').val('');
        $('#bookingSearch').val('');
        
        if (dataTables.bookings) {
            dataTables.bookings.search('').columns().search('').draw();
        }
        
        updateFilterButtonStates();
        console.log('🧹 Filters cleared');
    }
    
    /**
     * Update filter button states
     */
    function updateFilterButtonStates() {
        const hasActiveFilters = $('#statusFilter').val() || 
                                $('#riskLevelFilter').val() || 
                                $('#professionalFilter').val() || 
                                $('#fromDateFilter').val() || 
                                $('#toDateFilter').val() || 
                                $('#bookingSearch').val();
        
        if (hasActiveFilters) {
            $('#clearFiltersBtn').removeClass('btn-outline-secondary').addClass('btn-secondary');
            $('#applyFiltersBtn').removeClass('btn-outline-primary').addClass('btn-primary');
        } else {
            $('#clearFiltersBtn').removeClass('btn-secondary').addClass('btn-outline-secondary');
            $('#applyFiltersBtn').removeClass('btn-primary').addClass('btn-outline-primary');
        }
    }
    
    /**
     * Complete booking action
     */
    window.completeBooking = function(bookingId) {
        Swal.fire({
            title: 'Complete Booking',
            text: 'Are you sure you want to mark this booking as completed?',
            icon: 'question',
            showCancelButton: true,
            confirmButtonText: 'Yes, Complete',
            cancelButtonText: 'Cancel',
            confirmButtonColor: '#10b981'
        }).then((result) => {
            if (result.isConfirmed) {
                // Update booking status
                updateBookingStatus(bookingId, 'completed');
            }
        });
    };
    
    /**
     * Cancel booking action
     */
    window.cancelBooking = function(bookingId) {
        Swal.fire({
            title: 'Cancel Booking',
            text: 'Are you sure you want to cancel this booking?',
            icon: 'warning',
            showCancelButton: true,
            confirmButtonText: 'Yes, Cancel',
            cancelButtonText: 'Keep Booking',
            confirmButtonColor: '#ef4444'
        }).then((result) => {
            if (result.isConfirmed) {
                // Update booking status
                updateBookingStatus(bookingId, 'cancelled');
            }
        });
    };
    
    /**
     * Update booking status
     */
    function updateBookingStatus(bookingId, newStatus) {
        console.log(`📝 Updating booking ${bookingId} to ${newStatus}`);
        
        // Show loading
        Swal.fire({
            title: 'Updating...',
            text: 'Please wait while we update the booking status.',
            allowOutsideClick: false,
            showConfirmButton: false,
            didOpen: () => {
                Swal.showLoading();
            }
        });
        
        // Simulate API call (replace with actual API call)
        setTimeout(() => {
            Swal.fire({
                title: 'Success!',
                text: `Booking has been ${newStatus}.`,
                icon: 'success',
                confirmButtonText: 'OK'
            }).then(() => {
                // Reload bookings
                loadBookings();
            });
        }, 1000);
    }

})();

