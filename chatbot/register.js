(() => {
    const API_ROOT = "http://localhost:5057"; // Flask API server
    
    // Elements
    const registerForm = document.getElementById('registerForm');
    const registerBtn = document.getElementById('registerBtn');
    
    // Validation state
    let validationErrors = {};
    let isSubmitting = false;
    
    // API helper
    async function api(path, opts) {
        const url = API_ROOT + path;
        const res = await fetch(url, opts);
        if (!res.ok) {
            let errorData;
            try {
                errorData = await res.json();
            } catch (e) {
                const txt = await res.text();
                errorData = { error: txt || res.statusText };
            }
            throw new Error(JSON.stringify(errorData));
        }
        return res.json();
    }
    
    // Show message
    function showMessage(text, type = 'error') {
        const existing = document.querySelector('.error-message, .success-message');
        if (existing) existing.remove();
        
        const message = document.createElement('div');
        message.className = type === 'error' ? 'error-message' : 'success-message';
        message.textContent = text;
        
        registerForm.insertBefore(message, registerForm.firstChild);
        
        setTimeout(() => message.remove(), 5000);
    }
    
    // Show field error
    function showFieldError(fieldId, message) {
        const errorElement = document.getElementById(fieldId + 'Error');
        if (errorElement) {
            errorElement.textContent = message;
            errorElement.classList.add('show');
        }
        
        const formGroup = document.getElementById(fieldId).closest('.form-group');
        if (formGroup) {
            formGroup.classList.add('error');
            formGroup.classList.remove('success');
        }
    }
    
    // Clear field error
    function clearFieldError(fieldId) {
        const errorElement = document.getElementById(fieldId + 'Error');
        if (errorElement) {
            errorElement.textContent = '';
            errorElement.classList.remove('show');
        }
        
        const formGroup = document.getElementById(fieldId).closest('.form-group');
        if (formGroup) {
            formGroup.classList.remove('error');
            formGroup.classList.add('success');
        }
    }
    
    // Clear all field errors
    function clearAllFieldErrors() {
        const fieldIds = ['regUsername', 'regEmail', 'regFullname', 'regTelephone', 'regProvince', 'regDistrict', 'regPassword', 'regConfirmPassword', 'agreeTerms'];
        fieldIds.forEach(fieldId => clearFieldError(fieldId));
    }
    
    // Clear all generic error messages
    function clearAllGenericMessages() {
        const existing = document.querySelector('.error-message, .success-message');
        if (existing) existing.remove();
    }
    
    // Validate username
    function validateUsername(username) {
        if (!username || username.trim() === '') {
            return 'Username is required';
        }
        
        if (username.length < 3) {
            return 'Username must be at least 3 characters';
        }
        
        if (username.length > 50) {
            return 'Username must be less than 50 characters';
        }
        
        if (!/^[a-zA-Z0-9_]+$/.test(username)) {
            return 'Username can only contain letters, numbers, and underscores';
        }
        
        // Check for reserved usernames
        const reservedUsernames = ['admin', 'administrator', 'root', 'system', 'api', 'test', 'user', 'guest', 'null', 'undefined'];
        if (reservedUsernames.includes(username.toLowerCase())) {
            return 'This username is reserved and cannot be used';
        }
        
        return null;
    }
    
    // Validate email
    function validateEmail(email) {
        if (!email || email.trim() === '') {
            return 'Email address is required';
        }
        
        if (email.length > 100) {
            return 'Email address must be less than 100 characters';
        }
        
        const emailPattern = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
        if (!emailPattern.test(email)) {
            return 'Please enter a valid email address';
        }
        
        // Check for common email providers
        const commonDomains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'icloud.com'];
        const domain = email.split('@')[1]?.toLowerCase();
        if (domain && !commonDomains.includes(domain) && !domain.includes('.')) {
            return 'Please enter a valid email address';
        }
        
        return null;
    }
    
    // Validate full name
    function validateFullName(fullname) {
        if (!fullname || fullname.trim() === '') {
            return 'Full name is required';
        }
        
        if (fullname.length < 2) {
            return 'Full name must be at least 2 characters';
        }
        
        if (fullname.length > 100) {
            return 'Full name must be less than 100 characters';
        }
        
        if (!/^[a-zA-Z\s\-'\.]+$/.test(fullname)) {
            return 'Full name can only contain letters, spaces, hyphens, apostrophes, and periods';
        }
        
        // Check for minimum words
        const words = fullname.trim().split(/\s+/);
        if (words.length < 2) {
            return 'Please enter your complete name (first and last name)';
        }
        
        return null;
    }
    
    // Validate phone number
    function validatePhone(telephone) {
        if (!telephone || telephone.trim() === '') {
            return 'Phone number is required';
        }
        
        // Remove all spaces and special characters except + and digits
        const cleanPhone = telephone.replace(/[^\d+]/g, '');
        
        // Check Rwanda phone number format
        const phonePattern = /^(\+250|0)[0-9]{9}$/;
        if (!phonePattern.test(cleanPhone)) {
            return 'Please enter a valid Rwanda phone number (+250XXXXXXXXX or 07XXXXXXXX)';
        }
        
        // Additional validation for specific prefixes
        if (cleanPhone.startsWith('0')) {
            const prefix = cleanPhone.substring(0, 3);
            const validPrefixes = ['078', '079', '072', '073', '074', '075', '076', '077'];
            if (!validPrefixes.includes(prefix)) {
                return 'Please enter a valid Rwanda mobile number';
            }
        }
        
        return null;
    }
    
    // Validate password
    function validatePassword(password) {
        if (!password || password === '') {
            return 'Password is required';
        }
        
        if (password.length < 8) {
            return 'Password must be at least 8 characters long';
        }
        
        if (password.length > 128) {
            return 'Password must be less than 128 characters';
        }
        
        // Check for at least one letter and one number
        if (!/[a-zA-Z]/.test(password)) {
            return 'Password must contain at least one letter';
        }
        
        if (!/[0-9]/.test(password)) {
            return 'Password must contain at least one number';
        }
        
        // Check for common weak passwords
        const weakPasswords = ['password', '123456', '12345678', 'qwerty', 'abc123', 'password123', 'admin', 'letmein'];
        if (weakPasswords.includes(password.toLowerCase())) {
            return 'This password is too common. Please choose a stronger password';
        }
        
        return null;
    }
    
    // Validate password confirmation
    function validatePasswordConfirmation(password, confirmPassword) {
        if (!confirmPassword || confirmPassword === '') {
            return 'Please confirm your password';
        }
        
        if (password !== confirmPassword) {
            return 'Passwords do not match';
        }
        
        return null;
    }
    
    // Validate province
    function validateProvince(province) {
        if (!province || province === '') {
            return 'Please select a province';
        }
        
        const validProvinces = ['Kigali', 'Eastern', 'Northern', 'Southern', 'Western'];
        if (!validProvinces.includes(province)) {
            return 'Please select a valid province';
        }
        
        return null;
    }
    
    // Validate district
    function validateDistrict(district, province) {
        if (!district || district === '') {
            return 'Please select a district';
        }
        
        if (!province || province === '') {
            return 'Please select a province first';
        }
        
        const validDistricts = getDistrictsForProvince(province);
        if (!validDistricts.includes(district)) {
            return 'Please select a valid district for the selected province';
        }
        
        return null;
    }
    
    // Validate terms agreement
    function validateTerms(agreeTerms) {
        if (!agreeTerms) {
            return 'You must agree to the Terms of Service and Privacy Policy';
        }
        
        return null;
    }
    
    // Get districts for province
    function getDistrictsForProvince(province) {
        const provinceDistricts = {
            'Kigali': ['Gasabo', 'Kicukiro', 'Nyarugenge'],
            'Eastern': ['Bugesera', 'Gatsibo', 'Kayonza', 'Kirehe', 'Ngoma', 'Nyagatare', 'Rwamagana'],
            'Northern': ['Burera', 'Gakenke', 'Gicumbi', 'Musanze', 'Rulindo'],
            'Southern': ['Gisagara', 'Huye', 'Kamonyi', 'Muhanga', 'Nyamagabe', 'Nyanza', 'Nyaruguru', 'Ruhango'],
            'Western': ['Karongi', 'Ngororero', 'Nyabihu', 'Nyamasheke', 'Rubavu', 'Rusizi', 'Rutsiro']
        };
        return provinceDistricts[province] || [];
    }
    
    // Calculate password strength
    function calculatePasswordStrength(password) {
        let score = 0;
        
        if (password.length >= 8) score += 1;
        if (password.length >= 12) score += 1;
        if (/[a-z]/.test(password)) score += 1;
        if (/[A-Z]/.test(password)) score += 1;
        if (/[0-9]/.test(password)) score += 1;
        if (/[^a-zA-Z0-9]/.test(password)) score += 1;
        
        if (score <= 2) return 'weak';
        if (score <= 4) return 'medium';
        return 'strong';
    }
    
    // Update password strength indicator
    function updatePasswordStrength(password) {
        const strength = calculatePasswordStrength(password);
        const strengthElement = document.querySelector('.password-strength');
        const strengthBar = document.querySelector('.password-strength-bar');
        
        if (strengthElement && strengthBar) {
            strengthElement.textContent = `Password strength: ${strength}`;
            strengthElement.className = `password-strength ${strength}`;
            strengthBar.className = `password-strength-bar ${strength}`;
        }
    }
    
    // Real-time validation
    function setupRealTimeValidation() {
        // Username validation
        document.getElementById('regUsername').addEventListener('blur', function() {
            const error = validateUsername(this.value);
            if (error) {
                showFieldError('regUsername', error);
            } else {
                clearFieldError('regUsername');
            }
        });
        
        // Email validation
        document.getElementById('regEmail').addEventListener('blur', function() {
            const error = validateEmail(this.value);
            if (error) {
                showFieldError('regEmail', error);
            } else {
                clearFieldError('regEmail');
            }
        });
        
        // Full name validation
        document.getElementById('regFullname').addEventListener('blur', function() {
            const error = validateFullName(this.value);
            if (error) {
                showFieldError('regFullname', error);
            } else {
                clearFieldError('regFullname');
            }
        });
        
        // Phone validation
        document.getElementById('regTelephone').addEventListener('blur', function() {
            const error = validatePhone(this.value);
            if (error) {
                showFieldError('regTelephone', error);
            } else {
                clearFieldError('regTelephone');
            }
        });
        
        // Province validation
        document.getElementById('regProvince').addEventListener('change', function() {
            const error = validateProvince(this.value);
            if (error) {
                showFieldError('regProvince', error);
            } else {
                clearFieldError('regProvince');
            }
        });
        
        // District validation
        document.getElementById('regDistrict').addEventListener('change', function() {
            const province = document.getElementById('regProvince').value;
            const error = validateDistrict(this.value, province);
            if (error) {
                showFieldError('regDistrict', error);
            } else {
                clearFieldError('regDistrict');
            }
        });
        
        // Password validation
        document.getElementById('regPassword').addEventListener('input', function() {
            updatePasswordStrength(this.value);
            const error = validatePassword(this.value);
            if (error) {
                showFieldError('regPassword', error);
            } else {
                clearFieldError('regPassword');
            }
        });
        
        // Password confirmation validation
        document.getElementById('regConfirmPassword').addEventListener('blur', function() {
            const password = document.getElementById('regPassword').value;
            const error = validatePasswordConfirmation(password, this.value);
            if (error) {
                showFieldError('regConfirmPassword', error);
            } else {
                clearFieldError('regConfirmPassword');
            }
        });
        
        // Terms validation
        document.getElementById('agreeTerms').addEventListener('change', function() {
            const error = validateTerms(this.checked);
            if (error) {
                showFieldError('agreeTerms', error);
            } else {
                clearFieldError('agreeTerms');
            }
        });
    }
    
    // Validate all fields
    function validateAllFields() {
        const username = document.getElementById('regUsername').value.trim();
        const email = document.getElementById('regEmail').value.trim();
        const fullname = document.getElementById('regFullname').value.trim();
        const telephone = document.getElementById('regTelephone').value.trim();
        const province = document.getElementById('regProvince').value;
        const district = document.getElementById('regDistrict').value;
        const password = document.getElementById('regPassword').value;
        const confirmPassword = document.getElementById('regConfirmPassword').value;
        const agreeTerms = document.getElementById('agreeTerms').checked;
        
        validationErrors = {};
        
        // Validate each field
        const usernameError = validateUsername(username);
        if (usernameError) validationErrors.username = usernameError;
        
        const emailError = validateEmail(email);
        if (emailError) validationErrors.email = emailError;
        
        const fullnameError = validateFullName(fullname);
        if (fullnameError) validationErrors.fullname = fullnameError;
        
        const telephoneError = validatePhone(telephone);
        if (telephoneError) validationErrors.telephone = telephoneError;
        
        const provinceError = validateProvince(province);
        if (provinceError) validationErrors.province = provinceError;
        
        const districtError = validateDistrict(district, province);
        if (districtError) validationErrors.district = districtError;
        
        const passwordError = validatePassword(password);
        if (passwordError) validationErrors.password = passwordError;
        
        const confirmPasswordError = validatePasswordConfirmation(password, confirmPassword);
        if (confirmPasswordError) validationErrors.confirmPassword = confirmPasswordError;
        
        const termsError = validateTerms(agreeTerms);
        if (termsError) validationErrors.terms = termsError;
        
        // Show all errors
        Object.keys(validationErrors).forEach(field => {
            showFieldError('reg' + field.charAt(0).toUpperCase() + field.slice(1), validationErrors[field]);
        });
        
        return Object.keys(validationErrors).length === 0;
    }
    
    // Redirect to main app
    function redirectToApp(account = null) {
        if (account) {
            localStorage.setItem('aimhsa_account', account);
        }
        window.location.href = '/index.html';
    }
    
    // Registration form submission
    registerForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if (isSubmitting) return;
        
        // Clear previous errors
        clearAllFieldErrors();
        clearAllGenericMessages();
        
        // Validate all fields
        if (!validateAllFields()) {
            showMessage('Please correct the errors below');
            return;
        }
        
        const username = document.getElementById('regUsername').value.trim();
        const email = document.getElementById('regEmail').value.trim();
        const fullname = document.getElementById('regFullname').value.trim();
        const telephone = document.getElementById('regTelephone').value.trim();
        const province = document.getElementById('regProvince').value;
        const district = document.getElementById('regDistrict').value;
        const password = document.getElementById('regPassword').value;
        
        isSubmitting = true;
        registerBtn.disabled = true;
        registerBtn.textContent = 'Creating account...';
        
        try {
            const response = await api('/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    username, 
                    email, 
                    fullname, 
                    telephone, 
                    province, 
                    district, 
                    password 
                })
            });
            
            showMessage('Account created successfully! Redirecting...', 'success');
            setTimeout(() => redirectToApp(username), 1500);
        } catch (err) {
            console.log('Registration error:', err);
            
            // Parse server error response for specific field errors
            let serverErrors = {};
            let genericError = 'Registration failed. Please check the errors below.';
            
            try {
                // Try to parse JSON error response
                const errorData = JSON.parse(err.message);
                console.log('Parsed error data:', errorData);
                
                if (errorData.errors) {
                    serverErrors = errorData.errors;
                    console.log('Server errors:', serverErrors);
                } else if (errorData.error) {
                    genericError = errorData.error;
                }
            } catch (parseError) {
                console.log('Could not parse error as JSON:', parseError);
                // If not JSON, check for specific error patterns
                const errorText = err.message.toLowerCase();
                
                if (errorText.includes('username')) {
                    if (errorText.includes('already exists') || errorText.includes('taken')) {
                        serverErrors.username = 'This username is already taken. Please choose another.';
                    } else if (errorText.includes('invalid')) {
                        serverErrors.username = 'Invalid username format.';
                    }
                } else if (errorText.includes('email')) {
                    if (errorText.includes('already exists') || errorText.includes('taken')) {
                        serverErrors.email = 'This email is already registered. Please use a different email.';
                    } else if (errorText.includes('invalid')) {
                        serverErrors.email = 'Invalid email format.';
                    }
                } else if (errorText.includes('phone') || errorText.includes('telephone')) {
                    serverErrors.telephone = 'Invalid phone number format.';
                } else if (errorText.includes('password')) {
                    serverErrors.password = 'Password does not meet requirements.';
                } else if (errorText.includes('province')) {
                    serverErrors.province = 'Please select a valid province.';
                } else if (errorText.includes('district')) {
                    serverErrors.district = 'Please select a valid district.';
                }
            }
            
            // Show specific field errors
            if (Object.keys(serverErrors).length > 0) {
                console.log('Showing field errors:', serverErrors);
                
                // Clear any existing generic error messages
                clearAllGenericMessages();
                
                Object.keys(serverErrors).forEach(field => {
                    const fieldId = 'reg' + field.charAt(0).toUpperCase() + field.slice(1);
                    console.log(`Showing error for field ${fieldId}: ${serverErrors[field]}`);
                    showFieldError(fieldId, serverErrors[field]);
                });
                
                // Don't show generic message if we have specific field errors
                return; // Exit without showing generic message
            }
            
            // Only show generic message if no specific field errors
            console.log('Showing generic error:', genericError);
            showMessage(genericError);
        } finally {
            isSubmitting = false;
            registerBtn.disabled = false;
            registerBtn.textContent = 'Create Account';
        }
    });
    
    // Province/District mapping for Rwanda
    const provinceDistricts = {
        'Kigali': ['Gasabo', 'Kicukiro', 'Nyarugenge'],
        'Eastern': ['Bugesera', 'Gatsibo', 'Kayonza', 'Kirehe', 'Ngoma', 'Nyagatare', 'Rwamagana'],
        'Northern': ['Burera', 'Gakenke', 'Gicumbi', 'Musanze', 'Rulindo'],
        'Southern': ['Gisagara', 'Huye', 'Kamonyi', 'Muhanga', 'Nyamagabe', 'Nyanza', 'Nyaruguru', 'Ruhango'],
        'Western': ['Karongi', 'Ngororero', 'Nyabihu', 'Nyamasheke', 'Rubavu', 'Rusizi', 'Rutsiro']
    };
    
    // Handle province change to filter districts
    document.getElementById('regProvince').addEventListener('change', function() {
        const province = this.value;
        const districtSelect = document.getElementById('regDistrict');
        
        // Clear existing options except the first one
        districtSelect.innerHTML = '<option value="">Select District</option>';
        
        if (province && provinceDistricts[province]) {
            provinceDistricts[province].forEach(district => {
                const option = document.createElement('option');
                option.value = district;
                option.textContent = district;
                districtSelect.appendChild(option);
            });
        }
        
        // Clear district error when province changes
        clearFieldError('regDistrict');
    });
    
    // Initialize real-time validation
    setupRealTimeValidation();
    
    // Check if already logged in
    const account = localStorage.getItem('aimhsa_account');
    if (account && account !== 'null') {
        redirectToApp(account);
    }
})();