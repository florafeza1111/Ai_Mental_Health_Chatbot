(() => {
    const API_ROOT = "http://localhost:5057"; // Flask API server
    
    // Elements
    const loginForm = document.getElementById('loginForm');
    const signInBtn = document.getElementById('signInBtn');
    const anonBtn = document.getElementById('anonBtn');
    const usernameInput = document.getElementById('loginUsername');
    const usernameHint = document.getElementById('usernameHint');
    const passwordInput = document.getElementById('loginPassword');
    const togglePasswordBtn = document.getElementById('togglePassword');
    const meter = document.getElementById('passwordMeter');
    const meterBar = document.getElementById('passwordMeterBar');
    const capsLockIndicator = document.getElementById('capsLockIndicator');
    const rememberMe = document.getElementById('rememberMe');
    const forgotLink = document.getElementById('forgotLink');
    // Forgot password elements
    const fpModal = document.getElementById('fpModal');
    const fpBackdrop = document.getElementById('fpBackdrop');
    const fpClose = document.getElementById('fpClose');
    const fpUsername = document.getElementById('fpUsername');
    const fpRequestBtn = document.getElementById('fpRequestBtn');
    const fpStep1 = document.querySelector('.fp-step-1');
    const fpStep2 = document.querySelector('.fp-step-2');
    const fpCode = document.getElementById('fpCode');
    const fpNewPassword = document.getElementById('fpNewPassword');
    const fpApplyBtn = document.getElementById('fpApplyBtn');
    const fpResendBtn = document.getElementById('fpResendBtn');
    const fpMessage = document.getElementById('fpMessage');
    // MFA elements
    const mfaModal = document.getElementById('mfaModal');
    const mfaBackdrop = document.getElementById('mfaBackdrop');
    const mfaClose = document.getElementById('mfaClose');
    const mfaCode = document.getElementById('mfaCode');
    const mfaVerifyBtn = document.getElementById('mfaVerifyBtn');
    const mfaResendBtn = document.getElementById('mfaResendBtn');
    const mfaMessage = document.getElementById('mfaMessage');
    
    // API helper
    async function api(path, opts) {
        const url = API_ROOT + path;
        const res = await fetch(url, opts);
        if (!res.ok) {
            const txt = await res.text();
            throw new Error(txt || res.statusText);
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
        message.style.cssText = `
            padding: 12px 16px;
            margin: 16px 0;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 500;
            ${type === 'error' ? 
                'background: rgba(239, 68, 68, 0.1); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.2);' : 
                'background: rgba(16, 185, 129, 0.1); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.2);'
            }
        `;
        
        loginForm.insertBefore(message, loginForm.firstChild);
        
        setTimeout(() => message.remove(), 5000);
    }
    
    // Redirect to main app
    function redirectToApp(account = null) {
        if (account) {
            localStorage.setItem('aimhsa_account', account);
        }
        window.location.href = '/index.html';
    }

    // Utility: simple password strength score (0..4)
    function getPasswordStrengthScore(pw) {
        let score = 0;
        if (!pw) return 0;
        if (pw.length >= 8) score++;
        if (/[A-Z]/.test(pw)) score++;
        if (/[a-z]/.test(pw)) score++;
        if (/[0-9]/.test(pw)) score++;
        if (/[^A-Za-z0-9]/.test(pw)) score++;
        return Math.min(score, 4);
    }

    function updatePasswordMeter() {
        const pw = passwordInput.value;
        const score = getPasswordStrengthScore(pw);
        const pct = (score / 4) * 100;
        meterBar.style.width = pct + '%';
        let color = '#ef4444';
        if (score >= 3) color = '#f59e0b';
        if (score >= 4) color = '#10b981';
        meterBar.style.background = color;
        meter.setAttribute('aria-hidden', pw ? 'false' : 'true');
    }

    // Toggle password visibility
    togglePasswordBtn?.addEventListener('click', () => {
        const isPassword = passwordInput.type === 'password';
        passwordInput.type = isPassword ? 'text' : 'password';
        togglePasswordBtn.setAttribute('aria-pressed', String(isPassword));
    });

    // Caps lock indicator
    function handleKeyEventForCaps(e) {
        if (typeof e.getModifierState === 'function') {
            const on = e.getModifierState('CapsLock');
            if (on) {
                capsLockIndicator?.removeAttribute('hidden');
            } else {
                capsLockIndicator?.setAttribute('hidden', '');
            }
        }
    }
    passwordInput.addEventListener('keydown', handleKeyEventForCaps);
    passwordInput.addEventListener('keyup', handleKeyEventForCaps);
    passwordInput.addEventListener('input', () => {
        updatePasswordMeter();
    });

    // Remember me: prefill username
    const savedUsername = localStorage.getItem('aimhsa_saved_username');
    if (savedUsername) {
        usernameInput.value = savedUsername;
        rememberMe.checked = true;
    }

    // Username basic validation hint
    usernameInput.addEventListener('input', () => {
        const v = usernameInput.value.trim();
        if (!v) {
            usernameHint.textContent = '';
            return;
        }
        usernameHint.textContent = v.length < 3 ? 'Username is too short' : '';
    });

    // Forgot password (client-side placeholder)
    forgotLink.addEventListener('click', (e) => {
        e.preventDefault();
        openFpModal();
    });

    // Simple cooldown after repeated failures
    const COOLDOWN_KEY = 'aimhsa_login_cooldown_until';
    function isInCooldown() {
        const until = Number(localStorage.getItem(COOLDOWN_KEY) || '0');
        return Date.now() < until;
    }
    function applyCooldown(seconds) {
        const until = Date.now() + seconds * 1000;
        localStorage.setItem(COOLDOWN_KEY, String(until));
    }
    function getCooldownRemainingMs() {
        const until = Number(localStorage.getItem(COOLDOWN_KEY) || '0');
        return Math.max(0, until - Date.now());
    }
    function updateCooldownUI() {
        const remaining = getCooldownRemainingMs();
        if (remaining > 0) {
            signInBtn.disabled = true;
            const secs = Math.ceil(remaining / 1000);
            signInBtn.textContent = `Try again in ${secs}s`;
        } else {
            signInBtn.disabled = false;
            signInBtn.textContent = 'Sign In';
        }
    }
    if (isInCooldown()) {
        updateCooldownUI();
        const timer = setInterval(() => {
            updateCooldownUI();
            if (!isInCooldown()) clearInterval(timer);
        }, 500);
    }
    
    // Login form submission
    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (isInCooldown()) {
            updateCooldownUI();
            return;
        }
        
        const username = usernameInput.value.trim();
        const password = passwordInput.value;
        
        if (!username || !password) {
            showMessage('Please enter both username and password');
            return;
        }
        if (rememberMe.checked) {
            localStorage.setItem('aimhsa_saved_username', username);
        } else {
            localStorage.removeItem('aimhsa_saved_username');
        }
        
        signInBtn.disabled = true;
        signInBtn.textContent = 'Signing in...';
        
        try {
            // Try user login first
            try {
                console.log('Trying user login for:', username);
                const res = await api('/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, password })
                });
                
                if (res && res.mfa_required) {
                    // Launch MFA modal
                    openMfaModal({ flow: 'user', username, token: res.mfa_token });
                } else {
                    showMessage('Successfully signed in as user!', 'success');
                    setTimeout(() => redirectToApp(res.account || username), 1000);
                }
                return;
            } catch (userErr) {
                console.log('User login failed:', userErr.message);
                console.log('Trying professional login...');
            }
            
            // Try professional login
            try {
                console.log('Trying professional login for:', username);
                const res = await api('/professional/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, password })
                });
                
                // Store professional data
                localStorage.setItem('aimhsa_professional', JSON.stringify(res));
                if (res && res.mfa_required) {
                    openMfaModal({ flow: 'professional', username, token: res.mfa_token });
                } else {
                    showMessage('Successfully signed in as professional!', 'success');
                    setTimeout(() => {
                        window.location.href = '/professional_dashboard.html';
                    }, 1000);
                }
                return;
            } catch (profErr) {
                console.log('Professional login failed:', profErr.message);
                console.log('Trying admin login...');
            }
            
            // Try admin login
            try {
                console.log('Trying admin login for:', username);
                const res = await api('/admin/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, password })
                });
                
                console.log('Admin login successful:', res);
                // Store admin data
                localStorage.setItem('aimhsa_admin', JSON.stringify(res));
                if (res && res.mfa_required) {
                    openMfaModal({ flow: 'admin', username, token: res.mfa_token });
                } else {
                    showMessage('Successfully signed in as admin!', 'success');
                    setTimeout(() => {
                        window.location.href = '/admin_dashboard.html';
                    }, 1000);
                }
                return;
            } catch (adminErr) {
                console.log('Admin login failed:', adminErr.message);
            }
            
            // If all login attempts failed
            showMessage('Invalid username or password. Please check your credentials.');
            // backoff: 10s cooldown after aggregated failure
            applyCooldown(10);
            updateCooldownUI();
            
        } catch (err) {
            console.error('Login error:', err);
            showMessage('Login failed. Please try again.');
        } finally {
            signInBtn.disabled = false;
            signInBtn.textContent = 'Sign In';
        }
    });
    
    // Anonymous access
    anonBtn.addEventListener('click', () => {
        localStorage.setItem('aimhsa_account', 'null');
        window.location.href = '/index.html';
    });
    
    // Check if already logged in
    const account = localStorage.getItem('aimhsa_account');
    if (account && account !== 'null') {
        redirectToApp(account);
    }

    // --- MFA helpers ---
    function openMfaModal(context) {
        mfaModal.classList.add('open');
        mfaModal.setAttribute('aria-hidden', 'false');
        mfaCode.value = '';
        mfaMessage.textContent = '';
        mfaCode.focus();

        function close() {
            mfaModal.classList.remove('open');
            mfaModal.setAttribute('aria-hidden', 'true');
        }

        function onClose() {
            close();
            cleanup();
        }

        async function verify() {
            const code = mfaCode.value.trim();
            if (!/^[0-9]{6}$/.test(code)) {
                mfaMessage.textContent = 'Please enter a valid 6-digit code.';
                return;
            }
            mfaVerifyBtn.disabled = true;
            mfaVerifyBtn.textContent = 'Verifying...';
            try {
                const endpoint =
                    context.flow === 'admin' ? '/admin/mfa/verify' :
                    context.flow === 'professional' ? '/professional/mfa/verify' :
                    '/mfa/verify';
                const res = await api(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username: context.username, code, token: context.token })
                });
                mfaMessage.textContent = 'MFA verified. Redirecting...';
                setTimeout(() => {
                    if (context.flow === 'admin') {
                        window.location.href = '/admin_dashboard.html';
                    } else if (context.flow === 'professional') {
                        window.location.href = '/professional_dashboard.html';
                    } else {
                        redirectToApp(context.username);
                    }
                }, 600);
            } catch (err) {
                mfaMessage.textContent = 'Invalid or expired code. Please try again.';
            } finally {
                mfaVerifyBtn.disabled = false;
                mfaVerifyBtn.textContent = 'Verify';
            }
        }

        async function resend() {
            try {
                const endpoint =
                    context.flow === 'admin' ? '/admin/mfa/resend' :
                    context.flow === 'professional' ? '/professional/mfa/resend' :
                    '/mfa/resend';
                await api(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username: context.username, token: context.token })
                });
                mfaMessage.textContent = 'Code resent.';
            } catch (err) {
                mfaMessage.textContent = 'Could not resend code. Try again later.';
            }
        }

        function cleanup() {
            mfaBackdrop.removeEventListener('click', onClose);
            mfaClose.removeEventListener('click', onClose);
            mfaVerifyBtn.removeEventListener('click', verify);
            mfaResendBtn.removeEventListener('click', resend);
        }

        mfaBackdrop.addEventListener('click', onClose);
        mfaClose.addEventListener('click', onClose);
        mfaVerifyBtn.addEventListener('click', verify);
        mfaResendBtn.addEventListener('click', resend);
    }
})();

    // --- Forgot Password helpers ---
    function openFpModal() {
        fpModal.classList.add('open');
        fpModal.setAttribute('aria-hidden', 'false');
        fpMessage.textContent = '';
        fpStep1.classList.remove('hidden');
        fpStep2.classList.add('hidden');
        fpUsername.value = usernameInput.value.trim();
        setTimeout(() => fpUsername.focus(), 0);

        function close() {
            fpModal.classList.remove('open');
            fpModal.setAttribute('aria-hidden', 'true');
        }

        function onClose() {
            close();
            cleanup();
        }

        async function requestCode() {
            const u = fpUsername.value.trim();
            if (!u) {
                fpMessage.textContent = 'Please enter your username.';
                return;
            }
            fpRequestBtn.disabled = true;
            fpRequestBtn.textContent = 'Sending...';
            try {
                const res = await api('/forgot_password', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username: u })
                });
                // For demo, server may return token; show a subtle message
                fpMessage.textContent = res.token ? `Reset code: ${res.token} (demo)` : 'If the username exists, a code was sent.';
                fpStep1.classList.add('hidden');
                fpStep2.classList.remove('hidden');
                fpCode.value = '';
                fpNewPassword.value = '';
                setTimeout(() => fpCode.focus(), 0);
            } catch (err) {
                fpMessage.textContent = 'Could not initiate reset. Try again later.';
            } finally {
                fpRequestBtn.disabled = false;
                fpRequestBtn.textContent = 'Send code';
            }
        }

        async function applyReset() {
            const u = fpUsername.value.trim();
            const code = fpCode.value.trim();
            const newPw = fpNewPassword.value;
            if (!/^[0-9A-Z]{6}$/.test(code)) {
                fpMessage.textContent = 'Please enter the 6-character code.';
                return;
            }
            if (newPw.length < 6) {
                fpMessage.textContent = 'New password must be at least 6 characters.';
                return;
            }
            fpApplyBtn.disabled = true;
            fpApplyBtn.textContent = 'Resetting...';
            try {
                await api('/reset_password', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username: u, token: code, new_password: newPw })
                });
                fpMessage.textContent = 'Password updated. You can now sign in.';
                setTimeout(() => {
                    onClose();
                    usernameInput.value = u;
                    passwordInput.focus();
                }, 800);
            } catch (err) {
                fpMessage.textContent = 'Invalid code or error updating password.';
            } finally {
                fpApplyBtn.disabled = false;
                fpApplyBtn.textContent = 'Reset password';
            }
        }

        async function resendCode() {
            // Reuse forgot_password to resend
            requestCode();
        }

        function cleanup() {
            fpBackdrop.removeEventListener('click', onClose);
            fpClose.removeEventListener('click', onClose);
            fpRequestBtn.removeEventListener('click', requestCode);
            fpApplyBtn.removeEventListener('click', applyReset);
            fpResendBtn.removeEventListener('click', resendCode);
        }

        fpBackdrop.addEventListener('click', onClose);
        fpClose.addEventListener('click', onClose);
        fpRequestBtn.addEventListener('click', requestCode);
        fpApplyBtn.addEventListener('click', applyReset);
        fpResendBtn.addEventListener('click', resendCode);
    }
