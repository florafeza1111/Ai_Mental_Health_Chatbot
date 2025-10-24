(() => {
    const API_ROOT = ""; // Same port as frontend
    
    // Elements
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const authForm = document.getElementById('authForm');
    const signInBtn = document.getElementById('signInBtn');
    const registerBtn = document.getElementById('registerBtn');
    const anonBtn = document.getElementById('anonBtn');
    
    // Tab switching
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.dataset.tab;
            
            // Update active tab
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Show target content
            tabContents.forEach(content => {
                if (content.id === targetTab) {
                    content.classList.remove('hidden');
                } else {
                    content.classList.add('hidden');
                }
            });
        });
    });
    
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
        
        authForm.insertBefore(message, authForm.firstChild);
        
        setTimeout(() => message.remove(), 5000);
    }
    
    // Redirect to main app
    function redirectToApp(account = null) {
        if (account) {
            localStorage.setItem('aimhsa_account', account);
        }
        window.location.href = 'index.html';
    }
    
    // Form submission
    authForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const activeTab = document.querySelector('.tab-content:not(.hidden)').id;
        
        if (activeTab === 'signin') {
            const username = document.getElementById('loginUsername').value.trim();
            const password = document.getElementById('loginPassword').value;
            
            if (!username || !password) {
                showMessage('Please enter both username and password');
                return;
            }
            
            signInBtn.disabled = true;
            signInBtn.textContent = 'Signing in...';
            
            try {
                const res = await api('/api/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, password })
                });
                
                showMessage('Successfully signed in!', 'success');
                setTimeout(() => redirectToApp(res.account || username), 1000);
            } catch (err) {
                showMessage('Invalid username or password');
            } finally {
                signInBtn.disabled = false;
                signInBtn.textContent = 'Sign In';
            }
        } else {
            const username = document.getElementById('regUsername').value.trim();
            const password = document.getElementById('regPassword').value;
            const confirmPassword = document.getElementById('regConfirmPassword').value;
            
            if (!username || !password || !confirmPassword) {
                showMessage('Please fill in all fields');
                return;
            }
            
            if (password !== confirmPassword) {
                showMessage('Passwords do not match');
                return;
            }
            
            if (password.length < 6) {
                showMessage('Password must be at least 6 characters');
                return;
            }
            
            registerBtn.disabled = true;
            registerBtn.textContent = 'Creating account...';
            
            try {
                await api('/api/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, password })
                });
                
                showMessage('Account created successfully!', 'success');
                setTimeout(() => redirectToApp(username), 1000);
            } catch (err) {
                showMessage('Username already exists or registration failed');
            } finally {
                registerBtn.disabled = false;
                registerBtn.textContent = 'Create Account';
            }
        }
    });
    
    // Anonymous access
    anonBtn.addEventListener('click', () => {
        localStorage.removeItem('aimhsa_account');
        redirectToApp();
    });
    
    // Check if already logged in
    const account = localStorage.getItem('aimhsa_account');
    if (account) {
        redirectToApp(account);
    }
})();
