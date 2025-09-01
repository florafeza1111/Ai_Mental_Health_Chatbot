(() => {
    const API_ROOT = "http://localhost:5057"; // Flask API server
    
    // Elements
    const registerForm = document.getElementById('registerForm');
    const registerBtn = document.getElementById('registerBtn');
    
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
        
        registerForm.insertBefore(message, registerForm.firstChild);
        
        setTimeout(() => message.remove(), 5000);
    }
    
    // Redirect to main app
    function redirectToApp(account = null) {
        if (account) {
            localStorage.setItem('aimhsa_account', account);
        }
        window.location.href = '/';
    }
    
    // Registration form submission
    registerForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
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
            await api('/register', {
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
    });
    
    // Check if already logged in
    const account = localStorage.getItem('aimhsa_account');
    if (account && account !== 'null') {
        redirectToApp(account);
    }
})();
