(() => {
    const API_ROOT = "http://localhost:5057"; // Flask API server
    
    // Elements
    const loginForm = document.getElementById('loginForm');
    const signInBtn = document.getElementById('signInBtn');
    const anonBtn = document.getElementById('anonBtn');
    
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
        
        loginForm.insertBefore(message, loginForm.firstChild);
        
        setTimeout(() => message.remove(), 5000);
    }
    
    // Redirect to main app
    function redirectToApp(account = null) {
        if (account) {
            localStorage.setItem('aimhsa_account', account);
        }
        window.location.href = '/';
    }
    
    // Login form submission
    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const username = document.getElementById('loginUsername').value.trim();
        const password = document.getElementById('loginPassword').value;
        
        if (!username || !password) {
            showMessage('Please enter both username and password');
            return;
        }
        
        signInBtn.disabled = true;
        signInBtn.textContent = 'Signing in...';
        
        try {
            const res = await api('/login', {
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
    });
    
    // Anonymous access
    anonBtn.addEventListener('click', () => {
        localStorage.setItem('aimhsa_account', 'null');
        redirectToApp();
    });
    
    // Check if already logged in
    const account = localStorage.getItem('aimhsa_account');
    if (account && account !== 'null') {
        redirectToApp(account);
    }
})();
