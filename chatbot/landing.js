// AIMHSA Landing Page JavaScript
(() => {
    'use strict';

    // Smooth scrolling for navigation links
    function initSmoothScrolling() {
        const navLinks = document.querySelectorAll('a[href^="#"]');
        
        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const targetId = link.getAttribute('href');
                const targetElement = document.querySelector(targetId);
                
                if (targetElement) {
                    const offsetTop = targetElement.offsetTop - 80; // Account for fixed navbar
                    window.scrollTo({
                        top: offsetTop,
                        behavior: 'smooth'
                    });
                }
            });
        });
    }

    // Navbar scroll effect
    function initNavbarScroll() {
        const navbar = document.querySelector('.navbar');
        let lastScrollY = window.scrollY;

        window.addEventListener('scroll', () => {
            const currentScrollY = window.scrollY;
            
            if (currentScrollY > 100) {
                navbar.style.background = 'rgba(15, 23, 42, 0.98)';
                navbar.style.backdropFilter = 'blur(15px)';
            } else {
                navbar.style.background = 'rgba(15, 23, 42, 0.95)';
                navbar.style.backdropFilter = 'blur(10px)';
            }

            // Hide/show navbar on scroll
            if (currentScrollY > lastScrollY && currentScrollY > 200) {
                navbar.style.transform = 'translateY(-100%)';
            } else {
                navbar.style.transform = 'translateY(0)';
            }

            lastScrollY = currentScrollY;
        });
    }

    // Intersection Observer for animations
    function initScrollAnimations() {
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate-in');
                }
            });
        }, observerOptions);

        // Observe elements for animation
        const animateElements = document.querySelectorAll('.feature-card, .stat-item, .about-text, .about-visual');
        animateElements.forEach(el => {
            observer.observe(el);
        });
    }

    // Typing animation for chat preview
    function initTypingAnimation() {
        const typingIndicator = document.querySelector('.typing-indicator');
        if (!typingIndicator) return;

        // Show typing indicator after a delay
        setTimeout(() => {
            typingIndicator.style.display = 'flex';
            
            // Hide typing indicator and show new message
            setTimeout(() => {
                typingIndicator.style.display = 'none';
                addNewMessage();
            }, 2000);
        }, 3000);
    }

    // Add new message to chat preview
    function addNewMessage() {
        const chatMessages = document.querySelector('.chat-messages');
        if (!chatMessages) return;

        const newMessage = document.createElement('div');
        newMessage.className = 'message bot-message';
        newMessage.innerHTML = `
            <div class="message-content">
                <p>I'm here to listen and support you. What's on your mind today?</p>
            </div>
        `;

        chatMessages.appendChild(newMessage);
        
        // Scroll to bottom of chat
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Parallax effect for hero background
    function initParallax() {
        const heroPattern = document.querySelector('.hero-pattern');
        if (!heroPattern) return;

        window.addEventListener('scroll', () => {
            const scrolled = window.pageYOffset;
            const rate = scrolled * -0.5;
            heroPattern.style.transform = `translateY(${rate}px)`;
        });
    }

    // Counter animation for stats
    function initCounterAnimation() {
        const counters = document.querySelectorAll('.stat-number');
        
        const animateCounter = (counter) => {
            const target = counter.textContent;
            const isNumeric = !isNaN(target);
            
            if (!isNumeric) return;
            
            const increment = parseInt(target) / 50;
            let current = 0;
            
            const updateCounter = () => {
                if (current < parseInt(target)) {
                    current += increment;
                    counter.textContent = Math.ceil(current);
                    requestAnimationFrame(updateCounter);
                } else {
                    counter.textContent = target;
                }
            };
            
            updateCounter();
        };

        const counterObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    animateCounter(entry.target);
                    counterObserver.unobserve(entry.target);
                }
            });
        }, { threshold: 0.5 });

        counters.forEach(counter => {
            counterObserver.observe(counter);
        });
    }

    // Mobile menu toggle (if needed for mobile)
    function initMobileMenu() {
        const navLinks = document.querySelector('.nav-links');
        const navToggle = document.createElement('button');
        navToggle.className = 'nav-toggle';
        navToggle.innerHTML = '<i class="fas fa-bars"></i>';
        navToggle.style.display = 'none';
        
        const navContainer = document.querySelector('.nav-container');
        navContainer.appendChild(navToggle);

        navToggle.addEventListener('click', () => {
            navLinks.classList.toggle('active');
        });

        // Show/hide mobile menu based on screen size
        const checkScreenSize = () => {
            if (window.innerWidth <= 768) {
                navToggle.style.display = 'block';
                navLinks.style.display = navLinks.classList.contains('active') ? 'flex' : 'none';
            } else {
                navToggle.style.display = 'none';
                navLinks.style.display = 'flex';
                navLinks.classList.remove('active');
            }
        };

        window.addEventListener('resize', checkScreenSize);
        checkScreenSize();
    }

    // Form validation for CTA buttons
    function initFormValidation() {
        const ctaButtons = document.querySelectorAll('.btn-primary, .btn-secondary');
        
        ctaButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                // Add click animation
                button.style.transform = 'scale(0.95)';
                setTimeout(() => {
                    button.style.transform = 'scale(1)';
                }, 150);
            });
        });
    }

    // Loading animation
    function initLoadingAnimation() {
        // Add loading class to body
        document.body.classList.add('loading');
        
        // Remove loading class when page is fully loaded
        window.addEventListener('load', () => {
            document.body.classList.remove('loading');
        });
    }

    // Emergency contact click tracking and functionality
    function initEmergencyTracking() {
        const emergencyContacts = document.querySelectorAll('.contact-item');

        emergencyContacts.forEach(contact => {
            contact.addEventListener('click', () => {
                // Track emergency contact clicks
                console.log('Emergency contact clicked:', contact.textContent);

                // Add visual feedback
                contact.style.background = 'rgba(255, 255, 255, 0.1)';
                setTimeout(() => {
                    contact.style.background = '';
                }, 200);
            });
        });
    }

    // Emergency action functions
    function callEmergency(number) {
        // Try to initiate phone call
        if (typeof window !== 'undefined' && window.location) {
            // Mobile devices will handle tel: links
            window.location.href = `tel:${number}`;
        } else {
            // Fallback: show alert with number
            alert(`Please call emergency number: ${number}\n\nIf you're unable to call, please seek help from someone nearby or go to the nearest emergency room.`);
        }

        // Track emergency call attempt
        console.log('Emergency call initiated:', number);
        logEmergencyAction('call', number);
    }

    function sendEmergencySMS() {
        const message = "HELP - I need mental health support";
        const number = "105";

        // Try to open SMS app
        if (typeof window !== 'undefined' && window.location) {
            const smsUrl = `sms:${number}?body=${encodeURIComponent(message)}`;
            window.location.href = smsUrl;
        } else {
            alert(`Please send SMS to ${number} with message: "${message}"`);
        }

        // Track emergency SMS attempt
        console.log('Emergency SMS initiated');
        logEmergencyAction('sms', number);
    }

    function findNearbyHelp() {
        // Check if geolocation is available
        if ("geolocation" in navigator) {
            navigator.geolocation.getCurrentPosition(
                (position) => {
                    const lat = position.coords.latitude;
                    const lng = position.coords.longitude;

                    // Open Google Maps with search for mental health facilities
                    const query = encodeURIComponent("mental health hospital Rwanda");
                    const mapsUrl = `https://www.google.com/maps/search/${query}/@${lat},${lng},15z`;

                    window.open(mapsUrl, '_blank');

                    // Track location access
                    logEmergencyAction('location', `${lat},${lng}`);
                },
                (error) => {
                    console.error('Geolocation error:', error);
                    // Fallback: show general Rwanda mental health resources
                    const fallbackUrl = "https://www.google.com/maps/search/mental+health+hospital/@-1.9403,29.8739,8z/data=!3m1!4b1";
                    window.open(fallbackUrl, '_blank');
                    logEmergencyAction('location_fallback', 'Rwanda');
                },
                {
                    enableHighAccuracy: true,
                    timeout: 10000,
                    maximumAge: 300000 // 5 minutes
                }
            );
        } else {
            // Geolocation not available
            alert("Location services are not available. Showing general Rwanda mental health resources.");
            const fallbackUrl = "https://www.google.com/maps/search/mental+health+hospital/@-1.9403,29.8739,8z/data=!3m1!4b1";
            window.open(fallbackUrl, '_blank');
            logEmergencyAction('location_unavailable', 'Rwanda');
        }
    }

    function logEmergencyAction(action, details) {
        // Log emergency actions for analytics and support
        const emergencyLog = {
            action: action,
            details: details,
            timestamp: new Date().toISOString(),
            userAgent: navigator.userAgent,
            referrer: document.referrer || 'direct'
        };

        // Store in localStorage for now (could be sent to server later)
        const logs = JSON.parse(localStorage.getItem('aimhsa_emergency_logs') || '[]');
        logs.push(emergencyLog);

        // Keep only last 10 logs
        if (logs.length > 10) {
            logs.shift();
        }

        localStorage.setItem('aimhsa_emergency_logs', JSON.stringify(logs));

        // Could send to analytics server here
        console.log('Emergency action logged:', emergencyLog);
    }

    // Community interaction tracking
    function initCommunityTracking() {
        const communityCards = document.querySelectorAll('.community-card');

        communityCards.forEach(card => {
            card.addEventListener('click', () => {
                const cardType = card.querySelector('h3').textContent.toLowerCase().replace(' ', '_');
                console.log('Community card clicked:', cardType);

                // Add ripple effect
                const ripple = document.createElement('div');
                ripple.className = 'ripple-effect';
                ripple.style.position = 'absolute';
                ripple.style.borderRadius = '50%';
                ripple.style.background = 'rgba(255, 255, 255, 0.3)';
                ripple.style.transform = 'scale(0)';
                ripple.style.animation = 'ripple 0.6s linear';
                ripple.style.left = '50%';
                ripple.style.top = '50%';
                ripple.style.width = '20px';
                ripple.style.height = '20px';
                ripple.style.marginLeft = '-10px';
                ripple.style.marginTop = '-10px';

                card.appendChild(ripple);

                setTimeout(() => {
                    ripple.remove();
                }, 600);
            });
        });
    }

    // Resource interaction tracking
    function initResourceTracking() {
        const resourceLinks = document.querySelectorAll('.resource-link');

        resourceLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                const resourceType = link.closest('.resource-card').querySelector('h3').textContent;
                console.log('Resource accessed:', resourceType);

                // Add loading state
                link.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...';
                link.style.pointerEvents = 'none';

                // Reset after a short delay (simulating loading)
                setTimeout(() => {
                    link.innerHTML = link.getAttribute('data-original-text') || 'Access Resource <i class="fas fa-arrow-right"></i>';
                    link.style.pointerEvents = 'auto';
                }, 1000);
            });
        });
    }

    // Particle effect for hero section
    function initParticleEffect() {
        const hero = document.querySelector('.hero');
        if (!hero) return;

        const canvas = document.createElement('canvas');
        canvas.className = 'particle-canvas';
        canvas.style.position = 'absolute';
        canvas.style.top = '0';
        canvas.style.left = '0';
        canvas.style.width = '100%';
        canvas.style.height = '100%';
        canvas.style.pointerEvents = 'none';
        canvas.style.zIndex = '1';

        hero.appendChild(canvas);

        const ctx = canvas.getContext('2d');
        let particles = [];
        let animationId;

        function resizeCanvas() {
            canvas.width = hero.offsetWidth;
            canvas.height = hero.offsetHeight;
        }

        function createParticle() {
            return {
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                vx: (Math.random() - 0.5) * 0.5,
                vy: (Math.random() - 0.5) * 0.5,
                size: Math.random() * 2 + 1,
                opacity: Math.random() * 0.5 + 0.2
            };
        }

        function initParticles() {
            particles = [];
            for (let i = 0; i < 50; i++) {
                particles.push(createParticle());
            }
        }

        function animateParticles() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            particles.forEach(particle => {
                particle.x += particle.vx;
                particle.y += particle.vy;

                if (particle.x < 0 || particle.x > canvas.width) particle.vx *= -1;
                if (particle.y < 0 || particle.y > canvas.height) particle.vy *= -1;

                ctx.beginPath();
                ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(124, 58, 237, ${particle.opacity})`;
                ctx.fill();
            });

            animationId = requestAnimationFrame(animateParticles);
        }

        resizeCanvas();
        initParticles();
        animateParticles();

        window.addEventListener('resize', () => {
            resizeCanvas();
            initParticles();
        });
    }

    // Enhanced scroll animations with stagger effect
    function initEnhancedScrollAnimations() {
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach((entry, index) => {
                if (entry.isIntersecting) {
                    setTimeout(() => {
                        entry.target.classList.add('animate-in');
                    }, index * 100); // Stagger animation
                }
            });
        }, observerOptions);

        const animateElements = document.querySelectorAll('.feature-card, .stat-item, .about-text, .about-visual, .emergency-content, .cta-content');
        animateElements.forEach(el => {
            observer.observe(el);
        });
    }

    // Mouse follow effect for interactive elements
    function initMouseFollow() {
        const featureCards = document.querySelectorAll('.feature-card');
        const hero = document.querySelector('.hero');

        featureCards.forEach(card => {
            card.addEventListener('mousemove', (e) => {
                const rect = card.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;

                const centerX = rect.width / 2;
                const centerY = rect.height / 2;

                const rotateX = (y - centerY) / 10;
                const rotateY = (centerX - x) / 10;

                card.style.transform = `translateY(-8px) scale(1.02) rotateX(${rotateX}deg) rotateY(${rotateY}deg)`;
            });

            card.addEventListener('mouseleave', () => {
                card.style.transform = '';
            });
        });
    }

    // Initialize all features
    function init() {
        // Show loading screen initially
        const loadingScreen = document.getElementById('loading-screen');
        if (loadingScreen) {
            setTimeout(() => {
                loadingScreen.style.display = 'none';
            }, 2500); // Show loading for 2.5 seconds
        }

        initSmoothScrolling();
        initNavbarScroll();
        initEnhancedScrollAnimations();
        initTypingAnimation();
        initParallax();
        initCounterAnimation();
        initMobileMenu();
        initFormValidation();
        initLoadingAnimation();
        initEmergencyTracking();
        initParticleEffect();
        initMouseFollow();
        initCommunityTracking();
        initResourceTracking();
    }

    // Start when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    // Add CSS for animations
    const style = document.createElement('style');
    style.textContent = `
        .loading {
            overflow: hidden;
        }
        
        .loading::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: var(--background);
            z-index: 9999;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .loading::after {
            content: 'AIMHSA';
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 2rem;
            font-weight: 700;
            color: var(--primary);
            z-index: 10000;
        }
        
        .animate-in {
            animation: slideInUp 0.6s ease-out;
        }
        
        @keyframes slideInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .nav-toggle {
            background: none;
            border: none;
            color: var(--text);
            font-size: 1.5rem;
            cursor: pointer;
            padding: 0.5rem;
        }
        
        @media (max-width: 768px) {
            .nav-links {
                position: absolute;
                top: 100%;
                left: 0;
                right: 0;
                background: var(--surface);
                flex-direction: column;
                padding: 1rem;
                border-top: 1px solid var(--border);
                display: none;
            }
            
            .nav-links.active {
                display: flex;
            }
        }
    `;
    document.head.appendChild(style);

})();



