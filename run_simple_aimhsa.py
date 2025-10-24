#!/usr/bin/env python3
"""
AIMHSA Simple Launcher
Runs the simplified app.pybcp.py backend and frontend server
"""

import os
import sys
import time
import threading
import webbrowser
import argparse
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

class ChatBotHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress logging for Chrome DevTools and other browser noise
        if (len(args) > 0 and 
            ('.well-known' in str(args) or 
             'favicon.ico' in str(args) or
             'apple-touch-icon' in str(args) or
             'robots.txt' in str(args) or
             'sitemap.xml' in str(args))):
            return
        super().log_message(format, *args)
    
    def do_GET(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        # Handle browser requests silently
        if (path.startswith('/.well-known/') or 
            path.startswith('/favicon.ico') or 
            path.startswith('/apple-touch-icon') or
            path.startswith('/robots.txt') or
            path.startswith('/sitemap.xml')):
            self.send_response(404)
            self.end_headers()
            return
        
        # Handle routing
        if path == '/':
            self.serve_file('index.html')
        elif path == '/landing' or path == '/landing.html':
            self.serve_file('landing.html')
        elif path == '/login':
            self.serve_file('login.html')
        elif path == '/register':
            self.serve_file('register.html')
        elif path.startswith('/') and '.' in path:
            super().do_GET()
        else:
            self.serve_file('index.html')
    
    def serve_file(self, filename):
        try:
            if os.path.exists(filename):
                self.path = '/' + filename
                super().do_GET()
            else:
                self.send_error(404, f"File not found: {filename}")
        except Exception as e:
            self.send_error(500, f"Server error: {str(e)}")

def run_simple_backend(api_port=5057):
    """Run the simplified Flask backend"""
    print(f"🚀 Starting Simple AIMHSA Backend on port {api_port}...")
    
    try:
        # Import the simplified app
        import app as flask_app
        
        # Run the Flask app
        flask_app.app.run(host="0.0.0.0", port=api_port, debug=False, use_reloader=False)
        
    except ImportError as e:
        print(f"❌ Error importing Flask app: {e}")
        print("Make sure app.pybcp.py exists and is renamed to app.py")
    except Exception as e:
        print(f"❌ Error starting backend: {e}")

def run_frontend(frontend_port=8000):
    """Run the frontend HTTP server"""
    print(f"🌐 Starting Frontend on port {frontend_port}...")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    chatbot_dir = os.path.join(base_dir, "chatbot")
    
    if not os.path.isdir(chatbot_dir):
        print(f"❌ ERROR: chatbot/ directory not found at {chatbot_dir}")
        return
    
    os.chdir(chatbot_dir)
    
    addr = ("", frontend_port)
    httpd = ThreadingHTTPServer(addr, ChatBotHandler)
    url = f"http://localhost:{frontend_port}/"
    
    print(f"✅ Frontend running at {url}")
    print("Available routes:")
    print(f"  - {url} (main chat)")
    print(f"  - {url}login (login page)")
    print(f"  - {url}register (register page)")
    
    try:
        webbrowser.open(url)
    except Exception:
        pass
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("🛑 Shutting down...")
        httpd.server_close()

def main():
    parser = argparse.ArgumentParser(description="Simple AIMHSA Launcher")
    parser.add_argument("--api-port", "-a", type=int, default=5057, help="Backend port (default: 5057)")
    parser.add_argument("--frontend-port", "-f", type=int, default=8000, help="Frontend port (default: 8000)")
    parser.add_argument("--backend-only", "-b", action="store_true", help="Run only backend")
    parser.add_argument("--frontend-only", "-w", action="store_true", help="Run only frontend")
    args = parser.parse_args()
    
    print("="*50)
    print("🧠 AIMHSA Simple Launcher")
    print("="*50)
    print(f"Backend: http://localhost:{args.api_port}")
    print(f"Frontend: http://localhost:{args.frontend_port}")
    print("="*50)
    
    if args.backend_only:
        run_simple_backend(args.api_port)
    elif args.frontend_only:
        run_frontend(args.frontend_port)
    else:
        print("🚀 Starting both services...")
        
        # Start backend in thread
        backend_thread = threading.Thread(
            target=run_simple_backend, 
            args=(args.api_port,),
            daemon=True
        )
        backend_thread.start()
        
        time.sleep(2)
        run_frontend(args.frontend_port)

if __name__ == "__main__":
    main()
