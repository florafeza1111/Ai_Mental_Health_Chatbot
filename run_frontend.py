import os
import argparse
import webbrowser
import logging
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
        # Log other requests normally
        super().log_message(format, *args)
    
    def do_GET(self):
        # Parse the URL path
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        # Handle Chrome DevTools and other browser requests silently
        if (path.startswith('/.well-known/') or 
            path.startswith('/favicon.ico') or 
            path.startswith('/apple-touch-icon') or
            path.startswith('/robots.txt') or
            path.startswith('/sitemap.xml')):
            self.send_response(404)
            self.end_headers()
            return
        
        # Handle routing for SPA
        if path == '/':
            # Default to the new landing page
            self.serve_file('landing.html')
        elif path == '/landing' or path == '/landing.html':
            self.serve_file('landing.html')
        elif path == '/index.html':
            self.serve_file('index.html')
        elif path == '/login':
            self.serve_file('login.html')
        elif path == '/register':
            self.serve_file('register.html')
        elif path == '/admin_login.html':
            self.serve_file('admin_login.html')
        elif path == '/admin_dashboard.html':
            self.serve_file('admin_dashboard.html')
        elif path == '/professional_login.html':
            self.serve_file('professional_login.html')
        elif path == '/professional_dashboard.html':
            self.serve_file('professional_dashboard.html')
        elif path.startswith('/') and '.' in path:
            # Static file request (css, js, etc.)
            super().do_GET()
        else:
            # Fallback to index.html for SPA routing
            self.serve_file('index.html')
    
    def serve_file(self, filename):
        """Serve a specific HTML file"""
        try:
            if os.path.exists(filename):
                self.path = '/' + filename
                super().do_GET()
            else:
                self.send_error(404, f"File not found: {filename}")
        except Exception as e:
            self.send_error(500, f"Server error: {str(e)}")

def run_server(port: int):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    chatbot_dir = os.path.join(base_dir, "chatbot")
    if not os.path.isdir(chatbot_dir):
        print("ERROR: chatbot/ directory not found. Create c:\\aimhsa-rag\\chatbot with index.html, style.css, app.js")
        return
    
    # Change to chatbot directory so files are served correctly
    os.chdir(chatbot_dir)
    
    addr = ("", port)
    httpd = ThreadingHTTPServer(addr, ChatBotHandler)
    url = f"http://localhost:{port}/"
    print(f"Serving frontend at {url} (serving directory: {chatbot_dir})")
    print("Routes available:")
    print(f"  - {url} (landing)")
    print(f"  - {url}landing (landing page)")
    print(f"  - {url}login (login page)")
    print(f"  - {url}register (register page)")
    
    try:
        webbrowser.open(url)
    except Exception:
        pass
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down server...")
        httpd.server_close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve the chatbot frontend with proper routing.")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Port to serve the frontend on (default: 8000)")
    args = parser.parse_args()
    run_server(args.port)
