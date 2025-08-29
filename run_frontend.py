import os
import argparse
import webbrowser
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

def run_server(port: int):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    chatbot_dir = os.path.join(base_dir, "chatbot")
    if not os.path.isdir(chatbot_dir):
        print("ERROR: chatbot/ directory not found. Create c:\\aimhsa-rag\\chatbot with index.html, style.css, app.js")
        return
    os.chdir(chatbot_dir)
    addr = ("", port)
    handler = SimpleHTTPRequestHandler
    httpd = ThreadingHTTPServer(addr, handler)
    url = f"http://localhost:{port}/"
    print(f"Serving frontend at {url} (serving directory: {chatbot_dir})")
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
    parser = argparse.ArgumentParser(description="Serve the chatbot frontend (simple Python server).")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Port to serve the frontend on (default: 8000)")
    args = parser.parse_args()
    run_server(args.port)
