import http.server
import socketserver
import os
import sys

PORT = 5000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

if __name__ == "__main__":
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"Serving the ONNX web demo at http://127.0.0.1:{PORT}")
            print(f"Directory: {DIRECTORY}")
            print("Press Ctrl+C to stop.")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        sys.exit(0)