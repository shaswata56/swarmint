"""Local-only static server with correct MIME for .mjs (python's http.server
defaults to text/plain, which Chrome's module loader silently rejects).
Not part of the product; used only for local browser testing of web/."""
import http.server
import functools

Handler = http.server.SimpleHTTPRequestHandler
Handler.extensions_map[".mjs"] = "application/javascript"

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8791
    directory = sys.argv[2] if len(sys.argv) > 2 else "."
    handler = functools.partial(Handler, directory=directory)
    http.server.HTTPServer(("", port), handler).serve_forever()
