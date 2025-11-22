"""
WSGI entry point for production deployment
Use with: gunicorn wsgi:app
"""
from app import app

if __name__ == "__main__":
    app.run()
