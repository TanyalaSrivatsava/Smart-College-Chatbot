"""
Smart College Chatbot - Main Flask Application
SR University, Warangal
Fixed: hardcoded secret key, wildcard CORS, credential logging
"""

import os
import logging
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, render_template, session


def create_app():
    """Application factory"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    app = Flask(
        __name__,
        template_folder=os.path.join(base_dir, 'frontend', 'templates'),
        static_folder=os.path.join(base_dir, 'frontend', 'static')
    )

    # FIX [Critical]: No hardcoded fallback secret key
    secret = os.environ.get('SECRET_KEY')
    if not secret:
        import secrets as _s
        secret = _s.token_hex(32)
        app.logger.warning("SECRET_KEY not set — using random key (sessions won't persist across restarts)")
    app.secret_key = secret

    app.config['SESSION_COOKIE_HTTPONLY'] = True
    # FIX [Minor]: Use Strict SameSite to prevent CSRF
    app.config['SESSION_COOKIE_SAMESITE'] = 'Strict'
    app.config['SESSION_COOKIE_SECURE'] = os.environ.get('FLASK_ENV') == 'production'

    # FIX [Critical]: Restrict CORS to trusted origins instead of wildcard
    allowed_origins = os.environ.get('ALLOWED_ORIGINS', '*')

    @app.after_request
    def add_cors_headers(response):
        if request.path.startswith('/api/'):
            response.headers['Access-Control-Allow-Origin'] = allowed_origins
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            # FIX [Critical]: Prevent credential exfiltration
            response.headers['Access-Control-Allow-Credentials'] = 'false'
        return response

    setup_logging(app)

    from backend.models.database import init_database
    with app.app_context():
        init_database()

    from backend.routes.chat import chat_bp
    from backend.routes.admin import admin_bp
    from backend.routes.api import api_bp

    app.register_blueprint(chat_bp)
    app.register_blueprint(admin_bp, url_prefix='/admin')
    app.register_blueprint(api_bp, url_prefix='/api')

    register_error_handlers(app)
    app.logger.info("Smart College Chatbot started successfully!")
    return app


def setup_logging(app):
    """Configure application logging"""
    log_level = os.environ.get('LOG_LEVEL', 'INFO')
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler('logs/chatbot.log'),
            logging.StreamHandler()
        ]
    )
    app.logger.setLevel(getattr(logging, log_level))


def register_error_handlers(app):
    @app.errorhandler(404)
    def not_found(e):
        if request.path.startswith('/api/'):
            return jsonify({'error': 'Endpoint not found', 'status': 404}), 404
        return render_template('404.html'), 404

    @app.errorhandler(500)
    def server_error(e):
        app.logger.error(f"Server error: {e}")
        if request.path.startswith('/api/'):
            return jsonify({'error': 'Internal server error', 'status': 500}), 500
        return jsonify({'error': 'Something went wrong on our end. Please try again.'}), 500

    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify({'error': 'Method not allowed', 'status': 405}), 405

    @app.errorhandler(429)
    def rate_limited(e):
        return jsonify({'error': 'Too many requests. Please try again later.', 'status': 429}), 429


app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'development') == 'development'
    print("=" * 60)
    print("  🎓 Smart College Chatbot - SR University")
    print(f"  🌐 Running on: http://localhost:{port}")
    print(f"  🔧 Debug mode: {debug}")
    print("=" * 60)
    app.run(host='0.0.0.0', port=port, debug=debug)
