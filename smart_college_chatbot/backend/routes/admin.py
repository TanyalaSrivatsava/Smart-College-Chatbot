"""
Admin Routes - FAQ management and analytics dashboard
Fixed: CSRF protection via flask-wtf or manual token
"""

import logging
import secrets
from functools import wraps
from flask import Blueprint, request, jsonify, render_template, session, redirect, url_for

from backend.models.database import (
    get_all_faqs, get_faq_by_id, create_faq, update_faq, delete_faq,
    search_faqs, verify_admin, get_chat_stats, get_feedback_stats
)
from backend.nlp.classifier import get_classifier

logger = logging.getLogger(__name__)
admin_bp = Blueprint('admin', __name__)


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('admin_logged_in'):
            if request.path.startswith('/admin/api/'):
                return jsonify({'error': 'Unauthorized', 'redirect': '/admin/login'}), 401
            return redirect(url_for('admin.login'))
        # FIX [Minor]: CSRF check for state-changing requests
        if request.method in ('POST', 'PUT', 'DELETE') and request.is_json:
            token = request.headers.get('X-CSRF-Token', '')
            if token != session.get('csrf_token', ''):
                return jsonify({'error': 'Invalid CSRF token'}), 403
        return f(*args, **kwargs)
    return decorated


@admin_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json(silent=True) or request.form
        username = data.get('username', '').strip()
        password = data.get('password', '')

        if verify_admin(username, password):
            session['admin_logged_in'] = True
            session['admin_username'] = username
            session['csrf_token'] = secrets.token_hex(32)
            logger.info(f"Admin login: {username}")

            if request.is_json:
                return jsonify({'success': True, 'redirect': '/admin/', 'csrf_token': session['csrf_token']})
            return redirect(url_for('admin.dashboard'))
        else:
            if request.is_json:
                return jsonify({'success': False, 'error': 'Invalid credentials'}), 401
            return render_template('admin_login.html', error='Invalid username or password')

    return render_template('admin_login.html')


@admin_bp.route('/logout')
def logout():
    session.pop('admin_logged_in', None)
    session.pop('admin_username', None)
    session.pop('csrf_token', None)
    return redirect(url_for('admin.login'))


@admin_bp.route('/')
@admin_bp.route('/dashboard')
@admin_required
def dashboard():
    return render_template('admin.html', username=session.get('admin_username'))


@admin_bp.route('/api/faqs', methods=['GET'])
@admin_required
def api_get_faqs():
    category = request.args.get('category')
    faqs = get_all_faqs(category)
    return jsonify({'faqs': faqs, 'count': len(faqs)})


@admin_bp.route('/api/faqs', methods=['POST'])
@admin_required
def api_create_faq():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid request body'}), 400
    for field in ['category', 'question', 'answer']:
        if not data.get(field):
            return jsonify({'error': f'Field "{field}" is required'}), 400

    faq_id = create_faq(
        category=data['category'], question=data['question'],
        answer=data['answer'], keywords=data.get('keywords', ''))

    classifier = get_classifier()
    try:
        classifier.add_intent(
            tag=f"faq_{data['category']}_{faq_id}",
            patterns=[data['question']], responses=[data['answer']])
    except Exception as e:
        logger.warning(f"Could not add FAQ to NLP model: {e}")

    return jsonify({'success': True, 'id': faq_id}), 201


@admin_bp.route('/api/faqs/<int:faq_id>', methods=['GET'])
@admin_required
def api_get_faq(faq_id):
    faq = get_faq_by_id(faq_id)
    if not faq:
        return jsonify({'error': 'FAQ not found'}), 404
    return jsonify(faq)


@admin_bp.route('/api/faqs/<int:faq_id>', methods=['PUT'])
@admin_required
def api_update_faq(faq_id):
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid request body'}), 400
    success = update_faq(faq_id, data.get('category', ''), data.get('question', ''),
                         data.get('answer', ''), data.get('keywords', ''))
    if success:
        return jsonify({'success': True})
    return jsonify({'error': 'FAQ not found or update failed'}), 404


@admin_bp.route('/api/faqs/<int:faq_id>', methods=['DELETE'])
@admin_required
def api_delete_faq(faq_id):
    success = delete_faq(faq_id)
    if success:
        return jsonify({'success': True})
    return jsonify({'error': 'FAQ not found'}), 404


@admin_bp.route('/api/faqs/search', methods=['GET'])
@admin_required
def api_search_faqs():
    query = request.args.get('q', '')
    if not query:
        return jsonify({'error': 'Query parameter "q" is required'}), 400
    results = search_faqs(query)
    return jsonify({'results': results, 'count': len(results)})


@admin_bp.route('/api/stats', methods=['GET'])
@admin_required
def api_stats():
    chat_stats = get_chat_stats()
    feedback_stats = get_feedback_stats()
    classifier = get_classifier()
    return jsonify({
        'chat': chat_stats,
        'feedback': feedback_stats,
        'model': classifier.accuracy_info,
        'total_intents': len(classifier.get_all_intents())
    })


@admin_bp.route('/api/retrain', methods=['POST'])
@admin_required
def api_retrain():
    try:
        classifier = get_classifier()
        classifier.train()
        return jsonify({'success': True, 'accuracy_info': classifier.accuracy_info})
    except Exception as e:
        logger.error(f"Retrain failed: {e}")
        return jsonify({'error': f'Retrain failed: {str(e)}'}), 500
