"""
Chat Routes - Main chatbot endpoint
Fixed: session_id leak, rate limiting, chat_id None in feedback
"""

import uuid
import logging
import time
from datetime import datetime
from collections import defaultdict
from flask import Blueprint, request, jsonify, render_template, session

from backend.nlp.classifier import get_classifier
from backend.models.database import save_chat_message

logger = logging.getLogger(__name__)
chat_bp = Blueprint('chat', __name__)

# FIX [Major]: Simple in-memory rate limiter
_rate_limits = defaultdict(list)
RATE_LIMIT = 60  # requests per minute per IP


def _check_rate_limit(ip: str) -> bool:
    now = time.time()
    _rate_limits[ip] = [t for t in _rate_limits[ip] if now - t < 60]
    if len(_rate_limits[ip]) >= RATE_LIMIT:
        return False
    _rate_limits[ip].append(now)
    return True


def get_session_id():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']


@chat_bp.route('/')
def index():
    return render_template('index.html')


@chat_bp.route('/chat', methods=['POST'])
def chat():
    try:
        ip_address = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)

        # FIX [Major]: Rate limiting
        if not _check_rate_limit(ip_address):
            return jsonify({'error': 'Too many requests. Please try again later.'}), 429

        data = request.get_json(silent=True)
        if not data:
            return jsonify({'error': 'Request body must be JSON'}), 400

        user_message = data.get('message', '').strip()
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        if len(user_message) > 500:
            return jsonify({'error': 'Message too long (max 500 characters)'}), 400

        session_id = get_session_id()

        classifier = get_classifier()
        intent, confidence, response = classifier.predict(user_message)

        try:
            chat_id = save_chat_message(
                session_id=session_id,
                user_message=user_message,
                bot_response=response,
                intent=intent,
                confidence=confidence,
                ip_address=ip_address
            )
        except Exception as db_err:
            logger.warning(f"Failed to save chat to DB: {db_err}")
            chat_id = None

        # FIX [Major]: Don't leak session_id in response body
        result = {
            'response': response,
            'intent': intent,
            'confidence': round(confidence, 3),
            'chat_id': chat_id,
            'timestamp': datetime.now().isoformat(),
        }

        logger.info(f"[{session_id[:8]}] '{user_message[:50]}' -> {intent} ({confidence:.2f})")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return jsonify({
            'response': "I'm experiencing technical difficulties. Please try again or contact the college directly at +91-870-2427777.",
            'intent': 'error',
            'confidence': 0,
            'timestamp': datetime.now().isoformat()
        }), 500


@chat_bp.route('/feedback', methods=['POST'])
def submit_feedback():
    try:
        from backend.models.database import save_feedback
        data = request.get_json(silent=True)
        if not data:
            return jsonify({'error': 'Invalid request'}), 400

        chat_id = data.get('chat_id')
        rating = data.get('rating')
        comment = data.get('comment', '')

        # FIX [Minor]: Validate chat_id is a non-null integer
        if chat_id is None or rating is None:
            return jsonify({'error': 'chat_id and rating are required'}), 400

        try:
            chat_id = int(chat_id)
        except (ValueError, TypeError):
            return jsonify({'error': 'chat_id must be a valid integer'}), 400

        try:
            rating = int(rating)
        except (ValueError, TypeError):
            return jsonify({'error': 'rating must be a valid integer'}), 400

        if not (1 <= rating <= 5):
            return jsonify({'error': 'Rating must be between 1 and 5'}), 400

        feedback_id = save_feedback(chat_id, rating, comment)
        return jsonify({'success': True, 'feedback_id': feedback_id})

    except Exception as e:
        logger.error(f"Feedback error: {e}")
        return jsonify({'error': 'Failed to save feedback'}), 500


@chat_bp.route('/health')
def health():
    classifier = get_classifier()
    return jsonify({
        'status': 'healthy',
        'model_trained': classifier.is_trained,
        'timestamp': datetime.now().isoformat(),
        'accuracy_info': classifier.accuracy_info
    })
