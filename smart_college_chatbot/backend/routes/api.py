"""
Public API Routes - For external integrations
"""

import logging
from flask import Blueprint, request, jsonify

logger = logging.getLogger(__name__)
api_bp = Blueprint('api', __name__)


@api_bp.route('/chat', methods=['POST'])
def api_chat():
    from backend.routes.chat import chat
    return chat()


@api_bp.route('/intents', methods=['GET'])
def api_intents():
    from backend.nlp.classifier import get_classifier
    classifier = get_classifier()
    intents_info = [
        {'tag': intent['tag'], 'sample_patterns': intent['patterns'][:2]}
        for intent in classifier.get_all_intents()
    ]
    return jsonify({'intents': intents_info, 'total': len(intents_info)})


@api_bp.route('/faqs', methods=['GET'])
def api_faqs():
    from backend.models.database import get_all_faqs
    category = request.args.get('category')
    faqs = get_all_faqs(category)
    public_faqs = [
        {'id': f['id'], 'category': f['category'], 'question': f['question'], 'answer': f['answer']}
        for f in faqs
    ]
    return jsonify({'faqs': public_faqs, 'count': len(public_faqs)})


@api_bp.route('/status', methods=['GET'])
def api_status():
    from backend.nlp.classifier import get_classifier
    classifier = get_classifier()
    return jsonify({
        'status': 'online',
        'service': 'Smart College Chatbot',
        'version': '3.0.0',
        'university': 'SR University, Warangal',
        'model_ready': classifier.is_trained,
        'model_type': classifier.accuracy_info.get('model_type', 'unknown'),
        'endpoints': {
            'chat': 'POST /api/chat',
            'faqs': 'GET /api/faqs',
            'intents': 'GET /api/intents',
            'status': 'GET /api/status'
        }
    })
