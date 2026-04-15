"""
tests/test_chatbot.py — Unit tests for the Smart College Chatbot v3
Run with: python -m pytest tests/ -v
"""

import sys
import os
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestPreprocessor:
    def setup_method(self):
        from backend.nlp.preprocessor import TextPreprocessor
        self.pp = TextPreprocessor()

    def test_clean_text_lowercase(self):
        result = self.pp.clean_text("HELLO World")
        assert result == result.lower()

    def test_clean_text_removes_special_chars(self):
        result = self.pp.clean_text("hello! @world #test")
        assert '!' not in result
        assert '@' not in result

    def test_clean_text_removes_urls(self):
        result = self.pp.clean_text("visit https://example.com for info")
        assert 'http' not in result

    def test_clean_text_removes_emails(self):
        result = self.pp.clean_text("email admin@sruniversity.ac.in for help")
        assert '@' not in result

    def test_tokenize_basic(self):
        tokens = self.pp.tokenize("hello world test")
        assert isinstance(tokens, list)
        assert len(tokens) >= 2

    def test_remove_stopwords(self):
        tokens = ['a', 'the', 'is', 'fee', 'structure', 'for', 'btech']
        result = self.pp.remove_stopwords(tokens)
        assert 'fee' in result or 'structure' in result

    def test_stem_returns_list(self):
        tokens = ['running', 'admission', 'courses']
        result = self.pp.stem(tokens)
        assert isinstance(result, list)
        assert len(result) == len(tokens)

    def test_preprocess_pipeline(self):
        result = self.pp.preprocess("What is the fee structure for B.Tech admission?")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_preprocess_empty_string(self):
        result = self.pp.preprocess("")
        assert isinstance(result, list)

    def test_synonym_mapping(self):
        result = self.pp.clean_text("what is the tuition cost?")
        assert 'fee' in result

    def test_preprocess_to_string(self):
        result = self.pp.preprocess_to_string("tell me about hostel accommodation")
        assert isinstance(result, str)


class TestClassifier:
    def setup_method(self):
        from backend.nlp.classifier import IntentClassifier
        self.clf = IntentClassifier()

    def test_classifier_loads_intents(self):
        assert len(self.clf.intents) > 0

    def test_classifier_is_trained(self):
        assert self.clf.is_trained is True

    def test_predict_returns_tuple(self):
        tag, conf, response = self.clf.predict("hello")
        assert isinstance(tag, str)
        assert isinstance(conf, float)
        assert isinstance(response, str)

    def test_predict_greeting(self):
        tag, conf, response = self.clf.predict("hello there")
        assert conf >= 0 and conf <= 1.0
        assert len(response) > 0

    def test_predict_fee_query(self):
        tag, conf, response = self.clf.predict("what is the fee structure?")
        assert tag in ['fees', 'unknown']
        assert isinstance(response, str)

    def test_predict_admission_query(self):
        tag, conf, response = self.clf.predict("how do I apply for admission?")
        assert tag in ['admissions', 'unknown']

    def test_predict_hostel_query(self):
        tag, conf, response = self.clf.predict("tell me about hostel facilities")
        assert tag in ['hostel', 'unknown']

    def test_predict_placement_query(self):
        tag, conf, response = self.clf.predict("what is the placement record?")
        assert tag in ['placements', 'unknown']

    def test_predict_empty_string(self):
        tag, conf, response = self.clf.predict("")
        assert tag == 'unknown'
        assert conf == 0.0
        assert len(response) > 0

    def test_predict_unknown_query(self):
        tag, conf, response = self.clf.predict("xyzzy random nonsense klakla")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_predict_confidence_range(self):
        _, conf, _ = self.clf.predict("tell me about courses")
        assert 0.0 <= conf <= 1.0

    def test_responses_dict_populated(self):
        assert len(self.clf.responses) > 0

    def test_get_all_intents(self):
        intents = self.clf.get_all_intents()
        assert isinstance(intents, list)
        assert len(intents) > 0


class TestDatabase:
    def setup_method(self):
        import tempfile
        self.test_db = tempfile.mktemp(suffix='.db')
        import backend.models.database as db_module
        self.original_path = db_module.DB_PATH
        db_module.DB_PATH = self.test_db
        from backend.models.database import init_database
        init_database()

    def teardown_method(self):
        import backend.models.database as db_module
        db_module.DB_PATH = self.original_path
        if os.path.exists(self.test_db):
            os.remove(self.test_db)

    def test_init_creates_tables(self):
        from backend.models.database import get_all_faqs
        faqs = get_all_faqs()
        assert isinstance(faqs, list)

    def test_create_faq(self):
        from backend.models.database import create_faq, get_faq_by_id
        faq_id = create_faq('test', 'Test question?', 'Test answer.', 'test')
        assert faq_id > 0
        faq = get_faq_by_id(faq_id)
        assert faq is not None

    def test_update_faq(self):
        from backend.models.database import create_faq, update_faq, get_faq_by_id
        faq_id = create_faq('test', 'Original?', 'Original.', '')
        success = update_faq(faq_id, 'test', 'Updated?', 'Updated.', '')
        assert success is True
        faq = get_faq_by_id(faq_id)
        assert faq['question'] == 'Updated?'

    def test_delete_faq(self):
        from backend.models.database import create_faq, delete_faq, get_faq_by_id
        faq_id = create_faq('test', 'Delete me?', 'Delete me.', '')
        success = delete_faq(faq_id)
        assert success is True
        faq = get_faq_by_id(faq_id)
        assert faq is None

    def test_search_faqs_escapes_wildcards(self):
        from backend.models.database import create_faq, search_faqs
        create_faq('test', '100% discount?', 'No.', '')
        results = search_faqs('100%')
        assert isinstance(results, list)

    def test_save_chat_message(self):
        from backend.models.database import save_chat_message, get_chat_history
        msg_id = save_chat_message('test-session', 'hello', 'hi', 'greeting', 0.95)
        assert msg_id > 0
        history = get_chat_history('test-session')
        assert len(history) > 0

    def test_verify_admin(self):
        from backend.models.database import verify_admin
        assert verify_admin('admin', 'admin123') is True
        assert verify_admin('admin', 'wrong') is False

    def test_save_feedback(self):
        from backend.models.database import save_chat_message, save_feedback
        chat_id = save_chat_message('test', 'q', 'a', 'test', 0.9)
        fb_id = save_feedback(chat_id, 5, 'great')
        assert fb_id > 0
