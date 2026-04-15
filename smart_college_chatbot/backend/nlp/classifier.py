"""
Intent Classification Model — v3 with Sentence-Transformers (BERT)
------------------------------------------------------------------
Key upgrades:
  1. Uses sentence-transformers for BERT-based sentence embeddings
  2. Cosine similarity for intent matching (no sklearn classifier needed for inference)
  3. Proper data augmentation with non-deterministic seeding
  4. Preprocessor output is actually USED in the pipeline
  5. Train/test split on ORIGINAL patterns before augmentation (no data leakage)
  6. Falls back to TF-IDF + LinearSVC if sentence-transformers unavailable
  7. Falls back to keyword matching if sklearn also unavailable
"""

import json
import os
import logging
import random
import pickle
import re
import time
from typing import Dict, List, Tuple, Optional

import numpy as np

# --- Sentence Transformers (BERT) ---
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

# --- Sklearn fallback ---
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, accuracy_score
    from scipy.sparse import hstack
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from backend.nlp.preprocessor import get_preprocessor

logger = logging.getLogger(__name__)

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR     = os.path.join(BASE_DIR, 'data')
MODEL_DIR    = os.path.join(BASE_DIR, 'backend', 'models')
MODEL_PATH   = os.path.join(MODEL_DIR, 'chatbot_model.pkl')
INTENTS_PATH = os.path.join(DATA_DIR, 'intents.json')

# FIX [Minor]: Centralised confidence thresholds used everywhere
HIGH_CONFIDENCE   = 0.65
MEDIUM_CONFIDENCE = 0.40
LOW_CONFIDENCE    = 0.20

# -----------------------------------------------------------------------
AUGMENTATION_SYNONYMS: Dict[str, List[str]] = {
    'fee':         ['cost', 'charges', 'payment', 'tuition', 'expenses', 'price', 'amount'],
    'admission':   ['enrollment', 'joining', 'application', 'intake', 'apply', 'register'],
    'hostel':      ['dormitory', 'accommodation', 'residence', 'dorm', 'boarding'],
    'placement':   ['job', 'career', 'employment', 'hiring', 'recruitment', 'package', 'salary'],
    'faculty':     ['teacher', 'professor', 'staff', 'lecturer', 'instructor', 'hod'],
    'course':      ['program', 'degree', 'branch', 'department', 'subject', 'stream'],
    'exam':        ['test', 'assessment', 'evaluation', 'quiz'],
    'library':     ['books', 'reading room', 'e-library', 'journal'],
    'scholarship': ['financial aid', 'stipend', 'bursary', 'epass', 'fee waiver'],
    'transport':   ['bus', 'shuttle', 'conveyance', 'vehicle', 'travel'],
    'sports':      ['games', 'athletics', 'cricket', 'football', 'playground'],
    'lab':         ['laboratory', 'practical', 'workshop', 'computer lab'],
    'canteen':     ['food', 'cafeteria', 'mess', 'dining'],
    'club':        ['society', 'association', 'group', 'extracurricular'],
    'wifi':        ['internet', 'network', 'connectivity', 'broadband', 'online'],
    'attendance':  ['presence', 'absent', 'leave', 'shortage', 'percentage'],
    'location':    ['address', 'place', 'situated', 'where', 'directions', 'map'],
    'contact':     ['phone', 'call', 'email', 'reach', 'number', 'helpline'],
    'naac':        ['accreditation', 'rating', 'grade', 'ranking', 'ugc', 'aicte'],
}

QUESTION_TEMPLATES = [
    'what is {core}', 'tell me about {core}', 'i want to know about {core}',
    'can you tell me about {core}', 'any information on {core}',
    'details about {core}', 'what are {core}', 'how is {core}', 'explain {core}',
]

_STRIP_RE = re.compile(
    r"^(what('s| is| are)?\s+|how\s+|when\s+|where\s+|tell\s+me\s+about\s+"
    r"|can\s+you\s+|please\s+|is\s+there\s+|are\s+there\s+|do\s+you\s+|i\s+want\s+to\s+know\s+)",
    re.IGNORECASE
)


def _augment_patterns(patterns: List[str], rng: random.Random) -> List[str]:
    """Synonym substitution + question-template wrapping.
    FIX [Major]: uses instance RNG, not global random.seed(0)
    """
    augmented = [p.lower() for p in patterns]
    for p in patterns:
        p_lower = p.lower()
        for canonical, synonyms in AUGMENTATION_SYNONYMS.items():
            if canonical in p_lower:
                for syn in synonyms[:3]:
                    augmented.append(p_lower.replace(canonical, syn))
        core = _STRIP_RE.sub('', p_lower).strip()
        if core and len(core) > 3:
            for tmpl in rng.sample(QUESTION_TEMPLATES, min(3, len(QUESTION_TEMPLATES))):
                augmented.append(tmpl.format(core=core))
    return list(dict.fromkeys(augmented))


class IntentClassifier:
    """
    BERT-based (sentence-transformers) intent classifier with cosine similarity.
    Falls back to TF-IDF + SVC, then to keyword matching.
    """

    def __init__(self):
        self.preprocessor   = get_preprocessor()
        self.intents:       List[Dict] = []
        self.responses:     Dict[str, List[str]] = {}
        self.is_trained:    bool = False
        self.accuracy_info: Dict = {}
        self.model_type:    str = 'none'

        # BERT components
        self.sbert_model    = None
        self.intent_embeddings: Optional[np.ndarray] = None
        self.intent_labels: List[str] = []

        # Sklearn fallback components
        self.label_encoder  = None
        self.vec_word       = None
        self.vec_char       = None
        self.clf            = None

        os.makedirs(MODEL_DIR, exist_ok=True)
        self._load_intents()

        if not self._load_model():
            self.train()

    def _load_intents(self):
        try:
            with open(INTENTS_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.intents = data.get('intents', [])
            logger.info(f"Loaded {len(self.intents)} intents")
        except FileNotFoundError:
            logger.error(f"Intents file not found: {INTENTS_PATH}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")

    def train(self):
        logger.info("Training intent classifier (v3)...")
        start = time.time()
        if not self.intents:
            logger.error("No intents loaded. Cannot train.")
            return

        rng = random.Random()  # non-seeded for real stochastic augmentation

        # FIX [Major]: Split BEFORE augmentation to prevent data leakage
        # Collect original patterns with tags
        original_data = []
        self.responses = {}
        for intent in self.intents:
            tag = intent['tag']
            self.responses[tag] = intent['responses']
            for p in intent['patterns']:
                original_data.append((p.lower(), tag))

        orig_texts = [d[0] for d in original_data]
        orig_tags  = [d[1] for d in original_data]

        # Split original data for evaluation
        if len(set(orig_tags)) > 1 and len(orig_texts) > 10:
            train_texts, test_texts, train_tags, test_tags = train_test_split(
                orig_texts, orig_tags, test_size=0.2, stratify=orig_tags, random_state=42
            ) if SKLEARN_AVAILABLE else (orig_texts, [], orig_tags, [])
        else:
            train_texts, test_texts, train_tags, test_tags = orig_texts, [], orig_tags, []

        # Augment ONLY training data
        aug_corpus, aug_tags = [], []
        for intent in self.intents:
            tag = intent['tag']
            # Only augment patterns that are in train set
            train_patterns = [p for p in intent['patterns'] if p.lower() in train_texts]
            if not train_patterns:
                train_patterns = intent['patterns']  # fallback
            for p in _augment_patterns(train_patterns, rng):
                aug_corpus.append(p)
                aug_tags.append(tag)

        logger.info(f"Training: {len(aug_corpus)} augmented samples, {len(test_texts)} test samples")

        if SBERT_AVAILABLE:
            self._train_sbert(aug_corpus, aug_tags, test_texts, test_tags, start)
        elif SKLEARN_AVAILABLE:
            self._train_sklearn(aug_corpus, aug_tags, test_texts, test_tags, start)
        else:
            self._train_keyword(aug_corpus, aug_tags, start)

        self.is_trained = True
        self._save_model()

    def _train_sbert(self, corpus, tags, test_texts, test_tags, start):
        """Train using Sentence-BERT embeddings + cosine similarity."""
        logger.info("Using Sentence-BERT (all-MiniLM-L6-v2) ...")
        self.model_type = 'sbert'

        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

        # FIX [Major]: Preprocessor output IS used - clean text before encoding
        cleaned_corpus = [self.preprocessor.clean_text(t) for t in corpus]

        # Compute embeddings for all training patterns
        all_embeddings = self.sbert_model.encode(cleaned_corpus, show_progress_bar=False, normalize_embeddings=True)

        # Store per-intent mean embeddings for fast inference
        unique_tags = list(dict.fromkeys(tags))
        intent_embs = []
        for tag in unique_tags:
            indices = [i for i, t in enumerate(tags) if t == tag]
            mean_emb = np.mean(all_embeddings[indices], axis=0)
            mean_emb = mean_emb / np.linalg.norm(mean_emb)
            intent_embs.append(mean_emb)

        self.intent_embeddings = np.array(intent_embs)
        self.intent_labels = unique_tags

        # Also store all training embeddings for fine-grained matching
        self._all_train_embs = all_embeddings
        self._all_train_tags = tags

        # Evaluate on held-out test set
        train_acc = self._sbert_accuracy(cleaned_corpus, tags)
        test_acc = self._sbert_accuracy(
            [self.preprocessor.clean_text(t) for t in test_texts], test_tags
        ) if test_texts else 0.0

        self.accuracy_info = {
            'model_type': 'Sentence-BERT (all-MiniLM-L6-v2) + Cosine Similarity',
            'training_samples': len(corpus),
            'num_intents': len(self.responses),
            'training_accuracy': f"{train_acc*100:.2f}%",
            'holdout_accuracy': f"{test_acc*100:.2f}%" if test_texts else 'N/A',
            'training_time_sec': round(time.time() - start, 3),
        }
        logger.info(f"SBERT Done | Train: {train_acc*100:.1f}% | Hold-out: {test_acc*100:.1f}%")

    def _sbert_accuracy(self, texts, tags):
        if not texts:
            return 0.0
        embs = self.sbert_model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        correct = 0
        for emb, true_tag in zip(embs, tags):
            sims = emb @ self.intent_embeddings.T
            pred_idx = np.argmax(sims)
            if self.intent_labels[pred_idx] == true_tag:
                correct += 1
        return correct / len(tags)

    def _train_sklearn(self, corpus, tags, test_texts, test_tags, start):
        """Fallback: TF-IDF + CalibratedLinearSVC"""
        logger.info("Using TF-IDF + LinearSVC fallback...")
        self.model_type = 'sklearn'

        # FIX [Major]: preprocessor IS used for feature extraction
        cleaned_corpus = [self.preprocessor.clean_text(t) for t in corpus]

        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(tags)

        self.vec_word = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), sublinear_tf=True, min_df=1, max_features=8000)
        self.vec_char = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True, min_df=1, max_features=8000)

        X_word = self.vec_word.fit_transform(cleaned_corpus)
        X_char = self.vec_char.fit_transform(cleaned_corpus)
        X = hstack([X_word, X_char])

        base_svc = LinearSVC(C=0.8, max_iter=3000, class_weight='balanced')
        self.clf = CalibratedClassifierCV(base_svc, cv=min(5, min(np.bincount(y))))
        self.clf.fit(X, y)

        train_acc = (self.clf.predict(X) == y).mean()

        # Evaluate on held-out test
        test_acc = 0.0
        if test_texts:
            cleaned_test = [self.preprocessor.clean_text(t) for t in test_texts]
            Xt_w = self.vec_word.transform(cleaned_test)
            Xt_c = self.vec_char.transform(cleaned_test)
            Xt = hstack([Xt_w, Xt_c])
            y_test = self.label_encoder.transform(test_tags)
            test_acc = (self.clf.predict(Xt) == y_test).mean()

        self.accuracy_info = {
            'model_type': 'TF-IDF (word+char) + CalibratedLinearSVC',
            'training_samples': len(corpus),
            'num_intents': len(self.responses),
            'training_accuracy': f"{train_acc*100:.2f}%",
            'holdout_accuracy': f"{test_acc*100:.2f}%" if test_texts else 'N/A',
            'training_time_sec': round(time.time() - start, 3),
        }
        logger.info(f"Sklearn Done | Train: {train_acc*100:.1f}% | Hold-out: {test_acc*100:.1f}%")

    def _train_keyword(self, corpus, tags, start):
        self.model_type = 'keyword'
        self.accuracy_info = {
            'model_type': 'Keyword Matching (no ML libraries)',
            'training_samples': len(corpus),
            'num_intents': len(self.responses),
        }

    def predict(self, user_input: str) -> Tuple[str, float, str]:
        if not user_input or not user_input.strip():
            return ('unknown', 0.0, self._fallback_response())

        # FIX [Major]: preprocessor output IS used
        cleaned = self.preprocessor.clean_text(user_input.strip())

        if self.model_type == 'sbert' and self.sbert_model is not None:
            return self._predict_sbert(cleaned)
        elif self.model_type == 'sklearn' and self.clf is not None:
            return self._predict_sklearn(cleaned)
        else:
            return self._predict_keyword(cleaned)

    def _predict_sbert(self, query: str) -> Tuple[str, float, str]:
        emb = self.sbert_model.encode([query], normalize_embeddings=True)[0]
        sims = emb @ self.intent_embeddings.T
        best_idx = np.argmax(sims)
        confidence = float(sims[best_idx])
        # Map cosine similarity [0,1] to confidence
        confidence = max(0.0, min(1.0, confidence))

        if confidence < LOW_CONFIDENCE:
            return ('unknown', confidence, self._fallback_response())

        tag = self.intent_labels[best_idx]
        response = random.choice(self.responses.get(tag, [self._fallback_response()]))

        if confidence < MEDIUM_CONFIDENCE:
            response += "\n\n⚠️ I'm not very confident about this answer. Please verify with the college office."

        return (tag, confidence, response)

    def _predict_sklearn(self, query: str) -> Tuple[str, float, str]:
        X_w = self.vec_word.transform([query])
        X_c = self.vec_char.transform([query])
        X = hstack([X_w, X_c])

        probas = self.clf.predict_proba(X)[0]
        best_idx = np.argmax(probas)
        confidence = float(probas[best_idx])
        tag = self.label_encoder.inverse_transform([best_idx])[0]

        if confidence < LOW_CONFIDENCE:
            return ('unknown', confidence, self._fallback_response())

        response = random.choice(self.responses.get(tag, [self._fallback_response()]))

        if confidence < MEDIUM_CONFIDENCE:
            response += "\n\n⚠️ I'm not very confident about this answer. Please verify with the college office."

        return (tag, confidence, response)

    def _predict_keyword(self, query: str) -> Tuple[str, float, str]:
        best_tag, best_score = 'unknown', 0.0
        query_words = set(query.lower().split())
        for intent in self.intents:
            tag = intent['tag']
            for pattern in intent['patterns']:
                pattern_words = set(pattern.lower().split())
                if not pattern_words:
                    continue
                overlap = len(query_words & pattern_words)
                # FIX [Minor]: simple match count, not normalised-then-doubled
                score = overlap / max(len(query_words), len(pattern_words))
                if score > best_score:
                    best_score = score
                    best_tag = tag

        if best_score < LOW_CONFIDENCE:
            return ('unknown', best_score, self._fallback_response())

        response = random.choice(self.responses.get(best_tag, [self._fallback_response()]))
        return (best_tag, min(best_score, 1.0), response)

    def _fallback_response(self) -> str:
        return (
            "I'm not sure about that. You can:\n"
            "• Rephrase your question\n"
            "• Contact SR University: +91-870-2427777\n"
            "• Email: info@sruniversity.ac.in\n"
            "• Visit: sruniversity.ac.in"
        )

    def _save_model(self):
        try:
            payload = {
                'model_type': self.model_type,
                'responses': self.responses,
                'accuracy_info': self.accuracy_info,
                'intent_labels': self.intent_labels,
            }
            if self.model_type == 'sbert':
                payload['intent_embeddings'] = self.intent_embeddings
                payload['all_train_embs'] = self._all_train_embs
                payload['all_train_tags'] = self._all_train_tags
            elif self.model_type == 'sklearn':
                payload['vec_word'] = self.vec_word
                payload['vec_char'] = self.vec_char
                payload['clf'] = self.clf
                payload['label_encoder'] = self.label_encoder
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(payload, f)
            logger.info(f"Model saved → {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Save failed: {e}")

    def _load_model(self) -> bool:
        try:
            if not os.path.exists(MODEL_PATH):
                return False
            if os.path.getmtime(INTENTS_PATH) > os.path.getmtime(MODEL_PATH):
                logger.info("intents.json changed — retraining.")
                return False
            with open(MODEL_PATH, 'rb') as f:
                payload = pickle.load(f)

            self.model_type = payload.get('model_type', 'sklearn')
            self.responses = payload.get('responses', {})
            self.accuracy_info = payload.get('accuracy_info', {})
            self.intent_labels = payload.get('intent_labels', [])

            if self.model_type == 'sbert':
                if not SBERT_AVAILABLE:
                    logger.info("SBERT model saved but library not available — retraining")
                    return False
                self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.intent_embeddings = payload['intent_embeddings']
                self._all_train_embs = payload.get('all_train_embs')
                self._all_train_tags = payload.get('all_train_tags')
            elif self.model_type == 'sklearn':
                if not SKLEARN_AVAILABLE:
                    logger.info("Sklearn model saved but library not available — retraining")
                    return False
                self.vec_word = payload['vec_word']
                self.vec_char = payload['vec_char']
                self.clf = payload['clf']
                self.label_encoder = payload['label_encoder']

            self.is_trained = True
            logger.info(f"Loaded cached model (type={self.model_type})")
            return True
        except Exception as e:
            logger.warning(f"Could not load model: {e}")
            return False

    def add_intent(self, tag: str, patterns: List[str], responses: List[str]):
        self.responses[tag] = responses
        new_intent = {'tag': tag, 'patterns': patterns, 'responses': responses}
        self.intents.append(new_intent)
        self.train()

    def get_all_intents(self) -> List[Dict]:
        return self.intents


_classifier = None

def get_classifier() -> IntentClassifier:
    global _classifier
    if _classifier is None:
        _classifier = IntentClassifier()
    return _classifier
