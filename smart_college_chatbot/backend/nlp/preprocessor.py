"""
NLP Preprocessing Module
Fixed: NLTK download at import time, synonym order-dependent replacement
"""

import re
import logging
from typing import List

logger = logging.getLogger(__name__)

# FIX [Minor]: Lazy-load NLTK data instead of downloading at import time
_NLTK_READY = False
NLTK_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    logging.warning("NLTK not available. Using basic preprocessing.")


def _ensure_nltk_data():
    """Download NLTK data lazily on first use, not at import time."""
    global _NLTK_READY
    if _NLTK_READY or not NLTK_AVAILABLE:
        return
    resources = ['punkt', 'stopwords', 'wordnet', 'punkt_tab', 'averaged_perceptron_tagger']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass
    _NLTK_READY = True


KEEP_WORDS = {
    'not', 'no', 'when', 'how', 'what', 'where', 'who', 'why',
    'which', 'is', 'are', 'was', 'were', 'fee', 'exam', 'last'
}

# FIX [Minor]: Use a compiled multi-key regex to avoid order-dependent replacements
SYNONYMS = {
    'tuition': 'fee', 'cost': 'fee', 'price': 'fee', 'charge': 'fee', 'payment': 'fee',
    'professor': 'faculty', 'teacher': 'faculty', 'lecturer': 'faculty',
    'instructor': 'faculty', 'staff': 'faculty',
    'semester': 'exam', 'mid term': 'exam', 'midterm': 'exam',
    'test': 'exam', 'assessment': 'exam',
    'dormitory': 'hostel', 'accommodation': 'hostel', 'residence': 'hostel', 'dorm': 'hostel',
    'job': 'placement', 'employment': 'placement', 'career': 'placement', 'recruit': 'placement',
    'apply': 'admission', 'enroll': 'admission', 'register': 'admission', 'join': 'admission',
    'program': 'course', 'degree': 'course', 'branch': 'course', 'department': 'course',
    'btech': 'course', 'mtech': 'course', 'mba': 'course',
}

# Sort by length descending so longer synonyms match first (e.g., 'mid term' before 'term')
_SORTED_SYNS = sorted(SYNONYMS.keys(), key=len, reverse=True)
_SYN_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(s) for s in _SORTED_SYNS) + r')\b',
    re.IGNORECASE
)


def _syn_replace(match):
    return SYNONYMS[match.group(0).lower()]


class TextPreprocessor:
    def __init__(self):
        _ensure_nltk_data()
        self.stemmer = PorterStemmer() if NLTK_AVAILABLE else None
        self.lemmatizer = WordNetLemmatizer() if NLTK_AVAILABLE else None
        if NLTK_AVAILABLE:
            try:
                base_stops = set(stopwords.words('english'))
                self.stop_words = base_stops - KEEP_WORDS
            except Exception:
                self.stop_words = self._get_basic_stopwords()
        else:
            self.stop_words = self._get_basic_stopwords()
        logger.info(f"TextPreprocessor initialized. NLTK available: {NLTK_AVAILABLE}")

    def _get_basic_stopwords(self) -> set:
        return {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'as', 'it', 'its',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'shall',
            'my', 'your', 'our', 'their', 'this', 'that', 'these', 'those',
            'i', 'me', 'we', 'us', 'he', 'she', 'they', 'them', 'you'
        }

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = text.lower().strip()
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        # FIX: single-pass synonym replacement
        text = _SYN_PATTERN.sub(_syn_replace, text)
        return text

    def tokenize(self, text: str) -> List[str]:
        if NLTK_AVAILABLE:
            try:
                return word_tokenize(text)
            except Exception:
                pass
        return text.split()

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [t for t in tokens if t not in self.stop_words and len(t) > 1]

    def stem(self, tokens: List[str]) -> List[str]:
        if self.stemmer:
            return [self.stemmer.stem(t) for t in tokens]
        return [self._basic_stem(t) for t in tokens]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        if self.lemmatizer:
            return [self.lemmatizer.lemmatize(t) for t in tokens]
        return tokens

    def _basic_stem(self, word: str) -> str:
        suffixes = ['ing', 'tion', 'sion', 'ness', 'ment', 'er', 'es', 'ed', 's']
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) - len(suffix) > 3:
                return word[:-len(suffix)]
        return word

    def preprocess(self, text: str, use_stemming: bool = True) -> List[str]:
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        tokens = self.remove_stopwords(tokens)
        if use_stemming:
            tokens = self.stem(tokens)
        else:
            tokens = self.lemmatize(tokens)
        return tokens

    def preprocess_to_string(self, text: str) -> str:
        return ' '.join(self.preprocess(text))

    def get_ngrams(self, tokens: List[str], n: int = 2) -> List[str]:
        return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]


_preprocessor = None

def get_preprocessor() -> TextPreprocessor:
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = TextPreprocessor()
    return _preprocessor
