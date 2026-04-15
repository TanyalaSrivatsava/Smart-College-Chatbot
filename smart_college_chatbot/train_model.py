"""
train_model.py — Standalone script to train and evaluate the NLP model.
Fixed: uses centralised confidence thresholds, proper evaluation
"""

import sys
import os
import json
import argparse
import time

sys.path.insert(0, os.path.dirname(__file__))


def train():
    print("=" * 60)
    print("  Smart College Chatbot — NLP Model Training (v3)")
    print("=" * 60)

    print("\n📦 Loading dependencies...")
    from backend.nlp.classifier import get_classifier, MODEL_PATH, HIGH_CONFIDENCE

    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
        print("  ✓ Removed cached model (forcing retrain)")

    print("\n🧠 Training intent classifier...")
    start = time.time()
    classifier = get_classifier()

    print(f"\n✅ Training complete in {time.time() - start:.2f}s")
    print("\n📊 Model Statistics:")
    for key, val in classifier.accuracy_info.items():
        print(f"   {key:25s}: {val}")

    return classifier


def evaluate(classifier):
    from backend.nlp.classifier import HIGH_CONFIDENCE
    print("\n" + "=" * 60)
    print("  Evaluation — Test Queries")
    print("=" * 60)

    test_cases = [
        ("how do I apply for admission?", "admissions"),
        ("what is the fee structure for btech?", "fees"),
        ("which courses are offered?", "courses"),
        ("when are the semester exams?", "exams"),
        ("tell me about the faculty", "faculty"),
        ("what are the placement packages?", "placements"),
        ("is hostel available?", "hostel"),
        ("where is the college located?", "location"),
        ("are there scholarships available?", "scholarships"),
        ("what are the bus routes?", "transport"),
        ("library timings?", "library"),
        ("hi there", "greeting"),
        ("goodbye", "goodbye"),
        ("thanks for the help", "thanks"),
        ("xyz random query that makes no sense", "unknown"),
        ("tell me about sports facilities", "sports"),
        ("what clubs are available?", "clubs"),
        ("how is the wifi on campus?", "wifi"),
        ("tell me about the canteen food", "canteen"),
        ("what about attendance rules?", "attendance"),
    ]

    correct = 0
    results = []
    for query, expected in test_cases:
        intent, confidence, response = classifier.predict(query)
        is_correct = intent == expected or (expected == "unknown" and confidence < 0.25)
        if is_correct:
            correct += 1
        results.append((query, expected, intent, confidence, is_correct))

    print(f"\n{'Query':<45} {'Expected':<15} {'Predicted':<15} {'Conf':>6} {'✓'}")
    print("-" * 95)
    for query, expected, predicted, conf, ok in results:
        status = "✓" if ok else "✗"
        color = "\033[32m" if ok else "\033[31m"
        reset = "\033[0m"
        print(f"{color}{query[:43]:<45} {expected:<15} {predicted:<15} {conf:>5.2f}  {status}{reset}")

    accuracy = correct / len(test_cases) * 100
    print(f"\n{'='*95}")
    print(f"  Test Accuracy: {correct}/{len(test_cases)} = {accuracy:.1f}%")

    if accuracy >= 90:
        print("  🟢 Model performance: EXCELLENT")
    elif accuracy >= 80:
        print("  🟢 Model performance: GOOD")
    elif accuracy >= 60:
        print("  🟡 Model performance: FAIR")
    else:
        print("  🔴 Model performance: POOR")

    return accuracy


def test_query(classifier, query: str):
    from backend.nlp.classifier import HIGH_CONFIDENCE, MEDIUM_CONFIDENCE
    print(f"\n📨 Query: '{query}'")
    print("-" * 50)
    intent, confidence, response = classifier.predict(query)
    print(f"🎯 Intent:     {intent}")
    print(f"📊 Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
    level = "🟢 HIGH" if confidence >= HIGH_CONFIDENCE else "🟡 MEDIUM" if confidence >= MEDIUM_CONFIDENCE else "🔴 LOW"
    print(f"📈 Level:      {level}")
    print(f"\n💬 Response:\n{response}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and test the college chatbot NLP model')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--test', type=str, metavar='QUERY')
    args = parser.parse_args()

    classifier = train()
    if args.evaluate or not args.test:
        evaluate(classifier)
    if args.test:
        test_query(classifier, args.test)

    print("\n✅ Done! Model is ready to use.")
    print("   Start the server with: python app.py")
