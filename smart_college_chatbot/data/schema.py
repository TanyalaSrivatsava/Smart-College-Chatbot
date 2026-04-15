"""
Database initialization and schema for Smart College Chatbot
"""

SCHEMA_SQL = """
-- FAQs Table
CREATE TABLE IF NOT EXISTS faqs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    keywords TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active INTEGER DEFAULT 1
);

-- Chat History Table
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    user_message TEXT NOT NULL,
    bot_response TEXT NOT NULL,
    intent TEXT,
    confidence REAL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ip_address TEXT
);

-- Admin Users Table
CREATE TABLE IF NOT EXISTS admin_users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

-- Feedback Table
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_history_id INTEGER,
    rating INTEGER CHECK(rating BETWEEN 1 AND 5),
    comment TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (chat_history_id) REFERENCES chat_history(id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_chat_history_session ON chat_history(session_id);
CREATE INDEX IF NOT EXISTS idx_faqs_category ON faqs(category);
CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating);
"""

SAMPLE_FAQS = [
    {
        "category": "admissions",
        "question": "What is the last date for admission?",
        "answer": "Admissions are typically open from April to August. The last date for regular admissions is usually July 31st. Spot admissions may be available until September. Contact admissions@sruniversity.ac.in for exact dates.",
        "keywords": "last date, deadline, admission, apply"
    },
    {
        "category": "admissions",
        "question": "What documents are required for admission?",
        "answer": "Required documents: 1) 10th Marksheet & Certificate 2) 12th Marksheet & Certificate 3) Transfer Certificate 4) Migration Certificate 5) EAMCET/JEE Rank Card 6) Aadhar Card 7) Passport Photos (6) 8) Caste Certificate (if applicable) 9) Income Certificate (for scholarship) 10) Medical Certificate",
        "keywords": "documents, certificates, required, admission"
    },
    {
        "category": "fees",
        "question": "Are there any scholarships for SC/ST students?",
        "answer": "Yes! SC/ST students can avail TS ePass scholarship which covers full tuition fee and maintenance allowance. Apply at telanganaepass.cgg.gov.in. Additionally, the university provides special coaching and support for SC/ST students.",
        "keywords": "scholarship, sc, st, epass, fee waiver"
    },
    {
        "category": "exams",
        "question": "How are internal marks calculated?",
        "answer": "Internal marks (30 total): 1) Two Mid-Term exams: 10+10 = 20 marks (average of best scores) 2) Assignment/Seminar: 5 marks 3) Attendance: 5 marks (5 marks for 90%+, 4 for 85%+, etc.). You need minimum 12/30 in internals.",
        "keywords": "internal marks, mid term, assignment, attendance marks"
    },
    {
        "category": "placements",
        "question": "What is the minimum CGPA required for placements?",
        "answer": "Most companies require minimum 6.0 CGPA for placement eligibility. Some top companies (Google, Microsoft, Amazon) require 7.5+ CGPA. No active backlogs are allowed. Training & Placement cell provides guidance - visit them at Admin Block, 1st Floor.",
        "keywords": "cgpa, placement eligibility, minimum marks, backlog"
    },
    {
        "category": "hostel",
        "question": "Is there a curfew in the hostel?",
        "answer": "Yes, hostel timings: Boys: 10:30 PM curfew. Girls: 9:00 PM curfew. Gates close at these times. Late permissions can be obtained from warden for special occasions. Parents must give written consent for late permissions.",
        "keywords": "hostel curfew, timing, gate close, night out"
    },
    {
        "category": "courses",
        "question": "Is there lateral entry admission for diploma holders?",
        "answer": "Yes! Diploma holders can get direct admission to 2nd year B.Tech through lateral entry. Eligibility: Diploma in relevant branch with 60%+ marks. Seats: 20% of total intake. Apply through TS ECET examination.",
        "keywords": "lateral entry, diploma, ecet, second year"
      },
    {
        "category": "general",
        "question": "What are the college timings?",
        "answer": "College working hours: Monday to Friday: 9:00 AM to 5:00 PM. Classes: 9:15 AM to 4:15 PM (8 periods of 50 mins each). Office hours: 9:00 AM to 5:00 PM. Library: 8:00 AM to 9:00 PM.",
        "keywords": "college timings, class hours, working hours, schedule"
    }
]
