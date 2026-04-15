/**
 * Smart College Chatbot - Frontend JavaScript
 * SR University, Warangal
 */

// ─── State ────────────────────────────────────────────────────────────────────
const state = {
    isLoading: false,
    currentLanguage: 'en',
    isDarkMode: false,
    lastChatId: null,
    messageHistory: [],
    sidebarOpen: false
};

// ─── DOM References ───────────────────────────────────────────────────────────
const messagesArea = document.getElementById('messagesArea');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const clearBtn = document.getElementById('clearChat');
const charCount = document.getElementById('charCount');
const themeToggle = document.getElementById('themeToggle');
const sidebar = document.getElementById('sidebar');
const sidebarToggle = document.getElementById('sidebarToggle');
const voiceBtn = document.getElementById('voiceBtn');
const voiceOverlay = document.getElementById('voiceOverlay');
const welcomeScreen = document.getElementById('welcomeScreen');
const typingTemplate = document.getElementById('typingTemplate');

// ─── Initialization ───────────────────────────────────────────────────────────
function init() {
    // Restore theme
    const savedTheme = localStorage.getItem('theme') || 'light';
    if (savedTheme === 'dark') enableDarkMode(false);

    // Restore language
    const savedLang = localStorage.getItem('language') || 'en';
    setLanguage(savedLang, false);

    // Event listeners
    messageInput.addEventListener('input', onInputChange);
    messageInput.addEventListener('keydown', onKeyDown);
    sendBtn.addEventListener('click', sendMessage);
    clearBtn.addEventListener('click', clearChat);
    themeToggle.addEventListener('click', toggleTheme);
    sidebarToggle.addEventListener('click', toggleSidebar);

    // Quick topic buttons
    document.querySelectorAll('.topic-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            sendQuickQuery(btn.dataset.query);
            if (window.innerWidth <= 768) toggleSidebar();
        });
    });

    // Welcome chips
    document.querySelectorAll('.chip').forEach(chip => {
        chip.addEventListener('click', () => sendQuickQuery(chip.dataset.query));
    });

    // Close sidebar on outside click (mobile)
    document.addEventListener('click', (e) => {
        if (window.innerWidth <= 768 && 
            state.sidebarOpen && 
            !sidebar.contains(e.target) && 
            e.target !== sidebarToggle) {
            toggleSidebar();
        }
    });

    // Voice input setup
    setupVoiceInput();

    // Auto-resize textarea
    messageInput.addEventListener('input', autoResize);

    console.log('🎓 SR University Chatbot initialized');
}

// ─── Input Handling ───────────────────────────────────────────────────────────
function onInputChange() {
    const val = messageInput.value;
    sendBtn.disabled = val.trim().length === 0 || state.isLoading;
    charCount.textContent = val.length;

    // Character count color warning
    if (val.length > 450) {
        charCount.style.color = '#ef4444';
    } else if (val.length > 350) {
        charCount.style.color = '#f59e0b';
    } else {
        charCount.style.color = '';
    }
}

function onKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (!sendBtn.disabled) sendMessage();
    }
}

function autoResize() {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + 'px';
}

// ─── Send Message ─────────────────────────────────────────────────────────────
async function sendMessage() {
    const text = messageInput.value.trim();
    if (!text || state.isLoading) return;

    // Clear input
    messageInput.value = '';
    messageInput.style.height = 'auto';
    onInputChange();

    // Hide welcome screen
    if (welcomeScreen && welcomeScreen.parentNode) {
        welcomeScreen.style.opacity = '0';
        welcomeScreen.style.transform = 'translateY(-10px)';
        setTimeout(() => welcomeScreen.remove(), 300);
    }

    // Add user message
    addMessage('user', text);

    // Add typing indicator
    showTyping();

    // Set loading state
    state.isLoading = true;
    sendBtn.disabled = true;

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text })
        });

        const data = await response.json();

        // Remove typing indicator
        hideTyping();

        if (response.ok) {
            state.lastChatId = data.chat_id;
            addMessage('bot', data.response, {
                intent: data.intent,
                confidence: data.confidence,
                chatId: data.chat_id,
                timestamp: data.timestamp
            });

            // Save to history
            state.messageHistory.push({
                role: 'user', content: text, time: new Date().toISOString()
            });
            state.messageHistory.push({
                role: 'bot', content: data.response, 
                intent: data.intent, time: data.timestamp
            });
        } else {
            addMessage('bot', data.error || 'Something went wrong. Please try again.');
        }

    } catch (err) {
        hideTyping();
        console.error('Chat error:', err);
        addMessage('bot', '⚠️ Connection error. Please check your connection and try again.\n\nYou can also reach us directly:\n📞 +91-870-2427777');
    } finally {
        state.isLoading = false;
        sendBtn.disabled = messageInput.value.trim().length === 0;
    }
}

function sendQuickQuery(query) {
    messageInput.value = query;
    onInputChange();
    sendMessage();
}

// ─── Message Rendering ────────────────────────────────────────────────────────
function addMessage(role, content, meta = {}) {
    const messageEl = document.createElement('div');
    messageEl.className = `message ${role === 'bot' ? 'bot-message' : 'user-message'}`;

    const now = new Date();
    const timeStr = now.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' });

    if (role === 'bot') {
        const confPercent = meta.confidence ? Math.round(meta.confidence * 100) : null;

        messageEl.innerHTML = `
            <div class="message-avatar">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                    <circle cx="12" cy="8" r="4"/><path d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/>
                </svg>
            </div>
            <div>
                <div class="message-bubble">${formatBotMessage(content)}</div>
                <div class="message-meta">
                    <span class="message-time">${timeStr}</span>
                    ${confPercent ? `<span class="confidence-badge">${confPercent}% match</span>` : ''}
                    ${meta.chatId ? `
                    <div class="feedback-btns">
                        <button class="feedback-btn thumb-up" onclick="sendFeedback(${meta.chatId}, 5)" title="Helpful">👍</button>
                        <button class="feedback-btn thumb-down" onclick="sendFeedback(${meta.chatId}, 1)" title="Not helpful">👎</button>
                    </div>` : ''}
                </div>
            </div>
        `;
    } else {
        messageEl.innerHTML = `
            <div class="message-bubble">${escapeHtml(content)}</div>
            <div class="message-meta" style="justify-content:flex-end">
                <span class="message-time">${timeStr}</span>
            </div>
        `;
    }

    messagesArea.appendChild(messageEl);
    scrollToBottom();
}

function formatBotMessage(text) {
    // Convert markdown-like formatting to HTML
    return escapeHtml(text)
        // Bold **text**
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        // Italic *text*
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        // Bullet points: lines starting with •
        .replace(/^•\s(.+)$/gm, '<span style="display:block;padding-left:1em">• $1</span>')
        // Checkmarks: lines starting with ✅
        .replace(/^(✅|❌|📌|🔗|📅|📞|📧|⚠️|✔)\s(.+)$/gm, '<span style="display:block;padding:1px 0">$1 $2</span>')
        // Newlines
        .replace(/\n/g, '<br>');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.appendChild(document.createTextNode(text));
    return div.innerHTML;
}

// ─── Typing Indicator ─────────────────────────────────────────────────────────
function showTyping() {
    const node = typingTemplate.content.cloneNode(true);
    messagesArea.appendChild(node);
    scrollToBottom();
}

function hideTyping() {
    const indicator = document.getElementById('typingIndicator');
    if (indicator) {
        indicator.style.opacity = '0';
        setTimeout(() => indicator.remove(), 200);
    }
}

// ─── Feedback ─────────────────────────────────────────────────────────────────
async function sendFeedback(chatId, rating) {
    try {
        await fetch('/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ chat_id: chatId, rating })
        });
        // Visual feedback
        showToast(rating >= 4 ? '👍 Thanks for the feedback!' : '📝 Feedback noted. We\'ll improve!');
    } catch (err) {
        console.error('Feedback error:', err);
    }
}

// ─── Utility Functions ────────────────────────────────────────────────────────
function scrollToBottom() {
    requestAnimationFrame(() => {
        messagesArea.scrollTop = messagesArea.scrollHeight;
    });
}

function clearChat() {
    // Confirm
    if (state.messageHistory.length > 0) {
        if (!confirm('Clear all messages? This cannot be undone.')) return;
    }

    // Remove all messages except welcome screen placeholder
    while (messagesArea.firstChild) {
        messagesArea.removeChild(messagesArea.firstChild);
    }

    // Re-add welcome screen
    const ws = document.createElement('div');
    ws.className = 'welcome-screen';
    ws.id = 'welcomeScreen';
    ws.innerHTML = `
        <div class="welcome-glyph">
            <svg width="56" height="56" viewBox="0 0 56 56" fill="none">
                <path d="M28 4L52 16V40L28 52L4 40V16L28 4Z" stroke="currentColor" stroke-width="1.5" fill="none" opacity="0.3"/>
                <path d="M28 12L44 20V36L28 44L12 36V20L28 12Z" fill="currentColor" opacity="0.15"/>
                <circle cx="28" cy="28" r="7" fill="currentColor" opacity="0.6"/>
                <circle cx="28" cy="28" r="3" fill="currentColor"/>
            </svg>
        </div>
        <h1 class="welcome-title">Hello, I'm EduBot 👋</h1>
        <p class="welcome-subtitle">Your intelligent assistant for SR University, Warangal. Ask me anything about admissions, courses, fees, placements and more.</p>
        <div class="welcome-chips">
            <button class="chip" data-query="How to apply for admission?">How to apply?</button>
            <button class="chip" data-query="What is the fee structure?">Fee structure</button>
            <button class="chip" data-query="Tell me about placements">Placements</button>
            <button class="chip" data-query="What courses are offered?">Courses offered</button>
        </div>
    `;
    messagesArea.appendChild(ws);

    // Re-bind chips
    ws.querySelectorAll('.chip').forEach(chip => {
        chip.addEventListener('click', () => sendQuickQuery(chip.dataset.query));
    });

    state.messageHistory = [];
    state.lastChatId = null;
}

function toggleTheme() {
    state.isDarkMode ? disableDarkMode() : enableDarkMode();
}

function enableDarkMode(save = true) {
    document.documentElement.setAttribute('data-theme', 'dark');
    document.querySelector('.icon-sun').style.display = 'none';
    document.querySelector('.icon-moon').style.display = '';
    state.isDarkMode = true;
    if (save) localStorage.setItem('theme', 'dark');
}

function disableDarkMode(save = true) {
    document.documentElement.removeAttribute('data-theme');
    document.querySelector('.icon-sun').style.display = '';
    document.querySelector('.icon-moon').style.display = 'none';
    state.isDarkMode = false;
    if (save) localStorage.setItem('theme', 'light');
}

function toggleSidebar() {
    state.sidebarOpen = !state.sidebarOpen;
    sidebar.classList.toggle('open', state.sidebarOpen);
}

function setLanguage(lang, save = true) {
    state.currentLanguage = lang;
    document.getElementById('lang-en')?.classList.toggle('active', lang === 'en');
    document.getElementById('lang-te')?.classList.toggle('active', lang === 'te');
    if (save) localStorage.setItem('language', lang);

    if (lang === 'te') {
        messageInput.placeholder = 'మీ ప్రశ్న అడగండి...';
    } else {
        messageInput.placeholder = 'Ask about admissions, fees, courses, placements...';
    }
}

function showToast(message) {
    const toast = document.createElement('div');
    toast.style.cssText = `
        position: fixed; bottom: 100px; left: 50%; transform: translateX(-50%);
        background: var(--bot-bubble); color: var(--bot-bubble-text);
        padding: 10px 20px; border-radius: 100px; font-size: 0.85rem;
        box-shadow: var(--shadow-md); z-index: 9999;
        animation: fadeInUp 0.3s ease;
    `;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => { toast.style.opacity = '0'; setTimeout(() => toast.remove(), 300); }, 2500);
}

// ─── Voice Input ──────────────────────────────────────────────────────────────
let recognition = null;
let isRecording = false;

function setupVoiceInput() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    
    if (!SpeechRecognition) {
        voiceBtn.style.opacity = '0.3';
        voiceBtn.title = 'Voice input not supported in this browser';
        voiceBtn.disabled = true;
        return;
    }

    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = state.currentLanguage === 'te' ? 'te-IN' : 'en-IN';

    recognition.onresult = (event) => {
        const transcript = Array.from(event.results)
            .map(r => r[0].transcript).join('');
        messageInput.value = transcript;
        onInputChange();
        autoResize();
    };

    recognition.onend = () => {
        stopVoiceRecording();
        if (messageInput.value.trim()) {
            setTimeout(sendMessage, 500);
        }
    };

    recognition.onerror = (e) => {
        console.error('Speech error:', e.error);
        stopVoiceRecording();
        if (e.error !== 'aborted') {
            showToast('Voice input failed. Please try again.');
        }
    };

    voiceBtn.addEventListener('click', toggleVoiceRecording);
}

function toggleVoiceRecording() {
    if (isRecording) {
        stopVoiceRecording();
    } else {
        startVoiceRecording();
    }
}

function startVoiceRecording() {
    if (!recognition) return;
    recognition.lang = state.currentLanguage === 'te' ? 'te-IN' : 'en-IN';
    recognition.start();
    isRecording = true;
    voiceBtn.classList.add('recording');
    voiceOverlay.style.display = 'flex';
}

function stopVoiceRecording() {
    if (recognition && isRecording) {
        recognition.stop();
    }
    isRecording = false;
    voiceBtn.classList.remove('recording');
    voiceOverlay.style.display = 'none';
}

// Make stopVoiceRecording available globally (called from HTML)
window.stopVoiceRecording = stopVoiceRecording;
window.setLanguage = setLanguage;
window.sendFeedback = sendFeedback;

// ─── Start ────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', init);
