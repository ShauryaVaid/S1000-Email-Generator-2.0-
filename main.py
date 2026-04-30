"""AI Smart Email Reply Generator for Professionals - Optimized for Speed."""

import streamlit as st
import yaml
import os
import threading
import atexit
from pathlib import Path
from datetime import datetime

from ai_client import AIClient
from chroma_manager import ChromaManager
from email_retriever import EmailRetriever
from telegram_bot import TelegramBot, create_bot
from st_copy_to_clipboard import st_copy_to_clipboard
import subprocess
import sys
import signal

def cleanup_telegram_bot():
    """Clean up Telegram bot on exit."""
    temp_script = Path(__file__).parent / "telegram_runner_temp.py"
    if temp_script.exists():
        temp_script.unlink()

atexit.register(cleanup_telegram_bot)


# =============================================================================
# Configuration Loading (Cached)
# =============================================================================

@st.cache_resource
def load_config():
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_prompts():
    """Load prompts from prompts.yaml."""
    prompts_path = Path(__file__).parent / "prompts.yaml"
    with open(prompts_path, "r") as f:
        return yaml.safe_load(f)


# Load configurations once
config = load_config()
prompts = load_prompts()


# =============================================================================
# Initialize Clients (Cached)
# =============================================================================

@st.cache_resource
def get_ai_client_cached(provider: str, model: str, base_url: str, api_key: str) -> AIClient:
    """Get cached AI client."""
    ai_config = {
        'provider': provider,
        'model': model,
        'base_url': base_url,
        'api_key': api_key,
        'delay': config.get('generation', {}).get('delay', 0.05),
        'max_workers': config.get('generation', {}).get('max_workers', 50)
    }
    return AIClient(ai_config)


def get_ai_client(provider: str = None, model: str = None) -> AIClient:
    """Get configured AI client with fast lookup."""
    provider = provider or config.get('provider', 'ollama')

    if provider == 'ollama':
        model = model or config['ollama'].get('model', 'llama3.1')
        base_url = config['ollama']['base_url']
        api_key = config['ollama']['api_key']
    elif provider == 'ollama' or provider == 'ollama-cloud' or provider == 'minimax-cloud':
        if provider == 'ollama':
            model = model or config['ollama'].get('model', 'llama3.1')
            base_url = config['ollama']['base_url']
            api_key = config['ollama']['api_key']
        else:  # minimax-cloud or ollama-cloud
            model = model or 'minimax-m2.7:cloud'
            base_url = 'https://ollama.com'
            api_key = config['ollama'].get('cloud_api_key', '')
        provider = 'minimax-cloud' if provider == 'ollama-cloud' else provider
    elif provider == 'gemini':
        model = model or config['gemini']['model']
        base_url = 'https://generativelanguage.googleapis.com/v1beta'
        api_key = config['gemini']['api_key']
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return get_ai_client_cached(provider, model, base_url, api_key)


@st.cache_resource
def get_chroma_manager_cached():
    """Get cached ChromaDB manager."""
    chroma_config = config.get('chromadb', {})
    try:
        return ChromaManager(
            persist_directory=chroma_config.get('persist_directory', './chroma_db'),
            collection_name=chroma_config.get('collection_name', 'email_context')
        )
    except Exception as e:
        print(f"ChromaDB init error: {e}")
        return None


def get_chroma_manager():
    """Get ChromaDB manager (returns None if unavailable)."""
    return get_chroma_manager_cached()


def save_to_chroma_async(content: str, metadata: dict, email_id: str):
    """Save to ChromaDB in background thread (non-blocking)."""
    try:
        chroma = get_chroma_manager()
        if chroma:
            chroma.add_email(email_id=email_id, content=content, metadata=metadata)
    except Exception as e:
        print(f"Async ChromaDB save error: {e}")


# =============================================================================
# Email Generation Logic (Optimized)
# =============================================================================

def generate_email_fast(
    rough_message: str,
    tone: str,
    ai_client: AIClient
) -> str:
    """
    Generate email FAST - optimized for speed.
    Uses simple prompt without JSON overhead.
    """
    # Simplified system prompt for faster generation
    system_prompt = f"You are a professional email assistant. Write a {tone} email based on the notes below."

    # Direct prompt - no JSON parsing overhead
    prompt = f"""Write a professional email with a {tone} tone based on these notes:

Notes: {rough_message}

Include a clear subject line and proper greeting/sign-off.

Email:"""

    response = ai_client.generate_text(
        prompt=prompt,
        system_prompt=system_prompt
    )

    return response if response else "Error: Could not generate email"


# =============================================================================
# Streamlit UI (Optimized)
# =============================================================================

st.set_page_config(
    page_title="AI Email Generator S1000 - Fast",
    page_icon="⚡",
    layout="wide"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    /* Main gradient background */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }

    /* Title styling */
    .main-title {
        font-size: 2.8rem !important;
        font-weight: 700;
        background: linear-gradient(90deg, #00d4ff, #7b2cbf, #ff006e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: glow 2s ease-in-out infinite alternate;
    }

    @keyframes glow {
        from { filter: drop-shadow(0 0 5px #00d4ff); }
        to { filter: drop-shadow(0 0 20px #7b2cbf); }
    }

    /* Card styling */
    .stCard {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
    }

    /* Sidebar styling */
    .css-sidebar .stSidebar {
        background: rgba(26, 26, 46, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Button glow effect */
    .stButton > button:hover {
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
        transition: all 0.3s ease;
    }

    /* Input field focus */
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #00d4ff !important;
        box-shadow: 0 0 10px rgba(0, 212, 255, 0.3) !important;
    }

    /* Metric styling */
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Success/Info boxes */
    .stAlert {
        border-radius: 10px;
    }

    /* Smooth animations */
    .stSpinner > div {
        border-color: #00d4ff !important;
    }

    /* Radio button styling */
    .stRadio > div {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 10px;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(0, 212, 255, 0.5);
        border-radius: 4px;
    }

    /* Pulse animation for running status */
    .pulse {
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }

    /* Result box styling */
    .result-box {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(123, 44, 191, 0.1));
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">⚡ AI Email Generator S1000</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #888; font-size: 1.1rem;">✨ Transform rough notes into professional emails in seconds ✨</p>', unsafe_allow_html=True)
st.markdown("---")

# Initialize session state (lightweight)
if 'tone' not in st.session_state:
    st.session_state.tone = "Professional"
if 'provider' not in st.session_state:
    st.session_state.provider = config.get('provider', 'ollama')
if 'model' not in st.session_state:
    st.session_state.model = config['ollama'].get('model', 'llama3.1')
if 'ollama_cloud_api_key' not in st.session_state:
    st.session_state.ollama_cloud_api_key = config['ollama'].get('cloud_api_key', '96dc03fd305a421491186c268d334915.XE726r2RgxOc6UrFxhe7N3XG')
if 'telegram_bot_running' not in st.session_state:
    st.session_state.telegram_bot_running = False
if 'telegram_bot_thread' not in st.session_state:
    st.session_state.telegram_bot_thread = None

# Sidebar for settings
with st.sidebar:
    st.markdown("""
    <style>
        .sidebar-title {
            font-size: 1.5rem;
            font-weight: 600;
            color: #00d4ff;
            margin-bottom: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="sidebar-title">⚙️ Settings</p>', unsafe_allow_html=True)

    # Provider selection with cloud options
    provider = st.selectbox(
        "AI Provider",
        ["ollama-local", "ollama-cloud", "gemini"],
        format_func=lambda x: x.replace('-', ' ').title(),
        index=0 if st.session_state.provider == 'ollama' else (1 if st.session_state.provider == 'ollama-cloud' else 2 if st.session_state.provider == 'gemini' else 0)
    )

    # Map provider selection
    if provider == "ollama-local":
        provider_key = "ollama"
        available_models = ["llama3.1", "llama3.2", "mistral", "gemma2", "phi3"]
    elif provider == "ollama-cloud":
        provider_key = "minimax-cloud"
        # Minimax Cloud via Ollama.com API
        available_models = ["minimax-m2.7:cloud"]
    else:
        provider_key = "gemini"
        available_models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"]

    # Model selection
    selected_model = st.selectbox(
        "Model",
        available_models,
        index=0
    )

    # Update session state
    st.session_state.provider = provider_key
    st.session_state.model = selected_model
    if provider_key == "ollama-cloud":
        st.info("☁️ Using Ollama cloud - faster for heavy models")
    elif provider_key == "gemini":
        st.info("☁️ Using Google Gemini API")
    else:
        st.info("🏠 Using local Ollama - fastest for llama3.1")

    # Tone selection
    tone_options = ["Professional", "Polite", "Assertive", "Friendly", "GenZ", "Casual", "Brief"]
    st.session_state.tone = st.selectbox(
        "Default Tone",
        tone_options,
        index=tone_options.index(st.session_state.tone) if st.session_state.tone in tone_options else 0
    )

    st.markdown("---")

    # Telegram Bot Control
    st.markdown("""
    <style>
        .telegram-header {
            font-size: 1.2rem;
            font-weight: 600;
            color: #7b2cbf;
        }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<p class="telegram-header">🤖 Telegram Bot</p>', unsafe_allow_html=True)

    telegram_config = config.get('telegram', {})
    bot_enabled = telegram_config.get('enabled', False)
    bot_token = telegram_config.get('bot_token')

    if bot_enabled and bot_token:
        if st.session_state.telegram_bot_running:
            st.success("✅ Bot is running!")
            if st.button("⏹️ Stop Bot", use_container_width=True):
                # Terminate the bot process
                if st.session_state.telegram_bot_thread:
                    st.session_state.telegram_bot_thread.terminate()
                    st.session_state.telegram_bot_thread = None
                st.session_state.telegram_bot_running = False
                # Clean up temp script
                temp_script = Path(__file__).parent / "telegram_runner_temp.py"
                if temp_script.exists():
                    temp_script.unlink()
                st.rerun()
        else:
            st.info("⏸️ Bot is stopped")
            if st.button("▶️ Start Bot", use_container_width=True, type="primary"):
                st.session_state.telegram_bot_running = True
                st.rerun()
        # Show bot info
        bot_username = telegram_config.get('bot_username', '@YourBot')
        st.caption(f"Bot: {bot_username}")
    else:
        st.caption("⚙️ Enable in config.yaml to use")

    st.markdown("---")

    # ChromaDB status (lightweight check)
    st.subheader("💾 Memory")
    chroma = get_chroma_manager()
    if chroma:
        try:
            stats = chroma.get_collection_stats()
            st.metric("Stored Emails", stats['total_emails'])
        except Exception:
            st.caption("Memory active")
    else:
        st.caption("Memory disabled (faster)")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <style>
        .input-label {
            font-size: 1.2rem;
            font-weight: 600;
            color: #00d4ff;
            margin-bottom: 10px;
        }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<p class="input-label">📝 Write Your Rough Message</p>', unsafe_allow_html=True)
    rough_message = st.text_area(
        "",
        placeholder="e.g., tell boss im sick, wont come today, will finish report by monday",
        height=150,
        key="rough_input",
        label_visibility="collapsed"
    )

with col2:
    st.markdown("### 🎯 Tone")
    tone = st.radio(
        "Select:",
        ["Professional", "Polite", "Assertive", "Friendly", "GenZ", "Casual", "Brief"],
        index=tone_options.index(st.session_state.tone) if st.session_state.tone in tone_options else 0,
        label_visibility="collapsed"
    )

    st.markdown("""
    <style>
        .tips-box {
            background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(123, 44, 191, 0.1));
            border-radius: 10px;
            padding: 15px;
            margin-top: 15px;
        }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="tips-box">
        <p style="color: #00d4ff; font-weight: 600;">💡 Tips</p>
        <p style="color: #aaa; font-size: 0.9rem;">
        • Keep notes short for faster results<br>
        • Local llama3.1 = fastest<br>
        • Cloud minimax = smarter
        </p>
    </div>
    """, unsafe_allow_html=True)

# Generate button with speed optimization
st.markdown("")
col_btn, = st.columns([1])
with col_btn:
    st.markdown("""
    <style>
        .generate-btn > button {
            background: linear-gradient(135deg, #00d4ff, #7b2cbf);
            color: white;
            font-size: 1.2rem;
            font-weight: 600;
            padding: 15px 30px;
            border: none;
            border-radius: 15px;
            transition: all 0.3s ease;
            width: 100%;
        }
        .generate-btn > button:hover {
            transform: scale(1.02);
            box-shadow: 0 0 30px rgba(0, 212, 255, 0.6);
        }
    </style>
    """, unsafe_allow_html=True)

    if st.markdown('<div class="generate-btn">', unsafe_allow_html=True):
        pass

    if st.button("⚡ Generate Email", use_container_width=True, type="primary"):
        if not rough_message.strip():
            st.warning("⚠️ Enter your rough message first!")
        else:
            # Get AI client with selected provider/model
            ai_client = get_ai_client(provider=st.session_state.provider, model=st.session_state.model)

            with st.spinner(f"✨ Generating {tone.lower()} email..."):
                try:
                    # FAST generation - no JSON overhead
                    generated_email = generate_email_fast(
                        rough_message=rough_message.strip(),
                        tone=tone,
                        ai_client=ai_client
                    )

                    # Display result with styled box
                    st.markdown("""
                    <style>
                        .result-header {
                            font-size: 1.5rem;
                            font-weight: 600;
                            color: #00ff88;
                            margin-bottom: 15px;
                        }
                    </style>
                    <p class="result-header">✅ Generated Result</p>
                    """, unsafe_allow_html=True)

                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.code(generated_email, language="text")
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Copy button
                    st_copy_to_clipboard("📋 Copy to Clipboard", generated_email)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Save to ChromaDB in BACKGROUND (non-blocking)
                    import hashlib
                    email_id = hashlib.md5(f"{rough_message}{tone}".encode()).hexdigest()
                    threading.Thread(
                        target=save_to_chroma_async,
                        args=(generated_email, {'tone': tone, 'original': rough_message}, email_id),
                        daemon=True
                    ).start()

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("💡 Tip: Make sure Ollama is running (ollama serve) or check your internet connection")

# Quick stats
st.markdown("---")
col_stats1, col_stats2, col_stats3 = st.columns(3)
with col_stats1:
    st.metric("Provider", st.session_state.provider.title())
with col_stats2:
    st.metric("Model", st.session_state.model)
with col_stats3:
    st.metric("Tone", tone)

# Email retrieval section (collapsed by default)
with st.expander("📧 Fetch Gmail Messages", expanded=False):
    col_email1, col_email2, col_email3 = st.columns([2, 2, 1])

    with col_email1:
        email_address = st.text_input("Email Address", placeholder="your@gmail.com", key="email_addr")
    with col_email2:
        password = st.text_input("App Password", type="password", key="email_pass")
    with col_email3:
        fetch_btn = st.button("Fetch", use_container_width=True)

    if fetch_btn and email_address and password:
        retriever = EmailRetriever(config.get('email', {}))
        with st.spinner("Fetching emails..."):
            emails = retriever.fetch_emails(email_address=email_address, password=password, limit=5)
            if emails:
                st.success(f"Fetched {len(emails)} emails")
                for email_item in emails:
                    with st.expander(f"📩 {email_item['subject']} - {email_item['from']}"):
                        st.write(f"**Date:** {email_item['date']}")
                        st.write(email_item['body'][:500] + "...")
            else:
                st.warning("No emails found")

# Telegram Bot Runner (when started from sidebar)
st.markdown("---")
if st.session_state.telegram_bot_running:
    telegram_config = config.get('telegram', {})
    if telegram_config.get('enabled', False) and telegram_config.get('bot_token'):
        # Check if we need to start the bot
        if st.session_state.telegram_bot_thread is None:
            # Start bot in a separate process
            try:
                # Create a temporary runner script
                runner_script = Path(__file__).parent / "telegram_runner_temp.py"

                # Get provider-specific config
                provider = st.session_state.provider
                model = st.session_state.model

                if provider == 'ollama':
                    provider_base_url = config['ollama']['base_url']
                    provider_api_key = config['ollama']['api_key']
                elif provider == 'minimax-cloud' or provider == 'ollama-cloud':
                    provider_base_url = 'https://ollama.com'
                    provider_api_key = config['ollama'].get('cloud_api_key', '')
                    provider = 'minimax-cloud'
                else:  # gemini
                    provider_base_url = 'https://generativelanguage.googleapis.com/v1beta'
                    provider_api_key = config['gemini']['api_key']

                runner_content = f'''
import sys
import os
sys.path.insert(0, r"{Path(__file__).parent}")
os.chdir(r"{Path(__file__).parent}")

import asyncio
from telegram_bot import create_bot
from ai_client import AIClient
import yaml

config_path = os.path.join(r"{Path(__file__).parent}", "config.yaml")
prompts_path = os.path.join(r"{Path(__file__).parent}", "prompts.yaml")

with open(config_path) as f:
    config = yaml.safe_load(f)

with open(prompts_path) as f:
    prompts = yaml.safe_load(f)

ai_config = {{
    'provider': '{provider}',
    'model': '{model}',
    'base_url': '{provider_base_url}',
    'api_key': '{provider_api_key}',
}}
ai_client = AIClient(ai_config)

bot = create_bot(
    bot_token="{telegram_config['bot_token']}",
    ai_client=ai_client,
    prompts=prompts
)
print("Telegram Bot Started! Press Ctrl+C to stop.")
bot.start_sync()
'''
                with open(runner_script, "w") as f:
                    f.write(runner_content)

                # Start as subprocess
                proc = subprocess.Popen(
                    [sys.executable, str(runner_script)],
                    cwd=str(Path(__file__).parent),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                st.session_state.telegram_bot_thread = proc
                st.success("🤖 Telegram Bot started successfully!")
                st.caption("Bot is running in background. Use /start in Telegram to begin.")
            except Exception as e:
                st.error(f"Failed to start bot: {e}")
                st.session_state.telegram_bot_running = False
        else:
            st.success("🤖 Telegram Bot is running in background")
            st.caption("Message your bot on Telegram to generate emails!")
