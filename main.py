"""AI Smart Email Reply Generator for Professionals - Optimized for Speed & Token Efficiency."""

import streamlit as st
import yaml
import os
import threading
import atexit
from pathlib import Path
from datetime import datetime
import subprocess
import sys
import signal

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from ai_client import AIClient
from chroma_manager import ChromaManager
from email_retriever import EmailRetriever
from telegram_bot import TelegramBot, create_bot
from st_copy_to_clipboard import st_copy_to_clipboard


def cleanup_telegram_bot():
    """Clean up Telegram bot on exit."""
    temp_script = Path(__file__).parent / "telegram_runner_temp.py"
    if temp_script.exists():
        try:
            temp_script.unlink()
        except Exception:
            pass


atexit.register(cleanup_telegram_bot)


# =============================================================================
# Configuration Loading (Cached)
# =============================================================================

@st.cache_resource
def load_config():
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_prompts():
    """Load prompts from prompts.yaml."""
    prompts_path = Path(__file__).parent / "prompts.yaml"
    with open(prompts_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Load configurations once
config = load_config()
prompts = load_prompts()


def get_gemini_api_key() -> str:
    """
    Retrieve Gemini API key securely from the background.
    Never exposes key to frontend or URL query params.
    Order of precedence:
      1. Streamlit secrets (production cloud or local secrets.toml)
      2. Environment variables (GEMINI_API_KEY, GOOGLE_API_KEY)
      3. config.yaml (fallback)
    """
    # 1. Streamlit secrets
    try:
        if hasattr(st, "secrets"):
            if "GEMINI_API_KEY" in st.secrets:
                return str(st.secrets["GEMINI_API_KEY"]).strip()
            if "gemini" in st.secrets and isinstance(st.secrets["gemini"], dict) and "api_key" in st.secrets["gemini"]:
                return str(st.secrets["gemini"]["api_key"]).strip()
    except Exception:
        pass

    # 2. Environment variables
    for env_key in ["GEMINI_API_KEY", "GOOGLE_API_KEY"]:
        val = os.environ.get(env_key)
        if val:
            return val.strip()

    # 3. Fallback to config.yaml
    return config.get('gemini', {}).get('api_key', '').strip()


# =============================================================================
# Initialize Clients (Cached)
# =============================================================================

@st.cache_resource
def get_ai_client_cached(provider: str, model: str, base_url: str, api_key: str, max_output_tokens: int = 800) -> AIClient:
    """Get cached AI client instance with token constraints."""
    ai_config = {
        'provider': provider,
        'model': model,
        'embedding_model': config.get('gemini', {}).get('embedding_model', 'gemini-embedding-001'),
        'base_url': base_url,
        'api_key': api_key,
        'delay': config.get('generation', {}).get('delay', 0.05),
        'max_workers': config.get('generation', {}).get('max_workers', 20),
        'max_output_tokens': max_output_tokens,
        'temperature': config.get('generation', {}).get('temperature', 0.5)
    }
    return AIClient(ai_config)


def get_ai_client(provider: str = None, model: str = None) -> AIClient:
    """Get configured AI client with secure background key lookup."""
    provider = provider or config.get('provider', 'gemini')

    if provider == 'gemini':
        model = model or config.get('gemini', {}).get('model', 'gemini-2.5-flash')
        base_url = 'https://generativelanguage.googleapis.com/v1beta'
        api_key = get_gemini_api_key()
        max_output_tokens = config.get('gemini', {}).get('max_output_tokens', 800)
    elif provider in ['ollama', 'ollama-local']:
        provider = 'ollama'
        model = model or config.get('ollama', {}).get('model', 'llama3.1')
        base_url = config.get('ollama', {}).get('base_url', 'http://localhost:11434')
        api_key = config.get('ollama', {}).get('api_key', 'ollama')
        max_output_tokens = 800
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return get_ai_client_cached(provider, model, base_url, api_key, max_output_tokens)


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
# Email Generation Logic (Optimized for Speed & Strict Token Limits)
# =============================================================================

def generate_email_fast(
    rough_message: str,
    tone: str,
    ai_client: AIClient
) -> str:
    """
    Generate email with speed optimization, prompt hardening, and token control.
    """
    trimmed_notes = rough_message.strip()[:3000]

    system_prompt = (
        f"You are an executive email assistant. "
        f"Draft a crisp, high-impact email in a {tone.lower()} tone based on the user's notes. "
        f"Structure required: "
        f"1. A clean 'Subject:' line at the top. "
        f"2. A professional greeting. "
        f"3. A concise, clear body. "
        f"4. A courteous sign-off. "
        f"Keep the entire output under 250 words to conserve tokens. Do not add markdown commentary or explanations."
    )

    prompt = f"Tone: {tone}\nUser Notes:\n{trimmed_notes}\n\nDraft Email:"

    response = ai_client.generate_text(
        prompt=prompt,
        system_prompt=system_prompt
    )

    return response if response else "Error: Could not generate email response."


# =============================================================================
# Streamlit UI
# =============================================================================

st.set_page_config(
    page_title="AI Email Generator S1000 - Fast & Smart",
    page_icon="⚡",
    layout="wide"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    /* Main gradient background */
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0f2744 100%);
    }

    /* Title styling */
    .main-title {
        font-size: 2.8rem !important;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #00d4ff, #7b2cbf, #ff006e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem;
    }

    /* Subtitle styling */
    .sub-title {
        text-align: center;
        color: #8b949e;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }

    /* Card styling */
    .stCard {
        background: rgba(255, 255, 255, 0.04);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 20px;
    }

    /* Sidebar styling */
    .css-sidebar .stSidebar {
        background: rgba(13, 17, 23, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Button glow effect */
    .stButton > button {
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.4);
        transform: translateY(-1px);
    }

    /* Input field focus */
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #00d4ff !important;
        box-shadow: 0 0 12px rgba(0, 212, 255, 0.3) !important;
    }

    /* Metric styling */
    .stMetric {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 12px 18px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Result box styling */
    .result-box {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.06), rgba(123, 44, 191, 0.06));
        border: 1px solid rgba(0, 212, 255, 0.25);
        border-radius: 14px;
        padding: 20px;
        margin: 12px 0;
    }

    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 6px;
        background: rgba(0, 212, 255, 0.15);
        color: #00d4ff;
        border: 1px solid rgba(0, 212, 255, 0.3);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">⚡ AI Email Generator S1000</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">✨ Transform rough notes into polished, executive-ready emails in seconds ✨</p>', unsafe_allow_html=True)

# Initialize session state
if 'tone' not in st.session_state:
    st.session_state.tone = "Professional"
if 'provider' not in st.session_state:
    st.session_state.provider = config.get('provider', 'gemini')
if 'model' not in st.session_state:
    st.session_state.model = config.get('gemini', {}).get('model', 'gemini-2.5-flash')
if 'telegram_bot_running' not in st.session_state:
    st.session_state.telegram_bot_running = False
if 'telegram_bot_thread' not in st.session_state:
    st.session_state.telegram_bot_thread = None

# Sidebar for settings
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom: 15px;">
        <h3 style="color: #00d4ff; margin: 0;">⚙️ Engine Settings</h3>
        <p style="color: #8b949e; font-size: 0.85rem; margin-top: 4px;">Configure model & parameters</p>
    </div>
    """, unsafe_allow_html=True)

    # Provider selection: Google Gemini (Cloud) as primary
    provider_display_options = ["Google Gemini (Cloud)", "Ollama (Local)"]
    current_p_index = 0 if st.session_state.provider == 'gemini' else 1

    selected_provider_display = st.selectbox(
        "AI Provider",
        provider_display_options,
        index=current_p_index
    )

    if selected_provider_display == "Google Gemini (Cloud)":
        provider_key = "gemini"
        # Only working, verified Gemini models with high token efficiency
        available_models = [
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-flash-latest",
            "gemini-3.1-flash-lite",
            "gemini-3.5-flash",
            "gemini-3.8-flash"
        ]
    else:
        provider_key = "ollama"
        available_models = ["llama3.1", "llama3.2", "mistral", "gemma2", "phi3"]

    # Model selection
    default_m_index = 0
    if st.session_state.model in available_models:
        default_m_index = available_models.index(st.session_state.model)

    selected_model = st.selectbox(
        "Model",
        available_models,
        index=default_m_index
    )

    st.session_state.provider = provider_key
    st.session_state.model = selected_model

    # Status and Security indicators
    if provider_key == "gemini":
        has_key = bool(get_gemini_api_key())
        if has_key:
            st.success("🔒 Cloud API: Secured & Connected")
        else:
            st.warning("⚠️ API Key not detected in secrets or environment.")
        st.caption("⚡ Model has strict token limit (800 tokens max) to prevent hitting quota early.")
    else:
        st.info("🏠 Running locally via http://localhost:11434")

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
    <div style="margin-bottom: 10px;">
        <h4 style="color: #7b2cbf; margin: 0;">🤖 Telegram Bot</h4>
    </div>
    """, unsafe_allow_html=True)

    telegram_config = config.get('telegram', {})
    bot_enabled = telegram_config.get('enabled', False)
    bot_token = telegram_config.get('bot_token') or os.environ.get("TELEGRAM_BOT_TOKEN")

    if bot_enabled and bot_token:
        if st.session_state.telegram_bot_running:
            st.success("✅ Bot is running!")
            if st.button("⏹️ Stop Bot", use_container_width=True):
                if st.session_state.telegram_bot_thread:
                    st.session_state.telegram_bot_thread.terminate()
                    st.session_state.telegram_bot_thread = None
                st.session_state.telegram_bot_running = False
                cleanup_telegram_bot()
                st.rerun()
        else:
            st.info("⏸️ Bot is stopped")
            if st.button("▶️ Start Bot", use_container_width=True, type="primary"):
                st.session_state.telegram_bot_running = True
                st.rerun()
        bot_username = telegram_config.get('bot_username', '@YourBot')
        st.caption(f"Bot: {bot_username}")
    else:
        st.caption("⚙️ Set TELEGRAM_BOT_TOKEN to enable mobile bot")

    st.markdown("---")

    # Memory (ChromaDB) status
    st.subheader("💾 Context Memory")
    chroma = get_chroma_manager()
    if chroma:
        try:
            stats = chroma.get_collection_stats()
            st.metric("Stored Emails", stats['total_emails'])
        except Exception:
            st.caption("Memory active")
    else:
        st.caption("Memory in lightweight mode")

# Main content layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div style="margin-bottom: 8px;">
        <span class="badge-pill">1. Input Notes</span>
        <strong style="color: #00d4ff; font-size: 1.15rem;">Raw Thoughts & Bullet Points</strong>
    </div>
    """, unsafe_allow_html=True)
    rough_message = st.text_area(
        "",
        placeholder="e.g., need delay on sprint deliverables by 2 days due to API migration, QA team is on board, will deliver friday evening without fail",
        height=160,
        key="rough_input",
        label_visibility="collapsed"
    )

with col2:
    st.markdown("""
    <div style="margin-bottom: 8px;">
        <span class="badge-pill">2. Tone</span>
        <strong style="color: #7b2cbf; font-size: 1.15rem;">Select Persona</strong>
    </div>
    """, unsafe_allow_html=True)
    tone = st.radio(
        "Select Tone:",
        ["Professional", "Polite", "Assertive", "Friendly", "GenZ", "Casual", "Brief"],
        index=tone_options.index(st.session_state.tone) if st.session_state.tone in tone_options else 0,
        label_visibility="collapsed"
    )

    st.markdown("""
    <div style="background: rgba(0, 212, 255, 0.05); border: 1px solid rgba(0, 212, 255, 0.15); border-radius: 10px; padding: 12px; margin-top: 10px;">
        <p style="color: #00d4ff; font-weight: 600; font-size: 0.85rem; margin-bottom: 4px;">⚡ Smart Features</p>
        <p style="color: #8b949e; font-size: 0.8rem; margin: 0;">
        • Auto subject line & sign-off<br>
        • Token & rate-limit safe<br>
        • One-click clipboard copy
        </p>
    </div>
    """, unsafe_allow_html=True)

# Generate Button
st.markdown("")
if st.button("⚡ Generate Professional Email", use_container_width=True, type="primary"):
    if not rough_message.strip():
        st.warning("⚠️ Please enter your notes or thoughts first!")
    else:
        ai_client = get_ai_client(provider=st.session_state.provider, model=st.session_state.model)

        with st.spinner(f"✨ Crafting {tone.lower()} email using {st.session_state.model}..."):
            try:
                generated_email = generate_email_fast(
                    rough_message=rough_message.strip(),
                    tone=tone,
                    ai_client=ai_client
                )

                st.markdown("""
                <div style="margin-top: 20px; margin-bottom: 10px;">
                    <span class="badge-pill" style="background: rgba(0, 255, 136, 0.15); color: #00ff88; border-color: rgba(0, 255, 136, 0.3);">
                        ✅ Generated Draft
                    </span>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.code(generated_email, language="text")
                st.markdown('</div>', unsafe_allow_html=True)

                # Copy to clipboard widget
                st_copy_to_clipboard("📋 Copy Email to Clipboard", generated_email)

                # Save to ChromaDB in background thread
                import hashlib
                email_id = hashlib.md5(f"{rough_message}{tone}".encode()).hexdigest()
                threading.Thread(
                    target=save_to_chroma_async,
                    args=(generated_email, {'tone': tone, 'original': rough_message}, email_id),
                    daemon=True
                ).start()

            except Exception as e:
                st.error(f"❌ Error during generation: {str(e)}")

# Quick metrics
st.markdown("---")
col_stats1, col_stats2, col_stats3 = st.columns(3)
with col_stats1:
    st.metric("Engine", "Google Gemini" if st.session_state.provider == "gemini" else "Ollama Local")
with col_stats2:
    st.metric("Active Model", st.session_state.model)
with col_stats3:
    st.metric("Selected Tone", tone)

# Gmail retrieval section (collapsed by default)
with st.expander("📧 Fetch Gmail Messages (Optional Integration)", expanded=False):
    col_email1, col_email2, col_email3 = st.columns([2, 2, 1])

    with col_email1:
        email_address = st.text_input("Email Address", placeholder="your@gmail.com", key="email_addr")
    with col_email2:
        password = st.text_input("App Password", type="password", key="email_pass")
    with col_email3:
        fetch_btn = st.button("Fetch Emails", use_container_width=True)

    if fetch_btn and email_address and password:
        retriever = EmailRetriever(config.get('email', {}))
        with st.spinner("Fetching emails via IMAP..."):
            emails = retriever.fetch_emails(email_address=email_address, password=password, limit=5)
            if emails:
                st.success(f"Fetched {len(emails)} emails")
                for email_item in emails:
                    with st.expander(f"📩 {email_item['subject']} - {email_item['from']}"):
                        st.write(f"**Date:** {email_item['date']}")
                        st.write(email_item['body'][:500] + "...")
            else:
                st.warning("No emails found or could not authenticate.")

# Telegram Bot Runner Subprocess (if triggered)
if st.session_state.telegram_bot_running:
    telegram_config = config.get('telegram', {})
    bot_token = telegram_config.get('bot_token') or os.environ.get("TELEGRAM_BOT_TOKEN")
    if bot_token and st.session_state.telegram_bot_thread is None:
        try:
            runner_script = Path(__file__).parent / "telegram_runner_temp.py"
            provider = st.session_state.provider
            model = st.session_state.model

            if provider == 'ollama':
                provider_base_url = config['ollama']['base_url']
                provider_api_key = config['ollama']['api_key']
            else:
                provider_base_url = 'https://generativelanguage.googleapis.com/v1beta'
                provider_api_key = get_gemini_api_key()

            runner_content = f'''
import sys
import os
sys.path.insert(0, r"{Path(__file__).parent}")
os.chdir(r"{Path(__file__).parent}")

from telegram_bot import create_bot
from ai_client import AIClient
import yaml

prompts_path = os.path.join(r"{Path(__file__).parent}", "prompts.yaml")
with open(prompts_path, "r", encoding="utf-8") as f:
    prompts = yaml.safe_load(f)

ai_config = {{
    'provider': '{provider}',
    'model': '{model}',
    'base_url': '{provider_base_url}',
    'api_key': '{provider_api_key}',
    'max_output_tokens': 800,
}}
ai_client = AIClient(ai_config)

bot = create_bot(
    bot_token="{bot_token}",
    ai_client=ai_client,
    prompts=prompts
)
print("Telegram Bot Started! Press Ctrl+C to stop.")
bot.start_sync()
'''
            with open(runner_script, "w", encoding="utf-8") as f:
                f.write(runner_content)

            proc = subprocess.Popen(
                [sys.executable, str(runner_script)],
                cwd=str(Path(__file__).parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            st.session_state.telegram_bot_thread = proc
            st.success("🤖 Telegram Bot started in background!")
        except Exception as e:
            st.error(f"Failed to start bot: {e}")
            st.session_state.telegram_bot_running = False
