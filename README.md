# S1000-Email-Generator-2.0-
The **S1000 AI Email Generator** is a Python-based tool using **LangChain** and **Streamlit** to automate professional outreach. It features a document-aware backend that grounds drafts in specific project context. Integrated with **MongoDB**, it ensures high-quality, personalized replies, allowing users to focus on high-level strategy.
AI Email Generator S1000
High-Speed Professional Communication Suite

Project Overview
The AI Email Generator S1000 is a comprehensive automation tool designed to
transform rough, shorthand notes into polished, professional emails. Optimized for
speed and flexibility, it supports both local and cloud-based AI providers, features a
persistent memory system via vector databases, and offers multi-channel interaction
through a Streamlit web interface and a Telegram bot.

Core Features
Multimodal AI Support Local & Cloud LLMs Telegram Integration Vector Memory (RAG)
Gmail Sync Custom Tones
Dual Interface: Access the tool via a high-performance Streamlit dashboard or
a mobile-friendly Telegram bot.
Contextual Intelligence: Uses ChromaDB to store and retrieve past email
context, ensuring consistent communication.
Provider Flexibility: Seamlessly switch between local Ollama (for privacy),
Ollama Cloud (for speed), or Google Gemini .
Direct Email Retrieval: Connect your Gmail account via IMAP or Google API to
fetch and reply to existing threads.

Technical Architecture

Component Technology Purpose
Language Python 3.x Core logic and API interactions
•

•

•

•

AI
Orchestration

Custom AIClient

Handling prompt templates and provider
switching

Vector
Database

ChromaDB

Retrieval-Augmented Generation (RAG)
capabilities

Web UI Streamlit

Front-end dashboard with real-time
generation

Bot Interface

python-telegram-
bot

Asynchronous mobile interaction

Installation & Setup
1. Clone & Install Dependencies

pip install -r requirements.txt

2. Configuration
Edit config.yaml to set your API keys and preferred provider:

provider: ollama
ollama:
base_url: http://localhost:11434
model: llama3.1
telegram:
bot_token: "YOUR_TELEGRAM_BOT_TOKEN"
enabled: true

3. Running the Application
Launch the main dashboard:

streamlit run main.py

The Telegram bot can be started directly from the sidebar in the Streamlit UI or by
running the bot script independently.

Usage
Drafting: Input short notes like "tell team meeting moved to 4pm" into the text
area.
Tone Selection: Choose from Professional, Polite, Assertive, Friendly, GenZ, Casual,
or Brief.
Generation: Click "Generate Email" to receive a full draft including subject lines
and greetings.
Copy: Use the one-click clipboard tool to paste into your mail client.
