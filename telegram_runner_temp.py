
import sys
import os
sys.path.insert(0, r"C:\Users\shaur\OneDrive\Desktop\;-;\AI SMART EMAIL GENERATOR")
os.chdir(r"C:\Users\shaur\OneDrive\Desktop\;-;\AI SMART EMAIL GENERATOR")

import asyncio
from telegram_bot import create_bot
from ai_client import AIClient
import yaml

config_path = os.path.join(r"C:\Users\shaur\OneDrive\Desktop\;-;\AI SMART EMAIL GENERATOR", "config.yaml")
prompts_path = os.path.join(r"C:\Users\shaur\OneDrive\Desktop\;-;\AI SMART EMAIL GENERATOR", "prompts.yaml")

with open(config_path) as f:
    config = yaml.safe_load(f)

with open(prompts_path) as f:
    prompts = yaml.safe_load(f)

ai_config = {
    'provider': 'minimax-cloud',
    'model': 'minimax-m2.7:cloud',
    'base_url': 'https://ollama.com',
    'api_key': '96dc03fd305a421491186c268d334915.XE726r2RgxOc6UrFxhe7N3XG',
}
ai_client = AIClient(ai_config)

bot = create_bot(
    bot_token="8667694841:AAGPcm8iv91T4t6vpw7U70XMDTd3vXsjmeY",
    ai_client=ai_client,
    prompts=prompts
)
print("Telegram Bot Started! Press Ctrl+C to stop.")
bot.start_sync()
