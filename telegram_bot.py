"""Telegram bot integration for AI Email Generator."""

import asyncio
import logging
from typing import Optional, Dict, Any
from ai_client import AIClient

# Enable logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TelegramBot:
    """Telegram bot for interacting with AI Email Generator."""

    def __init__(
        self,
        bot_token: str,
        ai_client: AIClient,
        prompts: Dict[str, Any]
    ):
        """
        Initialize Telegram bot.

        Args:
            bot_token: Telegram bot token from @BotFather
            ai_client: Configured AIClient instance
            prompts: Prompts configuration dictionary
        """
        self.bot_token = bot_token
        self.ai_client = ai_client
        self.prompts = prompts
        self._running = False
        self._user_tones: Dict[int, str] = {}  # Store tone per user

    async def start(self) -> None:
        """Start the Telegram bot."""
        try:
            from telegram import Update, Bot
            from telegram.ext import (
                Application,
                CommandHandler,
                MessageHandler,
                filters,
                ContextTypes
            )
        except ImportError:
            logger.error("python-telegram-bot not installed")
            print(
                "python-telegram-bot not installed. "
                "Install with: pip install python-telegram-bot"
            )
            return

        bot = Bot(token=self.bot_token)
        logger.info(f"Bot initialized with token: {self.bot_token[:10]}...")

        async def start_command(
            update: Update,
            context: ContextTypes.DEFAULT_TYPE
        ) -> None:
            """Handle /start command with interactive keyboard."""
            user = update.effective_user
            logger.info(f"Start command from user: {user.id}")

            welcome_message = (
                f"👋 Welcome {user.first_name} to AI Email Generator!\n\n"
                "I can transform your rough notes into professional emails.\n\n"
                "🎯 *Quick Start:* Just send me your notes and I'll create an email!\n\n"
                "📝 *Example:* 'tell boss im sick wont come today'\n\n"
                "/help - Get help\n"
                "/tone - Change email tone\n"
                "/preview - See available tones"
            )

            await update.message.reply_text(welcome_message, parse_mode='Markdown')

        async def help_command(
            update: Update,
            context: ContextTypes.DEFAULT_TYPE
        ) -> None:
            """Handle /help command."""
            help_message = (
                "📚 *Help*\n\n"
                "*How to use:*\n"
                "1. Just send me your rough email notes\n"
                "2. I'll generate a professional email\n\n"
                "*Commands:*\n"
                "/start - Restart bot\n"
                "/tone [name] - Set tone (e.g., /tone professional)\n"
                "/preview - See tone examples\n"
                "/help - Show this message\n\n"
                "*Tones:* Professional, Polite, Assertive, Friendly, GenZ, Casual, Brief"
            )
            await update.message.reply_text(help_message, parse_mode='Markdown')

        async def preview_command(
            update: Update,
            context: ContextTypes.DEFAULT_TYPE
        ) -> None:
            """Show tone preview."""
            preview_message = (
                "🎨 *Available Tones*\n\n"
                "*Professional* - Formal business style\n"
                "*Polite* - Courteous and respectful\n"
                "*Assertive* - Direct and confident\n"
                "*Friendly* - Warm and approachable\n"
                "*GenZ* - Modern casual style\n"
                "*Casual* - Relaxed conversation\n"
                "*Brief* - Short and concise\n\n"
                "Use /tone [name] to set your preference"
            )
            await update.message.reply_text(preview_message, parse_mode='Markdown')

        async def tone_command(
            update: Update,
            context: ContextTypes.DEFAULT_TYPE
        ) -> None:
            """Handle /tone command."""
            if not context.args:
                current_tone = self._user_tones.get(update.effective_user.id, 'professional')
                await update.message.reply_text(
                    f"Current tone: *{current_tone.title()}*\n\n"
                    "Set a new tone with: /tone [name]\n"
                    "Use /preview to see available tones",
                    parse_mode='Markdown'
                )
                return

            tone = ' '.join(context.args).lower().strip()
            valid_tones = ['professional', 'polite', 'assertive', 'friendly', 'genz', 'casual', 'brief']

            if tone in valid_tones:
                self._user_tones[update.effective_user.id] = tone
                await update.message.reply_text(
                    f"✅ Tone set to: *{tone.capitalize()}*",
                    parse_mode='Markdown'
                )
            else:
                await update.message.reply_text(
                    f"❌ Invalid tone: '{tone}'\n"
                    "Use /preview to see available tones",
                    parse_mode='Markdown'
                )

        async def handle_message(
            update: Update,
            context: ContextTypes.DEFAULT_TYPE
        ) -> None:
            """Handle incoming messages and generate email."""
            if not update.message or not update.message.text:
                return

            rough_message = update.message.text.strip()
            user_id = update.effective_user.id
            tone = self._user_tones.get(user_id, 'professional')

            logger.info(f"Message from {user_id}: {rough_message[:50]}... (tone: {tone})")

            # Send typing indicator
            await update.message.chat.send_action('typing')

            try:
                system_prompt = self.prompts['email_generator']['system_template']
                user_template = self.prompts['email_generator']['user_template']

                user_prompt = user_template.format(
                    rough_message=rough_message,
                    tone=tone
                )

                response = self.ai_client.generate_text(
                    prompt=user_prompt,
                    system_prompt=system_prompt
                )

                if response:
                    # Send the generated email
                    await update.message.reply_text(
                        f"✨ *Generated Email ({tone.capitalize()}):*\n\n{response}",
                        parse_mode='Markdown'
                    )
                else:
                    await update.message.reply_text(
                        "❌ Sorry, I couldn't generate an email. Please try again."
                    )

            except Exception as e:
                logger.error(f"Error generating email: {e}")
                await update.message.reply_text(
                    f"❌ Error: {str(e)}\n\n"
                    "Make sure the AI service is running and try again."
                )

        # Create application
        application = Application.builder().token(self.bot_token).build()
        logger.info("Application built")

        # Add handlers - order matters! Commands first, then message handler
        application.add_handler(CommandHandler("start", start_command))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(CommandHandler("preview", preview_command))
        application.add_handler(CommandHandler("tone", tone_command))
        application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
        )
        logger.info("Handlers added")

        # Start polling with error handling
        self._running = True
        logger.info("Starting bot polling...")

        try:
            await application.run_polling(
                allowed_updates=Update.ALL_TYPES,
                drop_pending_updates=True
            )
        except Exception as e:
            logger.error(f"Polling error: {e}")
            raise

    def start_sync(self) -> None:
        """Start the bot synchronously."""
        logger.info("Starting bot synchronously...")
        asyncio.run(self.start())

    async def stop(self) -> None:
        """Stop the Telegram bot."""
        self._running = False
        logger.info("Bot stopped")


def create_bot(
    bot_token: str,
    ai_client: AIClient,
    prompts: Dict[str, Any]
) -> TelegramBot:
    """
    Factory function to create a Telegram bot.

    Args:
        bot_token: Telegram bot token
        ai_client: Configured AIClient instance
        prompts: Prompts configuration

    Returns:
        TelegramBot instance
    """
    return TelegramBot(bot_token, ai_client, prompts)