"""Email retrieval from Gmail using IMAP or Gmail API."""

import imaplib
import email
from email.header import decode_header
from typing import List, Dict, Any, Optional
import base64
import json


class EmailRetriever:
    """Retrieve emails from Gmail using IMAP or Gmail API."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize email retriever.

        Args:
            config: Configuration dictionary with email settings
        """
        self.method = config.get('method', 'imap')
        self.config = config

        if self.method == 'imap':
            self.imap_config = config.get('imap', {})
        elif self.method == 'gmail_api':
            self.gmail_api_config = config.get('gmail_api', {})
            self._gmail_service = None

    def connect_imap(self, email_address: str, password: str) -> imaplib.IMAP4_SSL:
        """
        Connect to Gmail IMAP server.

        Args:
            email_address: Gmail email address
            password: App password (not regular password)

        Returns:
            Connected IMAP client
        """
        server = self.imap_config.get('server', 'imap.gmail.com')
        port = self.imap_config.get('port', 993)
        use_ssl = self.imap_config.get('use_ssl', True)

        if use_ssl:
            imap_client = imaplib.IMAP4_SSL(server, port)
        else:
            imap_client = imaplib.IMAP4(server, port)

        imap_client.login(email_address, password)
        return imap_client

    def fetch_emails_imap(
        self,
        email_address: str,
        password: str,
        folder: str = "INBOX",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Fetch emails using IMAP.

        Args:
            email_address: Gmail email address
            password: App password
            folder: IMAP folder to fetch from
            limit: Maximum number of emails to fetch

        Returns:
            List of email dictionaries
        """
        emails = []

        try:
            imap_client = self.connect_imap(email_address, password)
            imap_client.select(folder)

            # Search for all emails
            status, messages = imap_client.search(None, "ALL")

            if status != "OK":
                return emails

            email_ids = messages[0].split()
            email_ids = email_ids[-limit:]  # Get most recent

            for email_id in reversed(email_ids):
                email_data = self._fetch_email_data(imap_client, email_id)
                if email_data:
                    emails.append(email_data)

            imap_client.close()
            imap_client.logout()

        except Exception as e:
            print(f"Error fetching emails via IMAP: {e}")

        return emails

    def _fetch_email_data(
        self,
        imap_client: imaplib.IMAP4_SSL,
        email_id: bytes
    ) -> Optional[Dict[str, Any]]:
        """Fetch individual email data."""
        try:
            status, msg = imap_client.fetch(email_id, "(RFC822)")

            if status != "OK":
                return None

            email_message = email.message_from_bytes(msg[0][1])

            # Decode subject
            subject, encoding = decode_header(email_message["Subject"])[0]
            if isinstance(subject, bytes):
                subject = subject.decode(encoding if encoding else 'utf-8')

            # Decode sender
            from_header, encoding = decode_header(email_message.get("From", ""))[0]
            if isinstance(from_header, bytes):
                from_header = from_header.decode(encoding if encoding else 'utf-8')

            # Get email body
            body = self._get_email_body(email_message)

            return {
                'id': email_id.decode(),
                'subject': subject,
                'from': from_header,
                'date': email_message.get("Date", ""),
                'body': body,
                'raw_message': str(email_message)
            }

        except Exception as e:
            print(f"Error fetching email data: {e}")
            return None

    def _get_email_body(self, email_message: email.message.Message) -> str:
        """Extract body text from email message."""
        body = ""

        if email_message.is_multipart():
            for part in email_message.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))

                # Skip attachments
                if "attachment" in content_disposition:
                    continue

                if content_type == "text/plain":
                    charset = part.get_content_charset() or 'utf-8'
                    try:
                        body = part.get_payload(decode=True).decode(charset)
                        break
                    except UnicodeDecodeError:
                        try:
                            body = part.get_payload(decode=True).decode('latin-1')
                            break
                        except Exception:
                            continue
        else:
            charset = email_message.get_content_charset() or 'utf-8'
            try:
                body = email_message.get_payload(decode=True).decode(charset)
            except UnicodeDecodeError:
                body = email_message.get_payload(decode=True).decode('latin-1', errors='ignore')

        return body

    def get_gmail_service(self):
        """Get authenticated Gmail API service."""
        if self._gmail_service:
            return self._gmail_service

        try:
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
            from google.auth.transport.requests import Request
            import os
            import pickle

            SCOPES = self.gmail_api_config.get(
                'scopes',
                ['https://www.googleapis.com/auth/gmail.readonly']
            )
            credentials_file = self.gmail_api_config.get(
                'credentials_file',
                'credentials.json'
            )
            token_file = self.gmail_api_config.get('token_file', 'token.json')

            creds = None

            if os.path.exists(token_file):
                creds = Credentials.from_authorized_user_file(token_file, SCOPES)

            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        credentials_file, SCOPES
                    )
                    creds = flow.run_local_server(port=0)

                with open(token_file, 'w') as f:
                    f.write(creds.to_json())

            self._gmail_service = build('gmail', 'v1', credentials=creds)
            return self._gmail_service

        except ImportError:
            print(
                "Gmail API libraries not installed. "
                "Install with: pip install google-auth google-auth-oauthlib "
                "google-auth-httplib2 google-api-python-client"
            )
            return None
        except Exception as e:
            print(f"Error initializing Gmail API: {e}")
            return None

    def fetch_emails_gmail_api(
        self,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Fetch emails using Gmail API.

        Args:
            limit: Maximum number of emails to fetch

        Returns:
            List of email dictionaries
        """
        service = self.get_gmail_service()
        if not service:
            return []

        emails = []

        try:
            results = service.users().messages().list(
                userId='me',
                maxResults=limit
            ).execute()

            messages = results.get('messages', [])

            for message in messages:
                msg = service.users().messages().get(
                    userId='me',
                    id=message['id'],
                    format='full'
                ).execute()

                headers = msg['payload']['headers']
                subject = next(
                    (h['value'] for h in headers if h['name'] == 'Subject'),
                    'No Subject'
                )
                from_header = next(
                    (h['value'] for h in headers if h['name'] == 'From'),
                    'Unknown'
                )
                date = next(
                    (h['value'] for h in headers if h['name'] == 'Date'),
                    ''
                )

                # Get body
                body = self._get_gmail_api_body(msg['payload'])

                emails.append({
                    'id': msg['id'],
                    'subject': subject,
                    'from': from_header,
                    'date': date,
                    'body': body
                })

        except Exception as e:
            print(f"Error fetching emails via Gmail API: {e}")

        return emails

    def _get_gmail_api_body(self, payload: Dict[str, Any]) -> str:
        """Extract body from Gmail API payload."""
        body = ""

        if 'body' in payload and 'data' in payload['body']:
            body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')

        if not body and 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain' and 'body' in part:
                    if 'data' in part['body']:
                        body = base64.urlsafe_b64decode(
                            part['body']['data']
                        ).decode('utf-8')
                        break

        return body

    def fetch_emails(
        self,
        email_address: Optional[str] = None,
        password: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Fetch emails using configured method.

        Args:
            email_address: Email address (for IMAP)
            password: Password (for IMAP)
            limit: Maximum number of emails to fetch

        Returns:
            List of email dictionaries
        """
        if self.method == 'imap':
            if not email_address or not password:
                print("Email address and password required for IMAP")
                return []
            return self.fetch_emails_imap(email_address, password, limit=limit)
        elif self.method == 'gmail_api':
            return self.fetch_emails_gmail_api(limit=limit)
        else:
            print(f"Unknown email retrieval method: {self.method}")
            return []
