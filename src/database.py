"""Database models and operations for Api Dzeck Ai Web API."""

import os
import psycopg2
import psycopg2.extras
import json
import time
from contextlib import contextmanager
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from uuid import uuid4

from werkzeug.security import generate_password_hash, check_password_hash

from config import config
from utils.exceptions import DatabaseError, ValidationError
from utils.logging import logger
from utils.validation import validate_username, validate_password
from utils.helpers import generate_uuid

@dataclass
class UserSettings:
    """User settings data model."""
    token: str
    provider: str = config.api.default_provider
    model: str = config.api.default_model
    system_prompt: str = ""
    message_history: bool = False
    username: str = ""
    password: str = ""
    chat_history: str = ""

@dataclass
class ServerSettings:
    """Server settings data model."""
    id: int = 1
    keyword: str = config.api.default_keyword
    file_input: bool = True
    port: str = str(config.server.port)
    provider: str = config.api.default_provider
    model: str = config.api.default_model
    cookie_file: str = config.files.cookies_file
    token: str = ""
    remove_sources: bool = True
    system_prompt: str = ""
    message_history: bool = False
    proxies: bool = False
    password: str = ""
    fast_api: bool = False
    virtual_users: bool = False
    chat_history: str = ""

class DatabaseManager:
    """Database manager for Api Dzeck Ai Web API."""
    
    def __init__(self, database_url: Optional[str] = None):
        """Initialize database manager.
        
        Args:
            database_url: PostgreSQL connection string
        """
        self.database_url = database_url or os.environ.get('DATABASE_URL', '') or config.database.database_url
        self.initialize_database()
    
    @contextmanager
    def get_connection(self):
        """Get database connection context manager.
        
        Yields:
            Database connection and cursor
        """
        conn = None
        try:
            conn = psycopg2.connect(self.database_url)
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            yield conn, cursor
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise DatabaseError(f"Database operation failed: {e}")
        finally:
            if conn:
                conn.close()
    
    def initialize_database(self):
        """Initialize database tables."""
        try:
            with self.get_connection() as (conn, cursor):
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS settings (
                        id SERIAL PRIMARY KEY,
                        keyword TEXT NOT NULL,
                        file_input BOOLEAN NOT NULL,
                        port TEXT NOT NULL,
                        provider TEXT NOT NULL,
                        model TEXT NOT NULL,
                        cookie_file TEXT NOT NULL,
                        token TEXT NOT NULL,
                        remove_sources BOOLEAN NOT NULL,
                        system_prompt TEXT NOT NULL,
                        message_history BOOLEAN NOT NULL,
                        proxies BOOLEAN NOT NULL,
                        password TEXT NOT NULL,
                        fast_api BOOLEAN NOT NULL,
                        virtual_users BOOLEAN NOT NULL,
                        chat_history TEXT NOT NULL
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS personal (
                        token TEXT PRIMARY KEY,
                        provider TEXT NOT NULL,
                        model TEXT NOT NULL,
                        system_prompt TEXT NOT NULL,
                        message_history BOOLEAN NOT NULL,
                        username TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL,
                        chat_history TEXT NOT NULL
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id TEXT PRIMARY KEY,
                        username TEXT NOT NULL,
                        title TEXT NOT NULL DEFAULT 'New Chat',
                        messages TEXT NOT NULL DEFAULT '[]',
                        created_at DOUBLE PRECISION NOT NULL,
                        updated_at DOUBLE PRECISION NOT NULL
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS api_keys (
                        id TEXT PRIMARY KEY,
                        api_key TEXT UNIQUE NOT NULL,
                        provider TEXT NOT NULL,
                        model TEXT NOT NULL DEFAULT '',
                        label TEXT NOT NULL DEFAULT '',
                        created_by TEXT NOT NULL,
                        created_at DOUBLE PRECISION NOT NULL,
                        last_used_at DOUBLE PRECISION,
                        usage_count INTEGER NOT NULL DEFAULT 0,
                        is_active BOOLEAN NOT NULL DEFAULT TRUE,
                        base_url TEXT NOT NULL DEFAULT ''
                    )
                """)
                
                cursor.execute("SELECT COUNT(*) AS cnt FROM settings")
                row = cursor.fetchone()
                if row is None or row['cnt'] == 0:
                    self._create_default_settings(cursor)
                
                conn.commit()
                logger.info("Database initialized successfully")
        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise DatabaseError(f"Database initialization failed: {e}")
    
    def _create_default_settings(self, cursor):
        """Create default settings."""
        default_settings = ServerSettings()
        cursor.execute("""
            INSERT INTO settings (
                id, keyword, file_input, port, provider, model, cookie_file,
                token, remove_sources, system_prompt, message_history, proxies,
                password, fast_api, virtual_users, chat_history
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            default_settings.id,
            default_settings.keyword,
            default_settings.file_input,
            default_settings.port,
            default_settings.provider,
            default_settings.model,
            default_settings.cookie_file,
            default_settings.token,
            default_settings.remove_sources,
            default_settings.system_prompt,
            default_settings.message_history,
            default_settings.proxies,
            default_settings.password,
            default_settings.fast_api,
            default_settings.virtual_users,
            default_settings.chat_history
        ))
        logger.info("Default settings created")
    
    def get_settings(self) -> Dict[str, Any]:
        """Get server settings.
        
        Returns:
            Dictionary with server settings
        """
        try:
            with self.get_connection() as (conn, cursor):
                cursor.execute("SELECT * FROM settings WHERE id = 1")
                row = cursor.fetchone()
                
                if not row:
                    raise DatabaseError("Settings not found")
                
                return {
                    "keyword": row["keyword"],
                    "file_input": bool(row["file_input"]),
                    "port": row["port"],
                    "provider": row["provider"],
                    "model": row["model"],
                    "cookie_file": row["cookie_file"],
                    "token": row["token"],
                    "remove_sources": bool(row["remove_sources"]),
                    "system_prompt": row["system_prompt"],
                    "message_history": bool(row["message_history"]),
                    "proxies": bool(row["proxies"]),
                    "password": row["password"],
                    "fast_api": bool(row["fast_api"]),
                    "virtual_users": bool(row["virtual_users"])
                }
        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"Failed to get settings: {e}")
            raise DatabaseError(f"Failed to get settings: {e}")
    
    def update_settings(self, settings: Dict[str, Any]):
        """Update server settings.
        
        Args:
            settings: Dictionary with settings to update
        """
        try:
            with self.get_connection() as (conn, cursor):
                update_fields = []
                values = []
                
                for key, value in settings.items():
                    if key == "password" and value:
                        value = generate_password_hash(value)
                    update_fields.append(f"{key} = %s")
                    values.append(value)
                
                if update_fields:
                    query = f"UPDATE settings SET {', '.join(update_fields)} WHERE id = 1"
                    cursor.execute(query, values)
                    conn.commit()
                    logger.info("Settings updated successfully")
        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"Failed to update settings: {e}")
            raise DatabaseError(f"Failed to update settings: {e}")
    
    def verify_admin_password(self, password: str) -> bool:
        """Verify admin password.
        
        Args:
            password: Password to verify
            
        Returns:
            True if password is correct, False otherwise
        """
        try:
            settings = self.get_settings()
            stored_password = settings.get("password", "")
            
            if not stored_password:
                logger.warning("Admin login attempted but no password is configured")
                return False
            
            is_valid = check_password_hash(stored_password, password)
            
            if is_valid:
                logger.info("Admin authentication successful")
            else:
                logger.warning(f"Admin authentication failed for password attempt")
            
            return is_valid
        except Exception as e:
            logger.error(f"Failed to verify admin password: {e}")
            return False
    
    def create_user(self, username: str, password: Optional[str] = None) -> str:
        """Create a new user.
        
        Args:
            username: Username
            password: Password (will be auto-generated if not provided)
            
        Returns:
            User token
            
        Raises:
            ValidationError: If username is invalid
            DatabaseError: If database operation fails
        """
        is_valid, error_msg = validate_username(username)
        if not is_valid:
            raise ValidationError(error_msg)
        
        if password is None:
            password = username
            is_valid, error_msg = validate_password(password, 1)
        else:
            is_valid, error_msg = validate_password(password, config.security.password_min_length)
        
        if not is_valid:
            raise ValidationError(error_msg)
        
        token = generate_uuid()
        hashed_password = generate_password_hash(password)
        
        user_settings = UserSettings(
            token=token,
            username=username,
            password=hashed_password
        )
        
        try:
            with self.get_connection() as (conn, cursor):
                cursor.execute("""
                    INSERT INTO personal (
                        token, provider, model, system_prompt, message_history,
                        username, password, chat_history
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    user_settings.token,
                    user_settings.provider,
                    user_settings.model,
                    user_settings.system_prompt,
                    user_settings.message_history,
                    user_settings.username,
                    user_settings.password,
                    user_settings.chat_history
                ))
                conn.commit()
                logger.info(f"User '{username}' created successfully")
                self.export_data_backup()
                return token
        except psycopg2.IntegrityError as e:
            if "username" in str(e):
                raise ValidationError(f"Username '{username}' already exists")
            raise DatabaseError(f"Failed to create user: {e}")
        except Exception as e:
            logger.error(f"Failed to create user '{username}': {e}")
            raise DatabaseError(f"Failed to create user: {e}")
    
    def get_user_by_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Get user by token.
        
        Args:
            token: User token
            
        Returns:
            User data dictionary or None if not found
        """
        try:
            with self.get_connection() as (conn, cursor):
                cursor.execute("SELECT * FROM personal WHERE token = %s", (token,))
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                return {
                    "token": row["token"],
                    "provider": row["provider"],
                    "model": row["model"],
                    "system_prompt": row["system_prompt"],
                    "message_history": bool(row["message_history"]),
                    "username": row["username"],
                    "password": row["password"],
                    "chat_history": row["chat_history"]
                }
        except Exception as e:
            logger.error(f"Failed to get user by token: {e}")
            return None
    
    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username.
        
        Args:
            username: Username
            
        Returns:
            User data dictionary or None if not found
        """
        try:
            with self.get_connection() as (conn, cursor):
                cursor.execute("SELECT * FROM personal WHERE username = %s", (username,))
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                return {
                    "token": row["token"],
                    "provider": row["provider"],
                    "model": row["model"],
                    "system_prompt": row["system_prompt"],
                    "message_history": bool(row["message_history"]),
                    "username": row["username"],
                    "password": row["password"],
                    "chat_history": row["chat_history"]
                }
        except Exception as e:
            logger.error(f"Failed to get user by username: {e}")
            return None
    
    def verify_user_password(self, username: str, password: str) -> bool:
        """Verify user password.
        
        Args:
            username: Username
            password: Password to verify
            
        Returns:
            True if password is correct, False otherwise
        """
        try:
            user = self.get_user_by_username(username)
            if not user:
                return False
            
            return check_password_hash(user["password"], password)
        except Exception as e:
            logger.error(f"Failed to verify user password: {e}")
            return False
    
    def update_user_settings(self, username: str, settings: Dict[str, Any]):
        """Update user settings.
        
        Args:
            username: Username
            settings: Settings to update
        """
        try:
            with self.get_connection() as (conn, cursor):
                update_fields = []
                values = []
                
                for key, value in settings.items():
                    if key == "password" and value:
                        value = generate_password_hash(value)
                    update_fields.append(f"{key} = %s")
                    values.append(value)
                
                if update_fields:
                    values.append(username)
                    query = f"UPDATE personal SET {', '.join(update_fields)} WHERE username = %s"
                    cursor.execute(query, values)
                    conn.commit()
                    logger.info(f"User '{username}' settings updated successfully")
        except Exception as e:
            logger.error(f"Failed to update user settings: {e}")
            raise DatabaseError(f"Failed to update user settings: {e}")
    
    def delete_user(self, username: str):
        """Delete user.
        
        Args:
            username: Username to delete
        """
        try:
            with self.get_connection() as (conn, cursor):
                cursor.execute("DELETE FROM personal WHERE username = %s", (username,))
                conn.commit()
                logger.info(f"User '{username}' deleted successfully")
                self.export_data_backup()
        except Exception as e:
            logger.error(f"Failed to delete user '{username}': {e}")
            raise DatabaseError(f"Failed to delete user: {e}")
    
    def get_all_users(self) -> List[Dict[str, Any]]:
        """Get all users.
        
        Returns:
            List of user dictionaries
        """
        try:
            with self.get_connection() as (conn, cursor):
                cursor.execute("SELECT * FROM personal")
                rows = cursor.fetchall()
                
                users = []
                for row in rows:
                    users.append({
                        "token": row["token"],
                        "provider": row["provider"],
                        "model": row["model"],
                        "system_prompt": row["system_prompt"],
                        "message_history": bool(row["message_history"]),
                        "username": row["username"],
                        "password": row["password"],
                        "chat_history": row["chat_history"]
                    })
                
                return users
        except Exception as e:
            logger.error(f"Failed to get all users: {e}")
            raise DatabaseError(f"Failed to get all users: {e}")
    
    def save_chat_history(self, username: str, chat_history: str):
        """Save chat history for user or admin.
        
        Args:
            username: Username ('admin' for admin user)
            chat_history: Chat history JSON string
        """
        try:
            with self.get_connection() as (conn, cursor):
                if username == "admin":
                    cursor.execute("UPDATE settings SET chat_history = %s WHERE id = 1", (chat_history,))
                else:
                    cursor.execute("UPDATE personal SET chat_history = %s WHERE username = %s", (chat_history, username))
                
                conn.commit()
                logger.debug(f"Chat history saved for user '{username}'")
        except Exception as e:
            logger.error(f"Failed to save chat history for user '{username}': {e}")
            raise DatabaseError(f"Failed to save chat history: {e}")
    
    def get_chat_history(self, username: str) -> str:
        """Get chat history for user or admin.
        
        Args:
            username: Username ('admin' for admin user)
            
        Returns:
            Chat history JSON string
        """
        try:
            with self.get_connection() as (conn, cursor):
                if username == "admin":
                    cursor.execute("SELECT chat_history FROM settings WHERE id = 1")
                else:
                    cursor.execute("SELECT chat_history FROM personal WHERE username = %s", (username,))
                
                row = cursor.fetchone()
                return row["chat_history"] if row else ""
        except Exception as e:
            logger.error(f"Failed to get chat history for user '{username}': {e}")
            return ""

    def create_conversation(self, username: str, title: str = 'New Chat') -> Dict[str, Any]:
        """Create a new conversation."""
        convo_id = str(uuid4())
        now = time.time()
        try:
            with self.get_connection() as (conn, cursor):
                cursor.execute(
                    "INSERT INTO conversations (id, username, title, messages, created_at, updated_at) VALUES (%s, %s, %s, %s, %s, %s)",
                    (convo_id, username, title, '[]', now, now)
                )
                conn.commit()
                return {"id": convo_id, "username": username, "title": title, "messages": [], "created_at": now, "updated_at": now}
        except Exception as e:
            logger.error(f"Failed to create conversation: {e}")
            raise DatabaseError(f"Failed to create conversation: {e}")

    def get_conversations(self, username: str) -> List[Dict[str, Any]]:
        """Get list of conversations for a user, ordered by updated_at DESC."""
        try:
            with self.get_connection() as (conn, cursor):
                cursor.execute(
                    "SELECT id, title, created_at, updated_at FROM conversations WHERE username = %s ORDER BY updated_at DESC",
                    (username,)
                )
                rows = cursor.fetchall()
                return [{"id": row["id"], "title": row["title"], "created_at": row["created_at"], "updated_at": row["updated_at"]} for row in rows]
        except Exception as e:
            logger.error(f"Failed to get conversations: {e}")
            return []

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get a full conversation including messages."""
        try:
            with self.get_connection() as (conn, cursor):
                cursor.execute("SELECT * FROM conversations WHERE id = %s", (conversation_id,))
                row = cursor.fetchone()
                if not row:
                    return None
                return {
                    "id": row["id"],
                    "username": row["username"],
                    "title": row["title"],
                    "messages": row["messages"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"]
                }
        except Exception as e:
            logger.error(f"Failed to get conversation: {e}")
            return None

    def update_conversation(self, conversation_id: str, messages_json: str, title: Optional[str] = None):
        """Update conversation messages and optionally title."""
        now = time.time()
        try:
            with self.get_connection() as (conn, cursor):
                if title is not None:
                    cursor.execute(
                        "UPDATE conversations SET messages = %s, title = %s, updated_at = %s WHERE id = %s",
                        (messages_json, title, now, conversation_id)
                    )
                else:
                    cursor.execute(
                        "UPDATE conversations SET messages = %s, updated_at = %s WHERE id = %s",
                        (messages_json, now, conversation_id)
                    )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to update conversation: {e}")
            raise DatabaseError(f"Failed to update conversation: {e}")

    def delete_conversation(self, conversation_id: str):
        """Delete a conversation."""
        try:
            with self.get_connection() as (conn, cursor):
                cursor.execute("DELETE FROM conversations WHERE id = %s", (conversation_id,))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to delete conversation: {e}")
            raise DatabaseError(f"Failed to delete conversation: {e}")

    def delete_all_conversations(self, username: str):
        """Delete all conversations for a user."""
        try:
            with self.get_connection() as (conn, cursor):
                cursor.execute("DELETE FROM conversations WHERE username = %s", (username,))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to delete all conversations: {e}")
            raise DatabaseError(f"Failed to delete all conversations: {e}")

    def create_api_key(self, provider: str, model: str, label: str, created_by: str, base_url: str = '') -> Dict[str, Any]:
        import secrets
        key_id = str(uuid4())
        api_key = f"sk-dzeck-{secrets.token_hex(24)}"
        now = time.time()
        try:
            with self.get_connection() as (conn, cursor):
                cursor.execute(
                    """INSERT INTO api_keys (id, api_key, provider, model, label, created_by, created_at, usage_count, is_active, base_url)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, 0, TRUE, %s)""",
                    (key_id, api_key, provider, model, label, created_by, now, base_url)
                )
                conn.commit()
                result = {"id": key_id, "api_key": api_key, "provider": provider, "model": model, "label": label, "created_by": created_by, "created_at": now, "usage_count": 0, "is_active": True, "base_url": base_url}
                self.export_data_backup()
                return result
        except Exception as e:
            logger.error(f"Failed to create API key: {e}")
            raise DatabaseError(f"Failed to create API key: {e}")

    def get_api_keys(self, created_by: Optional[str] = None) -> List[Dict[str, Any]]:
        try:
            with self.get_connection() as (conn, cursor):
                if created_by:
                    cursor.execute("SELECT * FROM api_keys WHERE created_by = %s ORDER BY created_at DESC", (created_by,))
                else:
                    cursor.execute("SELECT * FROM api_keys ORDER BY created_at DESC")
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get API keys: {e}")
            return []

    def get_api_key_by_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        try:
            with self.get_connection() as (conn, cursor):
                cursor.execute("SELECT * FROM api_keys WHERE api_key = %s AND is_active = TRUE", (api_key,))
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"Failed to get API key: {e}")
            return None

    def increment_api_key_usage(self, api_key: str):
        try:
            with self.get_connection() as (conn, cursor):
                cursor.execute(
                    "UPDATE api_keys SET usage_count = usage_count + 1, last_used_at = %s WHERE api_key = %s",
                    (time.time(), api_key)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to increment API key usage: {e}")

    def delete_api_key(self, key_id: str):
        try:
            with self.get_connection() as (conn, cursor):
                cursor.execute("DELETE FROM api_keys WHERE id = %s", (key_id,))
                conn.commit()
                self.export_data_backup()
        except Exception as e:
            logger.error(f"Failed to delete API key: {e}")
            raise DatabaseError(f"Failed to delete API key: {e}")

    def toggle_api_key(self, key_id: str, is_active: bool):
        try:
            with self.get_connection() as (conn, cursor):
                cursor.execute("UPDATE api_keys SET is_active = %s WHERE id = %s", (is_active, key_id))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to toggle API key: {e}")
            raise DatabaseError(f"Failed to toggle API key: {e}")

    def export_data_backup(self):
        backup_path = Path(__file__).parent / "data" / "db_backup.json"
        try:
            backup_data = {
                "users": [],
                "api_keys": [],
                "exported_at": time.time()
            }
            with self.get_connection() as (conn, cursor):
                cursor.execute("SELECT token, provider, model, system_prompt, message_history, username, password, chat_history FROM personal")
                for row in cursor.fetchall():
                    backup_data["users"].append(dict(row))

                cursor.execute("SELECT id, api_key, provider, model, label, created_by, created_at, last_used_at, usage_count, is_active, base_url FROM api_keys")
                for row in cursor.fetchall():
                    backup_data["api_keys"].append(dict(row))

            backup_path.parent.mkdir(parents=True, exist_ok=True)
            with open(backup_path, 'w') as f:
                json.dump(backup_data, f, indent=2, default=str)
            logger.info(f"Database backup exported to {backup_path} ({len(backup_data['users'])} users, {len(backup_data['api_keys'])} API keys)")
        except Exception as e:
            logger.warning(f"Failed to export database backup: {e}")

    def import_data_backup(self):
        backup_path = Path(__file__).parent / "data" / "db_backup.json"
        if not backup_path.exists():
            logger.info("No database backup file found, skipping import")
            return

        try:
            with open(backup_path, 'r') as f:
                backup_data = json.load(f)

            imported_users = 0
            imported_keys = 0

            with self.get_connection() as (conn, cursor):
                for user in backup_data.get("users", []):
                    cursor.execute("SELECT COUNT(*) AS cnt FROM personal WHERE username = %s", (user["username"],))
                    row_check = cursor.fetchone()
                    if row_check is None or row_check["cnt"] == 0:
                        cursor.execute(
                            """INSERT INTO personal (token, provider, model, system_prompt, message_history, username, password, chat_history)
                               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                            (user["token"], user.get("provider", "PollinationsAI"), user.get("model", "openai"),
                             user.get("system_prompt", ""), user.get("message_history", False),
                             user["username"], user["password"], user.get("chat_history", ""))
                        )
                        imported_users += 1

                for key in backup_data.get("api_keys", []):
                    cursor.execute("SELECT COUNT(*) AS cnt FROM api_keys WHERE api_key = %s", (key["api_key"],))
                    row_check = cursor.fetchone()
                    if row_check is None or row_check["cnt"] == 0:
                        cursor.execute(
                            """INSERT INTO api_keys (id, api_key, provider, model, label, created_by, created_at, last_used_at, usage_count, is_active, base_url)
                               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                            (key["id"], key["api_key"], key.get("provider", ""), key.get("model", ""),
                             key.get("label", ""), key.get("created_by", "admin"), key.get("created_at", time.time()),
                             key.get("last_used_at"), key.get("usage_count", 0), key.get("is_active", True),
                             key.get("base_url", ""))
                        )
                        imported_keys += 1

                conn.commit()

            if imported_users > 0 or imported_keys > 0:
                logger.info(f"Database backup restored: {imported_users} users, {imported_keys} API keys imported")
            else:
                logger.info("Database backup checked: all data already exists")
        except Exception as e:
            logger.warning(f"Failed to import database backup: {e}")

db_manager = DatabaseManager()
db_manager.import_data_backup()
