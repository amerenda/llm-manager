"""
Encrypted API key storage for cloud providers.

Keys are encrypted with Fernet (AES-128-CBC + HMAC) before storage in
PostgreSQL. The encryption key comes from the API_KEY_ENCRYPTION_KEY
env var, delivered via ExternalSecret.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)

_ENCRYPTION_KEY = os.environ.get("API_KEY_ENCRYPTION_KEY", "")


def _get_fernet() -> Fernet:
    if not _ENCRYPTION_KEY:
        raise ValueError("API_KEY_ENCRYPTION_KEY not configured")
    return Fernet(_ENCRYPTION_KEY.encode())


def encrypt_key(plaintext: str) -> str:
    """Encrypt an API key for DB storage."""
    return _get_fernet().encrypt(plaintext.encode()).decode()


def decrypt_key(ciphertext: str) -> str:
    """Decrypt an API key from DB storage."""
    try:
        return _get_fernet().decrypt(ciphertext.encode()).decode()
    except InvalidToken:
        logger.error("Failed to decrypt API key — encryption key may have changed")
        raise ValueError("Failed to decrypt API key")


def mask_key(plaintext: str) -> str:
    """Return a masked version of a key for display (first 8 chars + ****)."""
    if len(plaintext) <= 8:
        return "****"
    return plaintext[:8] + "****"


# ── DB operations ────────────────────────────────────────────────────────────

async def store_api_key(
    pool,
    provider: str,
    plaintext_key: str,
    label: str = "",
    user_id: Optional[int] = None,
) -> int:
    """Encrypt and store an API key. Returns the row ID."""
    encrypted = encrypt_key(plaintext_key)
    masked = mask_key(plaintext_key)
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            INSERT INTO api_keys (provider, user_id, encrypted_key, key_preview, label)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (provider, COALESCE(user_id, 0)) DO UPDATE SET
                encrypted_key = EXCLUDED.encrypted_key,
                key_preview = EXCLUDED.key_preview,
                label = EXCLUDED.label,
                created_at = NOW()
            RETURNING id
        """, provider, user_id, encrypted, masked, label)
    return row["id"]


async def get_api_key(
    pool,
    provider: str,
    user_id: Optional[int] = None,
) -> Optional[str]:
    """Retrieve and decrypt an API key. Returns None if not found."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT encrypted_key FROM api_keys
            WHERE provider = $1 AND (user_id = $2 OR ($2 IS NULL AND user_id IS NULL))
        """, provider, user_id)
    if not row:
        return None
    return decrypt_key(row["encrypted_key"])


async def list_api_keys(pool) -> list[dict]:
    """List all stored API keys (masked, never returns plaintext)."""
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, provider, user_id, key_preview, label, created_at
            FROM api_keys ORDER BY provider, id
        """)
    return [dict(r) for r in rows]


async def delete_api_key(pool, key_id: int) -> bool:
    """Delete an API key by ID."""
    async with pool.acquire() as conn:
        result = await conn.execute("DELETE FROM api_keys WHERE id = $1", key_id)
    return "DELETE 1" in result
