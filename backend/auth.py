"""
GitHub OAuth authentication for llm-manager admin UI.

Flow:
  1. GET /auth/login → redirect to GitHub
  2. GitHub redirects to GET /auth/callback?code=xxx
  3. Backend exchanges code for token, verifies username, sets session cookie
  4. All admin API calls include the session cookie
"""

import logging
import os
import time
from typing import Optional

import httpx
import jwt
from fastapi import Request

logger = logging.getLogger(__name__)

GITHUB_CLIENT_ID = os.environ.get("GITHUB_CLIENT_ID", "")
GITHUB_CLIENT_SECRET = os.environ.get("GITHUB_CLIENT_SECRET", "")
SESSION_SECRET = os.environ.get("SESSION_SECRET", "dev-secret-change-me")
GITHUB_ALLOWED_USERS = {
    u.strip() for u in os.environ.get("GITHUB_ALLOWED_USERS", "amerenda").split(",") if u.strip()
}
SESSION_TTL = 7 * 24 * 3600  # 7 days
COOKIE_NAME = "llm_session"


def create_session_token(username: str) -> str:
    """Create a signed JWT session token."""
    now = int(time.time())
    payload = {
        "sub": username,
        "iat": now,
        "exp": now + SESSION_TTL,
    }
    return jwt.encode(payload, SESSION_SECRET, algorithm="HS256")


def verify_session_token(token: str) -> Optional[dict]:
    """Verify and decode a session token. Returns payload or None."""
    try:
        return jwt.decode(token, SESSION_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def get_current_user(request: Request) -> Optional[str]:
    """Extract the authenticated username from the session cookie, or None."""
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None
    payload = verify_session_token(token)
    if not payload:
        return None
    return payload.get("sub")


async def exchange_code_for_user(code: str) -> Optional[str]:
    """Exchange GitHub OAuth code for access token, then get username."""
    async with httpx.AsyncClient(timeout=10) as client:
        # Exchange code for access token
        resp = await client.post(
            "https://github.com/login/oauth/access_token",
            json={
                "client_id": GITHUB_CLIENT_ID,
                "client_secret": GITHUB_CLIENT_SECRET,
                "code": code,
            },
            headers={"Accept": "application/json"},
        )
        if resp.status_code != 200:
            logger.error("GitHub token exchange failed: %s", resp.text)
            return None

        data = resp.json()
        access_token = data.get("access_token")
        if not access_token:
            logger.error("No access_token in GitHub response: %s", data)
            return None

        # Get user info
        user_resp = await client.get(
            "https://api.github.com/user",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json",
            },
        )
        if user_resp.status_code != 200:
            logger.error("GitHub user lookup failed: %s", user_resp.text)
            return None

        return user_resp.json().get("login")
