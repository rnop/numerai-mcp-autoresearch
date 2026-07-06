"""
Upload the latest weekly submission pickle to the TAILSPIN live model slot.

This is the upload step of the weekly loop. The custom weekly MCP (server.py) builds the
pickle but deliberately does not submit; submission goes through the *official* Numerai MCP
(HTTP, https://api-tournament.numer.ai/mcp). When that server isn't wired up as a session
tool, this script drives it in-process with fastmcp.Client over the same HTTP transport, so
"run weekly retrain and submission" works end to end.

It performs the documented handoff: get_upload_auth -> PUT bytes to the presigned URL ->
create -> poll until validated -> assign to TAILSPIN.

Run with the numerai_rag_env interpreter (Python 3.11; has fastmcp + requests):
    python custom_mcp/upload_to_tailspin.py [path/to/submission.pkl]

If no pickle path is given, the newest submissions/*_meta.json is used.

Credentials come from .env (gitignored), never the command line:
  NUMERAI_MCP_AUTH  -> Authorization: Token <...> header (gates the MCP connection)
  API_TOKEN         -> PUBLIC_ID$SECRET_KEY, passed as the apiToken param on each tool call
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import requests
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
ENV_PATH = PROJECT_ROOT / ".env"

MCP_URL = "https://api-tournament.numer.ai/mcp"
TARGET_MODEL_NAME = "tailspin"          # the weekly slot is fixed — never ANGOSTURA/PIXELATED
TOURNAMENT = 8                          # Classic
# Python 3.11 image — the pickle is 3.11 bytecode; the default 3.12 fails with "unknown opcode 0".
PY311_DOCKER_IMAGE = "4d39918c-a82b-42ea-8dc7-ed5a30e676c5"

VALIDATION_TIMEOUT_S = 300
POLL_INTERVAL_S = 15


def load_env(path: Path = ENV_PATH) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def latest_pickle() -> Path:
    metas = sorted(SUBMISSIONS_DIR.glob("*_meta.json"), key=lambda p: p.stat().st_mtime)
    if not metas:
        raise SystemExit("No submissions/*_meta.json found — run the weekly retrain first.")
    pkl = SUBMISSIONS_DIR / (metas[-1].stem.replace("_meta", "") + ".pkl")
    if not pkl.exists():
        raise SystemExit(f"Pickle missing for latest meta: {pkl.name}")
    return pkl


def make_client(env: dict[str, str]) -> Client:
    header_token = env.get("NUMERAI_MCP_AUTH") or env.get("API_TOKEN")
    if not header_token:
        raise SystemExit("No NUMERAI_MCP_AUTH / API_TOKEN in .env")
    transport = StreamableHttpTransport(
        url=MCP_URL,
        headers={"Authorization": f"Token {header_token}"},
    )
    return Client(transport)


def _data(result):
    """Plain data out of a fastmcp CallToolResult (data / structured / JSON content blocks)."""
    data = getattr(result, "data", None)
    if data is not None:
        return data
    structured = getattr(result, "structured_content", None)
    if structured:
        return structured
    parsed = []
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text is not None:
            try:
                parsed.append(json.loads(text))
            except json.JSONDecodeError:
                parsed.append(text)
    if len(parsed) == 1:
        return parsed[0]
    return parsed


async def _resolve_tailspin_id(client: Client, api_token: str) -> str:
    result = await client.call_tool("graphql_query", {
        "apiToken": api_token,
        "query": "query { account { models { id name tournament } } }",
    })
    models = (_data(result) or {}).get("account", {}).get("models", [])
    matches = [m for m in models if m.get("name") == TARGET_MODEL_NAME]
    if len(matches) != 1:
        raise SystemExit(
            f"Expected exactly one model named {TARGET_MODEL_NAME!r}, found {len(matches)}."
        )
    return matches[0]["id"]


async def upload(pkl_path: Path) -> int:
    env = load_env()
    api_token = env.get("API_TOKEN")
    if not api_token:
        raise SystemExit("No API_TOKEN in .env (needed as the apiToken param).")

    client = make_client(env)
    async with client:
        model_id = await _resolve_tailspin_id(client, api_token)
        print(f"TAILSPIN model: {model_id}")
        print(f"Pickle: {pkl_path.name} ({pkl_path.stat().st_size:,} bytes)")

        # 1. presigned upload URL
        auth = _data(await client.call_tool("upload_model", {
            "apiToken": api_token, "operation": "get_upload_auth",
            "modelId": model_id, "filename": pkl_path.name,
        }))
        if isinstance(auth, dict) and "computePickleUploadAuth" in auth:
            auth = auth["computePickleUploadAuth"]
        url = auth.get("url")
        server_filename = auth.get("filename", pkl_path.name)
        if not url:
            raise SystemExit(f"No presigned URL in get_upload_auth response: {list(auth)}")
        print(f"get_upload_auth -> {server_filename}")

        # 2. PUT raw bytes. The signed Content-Type is empty, so send no Content-Type header.
        put = requests.put(url, data=pkl_path.read_bytes())
        put.raise_for_status()
        print(f"PUT bytes -> HTTP {put.status_code}")

        # 3. register the uploaded pickle with the Python 3.11 runtime
        created = _data(await client.call_tool("upload_model", {
            "apiToken": api_token, "operation": "create",
            "modelId": model_id, "filename": server_filename,
            "dockerImageId": PY311_DOCKER_IMAGE,
        }))
        created = created.get("createComputePickleUpload", created) if isinstance(created, dict) else created
        pickle_id = created.get("id")
        print(f"create -> pickle {pickle_id} (validation {created.get('validationStatus')})")
        if not pickle_id:
            raise SystemExit(f"create returned no pickle id: {created}")

        # 4. poll until validated
        deadline = time.time() + VALIDATION_TIMEOUT_S
        status = created.get("validationStatus")
        while status != "validated":
            if time.time() > deadline:
                raise SystemExit(f"Validation timed out (last status: {status}).")
            time.sleep(POLL_INTERVAL_S)
            listing = _data(await client.call_tool("upload_model", {
                "apiToken": api_token, "operation": "list", "modelId": model_id,
            }))
            entries = listing.get("computePickles", listing) if isinstance(listing, dict) else listing
            entries = entries if isinstance(entries, list) else []
            entry = next((e for e in entries if e.get("id") == pickle_id), None)
            status = (entry or {}).get("validationStatus")
            print(f"  validationStatus: {status}")
            if status == "failed":
                raise SystemExit(f"Pickle {pickle_id} failed validation.")

        # 5. assign the validated pickle as the active TAILSPIN model
        assigned = _data(await client.call_tool("upload_model", {
            "apiToken": api_token, "operation": "assign",
            "modelId": model_id, "pickleId": pickle_id,
        }))
        assigned = assigned.get("assignPickleToModel", assigned) if isinstance(assigned, dict) else assigned
        ok = str(assigned).lower() == "true"
        print(f"assign -> {assigned}")
        print("SUCCESS: TAILSPIN updated." if ok else "WARNING: assign did not return true.")
        return 0 if ok else 1


if __name__ == "__main__":
    target = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else latest_pickle()
    sys.exit(asyncio.run(upload(target)))
