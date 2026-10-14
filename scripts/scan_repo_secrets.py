"""Security Audit: Scan tracked and untracked files for secrets, tokens, API keys, and credentials."""

import os
import subprocess
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

def main():
    print("=" * 70)
    print("STEP 1: SECURITY SCAN FOR API KEYS AND SECRETS")
    print("=" * 70)

    # 1. Get all tracked files via git
    try:
        git_files_raw = subprocess.check_output(["git", "ls-files"], cwd=REPO_ROOT).decode("utf-8")
        git_files = [f.strip() for f in git_files_raw.splitlines() if f.strip()]
    except Exception as e:
        print(f"Error running git ls-files: {e}")
        git_files = []

    print(f"Total tracked files in git: {len(git_files)}")

    secret_patterns = [
        ("Anthropic API Key", re.compile(r"sk-ant-[a-zA-Z0-9_\-]{20,}")),
        ("OpenAI API Key", re.compile(r"sk-[a-zA-Z0-9_\-]{32,}")),
        ("Groq API Key", re.compile(r"gsk_[a-zA-Z0-9_\-]{20,}")),
        ("GitHub Personal Access Token", re.compile(r"ghp_[a-zA-Z0-9_\-]{20,}")),
        ("Together API Key", re.compile(r"together_[a-zA-Z0-9_\-]{20,}")),
        ("HuggingFace Token", re.compile(r"hf_[a-zA-Z0-9_\-]{30,}")),
        ("Private Key Block", re.compile(r"-----BEGIN\s+([A-Z\s]+)-----")),
        ("Generic Hardcoded Secret", re.compile(r"(?i)(api[_-]?key|secret_key|access_token)\s*=\s*['\"][a-zA-Z0-9_\-\.]{24,}['\"]"))
    ]

    findings = []
    for rel in git_files:
        p = REPO_ROOT / rel
        if not p.is_file():
            continue
        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        
        for name, pat in secret_patterns:
            matches = pat.findall(content)
            if matches:
                # Filter out obvious false positives like dummy templates
                filtered = []
                for m in matches:
                    val = m if isinstance(m, str) else str(m)
                    if any(dummy in val.lower() for dummy in ["change_this", "your_", "placeholder", "gsk_your", "ollama"]):
                        continue
                    filtered.append(val)
                if filtered:
                    findings.append((rel, name, len(filtered)))

    print("\n--- Tracked Files Scan Result ---")
    if findings:
        print(f"ALERT: Potential secrets found in {len(findings)} tracked files:")
        for rel, name, count in findings:
            print(f"  [FOUND] File: {rel} | Type: {name} | Occurrences: {count}")
    else:
        print("PASS: 0 secrets, tokens, or API keys found in tracked git files.")

    # 2. Check untracked files like .env
    print("\n--- Untracked .env / Environment File Scan ---")
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        env_content = env_path.read_text(encoding="utf-8", errors="ignore")
        env_secrets = []
        for name, pat in secret_patterns:
            matches = pat.findall(env_content)
            if matches:
                filtered = [m for m in matches if not any(d in str(m).lower() for d in ["change_this", "your_", "placeholder", "ollama"])]
                if filtered:
                    env_secrets.append((name, len(filtered)))
        if env_secrets:
            print(f"NOTE: .env on disk contains credentials: {env_secrets}")
            # Verify .env is in .gitignore
            gitignore_text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8", errors="ignore")
            if ".env" in gitignore_text:
                print("PASS: .env is strictly ignored by .gitignore (Not committed to repository).")
            else:
                print("ALERT: .env is NOT in .gitignore!")
        else:
            print("NOTE: .env contains only placeholder configurations.")
    else:
        print("NOTE: No .env file on disk.")

    # 3. Check .env.example
    print("\n--- .env.example Sanity Check ---")
    env_ex_path = REPO_ROOT / ".env.example"
    if env_ex_path.exists():
        ex_content = env_ex_path.read_text(encoding="utf-8", errors="ignore")
        ex_secrets = []
        for name, pat in secret_patterns:
            matches = pat.findall(ex_content)
            filtered = [m for m in matches if not any(d in str(m).lower() for d in ["change_this", "your_", "placeholder", "ollama"])]
            if filtered:
                ex_secrets.append((name, len(filtered)))
        if ex_secrets:
            print(f"ALERT: .env.example contains real credentials: {ex_secrets}")
        else:
            print("PASS: .env.example contains exclusively dummy placeholders.")

if __name__ == "__main__":
    main()
