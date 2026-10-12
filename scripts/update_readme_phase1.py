from pathlib import Path

p = Path("README.md")
content = p.read_text(encoding="utf-8")

# 1. Update Gemini architecture mention
old_gemini = "(Google Gemini / Local Deterministic Fallback)"
new_gemini = "(Local Llama 3.2 via Ollama / OpenAI-Compatible / Offline Fallback)"
assert old_gemini in content, "Gemini text not found in README.md"
content = content.replace(old_gemini, new_gemini)

# 2. Update Model 4 XGBoost measured row
old_xgb = "| **Model 4** | **TF-IDF + XGBoost (tree_method=hist)** | **MEASURED** | `results` | `results` | `results` | `results` | `results` | CPU / GPU |"
new_xgb = "| **Model 4** | **TF-IDF + XGBoost (tree_method=hist)** | **MEASURED** | **0.5462** | **0.5427** | **0.4074** | **0.4252** | **0.0212 ms** | CPU Multicore / GPU |"
assert old_xgb in content, "XGBoost placeholder row not found in README.md"
content = content.replace(old_xgb, new_xgb)

p.write_text(content, encoding="utf-8")
print("README.md successfully updated with real measured numbers and provider-agnostic Llama!")
