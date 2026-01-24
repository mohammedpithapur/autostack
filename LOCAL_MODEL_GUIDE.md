# AutoStack - Local Model Support (Temporarily Paused)

Local/offline LLM support (llama.cpp/Ollama) is temporarily disabled in the CLI while we simplify the online-only flow. Until it returns, use cloud models (Claude, GPT-4.1, Gemini) with API keys.

## Current Status
- Local model selection is disabled in `autostack start`
- Any `local:` or `ollama:` model value will raise a friendly error
- Online providers remain fully supported

## Why Paused?
We are tightening reliability and speed for the online path first. Local support is planned to come back with a more robust install/check flow.

## Planned Re-Enablement (Roadmap)
- One-command local enablement with health checks
- Automatic GGUF download + caching
- llama.cpp CPU-first defaults, GPU when available
- Clear fallbacks to online models if local init fails

## Until Then
- Keep your API keys handy for Claude/GPT-4.1/Gemini
- Watch the changelog for the "Local LLM re-enabled" note
- Have feedback on local needs? Open an issue so we can prioritize the right configs
