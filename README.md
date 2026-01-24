# AutoStack: The AI-Native SaaS Framework

Generate production-ready SaaS apps instantly, powered by autonomous AI Agents.

## 🚀 What is AutoStack?

AutoStack is an AI-native framework designed to transform high-level ideas into fully functional SaaS applications rapidly, effortlessly, and at scale.

Powered by cutting-edge AI models, AutoStack automatically scaffolds full-stack React apps with Flask backend and SQLite database, built-in authentication, real-time features, complete TypeScript integration, and production-grade database schemas.

## ✨ Key Features

- 🤖 **AI-Powered Generation**: Instantly build full-stack React apps with Flask & SQLite
- 🏗️ **Complete Application Stack**: Frontend (React), backend (Flask), authentication , data persistence  all handled automatically
- 🔐 **Integrated Authentication**: Seamless Flask auth setup with protected routes
- 🎯 **TypeScript First**: Clean, strongly-typed React codebases by default
- ⚡ **Real-time Capabilities**: Real-time features via Flask integration
- 📊 **Automated Database Setup**: Auto-generated SQLite schemas and migrations


## 📦 Quick Start

### Installation

Install from PyPI:

```bash
pip install autostack_cli
```

Or install from source:

```bash
git clone https://github.com/mohammedpithapur/autostack.git
cd autostack
pip install -e .
```

## 🛠️ Usage

Start a new SaaS project effortlessly:

```bash
autostack start
```

### Interactive Setup Flow

Running `autostack start` guides you through:

✅ **Project Name & Description**

📋 **Template Selection**
- E-commerce Template
- SaaS Marketing Template
- CRM Template
- Default Template

🧩 **Tech Stack Selection**
- React (UI Only): For frontend-only projects
- React + Flask + SQLite: For full-stack projects with authentication, data persistence, and real-time features

🔑 **Database Configuration**
- SQLite database auto-setup

🤖 **AI Model Selection**
- Claude 3.7 Sonnet (Anthropic)
- GPT-4.1 (OpenAI)
- Gemini 2.5 Pro (Google)

## 🔑 API Keys Setup

Create a `.env` file with your API key(s):

```env
# Claude
ANTHROPIC_API_KEY=your_anthropic_api_key

# GPT-4.1
OPENAI_API_KEY=your_openai_api_key

# Gemini
GOOGLE_API_KEY=your_google_api_key
```

Only one API key (for your chosen model) is required.



## 🏗️ Behind-the-Scenes Build Steps

The AutoStack CLI handles:

| Task | Details |
|------|---------|
| Project Initialization | Scaffold React + Flask project with TypeScript integration |
| Authentication Setup | Integrated Flask Auth with UI & route protection |
| Database Schema | Automatic SQLite DB setup, models, and migrations |
| Dependency Installation | npm & pip dependencies and dev environment configuration |
| Real-time Setup | Native integration of Flask real-time capabilities |
| Development Server | Auto-start React and Flask dev servers for immediate preview |

## 📝 Example Output

```
✅ Build complete!
╭─────────────── Build Summary ───────────────╮
│                                             │
│ Project:       task-manager                 │
│ Description:   App for managing tasks       │
│ Tech Stack:    React + Flask + SQLite      │
│ Files Created: 42                           │
│ Status:        FINISHED                     │
│                                             │
╰─────────────────────────────────────────────╯
```

## 🌟 Our Vision

AutoStack aims to revolutionize application development, leveraging advanced AI to automate the creation of full-stack SaaS products, empowering developers to focus purely on innovation and unique business logic.

## 🎯 Our Mission

To build the most intuitive, powerful AI-native SaaS generation framework making software development dramatically faster, simpler, and more creative.

## 🤝 Contributing

Join our open-source community and help shape the future:

- 🌱 Fork and improve the repo
- 🛠️ Submit pull requests with features or fixes
- 💡 Share your suggestions and feedback on GitHub issues

## 📄 License

Licensed under MIT – see LICENSE for details.

## 🔗 Useful Links

- GitHub: [github.com/your-username/autostack](https://github.com/mohammedpithapur/autostack)
- PyPI: [pypi.org/project/autostack](https://pypi.org/project/autostack_cli)

⭐ Support the framework by starring the repo!

**AutoStack   SaaS app development reimagined.**

## 🧭 Future Plan
- Enable local LLM (llama.cpp/Ollama) with streamlined install checks, automatic GGUF downloads, and safe fallbacks to online providers when local init fails.
