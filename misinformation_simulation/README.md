# AI Multi-Agent Misinformation Simulation System

A Graph-Based Multi-Agent Platform for Social Network Information Dynamics

## Overview

This system simulates how misinformation spreads through social networks using five AI agents:
- **Misinformation Agent**: Generates realistic fake news claims using LLM
- **Neutral Agent**: Spreads content through the network using BFS
- **Fact-Checker Agent**: Verifies claims using LLM reasoning
- **Influencer Agent**: Rewrites content for viral spread or creates warnings
- **Moderator Agent**: Makes decisions to flag, block, or allow content

## Installation

1. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure API Key:**
   - Copy `.env.example` to `.env` (or edit the existing `.env`)
   - Add your Groq API key:
```
GROQ_API_KEY=your_groq_api_key_here
```
   - Get your API key at: https://console.groq.com/keys

## Usage

### Run the Gradio UI (Recommended)
```bash
python main.py
```
Then open http://localhost:7860 in your browser.

### Run CLI Mode
```bash
python main.py --cli
```

### Command Line Options
```bash
python main.py --help

Options:
  --cli         Run in CLI mode instead of UI
  --topic TEXT  Topic for claim generation (CLI mode)
  --nodes INT   Number of nodes in the social network (default: 15)
  --port INT    Port for Gradio UI (default: 7860)
  --share       Create a public share link for Gradio
```

## Project Structure

```
misinformation_simulation/
├── .env                    # Environment variables (API keys)
├── requirements.txt        # Python dependencies
├── main.py                 # Entry point
├── README.md              # This file
│
├── agents/                 # AI Agent modules
│   ├── __init__.py
│   ├── misinformation_agent.py
│   ├── neutral_agent.py
│   ├── fact_checker_agent.py
│   ├── influencer_agent.py
│   └── moderator_agent.py
│
├── graph/                  # Social network graph
│   ├── __init__.py
│   └── social_network.py
│
├── orchestration/          # LangGraph workflow
│   ├── __init__.py
│   └── workflow.py
│
└── ui/                     # Gradio interface
    ├── __init__.py
    └── gradio_app.py
```

## Agent Workflow

```
┌─────────────────┐
│  Misinformation │
│     Agent       │ → Generates claim
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Neutral      │
│     Agent       │ → Spreads through network (BFS)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Fact-Checker   │
│     Agent       │ → Verifies claim (REAL/FAKE/UNVERIFIED)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Influencer    │
│     Agent       │ → Rewrites content or creates warning
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Moderator     │
│     Agent       │ → BLOCK / FLAG / ALLOW decision
└─────────────────┘
```

## Technologies Used

- **LangGraph**: Multi-agent orchestration
- **LangChain + Groq**: LLM integration
- **NetworkX**: Social network graph modeling
- **Matplotlib**: Network visualization
- **Gradio**: Interactive web UI
- **Pydantic**: Data validation

## License

This is an educational project for MCA Final Year.
