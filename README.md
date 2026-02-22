<h1 align="center">🕸️ AI Multi-Agent Misinformation Simulation</h1>

<p align="center">
  <b>A Graph-Based Multi-Agent System for Simulating Misinformation Spread, Verification & Moderation</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LangGraph-Orchestration-orange?logo=langchain&logoColor=white" />
  <img src="https://img.shields.io/badge/Groq-Llama_3.3_70B-green?logo=meta&logoColor=white" />
  <img src="https://img.shields.io/badge/Gemini-2.5_Flash-yellow?logo=google&logoColor=white" />
</p>

---

## 📌 Overview

This project demonstrates how to build **autonomous AI agent pipelines** using [LangGraph](https://python.langchain.com/docs/langgraph). It simulates a social-media misinformation lifecycle — from **claim generation** to **fact-checking** to **content moderation** — orchestrated as a directed graph of five cooperating agents.

```
Misinformation ➜ Neutral User ➜ Fact-Checker ➜ Influencer ➜ Moderator
     Agent           Agent          Agent          Agent        Agent
```

Each agent is a node in a LangGraph `StateGraph`. State flows through the graph sequentially; every agent reads from and writes to a shared `AgentState` TypedDict.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   LANGGRAPH WORKFLOW                      │
│                                                          │
│  ┌───────────┐  ┌─────────┐  ┌──────────┐  ┌─────────┐ │
│  │ MISINFO   │→ │ NEUTRAL │→ │FACTCHECK │→ │INFLUENCR│ │
│  │  AGENT    │  │  AGENT  │  │  AGENT   │  │  AGENT  │ │
│  └───────────┘  └─────────┘  └──────────┘  └─────────┘ │
│       │                            │             │       │
│       ▼                            ▼             ▼       │
│   Groq LLM                    Groq LLM      Groq LLM    │
│  (Llama 3.3)                 (Llama 3.3)   (Llama 3.3)  │
│                                                          │
│                        │                                 │
│                        ▼                                 │
│                 ┌────────────┐                           │
│                 │ MODERATOR  │                           │
│                 │   AGENT    │                           │
│                 └────────────┘                           │
│                        │                                 │
│                        ▼                                 │
│                 ┌────────────┐                           │
│                 │   FINAL    │                           │
│                 │   REPORT   │                           │
│                 └────────────┘                           │
└──────────────────────────────────────────────────────────┘
```

---

## 🤖 Agent Pipeline

| # | Agent | Role | Tech |
|---|-------|------|------|
| 1 | **Misinformation** | Generates a plausible, news-style claim (real or fake) | Groq LLM |
| 2 | **Neutral User** | Shares the claim as-is without verification | Pass-through |
| 3 | **Fact-Checker** | Analyses the claim → returns `Real` / `Fake` / `Unverified` with evidence | Groq LLM |
| 4 | **Influencer** | Rewrites content: warning (fake), viral (real), or disclaimer (unverified) | Groq LLM |
| 5 | **Moderator** | Final decision: **Flag & Stop**, **Mark for Review**, or **Allow** | Rule-based |

### Moderation Policy

| Verdict | Decision | Effect |
|---------|----------|--------|
| Fake | 🚫 Flag & Stop | Content blocked |
| Unverified | ⚠️ Mark for Review | Needs human review |
| Real | ✅ Allow | Content approved |

---

## 📂 Project Structure

```
langgraph-tutorial/
├── 1_BASICS.ipynb          # LangGraph fundamentals — simple StateGraph with Gemini
├── pydantic_2.ipynb        # Gemini API setup & Pydantic exploration
├── agents_formation.ipynb  # ⭐ Full multi-agent pipeline (main notebook)
├── documentation.md        # In-depth system documentation
├── main.py                 # Entry point (scaffold)
├── pyproject.toml          # Project metadata & dependencies
├── .env                    # API keys (GROQ_API_KEY, GOOGLE_API_KEY)
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/langgraph-tutorial.git
cd langgraph-tutorial
```

### 2. Create a virtual environment & install dependencies

```bash
# Using uv (recommended)
uv venv && uv pip install -r pyproject.toml

# — or using pip —
python -m venv myenv
myenv\Scripts\activate        # Windows
pip install langgraph langchain-groq langchain-google-genai python-dotenv matplotlib
```

### 3. Configure API keys

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_...
GOOGLE_API_KEY=AIza...
```

| Key | Used By | Get it at |
|-----|---------|-----------|
| `GROQ_API_KEY` | agents_formation.ipynb (Llama 3.3 70B) | [console.groq.com](https://console.groq.com/) |
| `GOOGLE_API_KEY` | 1_BASICS.ipynb, pydantic_2.ipynb (Gemini 2.5 Flash) | [aistudio.google.dev](https://aistudio.google.dev/) |

### 4. Run the simulation

Open **agents_formation.ipynb** in VS Code / Jupyter and run all cells. The output includes:

- Step-by-step agent logs
- Agent workflow graph visualization
- A **Final Report** with claim, verdict, evidence, amplification score, and moderation decision

---

## 📓 Notebooks

### `1_BASICS.ipynb` — LangGraph Fundamentals

Learn the core concepts:
- Setting up `ChatGoogleGenerativeAI` (Gemini 2.5 Flash)
- Defining a `TypedDict` state schema
- Creating a `StateGraph` with nodes and edges
- Compiling and invoking a graph
- Visualizing the graph with Mermaid

### `agents_formation.ipynb` — Multi-Agent Pipeline ⭐

The main notebook that implements the full five-agent simulation:
1. Defines `AgentState` (shared state schema)
2. Implements each agent as a Python function
3. Wires agents into a `StateGraph` pipeline
4. Compiles, visualizes, and invokes the graph
5. Outputs a structured JSON final report

### `pydantic_2.ipynb` — Pydantic & Gemini Setup

Quick-start notebook for Gemini API verification and Pydantic model exploration.

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.13+ |
| Agent Orchestration | [LangGraph](https://python.langchain.com/docs/langgraph) |
| LLM (Agents) | [Groq](https://groq.com/) — Llama 3.3 70B Versatile |
| LLM (Basics) | [Google Gemini](https://ai.google.dev/) 2.5 Flash |
| State Management | Python `TypedDict` |
| Visualization | Mermaid (via LangGraph) + Matplotlib |
| Environment | python-dotenv |

---

## 📊 Sample Output

```
MISINFORMATION AGENT
Generated Claim: A new study reveals that 5G towers have been linked to ...

NEUTRAL AGENT
Sharing claim as-is: A new study reveals that 5G towers ...

FACT-CHECKER AGENT
Claim: A new study reveals that 5G towers ...
Verdict: Fake
Evidence: No credible peer-reviewed research supports a link between ...

INFLUENCER AGENT
Action: Rewriting as WARNING post
Influenced Content: ⚠️ MISLEADING CLAIM ALERT: Reports circulating about 5G ...
Amplification Score: 6/10

MODERATOR AGENT
Decision: FLAG & STOP - Content blocked

  FINAL REPORT
  Original Claim:   A new study reveals that 5G towers ...
  Verdict: Fake
  Evidence: No credible peer-reviewed research ...
  Amplification Score: 6/10
  Moderation Decision: Flag & Stop
```

---

## 📖 Documentation

See [documentation.md](documentation.md) for a deep dive into:
- Conceptual design & problem domain
- Barabási-Albert network model
- Agent specifications & prompt strategies
- Analytics methodology (spread velocity, penetration rate, verification impact)
- Full workflow diagrams

---

## 🗺️ Roadmap

- [ ] Add NetworkX social-network graph simulation (BFS spread)
- [ ] Streamlit interactive dashboard
- [ ] Spread velocity & penetration analytics
- [ ] Support additional LLM providers (OpenAI, Anthropic)
- [ ] Persistent logging & run comparison

---

## 📝 License

This project is for **educational purposes**. Feel free to use, modify, and learn from it.

---

<p align="center">
  Built with ❤️ using <a href="https://python.langchain.com/docs/langgraph">LangGraph</a> &
  <a href="https://groq.com/">Groq</a>
</p>