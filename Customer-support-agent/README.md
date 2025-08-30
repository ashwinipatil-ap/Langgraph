# Customer Support Agent with LangGraph

## Features
- Short-term and long-term memory
- Human-in-the-loop (HITL) escalation
- FAQ resolution using a mock LLM
- Modular LangGraph state handling

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python customer_support_agent.py
```

## Notes
- Persists memory to `user_memory.json`


## Streamlit Web App

```bash
streamlit run streamlit_app.py
```


## Exporting Chat History

Use the **"📤 Export Chat History"** button in the sidebar to download the current chat as a JSON file.
