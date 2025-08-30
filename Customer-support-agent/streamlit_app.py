# streamlit_app.py

import streamlit as st
import json
from datetime import datetime
from customer_support_agent import AgentState, state_reducer, support_agent, save_user_history

st.set_page_config(page_title="Customer Support Agent", layout="wide")
st.title("🤖 AI Customer Support Agent")

# -----------------------------
# Session state initialization
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_id" not in st.session_state:
    st.session_state.user_id = "user_123"

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Session Settings")
st.session_state.user_id = st.sidebar.text_input("User ID", st.session_state.user_id)

# Export chat history
if st.sidebar.button("📤 Export Chat History"):
    export_data = {
        "user_id": st.session_state.user_id,
        "messages": st.session_state.messages,
        "timestamp": datetime.now().isoformat()
    }
    export_json = json.dumps(export_data, indent=2)
    st.sidebar.download_button(
        label="Download JSON",
        data=export_json,
        file_name=f"chat_history_{st.session_state.user_id}.json",
        mime="application/json"
    )

# -----------------------------
# Chat Display
# -----------------------------
for msg in st.session_state.messages:
    role_icon = "🧑‍💼" if msg["role"] == "user" else "🤖"
    with st.chat_message(role_icon):
        st.markdown(msg["content"])

# -----------------------------
# Chat Input
# -----------------------------
user_input = st.chat_input("Ask a question...")
if user_input:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Build state and invoke agent
    agent_state = AgentState(user_id=st.session_state.user_id, messages=st.session_state.messages)
    updated_state = state_reducer(agent_state, {"role": "user", "content": user_input})
    final_state = support_agent.invoke(updated_state)

    # Use dictionary-style access (IMPORTANT)
    st.session_state.messages = final_state["messages"]

    # Save user history
    final_state["user_history"].append({
        "query": user_input,
        "response": final_state["messages"][-1]["content"]
    })

   # Save to long-term memory
    save_user_history(AgentState(
        user_id=final_state["user_id"],
        messages=final_state["messages"],
        user_history=final_state["user_history"]
    ))

    # Refresh UI
    st.rerun()
