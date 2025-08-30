# customer_support_agent.py

from typing import List, Optional, Dict
from dataclasses import dataclass, field
import os
import json
from langgraph.graph import StateGraph
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# -----------------------------
# Simple JSON-based Memory Saver
# -----------------------------
class MemorySaver:
    def __init__(self, file_path: str):
        self.file_path = file_path
        if not os.path.exists(self.file_path) or os.path.getsize(self.file_path) == 0:
            with open(self.file_path, 'w') as f:
                json.dump({}, f)

    def save(self, user_id: str, history: List[Dict[str, str]]):
        try:
            with open(self.file_path, 'r') as f:
                content = f.read().strip()
                data = json.loads(content) if content else {}
        except (json.JSONDecodeError, IOError):
            data = {}

        data[user_id] = history

        with open(self.file_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, user_id: str) -> List[Dict[str, str]]:
        if not os.path.exists(self.file_path):
            return []
        try:
            with open(self.file_path, 'r') as f:
                content = f.read().strip()
                if not content:
                    return []
                data = json.loads(content)
        except (json.JSONDecodeError, IOError):
            return []
        return data.get(user_id, [])


# -----------------------------
# Define Agent State
# -----------------------------
@dataclass
class AgentState:
    user_id: str
    thread_id: Optional[str] = None
    messages: List[Dict[str, str]] = field(default_factory=list)
    user_history: List[Dict[str, str]] = field(default_factory=list)

    def validate(self):
        assert isinstance(self.user_id, str), "user_id must be a string"
        assert isinstance(self.messages, list), "messages must be a list"

# -----------------------------
# Short-Term Memory Trimming
# -----------------------------
def trim_messages(messages: List[Dict[str, str]], limit: int = 5):
    return messages[-limit:]

# -----------------------------
# Message Filtering
# -----------------------------
def filter_messages(messages: List[Dict[str, str]]):
    ignore = ["hi", "hello", "good morning"]
    return [msg for msg in messages if msg['content'].lower() not in ignore]

# -----------------------------
# Long-Term Memory Saver
# -----------------------------
memory_saver = MemorySaver("user_memory.json")

def save_user_history(state: AgentState):
    memory_saver.save(state.user_id, state.user_history)

def load_user_history(user_id: str):
    return memory_saver.load(user_id) or []

# -----------------------------
# Mock LLM Responder
# -----------------------------
def mock_llm_response(query: str) -> str:
    faq = {
        "how do i change my password": "To change your password, go to Settings > Security > Change Password.",
        "how can i reset my password": "Click on 'Forgot Password' at login. A reset link will be sent to your email."
    }
    key = query.lower().strip("?.!")
    return faq.get(key, "I'm forwarding this to a human agent for further assistance.")

# -----------------------------
# LLM Node
# -----------------------------
def llm_node(state: AgentState) -> AgentState:
    last_message = state.messages[-1]['content']
    reply = mock_llm_response(last_message)
    state.messages.append({"role": "agent", "content": reply})
    return state

# -----------------------------
# History Node
# -----------------------------
def history_node(state: AgentState) -> AgentState:
    if not state.messages:
        return state
    history = load_user_history(state.user_id)
    if history:
        summary = f"Last time, you asked about: {history[-1]['query']}"
        if not any(summary in msg["content"] for msg in state.messages):
            state.messages.append({"role": "agent", "content": summary})
    return state

# -----------------------------
# HITL Escalation (Human Agent)
# -----------------------------
def hitl_node(state: AgentState) -> AgentState:
    print("Human in the loop")
    state.messages.append({"role": "agent", "content": "Escalating to a human agent. Please wait..."})
    return state

# -----------------------------
# State Reducer
# -----------------------------
def state_reducer(state: AgentState, new_message: Dict[str, str]) -> AgentState:
    filtered_new = filter_messages([new_message])
    if filtered_new:
        state.messages.append(filtered_new[0])
    state.messages = trim_messages(state.messages)
    state.validate()
    return state

# -----------------------------
# Build State Graph
# -----------------------------
graph = StateGraph(AgentState)
graph.add_node("HistoryNode", history_node)
graph.add_node("LLMNode", llm_node)
graph.add_node("HITLNode", hitl_node)

graph.set_entry_point("HistoryNode")
graph.add_conditional_edges("HistoryNode", {
    "LLMNode": lambda state: "password" in state.messages[-1]["content"].lower(),
    "HITLNode": lambda state: "password" not in state.messages[-1]["content"].lower()
})
graph.add_edge("LLMNode", "HITLNode")
graph.set_finish_point("HITLNode")


support_agent = graph.compile()

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    input_message = {"role": "user", "content": "How do I change my password?"}
    initial_state = AgentState(user_id="user_123")
    updated_state = state_reducer(initial_state, input_message)
    final_state = support_agent.invoke(updated_state)

    for msg in final_state.messages:
        print(f"{msg['role']}: {msg['content']}")

    # Save history
    final_state.user_history.append({
        "query": input_message['content'],
        "response": final_state.messages[-1]['content']
    })

    save_user_history(final_state)