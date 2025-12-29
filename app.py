import streamlit as st
import duckdb
import os
import uuid
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from pypdf import PdfReader
from PIL import Image
import google.generativeai as genai
import json
import io

# Load API Key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

# --- TASK 1: DATABASE LAYER ---
DB_FILE = "local_chat.ddb"

def get_db_connection():
    conn = duckdb.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_logs (
            session_id VARCHAR,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            role VARCHAR,
            content_text VARCHAR,
            has_attachment BOOLEAN
        )
    """)
    return conn

def save_message(session_id, role, text, has_attachment):
    conn = get_db_connection()
    conn.execute("INSERT INTO chat_logs (session_id, role, content_text, has_attachment) VALUES (?, ?, ?, ?)", 
                 (session_id, role, text, has_attachment))
    conn.close()

def load_history(session_id):
    conn = get_db_connection()
    result = conn.execute("SELECT role, content_text FROM chat_logs WHERE session_id = ? ORDER BY timestamp ASC", 
                          (session_id,)).fetchall()
    conn.close()
    return result

# NEW FUNCTION: Fetch all unique sessions for the sidebar history
def get_all_sessions():
    conn = get_db_connection()
    # Gets unique session IDs and the time of the first message in that session
    result = conn.execute("""
        SELECT session_id, MIN(timestamp) as start_time 
        FROM chat_logs 
        GROUP BY session_id 
        ORDER BY start_time DESC
    """).fetchall()
    conn.close()
    return result

# --- TASK 4 & 9: FILE PROCESSING ---
def process_file(uploaded_file, session_id):
    file_json_context = {}
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        text = "".join([page.extract_text() or "" for page in reader.pages])
        if len(text.strip()) > 10: 
            file_json_context = {"file_type": "Digital PDF", "filename": uploaded_file.name, "content": text}
        else:
            file_json_context = {"file_type": "Handwritten/Scanned PDF", "status": "OCR_REQUIRED"}
    elif uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        file_json_context = {"file_type": "CSV Data", "filename": uploaded_file.name, "data": df.to_dict(orient='records')}
    elif uploaded_file.type in ["image/png", "image/jpeg"]:
        file_json_context = {"file_type": "Image", "status": "VISION_REQUIRED"}
    return file_json_context

# --- SESSION MANAGEMENT ---
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

# LOOP FIX: Counter to reset audio widget
if "audio_key_index" not in st.session_state:
    st.session_state["audio_key_index"] = 0

# --- UI SETUP ---
st.set_page_config(page_title="Gemini + DuckDB Local Chat", layout="wide")
st.title("Gemini + DuckDB Local Chat")

# --- UPDATED SIDEBAR UI ---
# --- FIXED SIDEBAR UI ---
with st.sidebar:
    st.header("Chat Management")
    
    # 1. New Chat Button
    if st.button("➕ Start New Chat", use_container_width=True):
        st.session_state["session_id"] = str(uuid.uuid4())
        st.session_state["audio_key_index"] += 1
        st.rerun()

    st.write("---")

    # 2. Session Selector (History List)
    past_sessions = get_all_sessions()
    
    # Logic Fix: Create a list of all IDs, ensuring the CURRENT one is included
    session_ids = [s[0] for s in past_sessions]
    if st.session_state["session_id"] not in session_ids:
        # Add the new, unsaved session to the top of the list
        session_ids.insert(0, st.session_state["session_id"])
    
    # Build labels for the dropdown
    def get_label(sid):
        # Find if this ID has a timestamp in past_sessions
        for s in past_sessions:
            if s[0] == sid:
                return f"Chat {s[1].strftime('%b %d, %H:%M')}"
        return "✨ Current New Chat"

    st.subheader("📂 Past Conversations")
    selected_id = st.selectbox(
        "Select a session to load",
        options=session_ids,
        format_func=get_label,
        index=session_ids.index(st.session_state["session_id"])
    )

    # Only switch and rerun if the user actually clicks a different session
    if selected_id != st.session_state["session_id"]:
        st.session_state["session_id"] = selected_id
        st.rerun()

    st.write("---")
    st.write(f"**Active Session:**\n`{st.session_state['session_id']}`")
    
    system_prompt = st.text_area("System Prompt", value="You are a helpful assistant.")
    uploaded_file = st.file_uploader("Upload context", type=["pdf", "csv", "png", "jpg"])
    
    st.write("---")
    st.write("🎙️ **Voice Command**")
    audio_value = st.audio_input("Record message", key=f"audio_{st.session_state['audio_key_index']}")
# --- DISPLAY HISTORY ---
history = load_history(st.session_state["session_id"])
for role, text in history:
    with st.chat_message(role):
        st.markdown(text)

# --- CORE LOGIC ---
prompt = st.chat_input("Type your message here...")

if prompt or audio_value:
    user_input_mode = prompt if prompt else "Voice Command"
    
    file_context = {}
    if uploaded_file is not None:
        file_context = process_file(uploaded_file, st.session_state["session_id"])
        display_text = f"Attached: {uploaded_file.name}\n\n{user_input_mode}"
    else:
        display_text = user_input_mode

    gemini_history = [{"role": "user", "parts": [f"System Instruction: {system_prompt}"]}]
    for role, text in history:
        gemini_history.append({"role": "user" if role == "user" else "model", "parts": [text]})
    
    current_parts = []
    
    # Handle Audio
    if audio_value:
        current_parts.append({"mime_type": "audio/wav", "data": audio_value.getvalue()})
        current_parts.append("INSTRUCTIONS: User sent a voice command. Transcribe and answer.")

    # Handle Files/JSON
    if file_context:
        if "data" in file_context or "content" in file_context:
            current_parts.append(f"CONTEXT (JSON):\n{json.dumps(file_context, indent=2)}")
        if file_context.get("status") == "OCR_REQUIRED":
            current_parts.append("This is a scan. Transcribe to JSON and answer.")
            current_parts.append({"mime_type": "application/pdf", "data": uploaded_file.getvalue()})

    if prompt:
        current_parts.append(f"USER TEXT: {prompt}")

    gemini_history.append({"role": "user", "parts": current_parts})

    try:
        response = model.generate_content(gemini_history)
        bot_response = response.text
        
        save_message(st.session_state["session_id"], "user", display_text, bool(uploaded_file))
        save_message(st.session_state["session_id"], "assistant", bot_response, False)
        
        st.session_state["audio_key_index"] += 1
        st.rerun()

    except Exception as e:
        st.error(f"Error: {e}")