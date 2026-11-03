"""Streamlit Web Interface for the Smart NLP Chatbot.
Enables instant 1-click cloud deployment on Streamlit Community Cloud and Hugging Face Spaces.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from nlp import process_text_pipeline

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart NLP Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Dark UI
st.markdown("""
<style>
    .main { background-color: #09090B; }
    .stChatMessage { border-radius: 14px; margin-bottom: 12px; }
    .ner-badge { display: inline-block; padding: 2px 8px; border-radius: 6px; font-weight: 600; font-size: 12px; margin: 2px; }
    .ner-person { background-color: rgba(168, 85, 247, 0.25); border: 1px solid #A855F7; color: #DDD6FE; }
    .ner-loc { background-color: rgba(59, 130, 246, 0.25); border: 1px solid #3B82F6; color: #BFDBFE; }
    .ner-org { background-color: rgba(6, 182, 212, 0.25); border: 1px solid #06B6D4; color: #A5F3FC; }
    .ner-date { background-color: rgba(16, 185, 129, 0.25); border: 1px solid #10B981; color: #A7F3D0; }
    .ner-time { background-color: rgba(245, 158, 11, 0.25); border: 1px solid #F59E0B; color: #FDE68A; }
    .ner-money { background-color: rgba(234, 179, 8, 0.25); border: 1px solid #EAB308; color: #FEF08A; }
    .ner-pct { background-color: rgba(236, 72, 153, 0.25); border: 1px solid #EC4899; color: #FBCFE8; }
    .token-chip { display: inline-block; background-color: #18181B; border: 1px solid #27272A; padding: 3px 8px; border-radius: 6px; margin: 3px; font-family: monospace; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "latest_nlp" not in st.session_state:
    st.session_state.latest_nlp = None
if "latest_input" not in st.session_state:
    st.session_state.latest_input = ""

# -----------------------------------------------------------------------------
# Sidebar: Navigation & NLP Concepts
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🧠 Smart NLP")
    st.caption("100% Local & Offline NLP Processing Engine")
    st.divider()

    st.subheader("💡 Benchmark Examples")
    test_queries = [
        "Hello, how are you?",
        "Sachin is studying Artificial Intelligence in Bengaluru.",
        "I am very happy because I got a new job.",
        "This product is very bad and disappointing.",
        "The children are playing football.",
        "Google is an organization based in the technology industry.",
        "Sachin lives in Bengaluru and works at Google.",
        "He paid $250 on January 15 at 3:30 PM with a 15% discount."
    ]

    for q in test_queries:
        if st.button(f"💬 {q[:32]}...", key=f"btn_{q}"):
            res = process_text_pipeline(q)
            st.session_state.messages.append({"role": "user", "content": q})
            st.session_state.messages.append({"role": "assistant", "content": res["response"]})
            st.session_state.latest_nlp = res["nlp"]
            st.session_state.latest_input = q
            st.rerun()

    st.divider()
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.latest_nlp = None
        st.session_state.latest_input = ""
        st.rerun()

    st.markdown("### ⚙️ Engine Status")
    st.success("🟢 5 NLP Modules Active\n\n• Tokenization (NLTK punkt)\n• Stemming (Porter)\n• Lemmatization (WordNet)\n• NER (spaCy + NLTK)\n• Sentiment (VADER)")

# -----------------------------------------------------------------------------
# Main Layout: 2 Columns (Chat & Live NLP Dashboard)
# -----------------------------------------------------------------------------
col_chat, col_dash = st.columns([1.1, 1.2], gap="large")

# ==================== COLUMN 1: CHAT INTERACTION ====================
with col_chat:
    st.subheader("💬 Interactive NLP Chat")
    
    # Display message history
    if not st.session_state.messages:
        st.info("👋 Welcome! Type any sentence or select a benchmark chip from the sidebar to chat and run real-time NLP analysis.")
    else:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Chat input field
    user_input = st.chat_input("Type a message to chat and analyze...")
    if user_input:
        res = process_text_pipeline(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": res["response"]})
        st.session_state.latest_nlp = res["nlp"]
        st.session_state.latest_input = user_input
        st.rerun()

# ==================== COLUMN 2: LIVE NLP DASHBOARD ====================
with col_dash:
    st.subheader("📊 Live 5-Technique NLP Dashboard")

    if not st.session_state.latest_nlp:
        st.info("Awaiting input. Send a message to inspect the 5 NLP techniques in real time.")
    else:
        nlp = st.session_state.latest_nlp
        st.caption(f"**Analyzed Query:** *\"{st.session_state.latest_input}\"*")

        tab_tok, tab_stem, tab_lemma, tab_ner, tab_sent = st.tabs([
            "1. Tokenization",
            "2. Stemming",
            "3. Lemmatization",
            "4. NER",
            "5. Sentiment"
        ])

        # --- 1. TOKENIZATION ---
        with tab_tok:
            tok = nlp.get("tokenization", {})
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Tokens", tok.get("total_tokens", 0))
            m2.metric("Words", tok.get("word_count", 0))
            m3.metric("Punctuation", tok.get("punctuation_count", 0))
            
            st.markdown("##### Extracted Tokens")
            chips_html = "".join([f"<span class='token-chip'>{t}</span>" for t in tok.get("tokens", [])])
            st.markdown(chips_html, unsafe_allow_html=True)

        # --- 2. STEMMING ---
        with tab_stem:
            stem = nlp.get("stemming", {})
            st.caption("PorterStemmer heuristic suffix stripping")
            s_rows = stem.get("results", [])
            if s_rows:
                table_data = [{"Original Word": r["original"], "Stemmed Result": r["stemmed"], "Modified": "✅ Yes" if r["changed"] else "— No"} for r in s_rows]
                st.dataframe(table_data, use_container_width=True)
            else:
                st.write("No words to stem.")

        # --- 3. LEMMATIZATION ---
        with tab_lemma:
            lemma = nlp.get("lemmatization", {})
            st.caption("WordNetLemmatizer with POS tag mapping")
            l_rows = lemma.get("results", [])
            if l_rows:
                table_data = [{"Original": r["original"], "POS Tag": r["pos_tag"], "Category": r["pos_category"], "Lemma (Base Form)": r["lemma"], "Modified": "✅ Yes" if r["changed"] else "— No"} for r in l_rows]
                st.dataframe(table_data, use_container_width=True)
            else:
                st.write("No words to lemmatize.")

        # --- 4. NAMED ENTITY RECOGNITION (NER) ---
        with tab_ner:
            ner = nlp.get("ner", {})
            entities = ner.get("entities", [])
            st.metric("Detected Entities", len(entities))
            
            if entities:
                st.markdown("##### Highlighted Entity Elements")
                badges_html = ""
                for e in entities:
                    lbl = e["label"]
                    cls = "ner-org"
                    if lbl == "PERSON": cls = "ner-person"
                    elif lbl == "LOCATION": cls = "ner-loc"
                    elif lbl == "DATE": cls = "ner-date"
                    elif lbl == "TIME": cls = "ner-time"
                    elif lbl == "MONEY": cls = "ner-money"
                    elif lbl == "PERCENTAGE": cls = "ner-pct"
                    badges_html += f"<span class='ner-badge {cls}'>{e.get('icon', '')} {e['text']} → <strong>{lbl}</strong></span> "
                st.markdown(badges_html, unsafe_allow_html=True)
                
                st.divider()
                st.dataframe([{"Entity Text": e["text"], "Category": e["label"], "Position": f"{e['start']}:{e['end']}"} for e in entities], use_container_width=True)
            else:
                st.warning("No entities detected in this sentence.")

        # --- 5. SENTIMENT ANALYSIS ---
        with tab_sent:
            sent = nlp.get("sentiment", {})
            cat = sent.get("category", "NEUTRAL")
            compound = sent.get("compound", 0.0)
            
            c1, c2 = st.columns(2)
            c1.metric("Classification", f"{sent.get('icon', '')} {cat}")
            c2.metric("Compound Score", f"{compound:+.4f}")
            
            st.markdown(f"**Mood Assessment:** *{sent.get('mood', 'Neutral')}*")
            st.divider()
            
            st.write(f"**Positive:** {sent.get('pos_percent', 0.0)}%")
            st.progress(min(1.0, max(0.0, float(sent.get("pos", 0.0)))))
            
            st.write(f"**Neutral:** {sent.get('neu_percent', 100.0)}%")
            st.progress(min(1.0, max(0.0, float(sent.get("neu", 1.0)))))
            
            st.write(f"**Negative:** {sent.get('neg_percent', 0.0)}%")
            st.progress(min(1.0, max(0.0, float(sent.get("neg", 0.0)))))
