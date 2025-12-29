
# -------------------------------
# loop_ai_demo.py
# -------------------------------
import streamlit as st
from pyngrok import ngrok
from pathlib import Path
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Loop AI Agent Orchestration", layout="wide")
st.title("Loop Q - AI Agent Orchestration Prototype")
st.markdown("Select data source, strategy, model, and agent type. Workflow and results appear after execution.")

# -------------------------------
# Constants
# -------------------------------
VECTOR_DIR = "vector_db"
DATA_PATH = "/content/it_policy.pdf"
MODELS = {"Phi-3 Mini": "microsoft/phi-3-mini-4k-instruct"}

# -------------------------------
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None
if "model" not in st.session_state:
    st.session_state.model = None
if "workflow_generated" not in st.session_state:
    st.session_state.workflow_generated = False

# -------------------------------
# Helper Functions
# -------------------------------
def build_rag(pdf_path):
    os.makedirs(VECTOR_DIR, exist_ok=True)
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTOR_DIR)
    return db

def load_retriever():
    if not os.path.exists(VECTOR_DIR):
        return None
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
    return db.as_retriever(search_kwargs={"k": 1})

def load_slm(model_name):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    model_id = MODELS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    return tokenizer, model

def answer_question(question, retriever, tokenizer, model):
    # Retrieve context
    docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in docs])
    # Run model
    inputs = tokenizer(f"Context: {context}\nQuestion: {question}\nAnswer:", return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200,    use_cache=False)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# -------------------------------
# UI: Node Selection
# -------------------------------
st.subheader("1️⃣ Node Selection")
col1, col2 = st.columns(2)
with col1:
    data_source = st.selectbox("Select Data Source", ["PDF","SharePoint", "Jira", "Docs"])
    slm_strategy = st.selectbox("Select SLM Strategy", ["RAG", "Fine-tune"])
with col2:
    model_choice = st.selectbox("Select Model", ["Phi-3 Mini"])
    agent_type = st.selectbox("Select Agent Type", ["QA Agent", "Assistant"])

# -------------------------------
# Run Workflow
# -------------------------------
if st.button("Generate Workflow & Run", key="generate_workflow"):
    st.session_state.workflow_generated = True

    st.subheader("2️⃣ Node Workflow")
    workflow_dot = f"""
    digraph G {{
      rankdir=LR;
      node [shape=box, style=filled, color=lightgreen];
      data [label="{data_source}"];
      slm [label="{slm_strategy}"];
      model_node [label="{model_choice}"];
      agent [label="{agent_type}"];
      data -> slm -> model_node -> agent;
    }}
    """
    st.graphviz_chart(workflow_dot)

    st.subheader("3️⃣ Workflow Execution")
    status = st.empty()

    # Build/load RAG retriever only once
    status.info("Building RAG retriever from PDF...")
    if st.session_state.retriever is None:
        build_rag(DATA_PATH)
        st.session_state.retriever = load_retriever()
    status.success("RAG retriever ready!")

    # Load SLM model only once
    status.info(f"Loading {model_choice} model...")
    if st.session_state.model is None or st.session_state.tokenizer is None:
        tokenizer, model = load_slm(model_choice)
        st.session_state.tokenizer = tokenizer
        st.session_state.model = model
    status.success(f"{model_choice} loaded!")

# -------------------------------
# 4️⃣ Ask Question
# -------------------------------
if st.session_state.workflow_generated:
    st.subheader("4️⃣ Ask a Question")
    question = st.text_input("Enter your question for the agent:")

    if st.button("Run Agent", key="run_agent") and question.strip():
        status = st.empty()
        status.info("Running agent...")
        answer = answer_question(
            question,
            st.session_state.retriever,
            st.session_state.tokenizer,
            st.session_state.model
        )
        st.subheader("✅ Agent Response")
        st.write(answer)
        status.success("Workflow completed!")
