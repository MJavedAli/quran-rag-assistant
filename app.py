import streamlit as st
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Quran RAG Assistant", page_icon="🕌", layout="wide")
st.title("🕌 Quran RAG Assistant")
st.caption("Ask questions based on Quranic verses and Hadiths (runs fully local with Llama 2)")

with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded_file = st.file_uploader("Upload Quran/Hadith text file", type=["txt"])
    build_index = st.button("🔍 Build Knowledge Base")

# ----------------------------
# Step 1: Load Data
# ----------------------------
if build_index:
    if uploaded_file:
        with open("uploaded_data.txt", "wb") as f:
            f.write(uploaded_file.getvalue())
        data_path = "uploaded_data.txt"
    else:
        os.makedirs("data", exist_ok=True)
        data_path = "data/quran_english.txt"
        st.info(f"Using default file: {data_path}")

    loader = TextLoader(data_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    st.write(f"📚 Total text chunks created: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")
    db.persist()
    st.success("✅ Knowledge base created successfully!")

    st.session_state["db_ready"] = True


# ----------------------------
# Step 2: Retrieval + QA Chain
# ----------------------------
if "db_ready" in st.session_state:
    retriever = Chroma(
        persist_directory="chroma_db",
        embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    ).as_retriever(search_kwargs={"k": 3})

    llm = Ollama(model="llama2", temperature=0.2)

    system_prompt = (
        "You are a helpful Islamic scholar assistant. "
        "Use the provided Quran and Hadith context to answer the question. "
        "If the answer is not clear in the text, say you don't know. "
        "Keep responses concise (max 3 sentences). "
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(retriever, question_answer_chain)

    st.subheader("💬 Ask a Question")
    query = st.text_input("Type your question (e.g. 'What does the Quran say about patience?')")

    if st.button("Ask"):
        if not query:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                response = qa_chain.invoke({"input": query})
                st.markdown("### 🧭 Answer")
                st.write(response["answer"] if "answer" in response else response)
