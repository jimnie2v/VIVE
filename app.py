import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import os

# 1. 페이지 설정 및 iMessage 스타일 CSS
st.set_page_config(page_title="iOS Chatbot", page_icon="💬")

st.markdown("""
<style>
    .stApp { background-color: #ffffff; }
    .chat-bubble {
        padding: 10px 15px;
        border-radius: 20px;
        margin: 5px 0;
        max-width: 70%;
        font-family: -apple-system, sans-serif;
        font-size: 15px;
        line-height: 1.4;
    }
    .user-bubble {
        background-color: #007aff;
        color: white;
        align-self: flex-end;
        border-bottom-right-radius: 2px;
    }
    .bot-bubble {
        background-color: #e9e9eb;
        color: black;
        align-self: flex-start;
        border-bottom-left-radius: 2px;
    }
    .chat-row { display: flex; flex-direction: column; }
    [data-testid="stChatMessageAvatarBackground"] { display: none; }
</style>
""", unsafe_allow_html=True)

# API 보안 설정
if "GEMINI_API_KEY" not in st.secrets:
    st.error("Streamlit Cloud 설정에서 'GEMINI_API_KEY'를 추가해주세요.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]

# 2. RAG 엔진 (최신 Gemini 2.5 Flash 및 엄격 프롬프트)
@st.cache_resource
def init_rag():
    if not os.path.exists("test.pdf"):
        return None
    loader = PyPDFLoader("test.pdf")
    docs = loader.load()
    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)
    vectorstore = FAISS.from_documents(splits, GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"))
    
    # [필수] gemini-2.5-flash 모델 사용
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    template = """당신은 제공된 문서를 기반으로 답변하는 비서입니다.
    규칙:
    1. 반드시 제공된 컨텍스트(Context) 내용만 참고하세요.
    2. 문서에 없는 내용이라면 "죄송합니다. 해당 정보는 문서에서 찾을 수 없습니다."라고만 답하세요.
    3. 외부 지식을 절대 사용하지 마세요.

    Context: {context}
    Question: {question}
    Answer:"""
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PromptTemplate(template=template, input_variables=["context", "question"])}
    )

chain = init_rag()

# 3. 채팅 UI 구현
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    align = "flex-end" if msg["role"] == "user" else "flex-start"
    bubble_type = "user-bubble" if msg["role"] == "user" else "bot-bubble"
    st.markdown(f'<div style="display: flex; flex-direction: column; align-items: {align};"><div class="chat-bubble {bubble_type}">{msg["content"]}</div></div>', unsafe_allow_html=True)

if prompt := st.chat_input("메시지를 입력하세요..."):
    st.markdown(f'<div style="display: flex; flex-direction: column; align-items: flex-end;"><div class="chat-bubble user-bubble">{prompt}</div></div>', unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    if chain:
        with st.spinner(""):
            res = chain.invoke({"question": prompt})
            ans = res['answer']
            st.markdown(f'<div style="display: flex; flex-direction: column; align-items: flex-start;"><div class="chat-bubble bot-bubble">{ans}</div></div>', unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": ans})
    else:
        st.error("test.pdf 파일이 없습니다.")
