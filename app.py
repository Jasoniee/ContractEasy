import os
import pytesseract
from pdf2image import convert_from_path
from flask import Flask, request, jsonify, send_from_directory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import time
import sqlite3
import faiss
import numpy as np
from dashscope import Generation
from concurrent.futures import ThreadPoolExecutor
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

app = Flask(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['DASHSCOPE_API_KEY'] = 'sk-196ea76fc5e540768b0c899dbbd9a8d9'

def init_db():
    with sqlite3.connect('dialog_history_1.db', check_same_thread=False) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS dialog_history_1 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            sender TEXT NOT NULL,
            message TEXT NOT NULL
        );
        """)
    logging.info("Database initialized")

init_db()

# 使用适合英文合同向量化的模型
model = SentenceTransformer('all-MiniLM-L6-v2')
index = None  # 全局定义index变量
chunks = []  # 全局定义chunks变量

def extract_text_with_ocr(pdf_path):
    logging.info("Starting OCR extraction")
    pages = convert_from_path(pdf_path)
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page)
    logging.info("OCR extraction completed")
    return text

def analyze_chunk(chunk_text):
    logging.info(f"Analyzing chunk: {chunk_text[:30]}...")
    response = Generation.call(
        model="qwen-turbo",
        prompt=f'You are an expert in reviewing contracts and your duty is to help clients detect potential threats in the contract. analyze this contract section, your focus should be on the risks that might affect our client: {chunk_text}',
        max_tokens=1024
    )
    if 'status_code' in response and response['status_code'] == 200:
        output = response.get('output', {})
        choices = output.get('text', [])
        if choices:
            return choices
        else:
            return None
    else:
        return None

@app.route('/')
def serve_html():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/analyze-contract', methods=['POST'])
def analyze_contract():
    global index
    global chunks

    logging.info("Received request to analyze contract")
    start_time = time.time()
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    temp_dir = 'temp'
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)
    file.save(file_path)

    content = extract_text_with_ocr(file_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_text(content)
    logging.info("PDF loaded and split using OCR")

    # 将chunk插入到faiss本地向量数据库
    embeddings = [model.encode(chunk) for chunk in chunks]
    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(np.array(embeddings))
    faiss.write_index(index, 'LLM.faiss')
    logging.info("Chunks added to FAISS index")

    # 并行处理chunks
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(analyze_chunk, chunks))

    results = [res for res in results if res]  # 去除空结果
    logging.info("Chunks analyzed")

    summary_prompt = (
        "Now you receive analysis about different parts of a contract, you need to summary them，"
        "You need stand out the risk and weakness. Your analysis will start from here:\n\n"
    ) + "\n\n".join(results)

    all_results = "\n\n".join(results)
    final_text_splitter = RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=200)
    final_chunks = final_text_splitter.split_text(all_results)

    final_summaries = []
    for final_chunk in final_chunks:
        summary_response = Generation.call(
            model="qwen-turbo",
            prompt=f"You are an expert in reviewing contracts and your duty is to help clients detect potential threats in the contract. Now you receive analysis about different parts of a contract, you need to summary them,You need stand out the risk and weakness and your response should be within 150 words. Your analysis will start from here: {final_chunk}",
            max_tokens=1800
        )
        final_summary = summary_response.get('output', {}).get('text', '')
        final_summaries.append(final_summary)
    logging.info("Final summaries generated")

    if len(final_summaries) != 1:
        sum1 = "".join(final_summaries)
        summary_response = Generation.call(
            model="qwen-turbo",
            prompt=f"You are an expert in reviewing contracts and your duty is to help clients detect potential threats in the contract. Now you receive analysis about different parts of a contract, you need to summary them,You need stand out the risk and weakness and your response should be within 150 words. Your analysis will start from here: {sum1}",
            max_tokens=1800
        )
        end_time = time.time()
        logging.info(f"Contract analysis completed in {end_time - start_time} seconds")
        return jsonify({"summary": summary_response.get('output', {}).get('text', '')})

    if os.path.exists(file_path):
        os.remove(file_path)

    return jsonify({"summary": ''.join(final_summaries)})

@app.route('/chat', methods=['POST'])
def chat():
    global index
    global chunks

    data = request.json
    session_id = data.get('session_id')
    question = data.get('question')

    if not session_id or not question:
        return jsonify({"error": "Session ID or question not provided"}), 400

    # 将对话历史存储到数据库
    with sqlite3.connect('dialog_history_1.db', check_same_thread=False) as conn:
        cursor = conn.cursor()

        cursor.execute("INSERT INTO dialog_history_1 (session_id, timestamp, sender, message) VALUES (?, ?, ?, ?)",
                       (session_id, int(time.time()), 'user', question))
        conn.commit()

        # 获取对话历史
        cursor.execute("SELECT sender, message FROM dialog_history_1 WHERE session_id = ? ORDER BY timestamp", (session_id,))
        history = [{'sender': row[0], 'message': row[1]} for row in cursor.fetchall()]

        # 限制对话历史长度，只保留最近的10条对话
        max_history_length = 10
        if len(history) > max_history_length:
            history = history[-max_history_length:]

        # 构建对话历史上下文
        context = "\n".join([f"{entry['sender']}: {entry['message']}" for entry in history])

        # 确保index已经初始化
        if index is None:
            return jsonify({"error": "Index not initialized. Please upload and analyze a contract first."}), 500

        # 使用向量数据库搜索相关文档
        related_doc_indices = search_related_docs(question)
        related_docs = [chunks[i] for i in related_doc_indices]

        # 构建带有上下文的提示
        doc_context = "\n".join(related_docs)
        prompt = f"Conversation history:\n{context}\n\nUser question: {question}\n\nRelated document context:\n{doc_context} Your task it help the user as you can as possible"

        # 调用API获取回答
        response = Generation.call(
            model="qwen-turbo",
            prompt=prompt,
            max_tokens=1024
        )

        answer = response.get('output', {}).get('text', '')

        # 存储回答到数据库
        cursor.execute("INSERT INTO dialog_history_1 (session_id, timestamp, sender, message) VALUES (?, ?, ?, ?)",
                       (session_id, int(time.time()), 'bot', answer))
        conn.commit()

        return jsonify({"answer": answer})

def load_and_index_pdf(pdf_path):
    global index
    global chunks

    content = extract_text_with_ocr(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    chunks = text_splitter.split_text(content)
    logging.info("PDF loaded and split using OCR")

    embeddings = [model.encode(chunk) for chunk in chunks]
    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(np.array(embeddings))
    logging.info("Chunks added to FAISS index")

def search_related_docs(question, top_k=7):
    global index

    query_vector = model.encode([question])[0]
    D, I = index.search(np.array([query_vector]), top_k)
    return I[0]

if __name__ == '__main__':
    app.run(debug=True)
