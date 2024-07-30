import os
import pytesseract
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# 路径设置
pdf_dir = '/Users/yangqingyun/Documents/EasyContract/resource'
faiss_index_path = 'LLM.faiss'

# 初始化向量化模型
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_with_ocr(pdf_path):
    logging.info(f"Starting OCR extraction for {pdf_path}")
    pages = convert_from_path(pdf_path)
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page)
    logging.info(f"OCR extraction completed for {pdf_path}")
    return text

def process_pdf_files(pdf_dir, faiss_index_path):
    embeddings = []
    filenames = []

    # 遍历PDF目录并提取内容
    for filename in os.listdir(pdf_dir):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, filename)
            text = extract_text_with_ocr(pdf_path)
            embeddings.append(model.encode(text))
            filenames.append(filename)

    # 将嵌入向量存储到FAISS索引中
    if embeddings:
        embeddings = np.vstack(embeddings)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, faiss_index_path)
        logging.info(f"FAISS index saved to {faiss_index_path}")
    else:
        logging.info("No PDF files found in the directory.")

if __name__ == "__main__":
    process_pdf_files(pdf_dir, faiss_index_path)
