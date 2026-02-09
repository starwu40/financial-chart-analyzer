"""
Streamlit web application for financial chart analysis.
"""

import streamlit as st
import os
import subprocess
import uuid
import hashlib
import json
import sys

from financial_chart_analyzer.retrieval import ChartRetriever
from financial_chart_analyzer.analysis import ChartAnalyzer
from financial_chart_analyzer.config import config

# --- Page configuration ---
st.set_page_config(
    page_title="金融图表分析助手",
    page_icon="📊",
    layout="wide"
)

# --- Directory definitions ---
UPLOAD_DIR = str(config.upload_dir)
PREPROCESS_DIR = str(config.preprocess_dir)
CACHE_DIR = str(config.preprocess_cache_dir)

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PREPROCESS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


# --- Helper functions ---
def get_file_hash(file_path):
    """Calculate MD5 hash of a file for caching."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_cached_preprocess_dir(pdf_hash):
    """Get cache directory path for a specific PDF hash."""
    return os.path.join(CACHE_DIR, pdf_hash)


def is_pdf_preprocessed(pdf_hash):
    """Check if PDF has already been processed."""
    cache_dir = get_cached_preprocess_dir(pdf_hash)
    charts_json = os.path.join(cache_dir, "charts_index.json")
    return os.path.exists(charts_json)


# --- Core processing logic ---
def run_analysis_pipeline(pdf_path, query, pdf_hash):
    """
    Run the complete analysis pipeline: preprocessing -> retrieval -> LLM analysis.
    Uses caching to avoid reprocessing the same PDF.
    """
    cache_dir = get_cached_preprocess_dir(pdf_hash)
    os.makedirs(cache_dir, exist_ok=True)
    charts_json = os.path.join(cache_dir, "charts_index.json")
    index_bin = os.path.join(cache_dir, "chart_index.bin")

    # Step 1: Check cache, preprocess if needed
    if is_pdf_preprocessed(pdf_hash):
        st.success("✅ 检测到该PDF已处理过，直接使用缓存结果（节省时间）")
    else:
        st.info("步骤 1/3: 正在从PDF中提取图表...（首次处理需要一些时间）")
        try:
            # Use preprocess module directly instead of subprocess
            from financial_chart_analyzer.preprocessing import preprocess_pdf_charts
            preprocess_pdf_charts(
                pdf_path,
                out_dir=cache_dir,
                use_multiprocessing=config.use_multiprocessing
            )
            st.success("✅ PDF预处理完成，结果已缓存")
        except Exception as e:
            st.error(f"PDF预处理失败。错误信息: {e}")
            import traceback
            st.error(traceback.format_exc())
            return None, None

    # Step 2: Retrieve relevant charts
    st.info("步骤 2/3: 正在根据您的问题检索最相关的图表...")

    if not os.path.exists(charts_json):
        st.error(f"检索所需的文件 (charts_index.json) 未找到。路径: {charts_json}")
        return None, None

    retriever = ChartRetriever(charts_json, index_path=index_bin)
    retrieved_results = retriever.search(query, k=2)

    if not retrieved_results:
        st.warning("未能找到与您问题相关的图表。")
        return None, None

    # Step 3: LLM analysis
    st.info("步骤 3/3: 正在调用大语言模型进行分析和回答...")
    analyzer = ChartAnalyzer()
    if not analyzer.api_key:
        st.error("OpenAI API密钥未配置，无法进行分析。请设置 OPENAI_API_KEY 环境变量。")
        return None, None

    answer = analyzer.analyze_and_answer(query, retrieved_results)

    return answer, retrieved_results


# --- UI ---
st.title("📊 金融文档图表分析助手")
st.markdown("上传一份包含图表的金融PDF文档，然后提出您的问题，系统将自动分析并回答。")

# Session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'answer' not in st.session_state:
    st.session_state.answer = ""
if 'retrieved_charts' not in st.session_state:
    st.session_state.retrieved_charts = []
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False

# --- PDF upload section ---
with st.sidebar:
    st.header("1. 上传文档")
    uploaded_file = st.file_uploader("请选择一个PDF文件", type="pdf")

    if uploaded_file is not None:
        if not st.session_state.get('pdf_processed', False) or st.session_state.get('current_pdf_name') != uploaded_file.name:
            st.session_state.pdf_processed = False
            st.session_state.analysis_complete = False

            random_filename = f"{uuid.uuid4().hex}.pdf"
            pdf_path = os.path.join(UPLOAD_DIR, random_filename)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            pdf_hash = get_file_hash(pdf_path)

            st.session_state.pdf_path = pdf_path
            st.session_state.pdf_hash = pdf_hash
            st.session_state.current_pdf_name = uploaded_file.name

            if is_pdf_preprocessed(pdf_hash):
                st.success(f"✅ 文件 '{uploaded_file.name}' 上传成功！（该文件已处理过，可直接提问）")
            else:
                st.success(f"文件 '{uploaded_file.name}' 上传成功！")

# --- Main interaction area ---
if 'pdf_path' in st.session_state and os.path.exists(st.session_state.pdf_path):
    st.header("2. 提出您的问题")
    query = st.text_input("例如：'对比一下近三年的收入和利润增长情况' 或 '哪个业务分部的毛利率最高？'", key="query_input")

    analyze_button = st.button("开始分析", type="primary")

    if analyze_button and query:
        with st.spinner("正在处理，请稍候..."):
            pdf_hash = st.session_state.get('pdf_hash')
            if not pdf_hash:
                st.error("PDF哈希值缺失，请重新上传文件")
            else:
                answer, retrieved_charts = run_analysis_pipeline(
                    st.session_state.pdf_path,
                    query,
                    pdf_hash
                )
                st.session_state.answer = answer
                st.session_state.retrieved_charts = retrieved_charts
                st.session_state.analysis_complete = True
                st.session_state.pdf_processed = True

# --- Results display ---
if st.session_state.analysis_complete:
    st.divider()
    st.header("📈 分析结果")

    if st.session_state.answer:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("#### 模型回答")
            st.markdown(st.session_state.answer)

        with col2:
            st.markdown("#### 检索到的相关图表")
            if st.session_state.retrieved_charts:
                for i, (score, meta) in enumerate(st.session_state.retrieved_charts):
                    chart_path = meta.get("path")
                    if chart_path and os.path.exists(chart_path):
                        st.image(chart_path, caption=f"图表 {i+1}")
            else:
                st.info("没有找到用于生成回答的图表。")
    else:
        st.error("分析未能生成有效回答，请检查后台日志或调整您的问题。")
