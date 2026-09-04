"""Tóm tắt văn bản tiếng Việt dựa trên trích xuất từ khóa — đề tài 7, CTH625."""

import streamlit as st

from src.config import METHOD_KEYBERT, METHOD_TFIDF
from src.io_text import InputError, read_upload
from src.pipeline import run
from src.storage import list_recent

st.set_page_config(page_title="Tóm tắt theo từ khóa", layout="wide")
st.title("Tóm tắt văn bản dựa trên trích xuất từ khóa")
st.caption("Đề tài 7 — CTH625 Xử lý ngôn ngữ tự nhiên. Đầu vào: một văn bản tiếng Việt.")

with st.sidebar:
    st.header("Tham số")
    method_label = st.radio(
        "Phương pháp từ khóa",
        options=[METHOD_TFIDF, METHOD_KEYBERT],
        format_func=lambda value: (
            "TF-IDF + từ loại (baseline)"
            if value == METHOD_TFIDF
            else "KeyBERT + vietnamese-bi-encoder"
        ),
    )
    top_n = st.slider("Số từ khóa", min_value=5, max_value=20, value=10)
    do_summary = st.checkbox("Sinh tóm tắt bằng LLM", value=True)
    n_sentences = st.slider("Số câu tóm tắt", min_value=3, max_value=10, value=5)
    persist = st.checkbox("Lưu vào ChromaDB", value=True)
    st.caption("Bỏ tick thì lần chạy này không ghi vào lịch sử.")
    st.markdown(
        "KeyBERT lần đầu sẽ tải mô hình nhúng. "
        "Tóm tắt LLM cần `HF_TOKEN` (Hugging Face Inference Providers)."
    )

uploaded = st.file_uploader("Tải file .txt hoặc .pdf", type=["txt", "pdf"])
typed = st.text_area("Hoặc nhập văn bản tiếng Việt", height=220, placeholder="Dán văn bản tại đây...")

source_name = "nhap_lieu"
raw_text = typed
if uploaded is not None:
    try:
        raw_text = read_upload(uploaded.name, uploaded.getvalue())
        source_name = uploaded.name
        st.success(f"Đã đọc: {uploaded.name} ({len(raw_text)} ký tự)")
    except InputError as exc:
        st.error(str(exc))
        st.stop()

run_clicked = st.button("Trích xuất và tóm tắt", type="primary")

if run_clicked:
    if not (raw_text or "").strip():
        st.warning("Hãy nhập văn bản hoặc tải file.")
        st.stop()
    with st.spinner("Đang xử lý..."):
        try:
            result = run(
                raw_text,
                method=method_label,
                top_n=top_n,
                do_summary=do_summary,
                n_sentences=n_sentences,
                persist=persist,
                source=source_name,
            )
        except Exception as exc:
            st.error(str(exc))
            st.stop()

    left, right = st.columns(2)
    with left:
        st.subheader("Từ khóa")
        if result.keywords:
            st.dataframe(
                {
                    "từ khóa": [term for term, _ in result.keywords],
                    "điểm": [round(score, 4) for _, score in result.keywords],
                },
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("Không lấy được từ khóa sau khi lọc từ dừng và từ loại.")
        with st.expander("Token sau tách từ"):
            st.write(" ".join(result.tokens))
        with st.expander("Từ nội dung N/V/A"):
            st.write(", ".join(f"{token} ({pos})" for token, pos in result.content_words))

    with right:
        st.subheader("Tóm tắt")
        if result.summary:
            st.write(result.summary)
        elif do_summary:
            st.warning(result.warning or "Không sinh được tóm tắt.")
        else:
            st.caption("Đã tắt bước LLM.")
        if result.stored_id:
            st.caption(f"Đã lưu ChromaDB: `{result.stored_id}`")

def _history_summary(row: dict) -> str:
    summary = (row.get("summary") or "").strip()
    if summary:
        return summary[:160]
    note = (row.get("note") or "").strip()
    if note:
        return f"Chưa có tóm tắt ({note[:80]})"
    return "Chưa có tóm tắt"


st.divider()
st.subheader("Lịch sử gần đây")
st.caption("Chỉ hiện các lần đã bật «Lưu vào ChromaDB». Ô tóm tắt trống nghĩa là lúc lưu LLM lỗi hoặc đã tắt sinh tóm tắt.")
try:
    history = list_recent()
except Exception as exc:
    st.caption(f"Chưa đọc được ChromaDB: {exc}")
    history = []

if history:
    st.dataframe(
        [
            {
                "thời điểm": row.get("created_at_label") or row["created_at"],
                "nguồn": row["source"],
                "phương pháp": row["method"],
                "từ khóa": row["keywords"][:120],
                "tóm tắt": _history_summary(row),
            }
            for row in history
        ],
        hide_index=True,
        use_container_width=True,
    )
else:
    st.caption("Chưa có bản ghi.")
