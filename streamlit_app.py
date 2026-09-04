"""Tóm tắt văn bản tiếng Việt dựa trên trích xuất từ khóa — đề tài 7, CTH625."""

import streamlit as st

from src.config import METHOD_KEYBERT, METHOD_TFIDF
from src.io_text import InputError, read_upload
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
        key="method_label",
    )
    top_n = st.slider("Số từ khóa", min_value=5, max_value=20, value=10, key="top_n")
    do_summary = st.checkbox(
        "Sinh tóm tắt bằng LLM",
        value=True,
        help="Bỏ tick thì chỉ trích từ khóa, không gọi Hugging Face.",
        key="do_summary",
    )
    n_sentences = st.slider(
        "Số câu tóm tắt",
        min_value=3,
        max_value=10,
        value=5,
        disabled=not do_summary,
        key="n_sentences",
    )
    persist = st.checkbox("Lưu vào ChromaDB", value=True, key="persist")
    st.caption("Bỏ tick «Lưu vào ChromaDB» thì lần chạy này không ghi vào lịch sử.")
    st.markdown(
        "KeyBERT lần đầu sẽ tải mô hình nhúng. "
        "Tóm tắt LLM cần `HF_TOKEN` (Hugging Face Inference Providers)."
    )

def _on_file_change() -> None:
    st.session_state["_file_loading"] = True


def _read_uploaded(uploaded) -> None:
    file_key = (uploaded.name, uploaded.size)
    if st.session_state.get("_upload_key") == file_key:
        st.success(
            f"Đã đọc: {uploaded.name} ({len(st.session_state['_upload_text'])} ký tự). "
            "Bấm «Trích xuất và tóm tắt» để chạy."
        )
        return
    progress = st.status("Đang đọc file (tải lên máy chủ xong, đang tách chữ)…", expanded=True)
    try:
        text = read_upload(uploaded.name, uploaded.getvalue())
    except InputError as exc:
        progress.update(label="Không đọc được file", state="error")
        st.session_state["_file_loading"] = False
        st.error(str(exc))
        return
    st.session_state["_upload_key"] = file_key
    st.session_state["_upload_text"] = text
    st.session_state["_upload_name"] = uploaded.name
    st.session_state["_file_loading"] = False
    progress.update(
        label=f"Đã đọc {uploaded.name} ({len(text)} ký tự)",
        state="complete",
    )


uploaded = st.file_uploader(
    "Tải file .txt hoặc .pdf",
    type=["txt", "pdf"],
    on_change=_on_file_change,
)
st.caption(
    "Sau khi chọn file, đợi dòng «Đã đọc» bên dưới — lúc đó Streamlit đang tải file lên, "
    "chưa phải bước trích từ khóa. Chỉ bấm nút khi đã thấy «Đã đọc»."
)
if st.session_state.get("_file_loading") and uploaded is None:
    st.info("Đang tải file từ trình duyệt lên máy chủ…")
if uploaded is not None:
    _read_uploaded(uploaded)
else:
    st.session_state.pop("_upload_key", None)
    st.session_state.pop("_upload_text", None)
    st.session_state.pop("_upload_name", None)
    st.session_state["_file_loading"] = False

with st.form("extract_form", clear_on_submit=False, border=False):
    typed = st.text_area(
        "Hoặc nhập văn bản tiếng Việt",
        height=220,
        placeholder="Dán văn bản tại đây...",
    )
    submitted = st.form_submit_button("Trích xuất và tóm tắt", type="primary")

if submitted:
    source_name = "nhap_lieu"
    raw_text = typed or ""
    if st.session_state.get("_upload_text"):
        raw_text = st.session_state["_upload_text"]
        source_name = st.session_state.get("_upload_name") or source_name
    if not raw_text.strip():
        st.warning("Hãy nhập văn bản hoặc đợi file hiện «Đã đọc» rồi bấm lại.")
        st.stop()

    progress = st.status("Đang trích từ khóa / tóm tắt…", expanded=True)
    from src.pipeline import run

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
        progress.update(label="Lỗi khi xử lý", state="error")
        st.error(str(exc))
        st.stop()
    progress.update(label="Xong", state="complete")
    st.session_state["last_result"] = result
    st.session_state["last_source"] = source_name

result = st.session_state.get("last_result")
if result is not None:
    if st.session_state.get("last_source"):
        st.caption(f"Kết quả gần nhất: `{st.session_state['last_source']}`")
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
                column_config={
                    "điểm": st.column_config.NumberColumn(format="%.4f"),
                },
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
        elif result.warning:
            st.warning(result.warning)
        else:
            st.caption("Không có tóm tắt (đã tắt LLM).")
        if result.stored_id:
            st.caption(f"Đã lưu ChromaDB: `{result.stored_id}`")


def _history_summary_text(row: dict) -> str:
    summary = (row.get("summary") or "").strip()
    if summary:
        return summary
    note = (row.get("note") or "").strip()
    if note:
        return f"Chưa có tóm tắt ({note})"
    return "Chưa có tóm tắt"


st.divider()
st.subheader("Lịch sử gần đây")
st.caption("Bấm từng dòng để xem đủ từ khóa và tóm tắt. Chỉ hiện các lần đã bật «Lưu vào ChromaDB».")
try:
    history = list_recent()
except Exception as exc:
    st.caption(f"Chưa đọc được ChromaDB: {exc}")
    history = []

if history:
    for row in history:
        label = row.get("created_at_label") or row["created_at"]
        title = f"{label} · {row['source']} · {row['method']}"
        with st.expander(title):
            st.markdown("**Từ khóa**")
            st.write(row.get("keywords") or "—")
            st.markdown("**Tóm tắt**")
            st.write(_history_summary_text(row))
else:
    st.caption("Chưa có bản ghi.")
