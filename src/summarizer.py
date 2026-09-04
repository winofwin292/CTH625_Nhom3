"""Sinh tóm tắt bằng LLM, lấy từ khóa làm neo ngữ nghĩa trong prompt."""

from __future__ import annotations

from src.config import hf_provider, hf_token, llm_model

SYSTEM_PROMPT = (
    "Bạn là trợ lý tóm tắt văn bản tiếng Việt. "
    "Chỉ dùng thông tin có trong văn bản. Không bịa thêm sự kiện."
)


def build_user_prompt(text: str, keywords: list[str], n_sentences: int) -> str:
    joined = ", ".join(keywords) if keywords else "(không có từ khóa)"
    return (
        f"Từ khóa cốt lõi (neo ngữ nghĩa): {joined}\n\n"
        f"Văn bản:\n{text}\n\n"
        "Yêu cầu:\n"
        f"- Viết khoảng {n_sentences} câu tóm tắt bằng tiếng Việt, mạch lạc.\n"
        "- Ưu tiên các ý gắn với danh sách từ khóa, vẫn phải trung thực với văn bản.\n"
        "- Không liệt kê lại từ khóa. Không dùng gạch đầu dòng."
    )


def _chat_huggingface(messages: list[dict], max_tokens: int) -> str:
    from huggingface_hub import InferenceClient

    token = hf_token()
    if not token:
        raise RuntimeError(
            "Chưa có HF_TOKEN. Thêm token vào .env hoặc Streamlit Secrets để gọi LLM."
        )
    client_kwargs = {"token": token}
    provider = hf_provider()
    if provider:
        client_kwargs["provider"] = provider
    client = InferenceClient(**client_kwargs)
    try:
        response = client.chat_completion(
            messages=messages,
            model=llm_model(),
            max_tokens=max_tokens,
            temperature=0.3,
        )
    except Exception as exc:
        text = str(exc)
        if "model_not_supported" in text or "not supported by any provider" in text:
            raise RuntimeError(
                "Hugging Face không gọi được Qwen2.5-7B-Instruct vì chưa bật provider. "
                "Vào https://huggingface.co/settings/inference-providers, bật Featherless AI "
                "(Routed by HF, không cần tài khoản Featherless). "
                "Trong .env đặt HF_PROVIDER=featherless-ai rồi chạy lại Streamlit."
            ) from exc
        raise
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Hugging Face API trả về nội dung rỗng.")
    return content.strip()


def summarize(text: str, keywords: list[str], n_sentences: int = 5) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(text, keywords, n_sentences)},
    ]
    max_tokens = min(80 * max(n_sentences, 1), 800)
    return _chat_huggingface(messages, max_tokens)
