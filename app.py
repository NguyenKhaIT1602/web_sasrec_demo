import os
import re
import pickle
import random
import requests
import gdown
from dataclasses import dataclass
from collections import Counter

import torch
import torch.nn as nn
import streamlit as st
import pandas as pd


# ============================================================
# PATH
# ============================================================

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(APP_DIR, "datasets")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# MULTI DATASET CONFIG
# ============================================================

@dataclass
class DatasetConfig:
    name: str
    display_name: str
    description: str
    files: dict


DATASETS = {
    "beauty": DatasetConfig(
        name="beauty",
        display_name="💄 Beauty Recommendation",
        description="Đề xuất sản phẩm làm đẹp từ Amazon Beauty.",
        files={
            "sasrec_qwen_best.pt": "https://drive.google.com/file/d/18BE3T9Rt1UaEysriDHosAYsrfOCmGNCo/view?usp=sharing",
            "fused_item_emb_128.pt": "https://drive.google.com/file/d/1BD_amnmNO57GCpsNIbErLg9_FLXA6J9p/view?usp=sharing",
            "item_web_meta.pkl": "https://drive.google.com/file/d/1COgYR2ahgo66Vp2I_zyLCyrRptCNRkK8/view?usp=sharing",
            "user_history.pkl": "https://drive.google.com/file/d/1PSHhecbBQFcTUWw7ojzyZkGp5dvQfBu9/view?usp=sharing",
        }
    ),
    "movies": DatasetConfig(
        name="movies",
        display_name="🎬 Movies Recommendation",
        description="Đề xuất phim/sản phẩm Movies từ Amazon Movies.",
        files={
            "sasrec_qwen_best.pt": "https://drive.google.com/file/d/1lEtd22DNLlo3a1widi9KljC2mgzSN0w1/view?usp=sharing",
            "fused_item_emb_128.pt": "https://drive.google.com/file/d/1lEtd22DNLlo3a1widi9KljC2mgzSN0w1/view?usp=sharing",
            "item_web_meta.pkl": "https://drive.google.com/file/d/17lsY3PuowRM5cXeDzk9vSU5mODxuF0ri/view?usp=sharing",
            "user_history.pkl": "https://drive.google.com/file/d/1COgYR2ahgo66Vp2I_zyLCyrRptCNRkK8/view?usp=sharing",
        }
    ),
}


def get_dataset_dir(dataset_name):
    path = os.path.join(DATASET_ROOT, dataset_name)
    os.makedirs(path, exist_ok=True)
    return path


def extract_drive_id(file_id_or_url):
    file_id_or_url = str(file_id_or_url).strip()

    if "drive.google.com/file/d/" in file_id_or_url:
        return file_id_or_url.split("/file/d/")[1].split("/")[0]

    if "id=" in file_id_or_url:
        return file_id_or_url.split("id=")[1].split("&")[0]

    return file_id_or_url


def download_from_google_drive(dataset_name, filename, file_id):
    dataset_dir = get_dataset_dir(dataset_name)
    output_path = os.path.join(dataset_dir, filename)

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_path

    real_file_id = extract_drive_id(file_id)

    url = f"https://drive.google.com/uc?id={real_file_id}"

    result = gdown.download(
        url=url,
        output=output_path,
        quiet=False,
        fuzzy=True
    )

    if result is None or not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError(
            f"Không tải được file {filename} của dataset {dataset_name}. "
            "Hãy kiểm tra Google Drive đã bật quyền Anyone with the link -> Viewer chưa."
        )

    return output_path

def ensure_required_files(dataset_config):
    paths = {}

    for filename, file_id in dataset_config.files.items():
        if not file_id or "GOOGLE_DRIVE_ID" in file_id:
            raise RuntimeError(
                f"Bạn chưa thay Google Drive ID cho file: {filename} "
                f"trong dataset: {dataset_config.name}"
            )

        paths[filename] = download_from_google_drive(
            dataset_config.name,
            filename,
            file_id
        )

    return paths


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Commerce AI",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
    background: #f8fafc;
}

[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e2e8f0;
}

.hero {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    padding: 40px;
    border-radius: 24px;
    color: white;
    margin-bottom: 30px;
    box-shadow: 0 20px 25px -5px rgba(79, 70, 229, 0.1);
}

.hero h1 {
    font-size: 3rem;
    font-weight: 800;
    color: white !important;
    margin-bottom: 10px;
}

.hero p {
    font-size: 1.1rem;
    opacity: 0.95;
}

.nav-box {
    background: white;
    padding: 8px 12px;
    border-radius: 16px;
    border: 1px solid #e2e8f0;
    margin-bottom: 25px;
}

.section-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #1e293b;
    margin: 32px 0 16px 0;
    display: flex;
    align-items: center;
}

.section-title::before {
    content: "";
    width: 6px;
    height: 24px;
    background: #4f46e5;
    border-radius: 3px;
    margin-right: 12px;
}

.product-card, .rec-card, .history-card {
    background: white;
    border-radius: 20px;
    padding: 18px;
    border: 1px solid #f1f5f9;
    box-shadow: 0 4px 15px rgba(0,0,0,0.03);
    margin-bottom: 24px;
    transition: all 0.3s ease;
    display: flex;
    flex-direction: column;
}

.product-card {
    height: 520px;
}

.rec-card {
    height: 620px;
}

.history-card {
    height: 320px;
}

.product-card:hover, .rec-card:hover, .history-card:hover {
    transform: translateY(-8px);
    border-color: #4f46e5;
    box-shadow: 0 25px 50px rgba(79,70,229,0.12);
}

.title {
    color: #1e293b;
    font-size: 1.02rem;
    font-weight: 700;
    line-height: 1.4;
    margin-top: 14px;
    height: 45px;
    overflow: hidden;
}

.meta-item {
    margin-top: 12px;
}

.meta-label {
    color: #94a3b8;
    font-size: 0.65rem;
    font-weight: 800;
    text-transform: uppercase;
}

.meta-value {
    color: #64748b;
    font-size: 0.75rem;
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.price {
    color: #4f46e5;
    font-size: 1.35rem;
    font-weight: 800;
    margin-top: auto;
    padding-top: 12px;
}

.badge {
    display: inline-block;
    background: #4f46e5;
    color: white;
    padding: 4px 12px;
    border-radius: 8px;
    font-size: 0.75rem;
    font-weight: 700;
    margin-bottom: 12px;
}

.score {
    display: inline-block;
    background: #eef2ff;
    color: #4f46e5;
    border: 1px solid #e0e7ff;
    padding: 6px 12px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 0.8rem;
    margin-top: 12px;
}

.reason {
    background: #f8fafc;
    color: #475569;
    padding: 16px;
    border-radius: 12px;
    font-size: 0.95rem;
    border: 1px solid #e2e8f0;
    line-height: 1.6;
}

.no-img {
    height: 180px;
    border-radius: 12px;
    background: #f1f5f9;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #94a3b8;
    font-weight: 600;
}

.metric-box {
    background: white;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.04);
    height: 100%;
}

.tag {
    display: inline-block;
    background: #f1f5f9;
    color: #475569;
    padding: 6px 12px;
    border-radius: 8px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 4px;
    border: 1px solid #e2e8f0;
}

.stButton > button,
.stDownloadButton > button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
    border: 1px solid #e2e8f0 !important;
    background: white !important;
    color: #1e293b !important;
    width: 100% !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color: #4f46e5 !important;
    color: #4f46e5 !important;
    background: #f5f3ff !important;
}

.dialog-price {
    color: #4f46e5;
    font-size: 2rem;
    font-weight: 800;
    margin: 10px 0;
}

.dialog-label {
    color: #64748b;
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
}

.dialog-value {
    color: #1e293b;
    font-size: 1rem;
    font-weight: 500;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL
# ============================================================

class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(
            self.conv2(
                self.relu(
                    self.dropout1(
                        self.conv1(inputs.transpose(-1, -2))
                    )
                )
            )
        ).transpose(-1, -2)

        outputs += inputs
        return outputs


class WebSASRecQwen(nn.Module):
    def __init__(
        self,
        item_matrix,
        maxlen=50,
        hidden_units=128,
        num_blocks=2,
        num_heads=4,
        dropout_rate=0.2
    ):
        super().__init__()

        self.register_buffer("item_matrix", item_matrix.float())
        self.item_num = item_matrix.shape[0] - 1
        self.hidden_units = hidden_units
        self.maxlen = maxlen

        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

        for _ in range(num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.attention_layers.append(
                nn.MultiheadAttention(hidden_units, num_heads, dropout_rate)
            )
            self.forward_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.forward_layers.append(
                PointWiseFeedForward(hidden_units, dropout_rate)
            )

    def get_fused_item_emb(self, item_ids):
        return self.item_matrix[item_ids]

    def log2feats(self, log_seqs):
        seqs = self.get_fused_item_emb(log_seqs)
        seqs *= self.hidden_units ** 0.5

        positions = torch.arange(
            log_seqs.shape[1],
            device=log_seqs.device
        ).unsqueeze(0).expand(log_seqs.shape[0], -1)

        seqs += self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)

        timeline_mask = log_seqs == 0
        seqs *= ~timeline_mask.unsqueeze(-1)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(
            torch.ones((tl, tl), dtype=torch.bool, device=log_seqs.device)
        )

        for i in range(len(self.attention_layers)):
            seqs = seqs.transpose(0, 1)
            q = self.attention_layernorms[i](seqs)

            mha_outputs, _ = self.attention_layers(
                q, seqs, seqs, attn_mask=attention_mask
            ) if False else self.attention_layers[i](
                q, seqs, seqs, attn_mask=attention_mask
            )

            seqs = q + mha_outputs
            seqs = seqs.transpose(0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        return self.last_layernorm(seqs)

    @torch.no_grad()
    def get_user_vector(self, seq):
        seq = [int(x) for x in list(seq)[-self.maxlen:]]

        if len(seq) < self.maxlen:
            seq = [0] * (self.maxlen - len(seq)) + seq

        seq_tensor = torch.LongTensor(seq).unsqueeze(0).to(DEVICE)
        feats = self.log2feats(seq_tensor)

        return torch.nn.functional.normalize(feats[:, -1, :], dim=-1)


# ============================================================
# UTILS
# ============================================================

def render_metric_html(label, value):
    return f"""
    <div class="metric-box">
        <div style="color:#64748b;font-size:0.85rem;font-weight:600;margin-bottom:4px;">{label}</div>
        <div style="color:#1e293b;font-size:1.5rem;font-weight:700;">{value}</div>
    </div>
    """


def safe_str(x):
    if x is None:
        return ""
    if isinstance(x, list):
        return " ".join([str(i) for i in x])
    return str(x)


def clean_text(text):
    text = safe_str(text)
    text = re.sub(r'<style.*?>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'\{[^{}]*\}', '', text)
    text = re.sub(r'\.[a-zA-Z0-9_-]+\s*\{.*?\}', '', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]*>', '', text)
    text = text.replace('"', "'").replace('$', 'S')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def short_text(text, n=80):
    text = clean_text(text)
    return text if len(text) <= n else text[:n].rstrip() + "..."


def get_meta(item_meta, iid):
    return item_meta.get(int(iid), {}) or {}


def get_title(meta):
    title = clean_text(meta.get("title", ""))
    return title if title else "No Title"


def get_brand(meta):
    return clean_text(meta.get("brand", ""))


def get_category(meta):
    cat = meta.get("category", "") or meta.get("main_cat", "")
    if isinstance(cat, list):
        cat = " > ".join([safe_str(c) for c in cat[:4]])
    return clean_text(cat)


def get_main_cat(meta):
    return clean_text(meta.get("main_cat", ""))


def get_asin(meta):
    return clean_text(meta.get("asin", ""))


def get_feature(meta):
    feat = meta.get("feature", "")
    if isinstance(feat, list):
        feat = " | ".join([safe_str(f) for f in feat[:5]])
    return clean_text(feat)


def get_description(meta):
    desc = meta.get("description", "")
    if isinstance(desc, list):
        desc = " ".join([safe_str(d) for d in desc[:4]])
    return clean_text(desc)


def get_image(meta):
    for key in ["image", "imageURLHighRes", "imageURL", "image_url"]:
        img = meta.get(key, "")
        if isinstance(img, list) and len(img) > 0:
            img = img[0]
        if isinstance(img, str) and img.startswith("http"):
            return img
    return ""


def get_price(meta):
    price = clean_text(meta.get("price", ""))
    if "{" in price or "}" in price:
        return "Liên hệ"
    return price if price else "Liên hệ"


def render_img_html(image, height=180):
    if image and image.startswith("http"):
        return f"""
        <img src="{image}" 
             style="height:{height}px;width:100%;object-fit:contain;border-radius:12px;margin-bottom:10px;">
        """
    return f'<div class="no-img" style="height:{height}px;">No Image</div>'


# ============================================================
# LLM EXPLANATION
# ============================================================

@st.cache_data(show_spinner=False)
def get_llm_explanation(
    api_key,
    api_type,
    product_id,
    product_title,
    product_brand,
    product_cat,
    product_desc,
    history_titles
):
    if not api_key:
        return None

    hist_str = ", ".join(history_titles[-5:])

    prompt = f"""
Bạn là một chuyên gia tư vấn mua sắm AI thông minh.
Hãy giải thích lý do tại sao sản phẩm sau đây được đề xuất cho người dùng.

Lịch sử người dùng đã xem/mua:
{hist_str}

Sản phẩm đề xuất:
- Tiêu đề: {product_title}
- Thương hiệu: {product_brand}
- Danh mục: {product_cat}
- Mô tả: {product_desc[:300]}

Yêu cầu:
1. Trả lời ngắn gọn khoảng 2-3 câu.
2. Tập trung vào sự tương quan giữa lịch sử người dùng và sản phẩm mới.
3. Ngôn ngữ tiếng Việt, tự nhiên, thân thiện.
4. Không bắt đầu bằng cụm "Dựa trên".
"""

    try:
        if api_type == "Groq":
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7
            }
        else:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7
            }

        response = requests.post(url, headers=headers, json=data, timeout=15)
        res_json = response.json()

        if response.status_code == 429:
            return "QUOTA_EXCEEDED"

        if response.status_code != 200:
            error_msg = res_json.get("error", {}).get("message", "Lỗi API")
            return f"API Error ({response.status_code}): {error_msg}"

        return res_json["choices"][0]["message"]["content"].strip()

    except Exception as e:
        return f"Lỗi kết nối LLM: {str(e)}"


# ============================================================
# DIALOG
# ============================================================

@st.dialog("📦 Thông tin chi tiết", width="large")
def show_product_details(p_or_meta, is_meta=False):
    if is_meta:
        title = get_title(p_or_meta)
        brand = get_brand(p_or_meta)
        category = get_category(p_or_meta)
        main_cat = get_main_cat(p_or_meta)
        asin = get_asin(p_or_meta)
        desc = get_description(p_or_meta)
        feature = get_feature(p_or_meta)
        price = get_price(p_or_meta)
        image = get_image(p_or_meta)
        item_id = "N/A"
    else:
        title = p_or_meta["title"]
        brand = p_or_meta["brand"]
        category = p_or_meta["category"]
        main_cat = p_or_meta["main_cat"]
        asin = p_or_meta["asin"]
        desc = p_or_meta["description"]
        feature = p_or_meta["feature"]
        price = p_or_meta["price"]
        image = p_or_meta["image"]
        item_id = p_or_meta["item_id"]

    col1, col2 = st.columns([0.8, 1.2], gap="large")

    with col1:
        if image and image.startswith("http"):
            st.image(image)
        else:
            st.markdown(
                '<div class="no-img" style="height:250px;">No Image</div>',
                unsafe_allow_html=True
            )

    with col2:
        st.markdown(f"### {title}")
        st.markdown(f'<div class="dialog-price">{price}</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:
            st.markdown(
                f'<div class="dialog-label">Thương hiệu</div>'
                f'<div class="dialog-value">{brand if brand else "N/A"}</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="dialog-label">ASIN</div>'
                f'<div class="dialog-value"><code>{asin}</code></div>',
                unsafe_allow_html=True
            )

        with c2:
            st.markdown(
                f'<div class="dialog-label">Danh mục chính</div>'
                f'<div class="dialog-value">{main_cat if main_cat else "N/A"}</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="dialog-label">Item ID</div>'
                f'<div class="dialog-value"><code>{item_id}</code></div>',
                unsafe_allow_html=True
            )

        st.markdown(
            f'<div class="dialog-label">Danh mục chi tiết</div>'
            f'<div class="dialog-value">{category if category else "N/A"}</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    tab1, tab2 = st.tabs(["📄 Mô tả sản phẩm", "✨ Tính năng nổi bật"])

    with tab1:
        st.write(desc if desc else "Không có mô tả chi tiết.")

    with tab2:
        if feature:
            for f in feature.split(" | "):
                st.markdown(f"- {f}")
        else:
            st.write("Không có thông tin tính năng.")


@st.dialog("✨ AI Giải thích đề xuất", width="large")
def show_ai_dialog(api_key, api_type, p, history_titles):
    if not api_key:
        st.warning("Vui lòng nhập API Key ở thanh bên.")
        return

    with st.spinner("AI đang phân tích sở thích người dùng..."):
        explanation = get_llm_explanation(
            api_key,
            api_type,
            p["item_id"],
            p["title"],
            p["brand"],
            p["category"],
            p["description"],
            tuple(history_titles)
        )

    if explanation == "QUOTA_EXCEEDED":
        st.error("Tài khoản AI đã hết hạn mức.")
    elif explanation:
        st.markdown(
            f"""
            <div style="background:#f0fdf4;padding:20px;border-radius:16px;border:1px solid #bbf7d0;">
                <div style="color:#166534;font-size:1.1rem;line-height:1.6;">{explanation}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.error("Không thể nhận phản hồi từ AI.")


@st.dialog("💡 Lý do từ hệ thống", width="medium")
def show_system_dialog(reason):
    st.markdown(
        f"""
        <div class="reason" style="font-size:1.1rem;padding:20px;">
            {reason}
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# CARD UI
# ============================================================

def home_product_card(iid, meta, idx=0):
    title = get_title(meta)
    brand = get_brand(meta)
    category = get_category(meta)
    price = get_price(meta)
    image = get_image(meta)

    img_html = render_img_html(image, height=180)

    brand_html = (
        f'<div class="meta-item"><div class="meta-label">Thương hiệu</div>'
        f'<div class="meta-value">🏷️ {short_text(brand, 40)}</div></div>'
        if brand else ""
    )

    cat_html = (
        f'<div class="meta-item"><div class="meta-label">Danh mục</div>'
        f'<div class="meta-value">📦 {short_text(category, 45)}</div></div>'
        if category else ""
    )

    card_html = f"""
    <div class="product-card">
        {img_html}
        <div class="title">{short_text(title, 80)}</div>
        {brand_html}
        {cat_html}
        <div class="price">{price}</div>
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)

    if st.button("🔍 Xem chi tiết", key=f"btn_h_info_{iid}_{idx}", use_container_width=True):
        show_product_details(meta, is_meta=True)


def rec_product_card(p, api_key=None, api_type="Groq", history_titles=None, user_id=""):
    img_html = render_img_html(p["image"], height=200)

    brand_html = (
        f'<div class="meta-item"><div class="meta-label">Thương hiệu</div>'
        f'<div class="meta-value">🏷️ {short_text(p["brand"], 35)}</div></div>'
        if p["brand"] else ""
    )

    cat_html = (
        f'<div class="meta-item"><div class="meta-label">Danh mục</div>'
        f'<div class="meta-value">📦 {short_text(p["category"], 50)}</div></div>'
        if p["category"] else ""
    )

    card_html = f"""
    <div class="rec-card">
        <span class="badge">Rank #{p["rank"]}</span>
        {img_html}
        <div class="title">{short_text(p["title"], 90)}</div>
        {brand_html}
        {cat_html}
        <div class="price">{p["price"]}</div>
        <div class="score">Độ phù hợp: {p["score"]:.4f}</div>
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    rank = p.get("rank", 0)

    with c1:
        if st.button("🔍", key=f"btn_info_{user_id}_{p['item_id']}_{rank}", help="Thông tin sản phẩm"):
            show_product_details(p, is_meta=False)

    with c2:
        if st.button("✨", key=f"btn_ai_{user_id}_{p['item_id']}_{rank}", help="AI giải thích"):
            show_ai_dialog(api_key, api_type, p, history_titles or [])

    with c3:
        if st.button("💡", key=f"btn_sys_{user_id}_{p['item_id']}_{rank}", help="Lý do từ hệ thống"):
            show_system_dialog(p["reason"])


def history_card(iid, item_meta, idx=0):
    meta = get_meta(item_meta, iid)
    image = get_image(meta)
    title = get_title(meta)
    brand = get_brand(meta)

    img_html = render_img_html(image, height=130)

    brand_html = (
        f'<div class="meta-item"><div class="meta-label">Thương hiệu</div>'
        f'<div class="meta-value">🏷️ {short_text(brand, 35)}</div></div>'
        if brand else ""
    )

    card_html = f"""
    <div class="history-card">
        {img_html}
        <div class="title">{short_text(title, 65)}</div>
        {brand_html}
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)

    if st.button("🔍 Xem chi tiết", key=f"btn_hist_info_{iid}_{idx}", use_container_width=True):
        show_product_details(meta, is_meta=True)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_resource(show_spinner=True)
def load_all(dataset_name):
    dataset_config = DATASETS[dataset_name]
    paths = ensure_required_files(dataset_config)

    model_path = paths["sasrec_qwen_best.pt"]
    item_emb_path = paths["fused_item_emb_128.pt"]
    item_meta_path = paths["item_web_meta.pkl"]
    user_history_path = paths["user_history.pkl"]

    missing = []

    for p in [model_path, item_emb_path, item_meta_path, user_history_path]:
        if not os.path.exists(p):
            missing.append(p)

    if missing:
        return None, None, None, None, None, missing

    item_matrix = torch.load(item_emb_path, map_location=DEVICE)

    if isinstance(item_matrix, dict):
        for k in ["fused_item_emb", "item_emb", "emb", "item_matrix"]:
            if k in item_matrix:
                item_matrix = item_matrix[k]
                break

    item_matrix = item_matrix.float().to(DEVICE)
    item_matrix = torch.nn.functional.normalize(item_matrix, dim=-1)

    with open(item_meta_path, "rb") as f:
        item_meta = pickle.load(f)

    num_items = item_matrix.shape[0]

    titles = [""] * num_items
    brands = [""] * num_items
    categories = [""] * num_items
    search_blobs = [""] * num_items

    for iid, m in item_meta.items():
        iid = int(iid)

        if 0 <= iid < num_items:
            t = get_title(m)
            b = get_brand(m)
            c = get_category(m)

            titles[iid] = t
            brands[iid] = b
            categories[iid] = c
            search_blobs[iid] = f"{t} {b} {c}".lower()

    filter_data = {
        "titles": titles,
        "brands": brands,
        "categories": categories,
        "search_blobs": search_blobs
    }

    with open(user_history_path, "rb") as f:
        user_history = pickle.load(f)

    user_history = {
        int(k): [int(x) for x in v]
        for k, v in user_history.items()
    }

    ckpt = torch.load(model_path, map_location=DEVICE)

    maxlen = int(ckpt.get("maxlen", 50))
    hidden_units = int(ckpt.get("hidden_units", item_matrix.shape[1]))
    num_blocks = int(ckpt.get("num_blocks", 2))
    num_heads = int(ckpt.get("num_heads", 4))
    dropout_rate = float(ckpt.get("dropout_rate", 0.2))

    model = WebSASRecQwen(
        item_matrix=item_matrix,
        maxlen=maxlen,
        hidden_units=hidden_units,
        num_blocks=num_blocks,
        num_heads=num_heads,
        dropout_rate=dropout_rate
    ).to(DEVICE)

    state = ckpt.get("model_state_dict", ckpt)
    model_state = model.state_dict()

    filtered_state = {}

    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered_state[k] = v

    model.load_state_dict(filtered_state, strict=False)
    model.eval()

    return model, item_matrix, item_meta, user_history, filter_data, []


@st.cache_data(show_spinner=False)
def build_filter_options(filter_data):
    categories = set(filter_data["categories"])
    brands = set(filter_data["brands"])

    categories.discard("")
    brands.discard("")

    return (
        ["Tất cả"] + sorted(list(categories))[:1500],
        ["Tất cả"] + sorted(list(brands))[:1500]
    )


@st.cache_data(show_spinner=False)
def get_filtered_items(
    item_meta_keys,
    f_titles,
    f_brands,
    f_categories,
    f_search,
    home_category,
    home_brand,
    home_search_clean,
    num_display
):
    valid_items = []

    for iid in item_meta_keys:
        if 0 <= iid < len(f_titles):
            if home_category != "Tất cả" and f_categories[iid] != home_category:
                continue

            if home_brand != "Tất cả" and f_brands[iid] != home_brand:
                continue

            if home_search_clean and home_search_clean not in f_search[iid]:
                continue

            valid_items.append(iid)

            if len(valid_items) >= num_display:
                break

    return valid_items


# ============================================================
# RECOMMEND
# ============================================================

def build_user_profile(seq, item_meta):
    titles = []
    cats = []
    brands = []

    for iid in seq[-20:]:
        meta = get_meta(item_meta, iid)

        title = get_title(meta)
        cat = get_category(meta)
        brand = get_brand(meta)

        if title != "No Title":
            titles.append(title)

        if cat:
            cats.append(cat)

        if brand:
            brands.append(brand)

    return (
        titles,
        Counter(cats).most_common(5),
        Counter(brands).most_common(5)
    )


def make_reason(product, history_titles):
    hist = ", ".join(history_titles[-3:]) if history_titles else "các sản phẩm đã mua/xem trước đó"

    reasons = [
        f"người dùng từng tương tác với {hist}",
        "SASRec học xu hướng tiếp theo trong chuỗi hành vi",
        "Qwen embedding giúp hiểu nội dung tiêu đề, mô tả và danh mục sản phẩm"
    ]

    if product["category"]:
        reasons.append(f"sản phẩm thuộc nhóm {product['category']}")

    if product["brand"]:
        reasons.append(f"thương hiệu {product['brand']} phù hợp với hồ sơ sở thích")

    reasons.append(f"điểm phù hợp đạt {product['score']:.4f}")

    return "Hệ thống đề xuất sản phẩm này vì " + "; ".join(reasons) + "."


@torch.no_grad()
def recommend(
    model,
    item_matrix,
    item_meta,
    filter_data,
    user_history,
    user_id,
    top_k,
    category_filter,
    brand_filter,
    search_text,
    hide_seen=True,
    candidate_pool=5000
):
    seq = user_history.get(int(user_id), [])

    if not seq:
        return [], [], [], []

    user_vec = model.get_user_vector(seq)
    scores = torch.matmul(user_vec, item_matrix.T).squeeze(0)

    if scores.numel() > 0:
        scores[0] = -1e9

    if hide_seen:
        seen = [int(x) for x in set(seq) if 0 <= int(x) < scores.numel()]
        if seen:
            scores[torch.LongTensor(seen).to(DEVICE)] = -1e9

    pool = min(int(candidate_pool), scores.numel() - 1)
    top_scores, top_ids = torch.topk(scores, k=pool)

    history_titles, top_cats, top_brands = build_user_profile(seq, item_meta)
    search_text = safe_str(search_text).strip().lower()

    results = []

    f_titles = filter_data["titles"]
    f_categories = filter_data["categories"]
    f_brands = filter_data["brands"]
    f_search = filter_data["search_blobs"]

    top_ids_np = top_ids.detach().cpu().numpy()
    top_scores_np = top_scores.detach().cpu().numpy()

    for iid, sc in zip(top_ids_np, top_scores_np):
        iid = int(iid)
        sc = float(sc)

        if category_filter != "Tất cả" and f_categories[iid] != category_filter:
            continue

        if brand_filter != "Tất cả" and f_brands[iid] != brand_filter:
            continue

        if search_text and search_text not in f_search[iid]:
            continue

        meta = get_meta(item_meta, iid)

        p = {
            "item_id": iid,
            "asin": get_asin(meta),
            "score": sc,
            "title": f_titles[iid],
            "brand": f_brands[iid],
            "main_cat": get_main_cat(meta),
            "category": f_categories[iid],
            "description": get_description(meta),
            "feature": get_feature(meta),
            "price": get_price(meta),
            "image": get_image(meta)
        }

        p["reason"] = make_reason(p, history_titles)
        results.append(p)

        if len(results) >= top_k:
            break

    for i, p in enumerate(results, 1):
        p["rank"] = i

    return results, history_titles, top_cats, top_brands


# ============================================================
# MAIN UI
# ============================================================

st.markdown("""
<div class="hero">
    <h1>🛒 Smart Commerce AI</h1>
    <p>Hệ thống đề xuất sản phẩm sử dụng <b>SASRec</b> kết hợp <b>Qwen Embedding</b>.</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR DATASET SELECTOR
# ============================================================

st.sidebar.markdown("## 🗂️ Chọn bộ dữ liệu")

dataset_options = {
    key: cfg.display_name
    for key, cfg in DATASETS.items()
}

selected_dataset = st.sidebar.selectbox(
    "Loại đề xuất",
    options=list(dataset_options.keys()),
    format_func=lambda x: dataset_options[x],
    index=0
)

selected_config = DATASETS[selected_dataset]

st.sidebar.info(selected_config.description)

with st.spinner(f"🚀 Đang tải dữ liệu {selected_config.display_name}..."):
    try:
        model, item_matrix, item_meta, user_history, filter_data, missing = load_all(selected_dataset)
    except Exception as e:
        st.error(f"❌ Lỗi khi tải dataset `{selected_dataset}`")
        st.exception(e)
        st.stop()

if missing:
    st.error("⚠️ Thiếu các file cần thiết.")
    st.write(missing)
    st.stop()

st.success(f"✅ Đã load xong dữ liệu: {selected_config.display_name}")

categories, brands = build_filter_options(filter_data)
all_users = sorted(list(user_history.keys()))

if not all_users:
    st.error("Dataset này không có user_history.")
    st.stop()


# ============================================================
# NAVIGATION
# ============================================================

st.markdown('<div class="nav-box">', unsafe_allow_html=True)
page = st.radio(
    "Navigation",
    ["🏠 Khám phá", "🎯 Gợi ý"],
    horizontal=True,
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# PAGE: HOME
# ============================================================

if page == "🏠 Khám phá":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Bộ lọc tìm kiếm")

    home_search = st.sidebar.text_input("Tìm tên sản phẩm...", "")
    home_category = st.sidebar.selectbox("Danh mục", categories, index=0)
    home_brand = st.sidebar.selectbox("Thương hiệu", brands, index=0)

    total_items = item_matrix.shape[0] - 1

    num_display = st.sidebar.slider(
        "Số lượng hiển thị",
        min_value=20,
        max_value=max(20, total_items),
        value=min(100, max(20, total_items)),
        step=20
    )

    if num_display > 500:
        st.sidebar.warning("Số lượng hiển thị lớn có thể làm chậm trình duyệt.")

    st.markdown('<div class="section-title">🔥 Sản phẩm nổi bật</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)

    m1.markdown(render_metric_html("🗂️ Dataset", selected_config.display_name), unsafe_allow_html=True)
    m2.markdown(render_metric_html("👥 Tổng người dùng", f"{len(user_history):,}"), unsafe_allow_html=True)
    m3.markdown(render_metric_html("📦 Tổng sản phẩm", f"{item_matrix.shape[0] - 1:,}"), unsafe_allow_html=True)
    m4.markdown(render_metric_html("⚡ Thiết bị", DEVICE.upper()), unsafe_allow_html=True)

    home_search_clean = home_search.strip().lower()

    valid_items = get_filtered_items(
        tuple(sorted([int(x) for x in item_meta.keys()])),
        tuple(filter_data["titles"]),
        tuple(filter_data["brands"]),
        tuple(filter_data["categories"]),
        tuple(filter_data["search_blobs"]),
        home_category,
        home_brand,
        home_search_clean,
        num_display
    )

    if not valid_items:
        st.warning("Không tìm thấy sản phẩm.")
    else:
        cols = st.columns(5)

        for idx, iid in enumerate(valid_items):
            with cols[idx % 5]:
                home_product_card(iid, get_meta(item_meta, iid), idx)


# ============================================================
# PAGE: RECOMMENDATION
# ============================================================

else:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Cấu hình đề xuất")

    session_user_key = f"selected_user_{selected_dataset}"

    if session_user_key not in st.session_state:
        st.session_state[session_user_key] = all_users[0]

    selected_user = st.sidebar.selectbox(
        "👤 Chọn User ID",
        all_users,
        index=all_users.index(st.session_state[session_user_key])
        if st.session_state[session_user_key] in all_users else 0
    )

    st.session_state[session_user_key] = selected_user

    if st.sidebar.button("🎲 Chọn user ngẫu nhiên"):
        st.session_state[session_user_key] = random.choice(all_users)
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ✨ AI Reasoning")

    use_llm = st.sidebar.toggle("Kích hoạt LLM", value=True)
    api_type = st.sidebar.selectbox("AI Provider", ["Groq", "OpenAI"], index=0)
    api_key = st.sidebar.text_input(
        "API Key",
        value="",
        type="password",
        help="Nhập Groq/OpenAI API key nếu muốn dùng AI Reasoning."
    )

    top_k = st.sidebar.select_slider(
        "Top-K",
        options=[1, 5, 10, 20],
        value=10
    )

    search_text = st.sidebar.text_input("🔍 Tìm kiếm...", "")
    category_filter = st.sidebar.selectbox("📁 Lọc danh mục", categories, index=0)
    brand_filter = st.sidebar.selectbox("🏷️ Lọc thương hiệu", brands, index=0)
    hide_seen = st.sidebar.toggle("Ẩn sản phẩm đã xem", value=True)

    total_items = item_matrix.shape[0] - 1

    candidate_pool = st.sidebar.slider(
        "Kho ứng viên",
        min_value=500,
        max_value=max(500, total_items),
        value=min(5000, max(500, total_items)),
        step=500
    )

    seq = user_history.get(int(selected_user), [])
    history_titles, top_cats, top_brands = build_user_profile(seq, item_meta)

    st.markdown('<div class="section-title">👤 Hồ sơ người dùng</div>', unsafe_allow_html=True)

    u1, u2, u3, u4 = st.columns(4)

    u1.markdown(render_metric_html("🗂️ Dataset", selected_config.display_name), unsafe_allow_html=True)
    u2.markdown(render_metric_html("👤 User ID", selected_user), unsafe_allow_html=True)
    u3.markdown(render_metric_html("🛒 Tương tác", f"{len(seq)} sản phẩm"), unsafe_allow_html=True)
    u4.markdown(render_metric_html("🎯 Top-K", top_k), unsafe_allow_html=True)

    p1, p2 = st.columns(2)

    with p1:
        tags = "".join([
            f'<span class="tag">{cat} ({cnt})</span>'
            for cat, cnt in top_cats
        ])

        st.markdown(
            f'<div class="metric-box"><b>📂 Danh mục quan tâm</b><br>{tags}</div>',
            unsafe_allow_html=True
        )

    with p2:
        tags = "".join([
            f'<span class="tag">{brand} ({cnt})</span>'
            for brand, cnt in top_brands
        ])

        st.markdown(
            f'<div class="metric-box"><b>🏷️ Thương hiệu yêu thích</b><br>{tags}</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-title">🧾 Lịch sử tương tác</div>', unsafe_allow_html=True)

    recent = list(seq)[-20:][::-1]

    if recent:
        cols = st.columns(5)

        for idx, iid in enumerate(recent):
            with cols[idx % 5]:
                history_card(iid, item_meta, idx)
    else:
        st.info("User này chưa có lịch sử tương tác.")

    st.markdown('<div class="section-title">🎯 Gợi ý cho bạn</div>', unsafe_allow_html=True)

    with st.spinner("Đang tạo đề xuất..."):
        results, _, _, _ = recommend(
            model=model,
            item_matrix=item_matrix,
            item_meta=item_meta,
            filter_data=filter_data,
            user_history=user_history,
            user_id=selected_user,
            top_k=top_k,
            category_filter=category_filter,
            brand_filter=brand_filter,
            search_text=search_text,
            hide_seen=hide_seen,
            candidate_pool=candidate_pool
        )

    if not results:
        st.warning("Không tìm thấy sản phẩm phù hợp.")
    else:
        cols = st.columns(5)

        for idx, p in enumerate(results):
            with cols[idx % 5]:
                rec_product_card(
                    p,
                    api_key if use_llm else None,
                    api_type,
                    history_titles,
                    f"{selected_dataset}_{selected_user}"
                )

        st.markdown('<div class="section-title">📋 Bảng chi tiết</div>', unsafe_allow_html=True)

        export_data = []

        for p in results:
            export_data.append({
                "Dataset": selected_dataset,
                "Rank": p["rank"],
                "Item ID": p["item_id"],
                "ASIN": p["asin"],
                "Tiêu đề": p["title"],
                "Thương hiệu": p["brand"],
                "Main Cat": p["main_cat"],
                "Category": p["category"],
                "Mô tả": p["description"],
                "Tính năng": p["feature"],
                "Giá": p["price"],
                "Score": round(p["score"], 4),
                "Reason": p["reason"],
                "_image": p["image"],
            })

        df = pd.DataFrame(export_data)

        st.download_button(
            label="📥 Xuất dữ liệu đề xuất CSV",
            data=df.drop(columns=["_image"]).to_csv(index=False, encoding="utf-8-sig"),
            file_name=f"rec_{selected_dataset}_{selected_user}.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "_image": st.column_config.ImageColumn("Preview"),
                "Score": st.column_config.ProgressColumn(
                    "Score",
                    format="%.4f",
                    min_value=0,
                    max_value=float(df["Score"].max()) if not df.empty else 1.0
                ),
                "Reason": st.column_config.TextColumn("Reason", width="large"),
            }
        )
