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
# DATASET CONFIG
# ============================================================

@dataclass
class DatasetConfig:
    name: str
    display_name: str
    description: str
    files: dict


DATASETS = {
    "Beauty": DatasetConfig(
        name="Beauty",
        display_name="💄 Beauty Recommendation",
        description="Đề xuất sản phẩm Beauty.",
        files={
            "sasrec_qwen_best.pt":
                "https://drive.google.com/file/d/18BE3T9Rt1UaEysriDHosAYsrfOCmGNCo/view?usp=sharing",

            "fused_item_emb_128.pt":
                "https://drive.google.com/file/d/1BD_amnmNO57GCpsNIbErLg9_FLXA6J9p/view?usp=sharing",

            "item_web_meta.pkl":
                "https://drive.google.com/file/d/1COgYR2ahgo66Vp2I_zyLCyrRptCNRkK8/view?usp=sharing",

            "user_history.pkl":
                "https://drive.google.com/file/d/1PSHhecbBQFcTUWw7ojzyZkGp5dvQfBu9/view?usp=sharing",
        }
    ),

    "Movies": DatasetConfig(
        name="Movies",
        display_name="🎬 Movies Recommendation",
        description="Đề xuất Movies.",

        files={
            "sasrec_qwen_best.pt":
                "https://drive.google.com/file/d/1lEtd22DNLlo3a1widi9KljC2mgzSN0w1/view?usp=sharing",

            "fused_item_emb_128.pt":
                "https://drive.google.com/file/d/1lEtd22DNLlo3a1widi9KljC2mgzSN0w1/view?usp=sharing",

            "item_web_meta.pkl":
                "https://drive.google.com/file/d/17lsY3PuowRM5cXeDzk9vSU5mODxuF0ri/view?usp=sharing",

            "user_history.pkl":
                "https://drive.google.com/file/d/1COgYR2ahgo66Vp2I_zyLCyrRptCNRkK8/view?usp=sharing",
        }
    ),
}


# ============================================================
# GOOGLE DRIVE DOWNLOAD
# ============================================================

def extract_drive_id(file_id_or_url):
    file_id_or_url = str(file_id_or_url).strip()

    if "drive.google.com/file/d/" in file_id_or_url:
        return file_id_or_url.split("/file/d/")[1].split("/")[0]

    if "id=" in file_id_or_url:
        return file_id_or_url.split("id=")[1].split("&")[0]

    return file_id_or_url


def get_dataset_dir(dataset_name):
    path = os.path.join(DATASET_ROOT, dataset_name)
    os.makedirs(path, exist_ok=True)
    return path


def download_from_google_drive(dataset_name, filename, file_id_or_url):
    dataset_dir = get_dataset_dir(dataset_name)

    output_path = os.path.join(dataset_dir, filename)

    # đã tồn tại
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_path

    real_file_id = extract_drive_id(file_id_or_url)

    url = f"https://drive.google.com/uc?id={real_file_id}"

    try:
        result = gdown.download(
            url=url,
            output=output_path,
            quiet=False
        )

    except Exception as e:
        raise RuntimeError(
            f"""
Không thể tải file:
{filename}

Dataset:
{dataset_name}

Google Drive ID:
{real_file_id}

Lỗi:
{str(e)}

Kiểm tra:
1. File có public không
2. Anyone with the link -> Viewer
3. Có bị quota Google Drive không
"""
        )

    if (
        result is None
        or not os.path.exists(output_path)
        or os.path.getsize(output_path) == 0
    ):
        raise RuntimeError(
            f"""
Download thất bại:

File: {filename}
Dataset: {dataset_name}
"""
        )

    return output_path


def ensure_required_files(dataset_config):
    paths = {}

    for filename, file_id in dataset_config.files.items():

        if not file_id:
            raise RuntimeError(
                f"Thiếu Google Drive link cho file: {filename}"
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
    layout="wide"
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    padding: 40px;
    border-radius: 24px;
    color: white;
    margin-bottom: 30px;
}

.product-card {
    background: white;
    border-radius: 18px;
    padding: 18px;
    border: 1px solid #e2e8f0;
    margin-bottom: 20px;
    height: 500px;
}

.rec-card {
    background: white;
    border-radius: 18px;
    padding: 18px;
    border: 1px solid #e2e8f0;
    margin-bottom: 20px;
    height: 580px;
}

.title {
    font-size: 1rem;
    font-weight: 700;
    margin-top: 12px;
    height: 44px;
    overflow: hidden;
}

.price {
    color: #4f46e5;
    font-size: 1.3rem;
    font-weight: 800;
    margin-top: auto;
}

.badge {
    background: #4f46e5;
    color: white;
    padding: 4px 10px;
    border-radius: 8px;
    font-size: 0.75rem;
}

.score {
    margin-top: 10px;
    background: #eef2ff;
    padding: 6px 12px;
    border-radius: 8px;
    color: #4f46e5;
    font-weight: 700;
}

.no-img {
    height: 180px;
    background: #f1f5f9;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 12px;
}

.metric-box {
    background: white;
    padding: 18px;
    border-radius: 16px;
    border: 1px solid #e2e8f0;
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

            self.attention_layernorms.append(
                nn.LayerNorm(hidden_units, eps=1e-8)
            )

            self.attention_layers.append(
                nn.MultiheadAttention(
                    hidden_units,
                    num_heads,
                    dropout_rate
                )
            )

            self.forward_layernorms.append(
                nn.LayerNorm(hidden_units, eps=1e-8)
            )

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

            mha_outputs, _ = self.attention_layers[i](
                q,
                seqs,
                seqs,
                attn_mask=attention_mask
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

        return torch.nn.functional.normalize(
            feats[:, -1, :],
            dim=-1
        )


# ============================================================
# UTILS
# ============================================================

def clean_text(text):

    if text is None:
        return ""

    if isinstance(text, list):
        text = " ".join([str(x) for x in text])

    text = str(text)

    text = re.sub(r"<[^>]*>", "", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def short_text(text, n=80):

    text = clean_text(text)

    if len(text) <= n:
        return text

    return text[:n] + "..."


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
        cat = " > ".join([str(c) for c in cat[:4]])

    return clean_text(cat)


def get_image(meta):

    for key in [
        "image",
        "imageURLHighRes",
        "imageURL",
        "image_url"
    ]:

        img = meta.get(key, "")

        if isinstance(img, list) and len(img) > 0:
            img = img[0]

        if isinstance(img, str) and img.startswith("http"):
            return img

    return ""


def get_price(meta):

    price = clean_text(meta.get("price", ""))

    if price:
        return price

    return "Liên hệ"


def render_img_html(image, height=180):

    if image and image.startswith("http"):
        return f'''
        <img src="{image}"
             style="height:{height}px;
                    width:100%;
                    object-fit:contain;
                    border-radius:12px;">
        '''

    return f'''
    <div class="no-img" style="height:{height}px;">
        No Image
    </div>
    '''


# ============================================================
# LOAD
# ============================================================

@st.cache_resource(show_spinner=True)
def load_all(dataset_name):

    dataset_config = DATASETS[dataset_name]

    paths = ensure_required_files(dataset_config)

    model_path = paths["sasrec_qwen_best.pt"]
    item_emb_path = paths["fused_item_emb_128.pt"]
    item_meta_path = paths["item_web_meta.pkl"]
    user_history_path = paths["user_history.pkl"]

    item_matrix = torch.load(
        item_emb_path,
        map_location=DEVICE
    )

    if isinstance(item_matrix, dict):

        for k in [
            "fused_item_emb",
            "item_emb",
            "emb",
            "item_matrix"
        ]:

            if k in item_matrix:
                item_matrix = item_matrix[k]
                break

    item_matrix = item_matrix.float().to(DEVICE)

    item_matrix = torch.nn.functional.normalize(
        item_matrix,
        dim=-1
    )

    with open(item_meta_path, "rb") as f:
        item_meta = pickle.load(f)

    with open(user_history_path, "rb") as f:
        user_history = pickle.load(f)

    user_history = {
        int(k): [int(x) for x in v]
        for k, v in user_history.items()
    }

    ckpt = torch.load(
        model_path,
        map_location=DEVICE
    )

    maxlen = int(ckpt.get("maxlen", 50))
    hidden_units = int(ckpt.get("hidden_units", 128))
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

    return model, item_matrix, item_meta, user_history


# ============================================================
# RECOMMEND
# ============================================================

@torch.no_grad()
def recommend(
    model,
    item_matrix,
    item_meta,
    user_history,
    user_id,
    top_k=10
):

    seq = user_history.get(int(user_id), [])

    if not seq:
        return []

    user_vec = model.get_user_vector(seq)

    scores = torch.matmul(
        user_vec,
        item_matrix.T
    ).squeeze(0)

    seen = [int(x) for x in set(seq)]

    scores[torch.LongTensor(seen).to(DEVICE)] = -1e9

    top_scores, top_ids = torch.topk(scores, k=top_k)

    results = []

    for rank, (iid, score) in enumerate(
        zip(
            top_ids.cpu().numpy(),
            top_scores.cpu().numpy()
        ),
        1
    ):

        meta = get_meta(item_meta, iid)

        results.append({
            "rank": rank,
            "item_id": int(iid),
            "score": float(score),
            "title": get_title(meta),
            "brand": get_brand(meta),
            "category": get_category(meta),
            "price": get_price(meta),
            "image": get_image(meta),
        })

    return results


# ============================================================
# UI
# ============================================================

st.markdown("""
<div class="hero">
    <h1>🛒 Smart Commerce AI</h1>
    <p>
        SASRec + Qwen Embedding Recommendation System
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## 🗂️ Dataset")

dataset_options = {
    key: cfg.display_name
    for key, cfg in DATASETS.items()
}

selected_dataset = st.sidebar.selectbox(
    "Chọn dataset",
    options=list(dataset_options.keys()),
    format_func=lambda x: dataset_options[x]
)

selected_config = DATASETS[selected_dataset]

st.sidebar.info(selected_config.description)


# ============================================================
# LOAD
# ============================================================

with st.spinner(
    f"🚀 Đang tải {selected_config.display_name}..."
):

    model, item_matrix, item_meta, user_history = load_all(
        selected_dataset
    )

st.success(
    f"✅ Đã load xong {selected_config.display_name}"
)


# ============================================================
# USERS
# ============================================================

all_users = sorted(list(user_history.keys()))

selected_user = st.sidebar.selectbox(
    "👤 User ID",
    all_users
)

top_k = st.sidebar.select_slider(
    "Top-K",
    options=[1, 5, 10, 20],
    value=10
)


# ============================================================
# STATS
# ============================================================

m1, m2, m3 = st.columns(3)

m1.markdown(f"""
<div class="metric-box">
<h3>Dataset</h3>
<p>{selected_config.display_name}</p>
</div>
""", unsafe_allow_html=True)

m2.markdown(f"""
<div class="metric-box">
<h3>Users</h3>
<p>{len(user_history):,}</p>
</div>
""", unsafe_allow_html=True)

m3.markdown(f"""
<div class="metric-box">
<h3>Products</h3>
<p>{item_matrix.shape[0]-1:,}</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# RECOMMEND
# ============================================================

st.markdown("## 🎯 Recommended Products")

with st.spinner("Generating recommendations..."):

    results = recommend(
        model=model,
        item_matrix=item_matrix,
        item_meta=item_meta,
        user_history=user_history,
        user_id=selected_user,
        top_k=top_k
    )

if not results:

    st.warning("Không có đề xuất.")

else:

    cols = st.columns(5)

    for idx, p in enumerate(results):

        with cols[idx % 5]:

            img_html = render_img_html(
                p["image"],
                height=200
            )

            card_html = f"""
            <div class="rec-card">

                <span class="badge">
                    Rank #{p["rank"]}
                </span>

                {img_html}

                <div class="title">
                    {short_text(p["title"], 90)}
                </div>

                <p>
                    <b>Brand:</b>
                    {short_text(p["brand"], 30)}
                </p>

                <p>
                    <b>Category:</b>
                    {short_text(p["category"], 40)}
                </p>

                <div class="price">
                    {p["price"]}
                </div>

                <div class="score">
                    Score: {p["score"]:.4f}
                </div>

            </div>
            """

            st.markdown(
                card_html,
                unsafe_allow_html=True
            )


# ============================================================
# EXPORT
# ============================================================

if results:

    df = pd.DataFrame(results)

    st.download_button(
        label="📥 Download CSV",
        data=df.to_csv(
            index=False,
            encoding="utf-8-sig"
        ),
        file_name=f"{selected_dataset}_recommendation.csv",
        mime="text/csv"
    )
