"""
features.py — Tính 14 đặc trưng tĩnh từ channels_raw.csv (không leakage)
=======================================================================
Input  : data/collected/channels_raw.csv
Output : data/processed/features_final.csv
         data/processed/intervals.csv

Đồng bộ với config.py:
- Dùng FEATURE_COLS_STATIC để biết các feature cần tính.
- Dùng Paths.INTERVALS_FILE để lưu intervals.

Thay đổi so với bản gốc:
- Bỏ tính if_anomaly_score (sẽ do train.py tạo trong pipeline).
- Sửa sub_to_view_velocity_ratio an toàn với NaN.
- Lưu intervals riêng.
- Thêm tokenizer song ngữ để tính type_token_ratio và opening_repeat_ratio chính xác.
"""

import sys
import json
import re
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Thư viện xử lý ngôn ngữ ─────────────────────────────────────────────────
from langdetect import detect, DetectorFactory
from underthesea import word_tokenize as vi_tokenize

# Đảm bảo kết quả detect ổn định
DetectorFactory.seed = 0

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    SIMILARITY_WINDOW,
    TFIDF_MAX_FEATURES,
    MIN_UPLOAD_INTERVAL_DAYS,
    Paths,
    validate,
    FEATURE_COLS_STATIC,
)
from src.utils import _parse_timestamps, _parse_titles, _flatten_titles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Từ điển pattern
# ══════════════════════════════════════════════════════════════════════════════

CLICKBAIT_PATTERNS = [
    r"(?i)\byou (won'?t|will never) believe\b",
    r"(?i)\bthe (dark |hidden |shocking |untold |real |secret )?truth\b",
    r"(?i)\bwhat (nobody|no one) tells you\b",
    r"(?i)\bchanged (everything|the world|history)\b",
    r"(?i)\bmost people don'?t know\b",
    r"(?i)\bgenius of\b",
    r"(?i)\bexplained (simply|in \d+ minutes?)\b",
    r"(?i)\bthis (changes|will change) everything\b",
    r"(?i)\bthe (mind|brain) of\b",
    r"(?i)\bfeynman (explains|on|about|method)\b",
    r"(?i)\beinstein (explains|on|about)\b",
    r"(?i)\bcarl sagan (on|about|explains)\b",
    r"(?i)\bnot .{1,30} but\b",
    r"(?i)\bfew people (know|realize)\b",
    r"ít ai biết",
    r"sự thật (kinh hoàng|bí ẩn|đáng sợ|không ngờ)",
    r"không phải .{1,20} mà là",
    r"(bí mật|bí ẩn) (đằng sau|của|về)",
    r"tại sao (không ai|ít người)",
    r"review phim",
]

DASH_PATTERN = re.compile(r"—|--|–")


def _get_channel_age_days(timestamps_json: str) -> float:
    dts = _parse_timestamps(timestamps_json)
    if not dts:
        return -1.0
    first_video_date = min(dts)
    now = datetime.now(timezone.utc)
    channel_age_days = (now - first_video_date).total_seconds() / 86400
    return max(channel_age_days, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# Hàm xử lý ngôn ngữ (mới)
# ══════════════════════════════════════════════════════════════════════════════

def detect_channel_language(titles_json: str) -> str:
    """
    Xác định ngôn ngữ chính của kênh dựa trên toàn bộ titles.
    Trả về 'vi' hoặc 'en'. Mặc định 'en' nếu không rõ.
    """
    text = _flatten_titles(titles_json)
    if not text.strip():
        return "en"
    try:
        lang = detect(text)
        return "vi" if lang == "vi" else "en"
    except:
        # fallback heuristic: có >= 3 ký tự tiếng Việt -> 'vi'
        vi_chars = len(re.findall(
            r'[àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]',
            text.lower()
        ))
        return "vi" if vi_chars >= 3 else "en"


def tokenize(text: str, lang: str) -> List[str]:
    """Tách từ theo ngôn ngữ, trả về danh sách token."""
    if not text or not text.strip():
        return []
    if lang == "vi":
        # Dùng underthesea word_tokenize
        return vi_tokenize(text)
    else:
        # Tiếng Anh: tách bằng regex \w+ (bỏ dấu câu, đưa về lowercase)
        return re.findall(r'\b\w+\b', text.lower())


# ══════════════════════════════════════════════════════════════════════════════
# Các hàm tính feature (đã cập nhật F4, F11)
# ══════════════════════════════════════════════════════════════════════════════

def compute_time_interval_std(timestamps_json: str) -> float:
    dts = _parse_timestamps(timestamps_json)
    if len(dts) < 3:
        return -1.0
    dts.sort()
    intervals = [
        (dts[i+1] - dts[i]).total_seconds() / 86400
        for i in range(len(dts) - 1)
    ]
    intervals = [x for x in intervals if x <= 365]
    if len(intervals) < 2:
        return -1.0
    return float(np.std(intervals))


def compute_upload_burst_ratio(timestamps_json: str) -> float:
    dts = _parse_timestamps(timestamps_json)
    if len(dts) < 3:
        return 0.0
    dts.sort()
    intervals = [
        (dts[i+1] - dts[i]).total_seconds() / 86400
        for i in range(len(dts) - 1)
    ]
    intervals = [x for x in intervals if x <= 365]
    if not intervals:
        return 0.0
    threshold = np.median(intervals) * 0.3
    burst_count = sum(1 for x in intervals if x < threshold)
    return burst_count / len(intervals)


def compute_video_upload_frequency(
    timestamps_json: str,
    video_count: Optional[int] = None,
    n_videos_crawled: Optional[int] = None,
) -> float:
    dts = _parse_timestamps(timestamps_json)
    if not dts:
        return -1.0
    first_video_date = min(dts)
    now = datetime.now(timezone.utc)
    channel_age_days = (now - first_video_date).total_seconds() / 86400
    num_videos = n_videos_crawled if n_videos_crawled and n_videos_crawled > 0 else (
        video_count if video_count and video_count > 0 else len(dts)
    )
    if channel_age_days < 1:
        channel_age_days = 1.0
    return num_videos / channel_age_days


def compute_view_per_video(view_count: Optional[int], video_count: Optional[int]) -> float:
    try:
        views = float(view_count) if view_count is not None else 0.0
        videos = float(video_count) if video_count is not None else 0.0
        if videos <= 0:
            return 0.0
        return views / videos
    except (ValueError, TypeError):
        return 0.0

"""
Khi đọc channels_raw.csv, nếu cột subscriber_count hoặc view_count bị thiếu (pandas hiểu là NaN), 
biểu thức float(subscriber_count) không gây lỗi mà trả về NaN. Hàm hiện tại chỉ kiểm tra subs <= 0 
– điều kiện này sai với NaN (NaN <= 0 → False), 
nên nó lọt qua và thực hiện phép chia subs / channel_age_days, kết quả là NaN.
"""
def compute_subscriber_velocity(timestamps_json: str, subscriber_count: Optional[int]) -> float:
    channel_age_days = _get_channel_age_days(timestamps_json)
    if channel_age_days <= 0:          # bắt cả trường hợp 0 (dù _get_channel_age_days luôn >=1 hoặc -1)
        return -1.0
    try:
        subs = float(subscriber_count) if subscriber_count is not None else 0.0
        if np.isnan(subs):             # bắt NaN từ dữ liệu thô
            return -1.0
    except (ValueError, TypeError):
        return -1.0
    if subs <= 0:
        return 0.0
    return subs / channel_age_days


def compute_sub_to_view_velocity_ratio(subscriber_velocity: float, view_velocity: float) -> float:
    """Trả về 0.0 nếu đầu vào không hợp lệ thay vì NaN."""
    if not np.isfinite(subscriber_velocity) or not np.isfinite(view_velocity):
        return 0.0
    if subscriber_velocity <= 0 or view_velocity <= 0:
        return 0.0
    return float(np.log10((subscriber_velocity + 1) / (view_velocity + 1)))


def compute_dash_density(titles_json: str) -> float:
    text = _flatten_titles(titles_json)
    if not text:
        return 0.0
    words = text.split()   # vẫn dùng split cho đơn giản, vì chỉ để đếm mật độ dấu gạch ngang
    if not words:
        return 0.0
    n_dashes = len(DASH_PATTERN.findall(text))
    return n_dashes / len(words)


def compute_title_length_std(titles_json: str) -> float:
    titles = _parse_titles(titles_json)
    lengths = [len(t) for t in titles]
    if len(lengths) < 2:
        return 0.0
    return float(np.std(lengths))


def compute_capitalization_ratio(titles_json: str) -> float:
    titles = _parse_titles(titles_json)
    if not titles:
        return 0.0
    ratios = []
    for title in titles:
        letters = [c for c in title if c.isalpha()]
        if not letters:
            continue
        uppercase = sum(1 for c in letters if c.isupper())
        ratios.append(uppercase / len(letters))
    return float(np.mean(ratios)) if ratios else 0.0


def compute_opening_repeat_ratio(titles_json: str, n_gram: int = 3) -> float:
    titles = _parse_titles(titles_json)
    if len(titles) < 2:
        return 0.0
    lang = detect_channel_language(titles_json)  # xác định ngôn ngữ một lần
    openings = []
    for title in titles:
        words = tokenize(title, lang)[:n_gram]
        if words:
            openings.append(" ".join(words))
    if not openings:
        return 0.0
    unique_openings = len(set(openings))
    total_titles = len(openings)
    return 1.0 - (unique_openings / total_titles)


def compute_temporal_clickbait_ratio(titles_json: str, time_interval_std: float) -> float:
    titles = _parse_titles(titles_json)
    if not titles:
        return 0.0
    clickbait_markers = 0
    for title in titles:
        if re.search(r"[?!]", title) or re.search(r"\d+", title):
            clickbait_markers += 1
    ratio = clickbait_markers / len(titles)
    if time_interval_std < 0:
        return 0.0
    return ratio / (1.0 + time_interval_std)


def compute_type_token_ratio(titles_json: str) -> float:
    text = _flatten_titles(titles_json)
    if not text:
        return 0.0
    lang = detect_channel_language(titles_json)
    words = tokenize(text, lang)
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def compute_avg_title_similarity(titles_json: str, window: int = SIMILARITY_WINDOW) -> float:
    titles = _parse_titles(titles_json)
    titles = titles[:window]
    if len(titles) < 2:
        return 0.0
    try:
        vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
        )
        tfidf_matrix = vectorizer.fit_transform(titles)
        sim_matrix = cosine_similarity(tfidf_matrix)
        n = sim_matrix.shape[0]
        upper = [sim_matrix[i, j] for i in range(n) for j in range(i+1, n)]
        return float(np.mean(upper)) if upper else 0.0
    except Exception:
        return 0.0


def compute_sub_to_view_ratio(subscriber_count, view_count) -> float:
    try:
        sub = float(subscriber_count) if subscriber_count is not None else 0.0
        view = float(view_count) if view_count is not None else 0.0
        return sub / (view + 1)
    except (ValueError, TypeError):
        return 0.0


def compute_view_velocity(timestamps_json: str, view_count: Optional[int]) -> float:
    channel_age_days = _get_channel_age_days(timestamps_json)
    if channel_age_days <= 0:
        return -1.0
    try:
        views = float(view_count) if view_count is not None else 0.0
        if np.isnan(views):            # bắt NaN từ dữ liệu thô
            return -1.0
    except (ValueError, TypeError):
        return -1.0
    if views <= 0:
        return 0.0
    return views / channel_age_days


# ══════════════════════════════════════════════════════════════════════════════
# Hàm tính intervals
# ══════════════════════════════════════════════════════════════════════════════

def compute_intervals(timestamps_json: str) -> List[float]:
    dts = _parse_timestamps(timestamps_json)
    if len(dts) < 3:
        return []
    dts.sort()
    intervals = [
        (dts[i+1] - dts[i]).total_seconds() / 86400
        for i in range(len(dts) - 1)
    ]
    intervals = [x for x in intervals if x <= 365]
    return intervals


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline chính
# ══════════════════════════════════════════════════════════════════════════════

def build_features(df_raw: pd.DataFrame, verbose: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Tính toàn bộ feature (trừ if_anomaly_score) và intervals.
    Trả về (df_features, df_intervals).
    """
    log.info(f"Bắt đầu tính features cho {len(df_raw)} kênh...")
    rows = []
    intervals_rows = []

    for _, row in df_raw.iterrows():
        # --- Tính các feature cơ bản ---
        f1_time_std = compute_time_interval_std(row["video_timestamps"])
        f6_burst = compute_upload_burst_ratio(row["video_timestamps"])
        f14_freq = compute_video_upload_frequency(
            row["video_timestamps"],
            row.get("video_count"),
            row.get("n_videos_crawled")
        )

        # Text features
        f2_dash = compute_dash_density(row["video_titles"])
        f8_title_std = compute_title_length_std(row["video_titles"])
        f10_cap = compute_capitalization_ratio(row["video_titles"])
        f11_open = compute_opening_repeat_ratio(row["video_titles"])
        f12_temp_click = compute_temporal_clickbait_ratio(row["video_titles"], f1_time_std)

        f4_ttr = compute_type_token_ratio(row["video_titles"])
        f5_sim = compute_avg_title_similarity(row["video_titles"])

        f7_sub_view = compute_sub_to_view_ratio(
            row.get("subscriber_count", 0),
            row.get("view_count", 0)
        )

        # Velocity features
        view_vel = compute_view_velocity(row["video_timestamps"], row.get("view_count"))
        sub_vel = compute_subscriber_velocity(row["video_timestamps"], row.get("subscriber_count"))
        view_per_vid = compute_view_per_video(row.get("view_count"), row.get("video_count"))
        sub_view_vel_ratio = compute_sub_to_view_velocity_ratio(sub_vel, view_vel)

        # --- Intervals ---
        intervals = compute_intervals(row["video_timestamps"])
        intervals_rows.append({
            "channel_id": row["channel_id"],
            "intervals_json": json.dumps(intervals)
        })

        feat_row = {
            "channel_id": row["channel_id"],
            "title": row["title"],
            "label": row["label"],

            # Tất cả feature tĩnh (14)
            "time_interval_std": f1_time_std,
            "upload_burst_ratio": f6_burst,
            "video_upload_frequency": f14_freq,
            "view_per_video": view_per_vid,
            "dash_density": f2_dash,
            "title_length_std": f8_title_std,
            "capitalization_ratio": f10_cap,
            "opening_repeat_ratio": f11_open,
            "temporal_clickbait_ratio": f12_temp_click,
            "type_token_ratio": f4_ttr,
            "avg_title_similarity": f5_sim,
            "sub_to_view_ratio": f7_sub_view,
            "subscriber_velocity": sub_vel,
            "sub_to_view_velocity_ratio": sub_view_vel_ratio,

            # Meta
            "subscriber_count": row.get("subscriber_count", 0),
            "view_count": row.get("view_count", 0),
            "video_count": row.get("video_count", 0),
            "n_videos_crawled": row.get("n_videos_crawled", 0),
        }
        rows.append(feat_row)

        if verbose:
            label_str = "SLOP" if row["label"] == 1 else "GENUINE"
            log.info(
                f"  [{label_str}] {row['title'][:35]:<35} "
                f"std={f1_time_std:6.2f} burst={f6_burst:.3f} freq={f14_freq:.2f} "
                f"view/vid={view_per_vid:.1f} sub_vel={sub_vel:.1f} ratio={sub_view_vel_ratio:.3f}"
            )

    df_feat = pd.DataFrame(rows)
    # Kiểm tra cảnh báo về sub_to_view_velocity_ratio
    nan_count = df_feat["sub_to_view_velocity_ratio"].isna().sum()
    log.info(f"sub_to_view_velocity_ratio NaN count: {nan_count}")
    if nan_count > 0:
        log.info("Mẫu các dòng NaN:")
        log.info(df_feat[df_feat["sub_to_view_velocity_ratio"].isna()][["label", "subscriber_velocity", "view_per_video"]].head())
    
    df_intervals = pd.DataFrame(intervals_rows)
    return df_feat, df_intervals


def print_summary(df: pd.DataFrame) -> None:
    """In phân phối feature theo label, dùng danh sách từ config."""
    feature_cols = [c for c in FEATURE_COLS_STATIC if c in df.columns]
    print("\n" + "=" * 70)
    print(f"PHÂN PHỐI FEATURE THEO LABEL ({len(feature_cols)} features, không có if_anomaly_score)")
    print("=" * 70)

    for col in feature_cols:
        slop_vals = df.loc[df["label"] == 1, col]
        genuine_vals = df.loc[df["label"] == 0, col]

        # Lọc các giá trị >=0 để tính trung bình (loại bỏ missing -1 và NaN)
        slop_valid = slop_vals[slop_vals >= 0]
        genuine_valid = genuine_vals[genuine_vals >= 0]

        slop_mean = slop_valid.mean() if len(slop_valid) > 0 else float('nan')
        genuine_mean = genuine_valid.mean() if len(genuine_valid) > 0 else float('nan')

        # Định dạng hiển thị mean
        def fmt_mean(val):
            if pd.isna(val):
                return "N/A"
            else:
                return f"{val:10.4f}"

        direction = "↑slop" if slop_mean > genuine_mean else "↓slop"

        print(
            f"  {col:<28} "
            f"Slop={fmt_mean(slop_mean)}  Genuine={fmt_mean(genuine_mean)}  {direction}"
        )

    # Debug riêng cho sub_to_view_velocity_ratio
    if "sub_to_view_velocity_ratio" in df.columns:
        print("\n  🔍 Phân tích sub_to_view_velocity_ratio:")
        for label_val, label_name in [(0, "Genuine"), (1, "Slop")]:
            vals = df.loc[df["label"] == label_val, "sub_to_view_velocity_ratio"]
            n_total = len(vals)
            n_neg = (vals < 0).sum()
            n_zero = (vals == 0).sum()
            n_pos = (vals > 0).sum()
            n_nan = vals.isna().sum()
            print(f"     {label_name}: total={n_total}, <0={n_neg}, =0={n_zero}, >0={n_pos}, NaN={n_nan}")

    print()
    print(f"  Kênh thiếu data timestamps (time_interval_std=-1): "
          f"{(df['time_interval_std'] < 0).sum()} kênh")
    print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# Entrypoint
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Compute features from channels_raw.csv")
    parser.add_argument("--verbose", action="store_true",
                        help="In chi tiết feature của từng kênh")
    args = parser.parse_args()

    validate()

    if not Paths.CHANNELS_RAW.exists():
        log.error(f"Không tìm thấy {Paths.CHANNELS_RAW}. Hãy chạy crawl.py trước.")
        sys.exit(1)

    log.info(f"Đọc {Paths.CHANNELS_RAW.name}...")
    df_raw = pd.read_csv(Paths.CHANNELS_RAW)
    log.info(f"  {len(df_raw)} kênh | "
             f"Slop: {(df_raw['label']==1).sum()} | "
             f"Genuine: {(df_raw['label']==0).sum()}")

    df_feat, df_intervals = build_features(df_raw, verbose=args.verbose)

    Paths.PROCESSED.mkdir(parents=True, exist_ok=True)

    # Lưu features (không có if_anomaly_score)
    df_feat.to_csv(Paths.FEATURES_FINAL, index=False, encoding="utf-8")
    log.info(f"Saved features → {Paths.FEATURES_FINAL}")
    log.info(f"Shape: {df_feat.shape} | Columns: {list(df_feat.columns)}")

    # Lưu intervals (dùng đường dẫn từ config)
    df_intervals.to_csv(Paths.INTERVALS_FILE, index=False, encoding="utf-8")
    log.info(f"Saved intervals → {Paths.INTERVALS_FILE}")

    print_summary(df_feat)


if __name__ == "__main__":
    main()