"""
expand.py — Tự động mở rộng dataset (nhiều API key, lọc quốc gia, giới hạn video, cải thiện genuine)
=====================================================================================================
Cải thiện:
- Nếu API không trả về country, dùng query để đoán ngôn ngữ → tokenizer phù hợp.
- Giảm ngưỡng genuine xuống 0.15 để thu thập thêm kênh thật.
"""

import sys, json, logging, argparse, pickle
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    YOUTUBE_API_KEYS,
    VIDEOS_PER_CHANNEL,
    Paths, LABEL_SLOP, LABEL_GENUINE,
    AUTO_EXPAND_CONFIDENCE_THRESHOLD,
    AUTO_EXPAND_SLOP_HIGH_CONFIDENCE,
    AUTO_EXPAND_SLOP_EXCLUDE_KEYWORDS,
    AUTO_EXPAND_GENUINE_CONFIDENCE_THRESHOLD,   # mới
    AUTO_EXPAND_MAX_NEW_SLOP, AUTO_EXPAND_MAX_NEW_GENUINE,
    AUTO_EXPAND_MIN_VIEWS,  # mới
    AUTO_EXPAND_SLOP_QUERIES_VI, AUTO_EXPAND_SLOP_QUERIES_EN,
    AUTO_EXPAND_GENUINE_QUERIES_VI, AUTO_EXPAND_GENUINE_QUERIES_EN,
    AUTO_EXPAND_MIN_SUBSCRIBERS, AUTO_EXPAND_MIN_VIDEOS,
    AUTO_EXPAND_LOW_RESULT_QUERIES,
    FEATURE_COLS_STATIC, validate,
)
from src.utils import _parse_timestamps, _parse_titles, _flatten_titles
from src.train import IntervalAnomalyTransformer, DataFrameSelector
from src.features import (
    compute_time_interval_std, compute_upload_burst_ratio,
    compute_video_upload_frequency, compute_view_per_video,
    compute_dash_density, compute_title_length_std,
    compute_capitalization_ratio, compute_opening_repeat_ratio,
    compute_temporal_clickbait_ratio, compute_type_token_ratio,
    compute_avg_title_similarity, compute_sub_to_view_ratio,
    compute_subscriber_velocity, compute_view_velocity, compute_intervals,
)
from underthesea import word_tokenize as vi_tokenize

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ═══════════════════════════ Tokenizer theo quốc gia ═══════════════
def tokenize_for_country(text: str, country: str) -> List[str]:
    if not text or not text.strip():
        return []
    if country == "VN":
        return vi_tokenize(text)
    else:
        # Tiếng Anh hoặc ngôn ngữ khác: tách từ đơn giản
        return text.lower().split()


# ═══════════════════════════════ YouTube Service Manager ═══════════════
class YouTubeServiceManager:
    """Quản lý nhiều API key, tự động xoay vòng khi gặp lỗi quota."""
    def __init__(self, api_keys: List[str]):
        if not api_keys:
            raise ValueError("Cần ít nhất một API key.")
        self.api_keys = api_keys
        self.current_index = 0
        self.service = build("youtube", "v3", developerKey=self.current_key())
        first = self.api_keys[0]
        masked = first[:8] + "..." + first[-4:] if len(first) > 12 else first
        log.info(f"YouTubeServiceManager khởi tạo với {len(self.api_keys)} key (đầu tiên: {masked})")

    def current_key(self) -> str:
        return self.api_keys[self.current_index]

    def switch_to_next_key(self):
        self.current_index = (self.current_index + 1) % len(self.api_keys)
        self.service = build("youtube", "v3", developerKey=self.current_key())
        masked = self.current_key()[:8] + "..." + self.current_key()[-4:]
        log.warning(f"Chuyển sang API key mới (index {self.current_index}): {masked}")

    def get_service(self):
        return self.service

    def handle_quota_error(self, func, *args, **kwargs):
        """Thử gọi func. Nếu lỗi quota (403/429) thì đổi key và thử lại."""
        max_retries = len(self.api_keys)
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except HttpError as e:
                reason = e.error_details[0].get("reason", "") if e.error_details else ""
                if reason in ["rateLimitExceeded", "quotaExceeded", "dailyLimitExceeded"]:
                    log.warning(f"API key {self.current_key()[:8]}... hết quota, chuyển key...")
                    self.switch_to_next_key()
                    continue
                else:
                    raise
        raise Exception("Tất cả API key đã hết quota hoặc không hợp lệ.")


# ═══════════════════════════ Các hàm gọi API ═════════════════════════
def search_channels(manager: YouTubeServiceManager, query: str, max_results=10) -> List[str]:
    def _search():
        service = manager.get_service()
        request = service.search().list(
            part="snippet", q=query, type="channel",
            maxResults=max_results, order="relevance",
        )
        response = request.execute()
        return [item["snippet"]["channelId"] for item in response.get("items", [])]
    channel_ids = manager.handle_quota_error(_search)
    log.info(f"  Tìm thấy {len(channel_ids)} kênh từ query '{query}'")
    return channel_ids


def get_channel_info(manager: YouTubeServiceManager, channel_id: str) -> dict:
    """Lấy thông tin kênh, bao gồm country."""
    def _get():
        service = manager.get_service()
        request = service.channels().list(
            part="snippet,statistics",
            id=channel_id,
        )
        response = request.execute()
        if not response["items"]:
            return {}
        item = response["items"][0]
        snippet = item["snippet"]
        return {
            "channel_id": channel_id,
            "title": snippet["title"],
            "country": snippet.get("country", ""),   # thêm country
            "subscriber_count": int(item["statistics"].get("subscriberCount", 0)),
            "view_count": int(item["statistics"].get("viewCount", 0)),
            "video_count": int(item["statistics"].get("videoCount", 0)),
        }
    return manager.handle_quota_error(_get)


def get_channel_videos(manager: YouTubeServiceManager, channel_id: str, max_videos=VIDEOS_PER_CHANNEL) -> Tuple[list, list]:
    def _get():
        service = manager.get_service()
        request = service.search().list(
            part="snippet", channelId=channel_id, order="date",
            type="video", maxResults=max_videos,
        )
        response = request.execute()
        timestamps, titles = [], []
        for item in response.get("items", []):
            snippet = item["snippet"]
            timestamps.append(snippet["publishedAt"])
            titles.append(snippet["title"])
        return timestamps, titles
    return manager.handle_quota_error(_get)


# ═══════════════════════════ Features (có dùng tokenizer theo country) ═══════════════
def compute_channel_features(channel_info: dict, timestamps: list, titles: list) -> pd.DataFrame:
    """Tính 13 feature tĩnh + intervals, dùng tokenizer phù hợp với country."""
    ts_json = json.dumps(timestamps)
    titles_json = json.dumps(titles)
    country = channel_info.get("country", "")

    time_std = compute_time_interval_std(ts_json)
    burst = compute_upload_burst_ratio(ts_json)
    freq = compute_video_upload_frequency(ts_json, channel_info.get("video_count"), len(timestamps))
    view_per_vid = compute_view_per_video(channel_info.get("view_count"), channel_info.get("video_count"))

    dash = compute_dash_density(titles_json)
    title_std = compute_title_length_std(titles_json)
    cap = compute_capitalization_ratio(titles_json)

    # --- Các feature cần tokenizer ---
    # TTR và opening_repeat_ratio dùng tokenizer theo country
    # Chúng ta tính thủ công thay vì gọi hàm cũ để truyền đúng tokenizer
    # (Hoặc có thể sửa lại hàm cũ, nhưng để tránh đụng chạm, ta tự tính ở đây)

    # Tokenizer cho kênh
    def tokenize(text: str) -> List[str]:
        return tokenize_for_country(text, country)

    # TTR
    all_text = " ".join(titles)
    tokens = tokenize(all_text)
    ttr = len(set(tokens)) / len(tokens) if tokens else 0.0

    # Opening repeat ratio
    openings = []
    for t in titles:
        words = tokenize(t)[:3]  # 3-gram mở đầu
        if words:
            openings.append(" ".join(words))
    if len(openings) >= 2:
        opening = 1.0 - (len(set(openings)) / len(openings))
    else:
        opening = 0.0

    # Temporal clickbait (không cần tokenizer)
    temp_click = compute_temporal_clickbait_ratio(titles_json, time_std)

    # Avg similarity (dùng TfidfVectorizer mặc định, đã ổn cho cả hai ngôn ngữ)
    avg_sim = compute_avg_title_similarity(titles_json)

    sub_view = compute_sub_to_view_ratio(channel_info.get("subscriber_count"), channel_info.get("view_count"))
    sub_vel = compute_subscriber_velocity(ts_json, channel_info.get("subscriber_count"))
    view_vel = compute_view_velocity(ts_json, channel_info.get("view_count"))

    intervals = compute_intervals(ts_json)

    df = pd.DataFrame([{
        "channel_id": channel_info["channel_id"],
        "title": channel_info["title"],
        "label": -1,
        "time_interval_std": time_std,
        "upload_burst_ratio": burst,
        "video_upload_frequency": freq,
        "view_per_video": view_per_vid,
        "dash_density": dash,
        "title_length_std": title_std,
        "capitalization_ratio": cap,
        "opening_repeat_ratio": opening,
        "temporal_clickbait_ratio": temp_click,
        "type_token_ratio": ttr,
        "avg_title_similarity": avg_sim,
        "sub_to_view_ratio": sub_view,
        "subscriber_velocity": sub_vel,
        # meta
        "subscriber_count": channel_info.get("subscriber_count", 0),
        "view_count": channel_info.get("view_count", 0),
        "video_count": channel_info.get("video_count", 0),
        "n_videos_crawled": len(timestamps),
        "intervals": intervals,
    }])
    return df


# ═══════════════════════════ Model ═══════════════
def load_pipeline():
    with open(Paths.MODEL_RF, "rb") as f:
        model_data = pickle.load(f)
    return model_data["pipeline"], model_data["feature_cols"]


def predict_channel(pipeline, feature_cols: list, df_channel: pd.DataFrame) -> Tuple[int, float]:
    X = df_channel.copy()
    proba = pipeline.predict_proba(X)[0]
    pred = int(proba[1] >= 0.5)
    return pred, proba[1]


# ═══════════════════════════ Seed ═══════════════
def append_channel_id_to_file(filepath: Path, channel_id: str):
    if not filepath.exists():
        filepath.touch()
    existing = set(filepath.read_text(encoding="utf-8").splitlines())
    if channel_id not in existing:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(channel_id + "\n")

def save_results_to_csv(slops: list, genuines: list, filepath: Path):
    rows = []
    for cid, info in slops:
        rows.append({"channel_id": cid, "title": info.get("title", ""), "label": "slop"})
    for cid, info in genuines:
        rows.append({"channel_id": cid, "title": info.get("title", ""), "label": "genuine"})
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False, encoding="utf-8")
        log.info(f"Đã lưu danh sách kênh mới vào {filepath}")

def is_excluded_channel(info: dict, titles: list) -> bool:
    """Kiểm tra nếu tiêu đề kênh hoặc video chứa từ khóa loại trừ (nhạc/ASMR)."""
    text_to_check = info.get("title", "").lower()
    for t in titles[:5]:
        text_to_check += " " + t.lower()
    for kw in AUTO_EXPAND_SLOP_EXCLUDE_KEYWORDS:
        if kw in text_to_check:
            return True
    return False

# ═══════════════════════════ Main ═══════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-slops", type=int, default=AUTO_EXPAND_MAX_NEW_SLOP)
    parser.add_argument("--max-genuine", type=int, default=AUTO_EXPAND_MAX_NEW_GENUINE)
    parser.add_argument("--dry-run", action="store_true", help="Không ghi file")
    args = parser.parse_args()

    validate()

    if not Paths.MODEL_RF.exists():
        log.error(f"Không tìm thấy model {Paths.MODEL_RF}.")
        sys.exit(1)

    log.info("Đang load model...")
    pipeline, feature_cols = load_pipeline()
    log.info(f"Model loaded. Features: {feature_cols}")

    manager = YouTubeServiceManager(YOUTUBE_API_KEYS)

    new_slops = []
    new_genuines = []

    def process_query(query, target_label):
        nonlocal new_slops, new_genuines
        max_new = args.max_slops if target_label == LABEL_SLOP else args.max_genuine
        if target_label == LABEL_SLOP and len(new_slops) >= max_new:
            return
        if target_label == LABEL_GENUINE and len(new_genuines) >= max_new:
            return

        log.info(f"Search query: '{query}'")
        channel_ids = search_channels(manager, query, max_results=20)
        for cid in channel_ids:
            if target_label == LABEL_SLOP and len(new_slops) >= max_new:
                break
            if target_label == LABEL_GENUINE and len(new_genuines) >= max_new:
                break

            log.info(f"  Đang crawl kênh {cid}...")
            info = get_channel_info(manager, cid)
            if not info:
                continue

            # Gán country nếu thiếu dựa vào query
            if not info.get("country"):
                if any(c in query for c in 'àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ'):
                    info["country"] = "VN"

            # Lọc theo subscriber, video, view
            if info.get('subscriber_count', 0) < AUTO_EXPAND_MIN_SUBSCRIBERS or \
               info.get('video_count', 0) < AUTO_EXPAND_MIN_VIDEOS:
                log.info(f"    Bỏ qua vì không đạt ngưỡng (subs={info.get('subscriber_count')}, videos={info.get('video_count')})")
                continue
            if info.get('view_count', 0) < AUTO_EXPAND_MIN_VIEWS:
                log.info(f"    Bỏ qua vì view quá thấp (view={info.get('view_count')})")
                continue

            max_videos = 10 if query in AUTO_EXPAND_LOW_RESULT_QUERIES else VIDEOS_PER_CHANNEL
            timestamps, titles = get_channel_videos(manager, cid, max_videos=max_videos)
            if len(timestamps) < 3:
                log.info("    Bỏ qua vì ít video.")
                continue

            df_ch = compute_channel_features(info, timestamps, titles)
            _, prob = predict_channel(pipeline, feature_cols, df_ch)
            log.info(f"    Title: {info['title'][:50]}...  prob_slop={prob:.3f}")

            # Phân loại với ngưỡng và bộ lọc loại trừ
            if prob >= AUTO_EXPAND_CONFIDENCE_THRESHOLD:
                if prob < AUTO_EXPAND_SLOP_HIGH_CONFIDENCE:
                    if is_excluded_channel(info, titles):
                        log.info(f"    -> Loại trừ vì chứa từ khóa nhạc/ASMR")
                        continue
                new_slops.append((cid, info))
                log.info("    -> Thêm vào slop (tự tin cao)")
            elif prob <= AUTO_EXPAND_GENUINE_CONFIDENCE_THRESHOLD:
                new_genuines.append((cid, info))
                log.info("    -> Thêm vào genuine (tự tin cao)")
            else:
                log.info("    -> Không đủ tự tin, bỏ qua.")

    # Duyệt các query slop
    for q in AUTO_EXPAND_SLOP_QUERIES_VI + AUTO_EXPAND_SLOP_QUERIES_EN:
        if len(new_slops) >= args.max_slops:
            break
        process_query(q, LABEL_SLOP)

    # Duyệt các query genuine
    for q in AUTO_EXPAND_GENUINE_QUERIES_VI + AUTO_EXPAND_GENUINE_QUERIES_EN:
        if len(new_genuines) >= args.max_genuine:
            break
        process_query(q, LABEL_GENUINE)

    log.info(f"\nKết quả: {len(new_slops)} kênh slop mới, {len(new_genuines)} kênh genuine mới.")

    if args.dry_run:
        log.info("Dry-run: không ghi file.")
        return

    # Lưu CSV và cập nhật file seed
    save_results_to_csv(new_slops, new_genuines, Paths.PROCESSED / "new_channels.csv")
    slop_seed_file = Paths.SEED_SLOP_FILES[-1]
    genuine_seed_file = Paths.SEED_GENUINE_FILES[-1]
    for cid, _ in new_slops:
        append_channel_id_to_file(slop_seed_file, cid)
    for cid, _ in new_genuines:
        append_channel_id_to_file(genuine_seed_file, cid)
    log.info(f"Đã cập nhật file seed: {slop_seed_file}, {genuine_seed_file}")
    print("\nHoàn tất.")


if __name__ == "__main__":
    main()