"""
predict.py — Dự đoán kênh YouTube, có thể tự động thêm vào seed nếu tự tin cao
================================================================================
Input: URL kênh YouTube (channel/UC... hoặc @handle)
Output: Xác suất slop, nhãn dự đoán, thông tin kênh

Cách dùng:
    python src/predict.py "https://www.youtube.com/@ExampleHandle"
    python src/predict.py "https://www.youtube.com/channel/UC..."
    python src/predict.py "URL"                           # chỉ dự đoán
    python src/predict.py "URL" --add                     # hỏi trước khi thêm
    python src/predict.py "URL" --add --auto-add          # thêm ngay không hỏi
    python src/predict.py "URL" --label slop --auto-add   # ép nhãn slop
"""

import sys
import json
import pickle
import argparse
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    YOUTUBE_API_KEYS, VIDEOS_PER_CHANNEL, Paths,
    LABEL_SLOP, LABEL_GENUINE, FEATURE_COLS_STATIC, validate,
)
from src.crawl import parse_url, get_channel_info as crawl_get_channel_info
from src.expand import YouTubeServiceManager
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Ngưỡng tự tin để tự động thêm
AUTO_ADD_CONFIDENCE = 0.95


def load_model() -> Tuple[object, list]:
    with open(Paths.MODEL_RF, "rb") as f:
        model_data = pickle.load(f)
    return model_data["pipeline"], model_data["feature_cols"]


def crawl_channel_data(manager: YouTubeServiceManager, url: str) -> dict:
    """
    Từ URL kênh YouTube, crawl thông tin kênh và 50 video.
    Trả về dict chứa channel info, timestamps, titles.
    """
    parsed = parse_url(url)
    if parsed is None:
        raise ValueError("URL không hợp lệ. Cần dạng /channel/UC... hoặc /@handle")

    # Resolve channel_id
    def _resolve():
        yt = manager.get_service()
        if parsed["type"] == "id":
            return parsed["value"]
        resp = yt.channels().list(part="id", forHandle=parsed["value"]).execute()
        items = resp.get("items", [])
        if items:
            return items[0]["id"]
        raise Exception("Không tìm thấy kênh từ handle")

    channel_id = manager.handle_quota_error(_resolve)
    log.info(f"Resolved channel_id: {channel_id}")

    # Lấy thông tin kênh
    def _info():
        yt = manager.get_service()
        return crawl_get_channel_info(yt, channel_id)
    info = manager.handle_quota_error(_info)
    if not info:
        raise Exception("Không lấy được thông tin kênh.")
    log.info(f"Tên kênh: {info['title']}")

    # Lấy video
    def _videos():
        yt = manager.get_service()
        timestamps, titles = [], []
        playlist_id = info.get("uploads_playlist", "")
        if not playlist_id:
            return timestamps, titles
        next_page = None
        while len(timestamps) < VIDEOS_PER_CHANNEL:
            batch = min(50, VIDEOS_PER_CHANNEL - len(timestamps))
            resp = yt.playlistItems().list(
                part="snippet",
                playlistId=playlist_id,
                maxResults=batch,
                pageToken=next_page,
            ).execute()
            for item in resp.get("items", []):
                ts = item["snippet"].get("publishedAt")
                title = item["snippet"].get("title", "")
                if ts:
                    timestamps.append(ts)
                if title:
                    titles.append(title)
            next_page = resp.get("nextPageToken")
            if not next_page:
                break
        return timestamps, titles
    timestamps, titles = manager.handle_quota_error(_videos)
    log.info(f"Lấy được {len(timestamps)} timestamps, {len(titles)} titles")

    return {
        "channel_id": channel_id,
        "title": info["title"],
        "subscriber_count": info.get("subscriber_count", 0),
        "view_count": info.get("view_count", 0),
        "video_count": info.get("video_count", 0),
        "timestamps": timestamps,
        "titles": titles,
    }


def compute_features(channel_data: dict) -> pd.DataFrame:
    """Tính 14 đặc trưng tĩnh + intervals, trả về DataFrame 1 dòng."""
    ts_json = json.dumps(channel_data["timestamps"])
    titles_json = json.dumps(channel_data["titles"])

    time_std = compute_time_interval_std(ts_json)
    burst = compute_upload_burst_ratio(ts_json)
    freq = compute_video_upload_frequency(ts_json, channel_data.get("video_count"), len(channel_data["timestamps"]))
    view_per_vid = compute_view_per_video(channel_data.get("view_count"), channel_data.get("video_count"))

    dash = compute_dash_density(titles_json)
    title_std = compute_title_length_std(titles_json)
    cap = compute_capitalization_ratio(titles_json)
    opening = compute_opening_repeat_ratio(titles_json)
    temp_click = compute_temporal_clickbait_ratio(titles_json, time_std)

    ttr = compute_type_token_ratio(titles_json)
    avg_sim = compute_avg_title_similarity(titles_json)
    sub_view = compute_sub_to_view_ratio(channel_data.get("subscriber_count"), channel_data.get("view_count"))
    sub_vel = compute_subscriber_velocity(ts_json, channel_data.get("subscriber_count"))
    view_vel = compute_view_velocity(ts_json, channel_data.get("view_count"))

    intervals = compute_intervals(ts_json)

    df = pd.DataFrame([{
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
        # Meta
        "subscriber_count": channel_data.get("subscriber_count", 0),
        "view_count": channel_data.get("view_count", 0),
        "video_count": channel_data.get("video_count", 0),
        "n_videos_crawled": len(channel_data["timestamps"]),
        "intervals": intervals,
    }])
    return df


def predict(pipeline, df):
    proba = pipeline.predict_proba(df)[0]
    pred = int(proba[1] >= 0.5)
    return pred, proba[1]


def add_to_seed(channel_id, label):
    """Thêm channel_id vào file seed tương ứng (kiểm tra trùng)."""
    if label == LABEL_SLOP:
        seed_file = Paths.SEED_SLOP_FILES[-1]  # AI slop expand.txt
    else:
        seed_file = Paths.SEED_GENUINE_FILES[-1]  # non AI expand 2.txt

    seed_file.parent.mkdir(parents=True, exist_ok=True)
    if not seed_file.exists():
        seed_file.touch()

    existing = set()
    with open(seed_file, 'r', encoding='utf-8') as f:
        existing = set(line.strip() for line in f if line.strip())

    if channel_id in existing:
        log.warning(f"Kênh {channel_id} đã có trong {seed_file.name}, bỏ qua.")
        return False

    with open(seed_file, 'a', encoding='utf-8') as f:
        f.write(channel_id + "\n")
    log.info(f"Đã thêm {channel_id} vào {seed_file.name}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="URL kênh YouTube")
    parser.add_argument("--add", action="store_true", help="Cho phép thêm vào file seed nếu tự tin cao")
    parser.add_argument("--auto-add", action="store_true", help="Thêm ngay không cần hỏi (dùng với --add)")
    parser.add_argument("--label", choices=["slop", "genuine"], help="Ép nhãn (bỏ qua dự đoán)")
    args = parser.parse_args()

    validate()
    if not Paths.MODEL_RF.exists():
        log.error(f"Không tìm thấy model {Paths.MODEL_RF}.")
        sys.exit(1)

    log.info("Đang load model...")
    pipeline, feature_cols = load_model()

    manager = YouTubeServiceManager(YOUTUBE_API_KEYS)
    log.info(f"Đang crawl: {args.url}")
    channel_data = crawl_channel_data(manager, args.url)

    df = compute_features(channel_data)
    pred_label, prob_slop = predict(pipeline, df)

    print("\n" + "=" * 50)
    print(f"Kênh: {channel_data['title']}")
    print(f"Subscriber: {channel_data['subscriber_count']:,}")
    print(f"Videos: {channel_data['video_count']}")
    print(f"Xác suất Slop: {prob_slop:.4f}")

    if args.label:
        label_str = args.label
        label = LABEL_SLOP if label_str == "slop" else LABEL_GENUINE
        print(f"⚠️  Ép nhãn: {label_str.upper()}")
    else:
        label = pred_label
        print(f"Kết luận: {'🔴 AI SLOP' if label == 1 else '🟢 Genuine'}")
    print("=" * 50)

    if not args.add:
        return

    # Xác định có nên thêm không
    should_add = False
    if args.label:
        should_add = True   # ép nhãn thì luôn thêm
    else:
        if label == LABEL_SLOP and prob_slop >= AUTO_ADD_CONFIDENCE:
            should_add = True
        elif label == LABEL_GENUINE and prob_slop <= (1 - AUTO_ADD_CONFIDENCE):
            should_add = True

    if not should_add:
        log.info("Độ tự tin không đủ cao để tự động thêm (cần >= 0.95).")
        return

    if not args.auto_add:
        answer = input(f"Thêm kênh này vào seed ({'slop' if label==1 else 'genuine'})? (y/n): ")
        if answer.lower() != 'y':
            log.info("Đã hủy.")
            return

    add_to_seed(channel_data["channel_id"], label)


if __name__ == "__main__":
    main()