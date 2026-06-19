"""
crawl_auto_expand.py — Cào thông tin & video cho các kênh mới từ toàn bộ seed files
— Cào thông tin & video cho các kênh mới từ toàn bộ seed files
(đã hỗ trợ xoay vòng nhiều API key)
- Đọc tất cả file seed slop và genuine (gốc + expand).
- Chỉ lấy các dòng có channel ID hợp lệ (UC...), bỏ qua URL cũ.
- Tạo seed đúng cấu trúc (có 'parsed' từ parse_url) để tương thích crawl_channel.
- So sánh với channels_raw.csv, chỉ crawl những kênh chưa có.
- Gán label chính xác (1: slop, 0: genuine) dựa vào file nguồn.
- Bổ sung vào channels_raw.csv.
"""

import sys, re, time, logging, json
from pathlib import Path
import pandas as pd
from googleapiclient.errors import HttpError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    Paths, LABEL_SLOP, LABEL_GENUINE,
    YOUTUBE_API_KEYS,   # danh sách key
    validate
)
from src.crawl import crawl_channel, parse_url
from src.expand import YouTubeServiceManager   # tái sử dụng class quản lý key

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

OUTPUT_CSV = Paths.CHANNELS_RAW


def extract_channel_id(line: str) -> str:
    line = line.strip()
    if not line:
        return None
    if re.match(r'^UC[\w-]{22}$', line):
        return line
    m = re.search(r'channel/(UC[\w-]+)', line)
    if m:
        return m.group(1)
    return None


def load_seeds_with_labels():
    id_to_label = {}
    for f in Paths.SEED_SLOP_FILES:
        if f.exists():
            with open(f, 'r', encoding='utf-8') as fh:
                for line in fh:
                    cid = extract_channel_id(line)
                    if cid:
                        id_to_label[cid] = LABEL_SLOP
    for f in Paths.SEED_GENUINE_FILES:
        if f.exists():
            with open(f, 'r', encoding='utf-8') as fh:
                for line in fh:
                    cid = extract_channel_id(line)
                    if cid:
                        id_to_label[cid] = LABEL_GENUINE
    return id_to_label


def get_existing_ids(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    if "channel_id" not in df.columns:
        return set()
    return set(df["channel_id"].dropna().astype(str))


def crawl_channel_with_manager(manager, seed: dict):
    """
    Gọi crawl_channel, nếu gặp lỗi quota thì manager tự động đổi key và thử lại.
    """
    def _crawl():
        # Lấy service hiện tại từ manager
        yt = manager.get_service()
        return crawl_channel(yt, seed)
    return manager.handle_quota_error(_crawl)


def main():
    validate()

    seed_label_map = load_seeds_with_labels()
    log.info(f"Tổng ID kênh từ tất cả file seed: {len(seed_label_map)}")

    existing_ids = get_existing_ids(OUTPUT_CSV)
    log.info(f"Số kênh đã có trong CSV: {len(existing_ids)}")

    to_crawl = {cid: lab for cid, lab in seed_label_map.items() if cid not in existing_ids}
    if not to_crawl:
        log.info("Tất cả kênh đã được crawl. Hoàn tất.")
        return

    log.info(f"Phát hiện {len(to_crawl)} kênh mới cần crawl.")

    # Đọc dữ liệu cũ
    if OUTPUT_CSV.exists():
        df_old = pd.read_csv(OUTPUT_CSV)
        rows = df_old.to_dict("records")
    else:
        rows = []

    # Khởi tạo manager với danh sách key
    manager = YouTubeServiceManager(YOUTUBE_API_KEYS)

    new_count = 0
    fail_count = 0

    for i, (cid, label) in enumerate(sorted(to_crawl.items()), 1):
        log.info(f"[{i}/{len(to_crawl)}] Crawl {cid} (label={label})")
        raw_url = f"https://www.youtube.com/channel/{cid}"
        parsed = parse_url(raw_url)
        if parsed is None:
            log.error(f"Không parse được URL {raw_url}")
            fail_count += 1
            continue

        seed = {
            "parsed": parsed,
            "label": label,
            "raw_url": raw_url
        }

        try:
            row = crawl_channel_with_manager(manager, seed)
        except Exception as e:
            log.error(f"Lỗi không thể phục hồi: {e}")
            fail_count += 1
            continue

        if row is None:
            log.warning(f"Không crawl được {cid}")
            fail_count += 1
            continue

        row["label"] = label
        rows.append(row)
        new_count += 1

        if new_count % 10 == 0:
            pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
            log.info(f"Checkpoint: đã lưu {len(rows)} dòng")

        time.sleep(0.5)

    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    log.info(f"Hoàn tất! Thêm {new_count} kênh mới. ({fail_count} lỗi)")
    log.info(f"Tổng số kênh trong CSV: {len(rows)}")


if __name__ == "__main__":
    main()