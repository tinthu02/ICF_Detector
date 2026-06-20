"""
config.py — Cấu hình trung tâm cho đồ án ICF Detector
=======================================================
Tất cả các file khác đều import từ đây.
KHÔNG hardcode API key, path, hay hyperparameter ở chỗ khác.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env ──────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


# ══════════════════════════════════════════════════════════════════════════════
# 1. API keys
# ══════════════════════════════════════════════════════════════════════════════

# Key đơn (dùng cho các script crawl cũ)
YOUTUBE_API_KEY: str = os.getenv("YOUTUBE_API_KEY", "")

# Danh sách nhiều key (dùng trong expand.py, tự động xoay vòng)
_API_KEYS_RAW = os.getenv("YOUTUBE_API_KEYS", "")
if _API_KEYS_RAW:
    # Tách theo dấu phẩy, loại bỏ khoảng trắng thừa
    YOUTUBE_API_KEYS = [k.strip() for k in _API_KEYS_RAW.split(",") if k.strip()]
else:
    # Nếu không có danh sách, dùng key đơn (nếu có)
    YOUTUBE_API_KEYS = [YOUTUBE_API_KEY] if YOUTUBE_API_KEY else []

# Đảm bảo tương thích: nếu danh sách rỗng, thử lấy từ YOUTUBE_API_KEY một lần nữa
if not YOUTUBE_API_KEYS and YOUTUBE_API_KEY:
    YOUTUBE_API_KEYS = [YOUTUBE_API_KEY]

API_SEARCH_MAX_RESULTS: int = 50
VIDEOS_PER_CHANNEL: int = 50


# ══════════════════════════════════════════════════════════════════════════════
# 2. Đường dẫn (Paths)
# ══════════════════════════════════════════════════════════════════════════════

class Paths:
    ROOT        = _PROJECT_ROOT
    SRC         = ROOT / "src"
    DATA        = ROOT / "data"
    NOTEBOOKS   = ROOT / "notebooks"
    MODELS      = ROOT / "models"

    RAW         = DATA / "raw"
    COLLECTED   = DATA / "collected"
    PROCESSED   = DATA / "processed"

    # ── Seed files (nhiều file) ────────────────────────────────────────────
    SEED_SLOP_FILES = [
        RAW / "AI slop.txt",
        RAW / "AI slop expand.txt",
    ]
    SEED_GENUINE_FILES = [
        RAW / "non AI.txt",
        RAW / "non AI expand.txt",
        RAW / "non AI expand 2.txt",
    ]

    # Đường dẫn nhanh đến file chính đầu tiên (tương thích ngược)
    SEED_SLOP    = RAW / "AI slop.txt"
    SEED_GENUINE = RAW / "non AI.txt"

    # Output
    CHANNELS_RAW = COLLECTED / "channels_raw.csv"
    FEATURES_FINAL = PROCESSED / "features_final.csv"
    INTERVALS_FILE = PROCESSED / "intervals.csv"

    # Saved models
    MODEL_RF    = MODELS / "random_forest.pkl"
    MODEL_IF    = MODELS / "isolation_forest.pkl"


# ══════════════════════════════════════════════════════════════════════════════
# 3. Labels
# ══════════════════════════════════════════════════════════════════════════════

LABEL_SLOP    = 1
LABEL_GENUINE = 0


# ══════════════════════════════════════════════════════════════════════════════
# 4. Feature Engineering
# ══════════════════════════════════════════════════════════════════════════════

MIN_UPLOAD_INTERVAL_DAYS: float = 0.5
SIMILARITY_WINDOW: int = 10
SIMILARITY_HIGH_THRESHOLD: float = 0.8
TFIDF_MAX_FEATURES: int = 5000

# Danh sách feature tĩnh (chưa gồm if_anomaly_score – sẽ được pipeline thêm)
FEATURE_COLS_STATIC = [
    "time_interval_std",
    "upload_burst_ratio",
    "video_upload_frequency",
    "view_per_video",
    "dash_density",
    "title_length_std",
    "capitalization_ratio",
    "opening_repeat_ratio",
    "temporal_clickbait_ratio",
    "type_token_ratio",
    "avg_title_similarity",
    "sub_to_view_ratio",
    "subscriber_velocity",
    # "sub_to_view_velocity_ratio",   # đã loại bỏ
]


# ══════════════════════════════════════════════════════════════════════════════
# 5. Anomaly Detection (Isolation Forest)
# ══════════════════════════════════════════════════════════════════════════════

IF_CONTAMINATION: float = 0.15
IF_N_ESTIMATORS : int   = 100
IF_RANDOM_STATE : int   = 42


# ══════════════════════════════════════════════════════════════════════════════
# 6. Model (Random Forest)
# ══════════════════════════════════════════════════════════════════════════════

RF_N_ESTIMATORS : int   = 200
RF_MAX_DEPTH    : int   = 8
RF_RANDOM_STATE : int   = 42
RF_TEST_SIZE    : float = 0.2
RF_CV_FOLDS     : int   = 5


# ══════════════════════════════════════════════════════════════════════════════
# 7. Seed Expansion (expand.py)
# ══════════════════════════════════════════════════════════════════════════════

EXPANSION_AMBIGUOUS_THRESHOLD: float = 0.3
EXPANSION_MAX_CANDIDATES: int = 200


# ══════════════════════════════════════════════════════════════════════════════
# 8. Auto Expand (expand.py) — phân tách theo ngôn ngữ
# ══════════════════════════════════════════════════════════════════════════════

# Điều chỉnh ngưỡng slop chính
AUTO_EXPAND_CONFIDENCE_THRESHOLD = 0.92   # giảm nhẹ từ 0.95

# Ngưỡng slop tuyệt đối (luôn lấy)
AUTO_EXPAND_SLOP_HIGH_CONFIDENCE = 0.97

# Từ khóa loại trừ (nhạc/ASMR)
AUTO_EXPAND_SLOP_EXCLUDE_KEYWORDS = [
    "rain sound", "sleep music", "relaxing white noise", "lofi hip hop",
    "meditation music", "asmr", "deep sleep", "10 hours",
]
# Ngưỡng riêng cho genuine (thấp hơn để thu thập được nhiều hơn)
AUTO_EXPAND_GENUINE_CONFIDENCE_THRESHOLD = 0.15   # prob <= 0.15 → tự tin là genuine

AUTO_EXPAND_MAX_NEW_SLOP    = 50
AUTO_EXPAND_MAX_NEW_GENUINE = 50

AUTO_EXPAND_MIN_SUBSCRIBERS = 1000
AUTO_EXPAND_MIN_VIDEOS = 10

AUTO_EXPAND_MIN_VIEWS = 10000   # tổng view tối thiểu

# Mở rộng query slop tiếng Việt
AUTO_EXPAND_SLOP_QUERIES_VI = [
    "sự thật kinh hoàng về", "bí ẩn ít ai biết về", "giải mã bí ẩn vũ trụ", "lịch sử nhân loại chưa kể",
    "tư duy của người giàu", "quy luật ngầm của cuộc sống", "bí mật tài chính", "bài học đắt giá",
    "tóm tắt phim review", "cái kết bất ngờ của bộ phim", "review phim kinh dị",
    "truyện cổ tích kinh dị", "sự thật không ngờ về vị vua", "sẽ thế nào nếu",
    "nhạc lofi chill giảm stress", "âm thanh mưa ngủ ngon 8 tiếng",
    "câu chuyện có thật", "sự thật đáng sợ", "top 10 bí ẩn", "kể chuyện đêm khuya",
    "bói toán", "bí mật tâm linh",
]

# Mở rộng query slop tiếng Anh
AUTO_EXPAND_SLOP_QUERIES_EN = [
    "shocking facts you won't believe", "the truth about ancient", "mysteries explained simply",
    "what nobody tells you about", "rules of the rich", "the dark truth about money",
    "habits of successful people", "movie recap and review", "ending explained movie",
    "scary bedtime stories", "what would happen if", "unsolved mysteries of the world",
    "lofi hip hop radio to relax", "deep sleep rain sounds 10 hours",
    "unsolved mystery", "scary true stories", "top 10 creepiest", "real ghost stories",
    "psychic reading", "horror story compilation",
]

AUTO_EXPAND_GENUINE_QUERIES_VI = [
    # 1. Vlog đời sống / Ẩm thực (Quay camera thực tế)
    "vlog cuộc sống đời thường", "nhật ký hàng ngày", "nấu ăn tại nhà", "mukbang ăn sập",
    # 2. Thử thách & Giải trí / Du lịch trải nghiệm
    "thử thách 24h trốn trong", "vlog du lịch tự túc", "review quán ăn vỉa hè", "phóng sự thực tế",
    # 3. Học tập & Công nghệ / DIY (Có tương tác vật lý)
    "học cùng tôi study with me", "đập hộp công nghệ", "hướng dẫn tự làm diy", "decor phòng ngủ giá rẻ",
    # 4. Thú cưng / Hài kịch (Ghi hình đời thực)
    "một ngày nuôi chó mèo", "tiểu phẩm hài hước", "phỏng vấn đường phố"
]

AUTO_EXPAND_GENUINE_QUERIES_EN = [
    # 1. Daily Vlog / Cooking
    "daily life vlog", "day in my life vlog", "cooking at home recipe", "clean with me",
    # 2. Travel & Mukbang / Entertainment Challenges
    "24 hours challenge in", "traveling solo to", "street food tour", "honest review",
    # 3. Tech / DIY / Education
    "tech unboxing and setup", "homemade diy woodworking", "study with me live", "room makeover transformation",
    # 4. Gaming / Pets (Có giọng nói thật và tương tác webcam)
    "gameplay walkthrough no commentary", "funny cat videos compilation", "reaction to trend"
]

# Các query có nội dung dài (nhạc, ASMR) → chỉ lấy 10 video để tiết kiệm quota
AUTO_EXPAND_LOW_RESULT_QUERIES = {
    "nhạc lofi chill giảm stress",
    "âm thanh mưa ngủ ngon 8 tiếng",
    "lofi hip hop radio to relax",
    "deep sleep rain sounds 10 hours",
}

# ══════════════════════════════════════════════════════════════════════════════
# 9. Validation
# ══════════════════════════════════════════════════════════════════════════════

def validate() -> None:
    errors = []

    if not YOUTUBE_API_KEYS or all(not k for k in YOUTUBE_API_KEYS):
        errors.append(
            "Cần ít nhất một API key.\n"
            "  → Đặt YOUTUBE_API_KEY (key đơn) hoặc YOUTUBE_API_KEYS (danh sách, cách nhau dấu phẩy) trong file .env"
        )

    # Kiểm tra tất cả file seed
    for f in Paths.SEED_SLOP_FILES:
        if not f.exists():
            errors.append(f"Không tìm thấy seed file slop: {f}")
    for f in Paths.SEED_GENUINE_FILES:
        if not f.exists():
            errors.append(f"Không tìm thấy seed file genuine: {f}")

    # Tạo thư mục nếu chưa có
    for folder in [Paths.COLLECTED, Paths.PROCESSED, Paths.MODELS]:
        folder.mkdir(parents=True, exist_ok=True)

    if errors:
        raise EnvironmentError(
            "\n\n[config] Lỗi cấu hình:\n" + "\n".join(f"  • {e}" for e in errors)
        )

    # In thông tin seed files và API keys
    total_slop = sum(_count_lines(f) for f in Paths.SEED_SLOP_FILES if f.exists())
    total_genuine = sum(_count_lines(f) for f in Paths.SEED_GENUINE_FILES if f.exists())

    def file_list_str(files):
        parts = []
        for f in files:
            if f.exists():
                parts.append(f"{f.name} ({_count_lines(f)} kênh)")
        return ", ".join(parts) if parts else "Không có"

    # Hiển thị thông tin key (ẩn nội dung thật)
    key_info = f"{len(YOUTUBE_API_KEYS)} key(s)"
    if YOUTUBE_API_KEYS:
        first_key = YOUTUBE_API_KEYS[0]
        key_info += f" (đầu tiên: {first_key[:8]}...{first_key[-4:]})"

    print("[config] ✓ Cấu hình hợp lệ.")
    print(f"  PROJECT_ROOT : {Paths.ROOT}")
    print(f"  API keys     : {key_info}")
    print(f"  Seed slop    : {file_list_str(Paths.SEED_SLOP_FILES)} (tổng {total_slop} kênh)")
    print(f"  Seed genuine : {file_list_str(Paths.SEED_GENUINE_FILES)} (tổng {total_genuine} kênh)")


def _count_lines(path: Path) -> int:
    """Đếm số dòng không rỗng trong file txt."""
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


# ── Quick self-test khi chạy trực tiếp ────────────────────────────────────────
if __name__ == "__main__":
    validate()