"""
main.py — Entry point cho toàn bộ pipeline ICF Detector
==========================================================
Sử dụng:
    python main.py all              # Chạy toàn bộ pipeline hiện tại
    python main.py crawl            # Chỉ crawl/expand
    python main.py features         # Tính features
    python main.py train            # Huấn luyện mô hình
    python main.py predict URL      # Dự đoán 1 kênh

Sau khi có kết quả EDA & so sánh mô hình, script sẽ được cập nhật thêm.
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"


def run_script(script_name: str, args: list = None):
    """Chạy một script Python trong src/ với tham số."""
    cmd = [sys.executable, str(SRC / script_name)]
    if args:
        cmd.extend(args)
    print(f"\n▶ Chạy: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    extra_args = sys.argv[2:]

    if command == "all":
        print("=== ICF Detector Pipeline ===")
        print("1. Crawl mở rộng dataset...")
        run_script("expand.py", ["--max-slops", "20", "--max-genuine", "20"])
        run_script("crawl_expand.py")

        print("2. Tính features...")
        run_script("features.py")

        print("3. Huấn luyện mô hình...")
        run_script("train.py")

        print("4. (Tùy chọn) Chạy so sánh mô hình...")
        try:
            run_script("compare_models.py")
        except FileNotFoundError:
            print("⚠️  compare_models.py chưa sẵn sàng, bỏ qua.")

        print("\n✅ Pipeline hoàn tất!")

    elif command == "crawl":
        print("Chạy mở rộng & crawl dữ liệu...")
        run_script("expand.py", ["--max-slops", "20", "--max-genuine", "20"])
        run_script("crawl_expand.py")

    elif command == "features":
        run_script("features.py")

    elif command == "train":
        run_script("train.py")

    elif command == "predict":
        if not extra_args:
            print("Vui lòng cung cấp URL kênh YouTube.")
            print("VD: python main.py predict https://www.youtube.com/@Example")
            sys.exit(1)
        run_script("predict.py", extra_args)

    elif command == "eda":
        print("Chức năng EDA sẽ được bổ sung sau khi bạn [Tên bạn 1] hoàn thành notebook.")
        # Sau khi có notebook, có thể chạy: subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", "notebooks/EDA.ipynb"])
        pass

    elif command == "compare":
        print("Chức năng so sánh mô hình sẽ được bổ sung sau khi bạn [Tên bạn 2] hoàn thành.")
        # run_script("compare_models.py")  # đã có sẵn, có thể kích hoạt ngay
        run_script("compare_models.py")   # tạm thời gọi script có sẵn

    else:
        print(f"Lệnh không xác định: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()