import requests
import json
import os

# ==========================================
# 1. CẤU HÌNH
# ==========================================
# Cổng 8000 
API_URL = "http://127.0.0.1:8000/api/ai/analyze-image"

# Đường dẫn tới tấm ảnh bạn muốn test
IMAGE_PATH = r"D:\Information Technology\NCKH\App\food-ai-service\data\images\ga.jpg"

# ==========================================
# 2. HÀM GỬI YÊU CẦU
# ==========================================
def test_upload_image():
    print("=" * 50)
    print(" BẮT ĐẦU CHẠY SCRIPT TEST API")
    print("=" * 50)
    
    if not os.path.exists(IMAGE_PATH):
        print(f" LỖI: Không tìm thấy file ảnh tại đường dẫn:\n   {IMAGE_PATH}")
        print("   -> Vui lòng sửa lại biến IMAGE_PATH trong file test_api.py")
        return

    print(f" Đang gửi file: {IMAGE_PATH}")
    print(f" Tới địa chỉ : {API_URL}")
    print(" Đợi một lát để AI xử lý...\n")
    
    try:
        # Mở file ảnh dưới dạng nhị phân (binary)
        with open(IMAGE_PATH, 'rb') as img_file:
            # Đóng gói thành form-data (giống y hệt cách React/Flutter gửi file)
            files = {'image': ('test_image.jpg', img_file, 'image/jpeg')}
            
            # Gửi HTTP POST request
            response = requests.post(API_URL, files=files)
            
            # Kiểm tra mã trạng thái trả về
            if response.status_code == 200:
                print(" KẾT QUẢ TRẢ VỀ TỪ SERVER (Thành công):")
                # Parse JSON và in ra màn hình thật đẹp
                data = response.json()
                print(json.dumps(data, indent=4, ensure_ascii=False))
            else:
                print(f" LỖI TỪ SERVER (Mã {response.status_code}):")
                print(response.text)
                
    except requests.exceptions.ConnectionError:
        print(" LỖI KẾT NỐI: Không thể gọi tới Server.")
        print("   -> Bạn đã bật file run_ai.py (Server AI) lên chưa?")
    except Exception as e:
        print(f" CÓ LỖI XẢY RA: {e}")

if __name__ == "__main__":
    test_upload_image()