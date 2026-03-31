import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Khởi tạo các Services
from services.vision_service import VisionService
from services.chat_service import CookingLangChainService

load_dotenv()

app = Flask(__name__)
CORS(app) # Cho phép Backend khác port gọi qua

print("====================================")
print(" KHỞI ĐỘNG FOOD AI SERVICE ")
print("====================================")

# Khởi tạo Models (Chỉ chạy 1 lần lúc bật server để tối ưu RAM)
vision_model = VisionService()
ai_assistant = CookingLangChainService()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "AI Service is active and running!"}), 200

@app.route('/api/ai/analyze-image', methods=['POST'])
def analyze_image():
    # Kiểm tra xem có file ảnh gửi kèm không
    if 'image' not in request.files:
        return jsonify({"success": False, "message": "Missing image file"}), 400

    file = request.files['image']
    image_bytes = file.read()

    try:
        # Bước 1: Computer Vision (YOLO + ResNet)
        detected_items = vision_model.predict_image(image_bytes)
        
        if not detected_items:
            return jsonify({
                "success": True,
                "data": {
                    "ingredients": [],
                    "ai_suggestion": None
                },
                "message": "Không nhận diện được nguyên liệu nào."
            }), 200

        # Bước 2: AI Reasoning (LangChain + OpenAI)
        ingredient_names = [item['name'] for item in detected_items]
        suggestion = ai_assistant.get_suggestion(ingredient_names)

        # Bước 3: Trả về kết quả JSON tổng hợp
        return jsonify({
            "success": True,
            "data": {
                "ingredients": detected_items,
                "ai_suggestion": suggestion
            },
            "message": "Phân tích hình ảnh và tư vấn thành công!"
        }), 200

    except Exception as e:
        print(f"Lỗi Server: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

if __name__ == '__main__':
    # Đọc port từ file .env, mặc định là 8000
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port, debug=False) # Tắt debug khi chạy mô hình nặng