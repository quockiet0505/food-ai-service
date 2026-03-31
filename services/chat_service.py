import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

class CookingLangChainService:
    def __init__(self):
        print(" [LangChain] Đang khởi tạo AI Assistant...")

        api_key = os.getenv("OPENAI_API_KEY")
        print(" API KEY:", api_key[:10] + "..." if api_key else " NONE")

        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2
        )

        self.prompt = PromptTemplate(
            template="""
Bạn là một đầu bếp.

Dựa vào danh sách công thức và nguyên liệu hiện có, hãy chọn 1 món phù hợp nhất.

[DANH SÁCH CÔNG THỨC]:
{recipes}

[NGUYÊN LIỆU HIỆN CÓ]:
{ingredients}

YÊU CẦU:
- Chỉ chọn món trong danh sách
- Nếu nguyên liệu gần giống (ví dụ: "Thịt gà" ≈ "Gà nguyên con") thì vẫn coi là khớp

Trả lời đúng format:

Tên món: <tên>
Lý do: <ngắn gọn>
Nguyên liệu có: <liệt kê>
Nguyên liệu thiếu: <liệt kê>
""",
            input_variables=["recipes", "ingredients"]
        )

        self.recipes_data = self._load_recipes_json()

    def _load_recipes_json(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(base_dir, 'data', 'data.json')
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f" Không đọc được data.json: {e}")
            return "[]"

    def _parse_response(self, text):
        try:
            lines = text.split("\n")

            result = {
                "name": "",
                "description": "",
                "matched_ingredients": [],
                "missing_ingredients": []
            }

            for line in lines:
                line = line.strip()

                if line.startswith("Tên món"):
                    result["name"] = line.split(":", 1)[1].strip()

                elif line.startswith("Lý do"):
                    result["description"] = line.split(":", 1)[1].strip()

                elif line.startswith("Nguyên liệu có"):
                    items = line.split(":", 1)[1]
                    result["matched_ingredients"] = [i.strip() for i in items.split(",") if i.strip()]

                elif line.startswith("Nguyên liệu thiếu"):
                    items = line.split(":", 1)[1]
                    result["missing_ingredients"] = [i.strip() for i in items.split(",") if i.strip()]

            return result

        except Exception as e:
            print(" Parse lỗi:", e)
            return None

    def get_suggestion(self, detected_ingredients: list) -> dict:
        if not detected_ingredients:
            print(" Không có nguyên liệu đầu vào")
            return None

        ingredients_str = ", ".join(detected_ingredients)

        try:
            print(f" INPUT INGREDIENTS: {ingredients_str}")

            formatted_prompt = self.prompt.format(
                recipes=self.recipes_data,
                ingredients=ingredients_str
            )

            print("\n====== PROMPT ======")
            print(formatted_prompt[:500])
            print("====================\n")

            response = self.llm.invoke(formatted_prompt)

            print("\n====== RAW LLM OUTPUT ======")
            print(response.content)
            print("============================\n")

            parsed = self._parse_response(response.content)

            print(" PARSED:", parsed)

          
            if not parsed or not parsed["name"]:
                print(" Parse fail → fallback")

                return {
                    "name": "Không xác định",
                    "description": response.content,
                    "matched_ingredients": detected_ingredients,
                    "missing_ingredients": []
                }

            return parsed

        except Exception as e:
            print(f" Lỗi LangChain: {e}")

            return {
                "name": "Lỗi AI",
                "description": str(e),
                "matched_ingredients": detected_ingredients,
                "missing_ingredients": []
            }