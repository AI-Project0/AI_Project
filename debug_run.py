import sys
import os

# 確保當前路徑在 Python 搜尋路徑中
sys.path.append(os.getcwd())

print("🔍 開始診斷後端程式...")
print(f"📂 當前工作目錄: {os.getcwd()}")

try:
    print("⏳ 正在嘗試匯入 app.main...")
    from app.main import app
    print("✅ 成功匯入 app.main！後端邏輯看起來沒問題。")
    
    print("🚀 嘗試啟動 Uvicorn...")
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

except ImportError as e:
    print("\n❌ 匯入錯誤 (Import Error):")
    print(e)
    print("💡這通常是因為缺少套件，或 requirements.txt 安裝不完全。")
except SyntaxError as e:
    print("\n❌ 語法錯誤 (Syntax Error):")
    print(e)
    print("💡這代表程式碼有打錯字。")
except Exception as e:
    print("\n❌ 未知錯誤 (General Error):")
    import traceback
    traceback.print_exc()
    