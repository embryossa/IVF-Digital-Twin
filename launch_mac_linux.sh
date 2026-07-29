#!/bin/bash
# IVF Digital Twin v6.2 — launcher for Mac/Linux

echo ""
echo "============================================="
echo "  IVF DIGITAL TWIN v6.2"
echo "  Sergeev et al."
echo "============================================="
echo ""

# Переходим в папку скрипта
cd "$(dirname "$0")"

# Проверяем Python
if ! command -v python3 &> /dev/null; then
    echo "[ОШИБКА] Python3 не установлен"
    echo "Mac:   brew install python"
    echo "Linux: sudo apt install python3 python3-pip"
    exit 1
fi

# Зависимости
echo "Проверка зависимостей..."
pip3 install -r requirements.txt -q 2>/dev/null

# Запуск
echo "Запуск на http://localhost:8501 ..."
echo ""
python3 -m streamlit run app.py \
    --server.headless false \
    --server.port 8501 \
    --browser.gatherUsageStats false \
    --theme.primaryColor "#1B4F72" \
    --theme.backgroundColor "#f8fafc" \
    --theme.secondaryBackgroundColor "#e8f4f8" \
    --theme.textColor "#1a3a4a"
