# Импорт библиотек
import os
import pytest
from fastapi.testclient import TestClient
from st_app_1 import app

# Создание клиента для тестирования
client = TestClient(app)

# Тестируем endpoint /run_model/ с изображением
def test_run_model():
    test_image_path = 'data'  # Путь тестовый
    with open(test_image_path, "rb") as file:
        files = {'path': file,}
        response = client.post("/run_model/", data=files)
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/csv"
