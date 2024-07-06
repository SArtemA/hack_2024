# Импорт библиотек
import os
import pytest
from fastapi.testclient import TestClient
from st_app_1 import app

# Создание клиента для тестирования
client = TestClient(app)

# Тестируем endpoint /run_model/ с изображением
def test_run_model():
    # Путь тестовый
    files = {'data_pth': 'data'}
    response = client.post("/run_model/", json=files)
    print(response.content)
    assert response.status_code == 200
    assert response.headers["content-type"] == 'text/csv; charset=utf-8'
    
    
