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
    response = client.post("/run_model/", data=files)
    print(response.content)
    assert response.headers["content-type"] == 'application/json'
    assert response.status_code == 200
    
