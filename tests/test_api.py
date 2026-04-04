from fastapi.testclient import TestClient
from unittest.mock import patch
from backend.main import app

client = TestClient(app)


@patch("backend.api.router.process_chat_stream")
def test_chat_endpoint_success(mock_process_chat_stream):
    # Setup mock async generator response
    async def mock_stream(*args, **kwargs):
        yield "你好，"
        yield "我是厦小嘉！"

    mock_process_chat_stream.side_effect = mock_stream

    # Perform request
    response = client.post(
        "/api/chat",
        json={
            "messages": [{"role": "user", "content": "你好"}],
            "top_k": 3,
            "rerank_top_k": 1,
        },
    )

    # Assertions
    assert response.status_code == 200
    assert response.text == "你好，我是厦小嘉！"
    mock_process_chat_stream.assert_called_once()


def test_chat_endpoint_empty_messages():
    # Perform request with empty messages array
    response = client.post("/api/chat", json={"messages": []})

    # Assertions
    assert response.status_code == 400
    assert response.json()["detail"] == "Messages cannot be empty"


def test_chat_endpoint_invalid_payload():
    # Perform request with missing messages field
    response = client.post("/api/chat", json={"top_k": 5})

    # Assertions
    assert response.status_code == 422  # Unprocessable Entity
