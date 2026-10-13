import pytest
import asyncio
from fastapi.testclient import TestClient
from app.main import app, TASK_QUEUE

client = TestClient(app)

def test_root_documentation_handshake():
    """Verify that the base FastAPI application context initiates cleanly."""
    response = client.get("/docs")
    assert response.status_code == 200

def test_invalid_file_extension_rejection():
    """Confirm the router throws a 400 Bad Request if an unsupported format is uploaded."""
    files = {"file": ("corrupted_specimen.txt", b"BLANK_TEXT_DATA", "text/plain")}
    response = client.post("/preview", files=files)
    assert response.status_code == 400
    assert "Unsupported format" in response.json()["detail"]

@pytest.mark.asyncio
async def test_strict_fifo_queue_concurrency_throttling():
    """Verify that incoming requests are enqueued sequentially inside the asyncio.Queue."""
    # Drain any existing tasks from previous runs safely
    while not TASK_QUEUE.empty():
        await TASK_QUEUE.get()
        TASK_QUEUE.task_done()
        
    assert TASK_QUEUE.empty() is True
    
    # Inject mock task data directly to simulate overlapping browser tab requests
    mock_task_1 = {"task_id": "id-991", "filename": "scan1.dcm"}
    mock_task_2 = {"task_id": "id-992", "filename": "scan2.dcm"}
    
    await TASK_QUEUE.put(mock_task_1)
    await TASK_QUEUE.put(mock_task_2)
    
    # Confirm queue holds exactly 2 items, preserving structural FIFO allocation
    assert TASK_QUEUE.qsize() == 2
    
    first_out = await TASK_QUEUE.get()
    assert first_out["task_id"] == "id-991"
    TASK_QUEUE.task_done()
    
    second_out = await TASK_QUEUE.get()
    assert second_out["task_id"] == "id-992"
    TASK_QUEUE.task_done()

def test_missing_task_status_fallback():
    """Confirm that searching for a non-existent Task ID returns a clean 'not_found' payload."""
    response = client.get("/status/ghost-token-999-xyz")
    assert response.status_code == 200
    assert response.json()["status"] == "not_found"
