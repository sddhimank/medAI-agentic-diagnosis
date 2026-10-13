import os
import shutil
import base64
import logging
import uuid
import asyncio
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from app.core.processors import process_medical_scan
from app.core.agents import run_medical_agent_workflow
from app.templates.report_gen import generate_pdf_report

class ChronologicalLogFilter(logging.Filter):
    def filter(self, record):
        if "GET /status/" in record.getMessage() or "GET /reports/" in record.getMessage():
            return False
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record.msg = f"[{now}] {record.msg}"
        return True

for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
    sys_logger = logging.getLogger(logger_name)
    sys_logger.addFilter(ChronologicalLogFilter())

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="MedAI Strict Sequential Analytics Backend", version="1.3.0")

UPLOAD_DIR = "/tmp/medical_uploads"
REPORTS_DIR = "/tmp/medical_reports"
ANALYSIS_JOBS = {}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# FIXED: Strict FIFO asynchronous queue to throttle CPU resource allocation
TASK_QUEUE = asyncio.Queue()

def blocking_agent_calculation(task_id: str, file_path: str, filename: str, processed_img_path: str, annotated_img_path: str, clinical_metadata: dict):
    """Executes heavy matrix math inside an isolated OS thread."""
    try:
        logger.info(f"[ENGINE] Processing Task ID: {task_id} exclusively on CPU threads.")
        agent_assessment = run_medical_agent_workflow(processed_img_path, clinical_metadata)
        
        pdf_filename = f"Report_{task_id}.pdf"
        pdf_path = os.path.join(REPORTS_DIR, pdf_filename)
        
        generate_pdf_report(pdf_path, agent_assessment, clinical_metadata, annotated_img_path)
        
        ANALYSIS_JOBS[task_id] = {
            "status": "completed",
            "findings": agent_assessment["findings"],
            "diagnosis": agent_assessment["diagnosis"],
            "remedies": agent_assessment["remedies"],
            "internal_pdf_url": f"http://agent-api:8000/reports/{pdf_filename}"
        }
        logger.info(f"✓ Task ID {task_id} finalized.")
    except Exception as e:
        logger.error(f"❌ Execution crashed for Task ID {task_id}: {str(e)}")
        ANALYSIS_JOBS[task_id] = {"status": "failed", "error": str(e)}
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

async def queue_worker_loop():
    """Permanent background worker loop that processes tasks one at a time."""
    logger.info("[QUEUE MANAGER] Strict sequential processing worker initialized.")
    while True:
        task_data = await TASK_QUEUE.get()
        task_id = task_data["task_id"]
        
        # Safely offload to thread, blocking this worker loop from picking up task 2 early
        await asyncio.to_thread(
            blocking_agent_calculation,
            task_id, task_data["file_path"], task_data["filename"],
            task_data["processed_img_path"], task_data["annotated_img_path"],
            task_data["clinical_metadata"]
        )
        TASK_QUEUE.task_done()

@app.on_event("startup")
async def startup_event():
    # Spin up the permanent single-threaded consumer loop on app boot
    asyncio.create_task(queue_worker_loop())

@app.post("/preview")
async def generate_preview(file: UploadFile = File(...)):
    clean_name = file.filename.strip().strip('"').strip("'")
    filename_lower = clean_name.lower()
    
    if not (filename_lower.endswith('.dcm') or filename_lower.endswith('.png') or filename_lower.endswith('.jpg') or filename_lower.endswith('.jpeg')):
        raise HTTPException(status_code=400, detail="Unsupported format.")

    temp_id = str(uuid.uuid4())[:8]
    file_path = os.path.join(UPLOAD_DIR, f"prev_{temp_id}_{clean_name}")
    try:
        content = await file.read()
        with open(file_path, "wb") as buffer:
            buffer.write(content)
            
        processed_img_path = os.path.join(UPLOAD_DIR, f"proc_{temp_id}.png")
        annotated_img_path = os.path.join(UPLOAD_DIR, f"annotated_{temp_id}.png")
        
        _, clinical_metadata = process_medical_scan(file_path, processed_img_path, annotated_img_path)
        
        with open(annotated_img_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
            
        if os.path.exists(processed_img_path): os.remove(processed_img_path)
        if os.path.exists(annotated_img_path): os.remove(annotated_img_path)
            
        return {"preview_img": encoded_string, "metadata": clinical_metadata}
    except Exception as e:
        logger.error(f"Preview extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/analyze")
async def start_analysis(file: UploadFile = File(...)):
    clean_name = file.filename.strip().strip('"').strip("'")
    task_id = str(uuid.uuid4())
    logger.info(f"[ROUTER] Ingesting request. Enqueueing Task ID: {task_id}")
    
    file_path = os.path.join(UPLOAD_DIR, f"src_{task_id}_{clean_name}")
    try:
        content = await file.read()
        with open(file_path, "wb") as buffer:
            buffer.write(content)
            
        processed_img_path = os.path.join(UPLOAD_DIR, f"proc_{task_id}.png")
        annotated_img_path = os.path.join(UPLOAD_DIR, f"annotated_{task_id}.png")
        
        _, clinical_metadata = process_medical_scan(file_path, processed_img_path, annotated_img_path)
        ANALYSIS_JOBS[task_id] = {"status": "processing"}
        
        # Push parameters into the strict sequential loop
        await TASK_QUEUE.put({
            "task_id": task_id, "file_path": file_path, "filename": clean_name,
            "processed_img_path": processed_img_path, "annotated_img_path": annotated_img_path,
            "clinical_metadata": clinical_metadata
        })
        return {"status": "queued", "task_id": task_id}
    except Exception as e:
        logger.error(f"Analysis trigger failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{task_id}")
async def check_status(task_id: str):
    return ANALYSIS_JOBS.get(task_id, {"status": "not_found"})

@app.get("/reports/{filename}")
async def get_report(filename: str):
    file_path = os.path.join(REPORTS_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(file_path, media_type="application/pdf")