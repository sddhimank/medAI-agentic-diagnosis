# 🩺 MedAI: Multi-Agent Medical Imaging Ingestion, Analytics & Report Generation Engine

MedAI is a production-grade, asynchronous, multi-tenant automated screening and evaluation platform designed to ingest raw **16-bit medical DICOM matrices**, instantly extract 8-bit visual previews, and run advanced multi-agent inference loops entirely on CPU infrastructure. 

Optimized for high-performance execution on a **Standard_D4as_v5 (4 vCPUs, 16 GB RAM)** instance, the system implements client-side viewport transformations, advanced OpenCV pixel extraction algorithms, and a strict thread-safe FIFO task manager queue.

---

## 📋 Table of Contents
1. [Overview](#-overview)
2. [Objective](#-objective)
3. [Components](#-components)
4. [System Design & Architecture](#-system-design--architecture)
5. [Deployment & Ingestion Verification](#-deployment--ingestion-verification)
6. [Mandatory System Disclosure](#-mandatory-system-disclosure)

---

## 🔍 Overview

Web browsers cannot natively decode or display raw 16-bit medical DICOM pixel matrices (`.dcm` files). Furthermore, traditional multi-modal Vision-Language Models (VLMs) and Large Language Models (LLMs) place heavy computational demands on systems, often causing severe core bottlenecks, context thrashing, and socket drops when deployed on shared, CPU-bound virtual machines.

**MedAI resolves these challenges through decoupled asynchronous processing:**
* **Instant Preview Layer:** Decouples heavy VLM math from file ingestion. On file upload, a dedicated micro-endpoint extracts DICOM tags, applies Hounsfield scaling, windowing, and an OpenCV thresholding engine to generate an 8-bit color-annotated preview layout in under **200 milliseconds**.
* **Strict Sequential Queue:** Forces multi-tenant requests from distinct browser tabs into a thread-safe First-In-First-Out (FIFO) queue managed by an asynchronous loop worker. This ensures that heavy 7B model calculations run one after another, maintaining system stability.
* **CPU Affinity Binding:** Parallel matrix operations utilize all **4 logical vCPUs**, optimizing token generation times on the host machine.

---

## 🎯 Objective

* **Decouple Ingestion from Inference:** Allow immediate visual validation of medical scans upon upload, eliminating loading lockups.
* **Provide Rich Workspace Interactive Layouts:** Implement real-time canvas mouse-drag panning and slider-less zoom controls directly inside the web viewport.
* **Ensure Multi-Tenant Isolation:** Securely process overlapping requests from separate browser tabs using unique Task UUID identifiers and isolated temporary filesystem storage paths.
* **Automate Document Compilation:** Generate clinical-grade, multi-page ReportLab PDF artifacts with embedded anomaly-highlighted scans, statutory warnings, and physical reviewer signing blocks.

---

## 📦 Components

The overall project structure is organized into decoupled microservices and decoupled logic modules:

| Sub-Module Component | Machine File Location Path | Operational Purpose |
| :--- | :--- | :--- |
| **`FastAPI Router`** | `app/main.py` | Hosts non-blocking endpoints, async content buffers, and the permanent FIFO task queue loop worker. |
| **`Streamlit UI`** | `ui/app.py` | Implements the web dashboard, custom HTML5/JS drag-panning canvas, micro-timer widget, and sample dropdown. |
| **`Image Preprocessor`** | `app/core/processors.py` | Extracts metadata tags, runs Hounsfield scaling, and uses OpenCV contours to highlight anomalies. |
| **`Multi-Agent Orchestrator`** | `app/core/agents.py` | Handles sequential LLM context handoffs: Stage 1 (Llava:7b visual analysis) ➔ Stage 2 (Mistral:7b clinical reasoning). |
| **`PDF Template Engine`** | `app/templates/report_gen.py` | Programmatically compiles the final multi-page document layout using ReportLab. |
| **`Ollama CPU Server`** | Containerized Daemon | Serves the quantized neural weights and controls logical core task affinity allocation constraints. |

---

## 🏗️ System Design & Architecture

MedAI operates as a microservice cluster using isolated Docker networking structures:

```text
 ┌──────────────────────────────────────────────────────────────────────────────────┐
 │                                 STREAMLIT DASHBOARD (UI)                         │
 │  ┌──────────────────────────────────────────┬─────────────────────────────────┐  │
 │  │      Instant Drag-Panning Canvas         │  Sleek Micro-Timer & Markdown   │  │
 │  └──────────────────────────────────────────┴─────────────────────────────────┘  │
 └───────────────────────┬───────────────────────────────────────────▲──────────────┘
                         │ (1) /preview JSON Payload                 │ (4) Polling Status Loop
                         ▼                                           │     /status/{task_id}
 ┌───────────────────────────────────────────────────────────────────┴──────────────┐
 │                                  FASTAPI API BACKEND                             │
 │                                                                                  │
 │    ┌─────────────────────────┐               ┌──────────────────────────────┐    │
 │    │     /preview Endpoint   │               │      /analyze Endpoint       │    │
 │    └────────────┬────────────┘               └──────────────┬───────────────┘    │
 │                 │                                           │ (2) Push Token     │
 │                 ▼                                           ▼                    │
 │     [ OpenCV Core Processor ]                     [ asyncio.Queue FIFO ]         │
 │     - 16-bit Matrix Windowing                     - Non-blocking Thread Spawn    │
 │     - Center-slice Extraction                     - Sequential Task Isolation    │
 │     - Color Contour Annotation                              │                    │
 │                                                             ▼                    │
 │                                                   [ Multi-Agent Loop Engine ]    │
 │                                                   - Stage 1: Llava:7b (VLM)      │
 │                                                   - Stage 2: Mistral:7b (LLM)    │
 │                                                             │                    │
 │                                                             ▼                    │
 │                                                   [ ReportLab Template Engine ]  │
 │                                                   - Embed Marked Scan Image      │
 │                                                   - Insert Statutory Warnings    │
 │                                                   - Append Signing Tracks        │
 └───────────────────────┬───────────────────────────────────────────▲──────────────┘
                         │                                           │
                         │ (3) Internal Vector Inference Handshake   │
                         ▼                                           │
 ┌───────────────────────────────────────────────────────────────────┴──────────────┐
 │                               OLLAMA INFERENCE DAEMON                            │
 │  [ ENV: OLLAMA_NUM_PARALLEL=1 ]               [ ENV: OLLAMA_KEEP_ALIVE=0 ]       │
 │  Forces true sequential tracking to           Enforces instant RAM cleanup       │
 │  prevent concurrent core thrashing.           to stop dual-server stacking.      │
 └──────────────────────────────────────────────────────────────────────────────────┘
```

### Data Pipeline Sequence
1. **Ingestion & Visual Validation:** The user uploads a file or selects one of the 10 built-in sample modalities. The UI streams the binary bytes to `/preview`. The preprocessor normalizes the shape dimensions, detects edge variations via OpenCV adaptive thresholding, and returns a Base64 preview string to the frontend in milliseconds.
2. **Task Enqueueing:** When the user clicks "Run Agentic Diagnosis", the file parameters are packaged with a unique `task_id` (UUID) and pushed to the async FIFO queue. The API returns a `queued` response immediately, switching the UI into a loading state.
3. **Isolated Thread Processing:** The backend task loop worker extracts the job from the queue and offloads the heavy matrix operations to an isolated system thread via `asyncio.to_thread()`. This leaves the main FastAPI loop completely open and responsive.
4. **Sequential Model Handoff:** 
   * **Llava:7b** processes the image, writes a structured block of findings, and unloads from system memory immediately (`keep_alive=0`).
   * **Mistral:7b** processes the text block, generates an impression and recommended remedies, and unloads from memory.
5. **Document Rendering & Status Update:** ReportLab takes the unique text findings and the task-specific annotated image path, writes the final PDF, and flags the job status as `completed`. The Streamlit interface, which polls the status route every 4 seconds, automatically re-renders to display the results and unlock the download link.

---

## 🚀 Deployment & Ingestion Verification

### Prerequisites
* Host VM: **Standard_D4as_v5** (or an equivalent instance providing at least 4 vCPUs and 16 GB RAM).
* Operating System: **SUSE Linux Enterprise Server (SLES)** or Ubuntu Server LTS.
* Dependencies: Docker Engine (v20.10+) and Docker Compose (v2.20+).

### Step-by-Step Installation

#### 1. Clone your application repository and navigate to the project directory:
```bash
cd ~/medical-imaging-agent
```

#### 2. Configure your environment layers. Ensure your `Dockerfile.api` block contains the bytecode suppression parameters to force pure disc loading:
```env
PYTHONDONTWRITEBYTECODE=1
PYTHONUNBUFFERED=1
```

#### 3. Configure the Orchestration Matrix (`docker-compose.yml`)
Establish your environment stack deployment by setting up this unified multi-container blueprint file:

```yaml
version: '3.8'

services:
  ollama-cpu:
    image: ollama/ollama:latest
    container_name: ollama_cpu_server
    ports:
      - "11434:11434"
    environment:
      - OLLAMA_NUM_PARALLEL=1
      - OLLAMA_KEEP_ALIVE=0
      - OLLAMA_INTEL_GPU=false
    volumes:
      - ollama_storage:/root/.ollama
    restart: unless-stopped

  agent-api:
    build:
      context: .
      dockerfile: Dockerfile.api
    container_name: medical_agent_api
    ports:
      - "8000:8000"
      - "5678:5678"
    environment:
      - OLLAMA_HOST=http://ollama-cpu:11434
    volumes:
      - .:/workspace
    depends_on:
      - ollama-cpu
    restart: unless-stopped

  streamlit-ui:
    image: python:3.11-slim-bookworm
    container_name: streamlit_ui_portal
    ports:
      - "8501:8501"
    environment:
      - BACKEND_PREVIEW_URL=http://agent-api:8000/preview
      - BACKEND_API_URL=http://agent-api:8000/analyze
      - BACKEND_STATUS_URL=http://agent-api:8000/status
    volumes:
      - ./ui:/app/ui
    working_dir: /app
    command: sh -c "pip install streamlit==1.32.0 requests && streamlit run ui/app.py --server.port=8501 --server.address=0.0.0.0"
    depends_on:
      - agent-api
    restart: unless-stopped

volumes:
  ollama_storage:
```

#### 4. Seed Built-In Diagnostic Demo Samples
To unlock the pre-indexed multi-modality specimen dropdown panel natively inside the container layout, stage your clinical specimen files straight to the active storage directory tree using this command block:

```bash
# 1. Establish the clean workspace target directory bounds on your SUSE host
mkdir -p /home/azureuser/medical-imaging-agent/app/samples

# 2. Stage the 10 multi-modality verification specimen files into the repository core
cp /tmp/test_scans/ct-lung-screening-nlst-instance.dcm /home/azureuser/medical-imaging-agent/app/samples/ 2>/dev/null
cp /tmp/test_scans/dx-chest-lidc-idri-instance.dcm /home/azureuser/medical-imaging-agent/app/samples/ 2>/dev/null
cp /tmp/test_scans/mg-mammography-cbis-ddsm-instance.dcm /home/azureuser/medical-imaging-agent/app/samples/ 2>/dev/null
cp /tmp/test_scans/mg-mammography-cmmd-instance.dcm /home/azureuser/medical-imaging-agent/app/samples/ 2>/dev/null
cp /tmp/test_scans/mg-tomosynthesis-victre-instance.dcm /home/azureuser/medical-imaging-agent/app/samples/ 2>/dev/null
cp /tmp/test_scans/mr-brain-upenn-gbm-instance.dcm /home/azureuser/medical-imaging-agent/app/samples/ 2>/dev/null
cp /tmp/test_scans/mr-breast-ispy1-instance.dcm /home/azureuser/medical-imaging-agent/app/samples/ 2>/dev/null
cp /tmp/test_scans/mr-prostate-prostatex-instance.dcm /home/azureuser/medical-imaging-agent/app/samples/ 2>/dev/null
cp /tmp/test_scans/us-kidney-cmb-crc-instance.dcm /home/azureuser/medical-imaging-agent/app/samples/ 2>/dev/null
cp /tmp/test_scans/us-lymph-node-cmb-lca-instance.dcm /home/azureuser/medical-imaging-agent/app/samples/ 2>/dev/null

# 3. Synchronize storage read permissions across the container group
chmod -R 755 /home/azureuser/medical-imaging-agent/app/samples
```

#### 4. Launch the Docker Orchestration Stack
Compile, assemble, and launch the asynchronous execution network in background mode using the native Docker compose toolchain:

```bash
# Force a clean cache-free composition to synchronize volume paths cleanly
DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker-compose up --build --force-recreate -d
```

#### 5. Pre-download and Model Cache Initialization
Warm up the neural network layers inside the sandboxed database cluster to eliminate initial loading delays:

```bash
docker exec -it ollama_cpu_server ollama pull llava:7b
docker exec -it ollama_cpu_server ollama pull mistral:7b-instruct-q4_K_M
```

#### 6. Verification Checklist & Sanity Testing Logs
Validate system operations across the internal loop frameworks by executing these diagnostic command steps:

| Step | Command Verification Matrix | Expected Success Metric Indicator |
| :--- | :--- | :--- |
| **1** | `docker ps --format "table {{.Names}}\t{{.Status}}"` | Three active containers showing `Up` status bounds safely. |
| **2** | `curl -X 'GET' 'http://localhost:8000/docs'` | FastAPI OpenAPI documentation tree interface returns a `200 OK`. |
| **3** | `docker-compose logs -f agent-api` | Zero argument crashes or path mapping exceptions logged to output. |

---

## 📋 Interactive Operating Instructions

1. **Dashboard Access:** Open an active web browser session panel and navigate to the Streamlit proxy deployment link: `http://localhost:8501`.
2. **Instant Visual Preview:** Expand the sidebar menu. Select any of the **10 Built-in Demo Scans** or upload a custom raw `.dcm` image file. The optimized OpenCV matrix pipeline will downsample, normalize, and project the 8-bit visual specimen canvas onto your left column workspace **instantly**.
![Preview](/images/medAI-preview.png)
3. **Draggable Viewport Controls:** Left-click down anywhere inside the black medical viewport box and move your mouse to translate and pan across pixel density structures. Use the toolbar buttons (`🔍 +` and `🔍 -`) to adjust image zoom scaling smoothly.
4. **Queue Ingestion Processing:** Click the primary action button labeled **`🚀 Run Agentic Diagnosis`**. The system will launch a micro-sized **Inference Processing Timer** to track active calculation durations.
![Diagnosis](/images/medAI-diagnosis.png)
5. **Multi-User Stability Handshake:** If separate browser sessions call endpoints simultaneously, the requests are queued inside a single-threaded asynchronous FIFO array. Each task is executed sequentially, preventing high-overhead resource competition and stabilizing CPU core load averages.
6. **Clinical Review Sign-off Documentation:** Once the background worker loop finishes, the page refreshes to reveal detailed structured medical text justifications, highlights prominent warning disclaimers, and unlocks the physical report asset download path button: **`📥 Download PDF Report for Specialist Sign-off`**.
![Report](/images/medAI-report.png)

---

## 🩺 Mandatory System Disclosure
This application functions entirely as an automated evaluation application utilizing open-source models on CPU-bound resources. All generated visual token evaluations, text findings, diagnostic assertions, and treatment suggestions are structured for development review, technical validation, and screening research workflows. They are not a substitute for a definitive professional diagnosis and must be critically reviewed and manually verified by a licensed physician or medical specialist before acting on any patient diagnostic care pathways.
