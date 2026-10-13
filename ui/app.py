import streamlit as st
import requests
import base64
import os
import time

st.set_page_config(page_title="MedAI Analytics Dashboard", layout="wide")

st.title("🩺 Medical Imaging AI Analyzer & Report Generator")
st.markdown("---")

PREVIEW_URL = os.getenv("BACKEND_PREVIEW_URL", "http://agent-api:8000/preview")
ANALYZE_URL = os.getenv("BACKEND_API_URL", "http://agent-api:8000/analyze")
STATUS_URL = os.getenv("BACKEND_STATUS_URL", "http://agent-api:8000/status")

# Path to the samples directory inside the workspace container volume mount
SAMPLES_DIR = "app/samples"

# Initialize persistent session states safely
if 'current_file' not in st.session_state: st.session_state['current_file'] = None
if 'preview_img' not in st.session_state: st.session_state['preview_img'] = None
if 'active_task_id' not in st.session_state: st.session_state['active_task_id'] = None
if 'job_data' not in st.session_state: st.session_state['job_data'] = None
if 'inference_duration' not in st.session_state: st.session_state['inference_duration'] = None

st.sidebar.header("Patient Scan Ingestion")

# FIXED: Comprehensive Dictionary mapping clean titles to all 10 available specimen files
sample_mapping = {
    "None - Use Manual Upload": None,
    "🫁 Lung Cancer Screening (CT)": "ct-lung-screening-nlst-instance.dcm",
    "🫁 Chest Diagnostic Scan (DX)": "dx-chest-lidc-idri-instance.dcm",
    "🎀 CBIS-DDSM Mammography (MG)": "mg-mammography-cbis-ddsm-instance.dcm",
    "🎀 CMMD Screening Mammography (MG)": "mg-mammography-cmmd-instance.dcm",
    "🎀 VICTRE Tomosynthesis (MG)": "mg-tomosynthesis-victre-instance.dcm",
    "🧠 UPENN-GBM Glioblastoma Brain (MR)": "mr-brain-upenn-gbm-instance.dcm",
    "Breast Tumor ISPY1 Screening (MR)": "mr-breast-ispy1-instance.dcm",
    "🎯 PROSTATEX Prostate Diagnostic (MR)": "mr-prostate-prostatex-instance.dcm",
    "🧬 Kidney Lesion Assessment (US)": "us-kidney-cmb-crc-instance.dcm",
    "🧬 Lymph Node Evaluation (US)": "us-lymph-node-cmb-lca-instance.dcm"
}

selected_sample = st.sidebar.selectbox("🎯 Select Built-in Demo Sample Scan", list(sample_mapping.keys()))
uploaded_file = st.sidebar.file_uploader("Or Upload Medical Scan Manual", type=["dcm", "png", "jpg", "jpeg"])

col1, col2 = st.columns(2)

# Establish default baseline file stream data parameters
file_name = None
file_bytes = None
file_type = "application/octet-stream"

if selected_sample != "None - Use Manual Upload":
    target_sample_name = sample_mapping[selected_sample]
    sample_path = os.path.join(SAMPLES_DIR, target_sample_name)
    
    if os.path.exists(sample_path):
        file_name = target_sample_name
        with open(sample_path, "rb") as f:
            file_bytes = f.read()
    else:
        st.sidebar.error(f"Sample file not found on disk at: {sample_path}")
elif uploaded_file is not None:
    file_name = uploaded_file.name
    file_bytes = uploaded_file.getvalue()
    file_type = uploaded_file.type

if file_name is not None and file_bytes is not None:
    if st.session_state['current_file'] != file_name:
        st.session_state['current_file'] = file_name
        st.session_state['job_data'] = None
        st.session_state['active_task_id'] = None
        st.session_state['preview_img'] = None
        st.session_state['inference_duration'] = None
        
        with st.spinner("Decoding DICOM matrix structures for instant display..."):
            try:
                files = {"file": (file_name, file_bytes, file_type)}
                prev_resp = requests.post(PREVIEW_URL, files=files)
                if prev_resp.status_code == 200:
                    st.session_state['preview_img'] = prev_resp.json().get('preview_img')
                else:
                    st.error("Failed to parse visual preview structure.")
            except Exception as e:
                st.error(f"Preview connection error: {str(e)}")

    with col1:
        st.subheader("📸 Interactive Scan Viewport")
        if st.session_state['preview_img'] is not None:
            html_markup = f"""
            <div style="position:relative; width:100%; height:460px; overflow:hidden; border:2px solid #CBD5E0; border-radius:12px; background-color:#0A0F1D; display:flex; align-items:center; justify-content:center; user-select:none; -webkit-user-select:none;">
                <div style="position:absolute; top:15px; left:15px; z-index:100; background:rgba(10,15,30,0.85); padding:6px; border-radius:8px; display:flex; gap:6px; border:1px solid rgba(255,255,255,0.15); box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
                    <button onclick="zoomIn()" style="background:#2B6CB0; color:white; border:none; padding:6px 12px; border-radius:4px; cursor:pointer; font-weight:bold; font-size:12px;">🔍 +</button>
                    <button onclick="zoomOut()" style="background:#2B6CB0; color:white; border:none; padding:6px 12px; border-radius:4px; cursor:pointer; font-weight:bold; font-size:12px;">🔍 -</button>
                    <button onclick="resetViewport()" style="background:#4A5568; color:white; border:none; padding:6px 12px; border-radius:4px; cursor:pointer; font-size:12px;">✋ Reset</button>
                </div>
                <div id="dragContainer" style="position:absolute; width:100%; height:100%; display:flex; align-items:center; justify-content:center; cursor:grab; transform: scale(1) translate(0px, 0px); transform-origin: center center; opacity:0; transition: transform 0.05s linear, opacity 0.2s ease-in-out;">
                    <img id="medicalCanvas" src="data:image/png;base64,{st.session_state['preview_img']}" style="max-width:95%; max-height:95%; object-fit:contain; pointer-events: none;" />
                </div>
                <div id="hudIndicator" style="position:absolute; bottom:15px; left:15px; z-index:100; background:rgba(0,0,0,0.75); color:#E2E8F0; font-family:monospace; font-size:11px; padding:6px 12px; border-radius:20px; border:1px solid rgba(255,255,255,0.15);">
                    Scale: 1.0x | Offset: (0px, 0px)
                </div>
            </div>
            <script>
                const container = document.getElementById('dragContainer');
                const hud = document.getElementById('hudIndicator');
                let scale = 1.0; let panX = 0; let panY = 0; let isDragging = false; let startX, startY;
                window.addEventListener('DOMContentLoaded', () => {{ setTimeout(() => {{ container.style.opacity = "1"; }}, 150); }});
                function updateTransform() {{ container.style.transform = `scale(${{scale}}) translate(${{panX}}px, ${{panY}}px)`; hud.innerText = `Scale: ${{scale.toFixed(1)}}x | Offset: (${{Math.round(panX)}}px, ${{Math.round(panY)}}px)`; }}
                window.zoomIn = function() {{ scale = Math.min(scale + 0.2, 4.0); updateTransform(); }}
                window.zoomOut = function() {{ scale = Math.max(scale - 0.2, 1.0); if (scale === 1.0) {{ panX = 0; panY = 0; }} updateTransform(); }}
                window.resetViewport = function() {{ scale = 1.0; panX = 0; panY = 0; updateTransform(); }}
                container.addEventListener('mousedown', (e) => {{ isDragging = true; container.style.cursor = 'grabbing'; startX = e.clientX - (panX * scale); startY = e.clientY - (panY * scale); e.preventDefault(); }});
                window.addEventListener('mousemove', (e) => {{ if (!isDragging) return; panX = (e.clientX - startX) / scale; panY = (e.clientY - startY) / scale; updateTransform(); }});
                window.addEventListener('mouseup', () => {{ isDragging = false; container.style.cursor = 'grab'; }});
            </script>
            """
            st.components.v1.html(html_markup, height=480)
        else:
            st.info("Visualizing medical matrix arrays. Please wait...")

        if st.sidebar.button("🚀 Run Agentic Diagnosis", type="primary"):
            try:
                files = {"file": (file_name, file_bytes, file_type)}
                an_resp = requests.post(ANALYZE_URL, files=files)
                if an_resp.status_code == 200:
                    st.session_state['active_task_id'] = an_resp.json().get('task_id')
                    st.session_state['job_data'] = {"status": "processing"}
                    st.session_state['inference_duration'] = time.time()
                else:
                    st.sidebar.error("Failed to enqueue background worker thread.")
            except Exception as e:
                st.sidebar.error(f"Analysis loop exception: {str(e)}")

    with col2:
        st.subheader("📋 Generated Clinical Report")
        
        if st.session_state['job_data'] and st.session_state['job_data'].get('status') == 'processing':
            status_box = st.empty()
            status_box.info("Your request is queued. Executing CPU background workers sequentially...")
            
            elapsed = int(time.time() - st.session_state['inference_duration'])
            mins, secs = divmod(elapsed, 60)
            
            st.markdown(f"""
            <div style="background-color:#EDF2F7; border:1px solid #CBD5E0; border-radius:6px; padding:6px 12px; display:inline-flex; align-items:center; gap:8px; font-family:monospace; margin-bottom:15px; box-shadow:0 1px 2px rgba(0,0,0,0.05);">
                <span style="font-size:11px; color:#4A5568; font-weight:bold;">⏱️ RUNTIME COUNTER:</span>
                <span style="font-size:14px; color:#2B6CB0; font-weight:bold;">{mins:02d}:{secs:02d}</span>
            </div>
            """, unsafe_allow_html=True)
            
            time.sleep(4)
            try:
                chk = requests.get(f"{STATUS_URL}/{st.session_state['active_task_id']}").json()
                if chk.get("status") == "completed":
                    st.session_state['job_data'] = chk
                    total_seconds = int(time.time() - st.session_state['inference_duration'])
                    tm_m, tm_s = divmod(total_seconds, 60)
                    st.session_state['inference_duration'] = f"{tm_m:02d}:{tm_s:02d}"
                    status_box.empty()
                    
                    if hasattr(st, "runtime") and st.runtime.exists():
                        st.rerun()
                    else:
                        st.experimental_rerun()
                elif chk.get("status") == "failed":
                    st.session_state['job_data'] = None
                    status_box.error("Analysis task failed.")
            except Exception:
                pass
                
            if st.session_state['job_data'] and st.session_state['job_data'].get('status') == 'processing':
                if hasattr(st, "runtime") and st.runtime.exists():
                    st.rerun()
                else:
                    st.experimental_rerun()

        elif st.session_state['job_data'] and st.session_state['job_data'].get('status') == 'completed':
            data = st.session_state['job_data']
            clean_diagnosis = data.get('diagnosis', 'Inconclusive').replace("Diagnosis:", "").strip()
            
            if isinstance(st.session_state['inference_duration'], str):
                st.markdown(f"⚡ *Total Inference Duration:* **` {st.session_state['inference_duration']} `**")
            
            st.markdown("### **Clinical Impression & Diagnosis**")
            st.markdown(f"<div style='font-size:14px; color:#2D3748; background-color:#FEFCBF; padding:12px; border-left:4px solid #D69E2E; border-radius:4px;'><u>👉 <b>Major Medical Assessment:</b> {clean_diagnosis}</u></div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("#### **Extracted Physiological Findings**")
            st.write(data.get('findings'))
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("#### **Evidence-Based Remedies & Care Paths**")
            st.write(data.get('remedies'))
            st.markdown("---")
            
            st.warning(
                "⚠️ STATUTORY WARNING: This automated evaluation artifact was compiled by an AI multi-agent "
                "orchestration layer running open-source CPU inference models. It is structured for technical screening "
                "validation workflows and research development only. Do not rely on these observations for clinical "
                "care decisions. All findings must be critically cross-referenced, reviewed, and signed off "
                "by a certified Subject Matter Expert (SME) physician before acting on any patient diagnostic pathways."
            )
            st.markdown("<br>", unsafe_allow_html=True)
            
            if "internal_pdf_url" in data:
                st.success("Clinical-Grade PDF compiled successfully.")
                try:
                    pdf_bytes = requests.get(data["internal_pdf_url"]).content
                    st.download_button(
                        label="📥 Download PDF Report for Specialist Sign-off",
                        data=pdf_bytes,
                        file_name=f"Report_{st.session_state['active_task_id'][:8]}.pdf",
                        mime="application/pdf"
                    )
                except Exception as net_err:
                    st.error(f"Failed to fetch PDF payload internally: {str(net_err)}")
        else:
            st.info("Select a demo scan from the dropdown or upload your own file to initialize the models.")
else:
    st.session_state['current_file'] = None
    st.session_state['preview_img'] = None
    st.session_state['active_task_id'] = None
    st.session_state['job_data'] = None
    st.session_state['inference_duration'] = None
    st.info("Please choose a built-in demo scan or drop a file in the sidebar to begin analysis.")
