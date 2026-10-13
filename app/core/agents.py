import ollama
import logging

logger = logging.getLogger("uvicorn.error")

def run_medical_agent_workflow(image_path: str, metadata: dict) -> dict:
    logger.info("[STEP 2: AGENT RUNTIME] Initializing multi-agent orchestration loop...")
    
    # --- STAGE 1: 8-Thread Clinical Vision VLM ---
    logger.info("[AGENT] Loading 'llava:7b' into CPU RAM space. Starting visual analysis...")
    vision_prompt = (
        "You are an expert clinical computer vision agent. Analyze this medical scan image meticulously. "
        "List all visible physiological anomalies, densities, structural changes, anomalies or fractures. "
        "Provide factual observations only. Do not attempt to finalise a medical diagnosis yet."
    )
    
    vision_response = ollama.generate(
        model="llava:7b",
        prompt=vision_prompt,
        images=[image_path],
        options={
            "temperature": 0.0,
            "num_thread": 8,       # Core affinity optimization
            "num_ctx": 1024,       # ACCELERATION: Limits context footprint calculation size
            "num_predict": 180,    # ACCELERATION: Restricts tokens to eliminate CPU loop hanging
        },
        keep_alive=0 
    )
    extracted_findings = vision_response.get("response", "No anomalies extracted from visual data.")
    logger.info("✓ [AGENT] Stage 1 Vision analysis complete.")

    # --- STAGE 2: Reverted High-Performance Clinical SME Specialist ---
    logger.info("[AGENT] Loading 'mistral:7b-instruct-q4_K_M' into memory for clinical reasoning...")
    sme_prompt = f"""
    You are a Senior Medical Specialist AI model. Review the following raw optical findings from a patient's scan.
    
    [Patient Scan Modality]: {metadata['scan_type']}
    [Patient Demographic Metadata]: Age {metadata['patient_age']}
    [Vision Agent Observations]: {extracted_findings}
    
    Based on the above criteria:
    1. Formulate a tentative Diagnosis supported by clinical justification.
    2. Suggest evidence-based remedies, follow-up examinations, or standard care paths.
    
    Format your final response EXACTLY using these clear section headers, separating each with three dashes (---):
    FINDINGS:
    <Summarized findings here>
    ---
    DIAGNOSIS:
    <Finalized medical impression here>
    ---
    REMEDIES:
    <Actionable care recommendations here>
    """
    
    sme_response = ollama.generate(
        model="mistral:7b-instruct-q4_K_M",
        prompt=sme_prompt,
        options={
            "temperature": 0.1,
            "num_thread": 8,       # Core affinity optimization
            "num_ctx": 1536,       # ACCELERATION: Limits context footprint calculation size
            "num_predict": 250,    # ACCELERATION: Restricts long generation loops
            "stop": ["\n\n\n"]     
        },
        keep_alive=0              
    )
    raw_output = sme_response.get("response", "")
    logger.info("✓ [AGENT] Stage 2 Medical Specialist calculation complete.")
    
    # --- STAGE 3: Structuring & Parsing Agent Output ---
    parsed_report = {"findings": extracted_findings, "diagnosis": "Inconclusive", "remedies": "Consult specialist."}
    try:
        parts = raw_output.split("---")
        for part in parts:
            if "FINDINGS:" in part:
                parsed_report["findings"] = part.replace("FINDINGS:", "").strip()
            elif "DIAGNOSIS:" in part:
                parsed_report["diagnosis"] = part.replace("DIAGNOSIS:", "").strip()
            elif "REMEDIES:" in part:
                parsed_report["remedies"] = part.replace("REMEDIES:", "").strip()
    except Exception:
        parsed_report["diagnosis"] = raw_output
        
    logger.info("✓ [AGENT LOOP SUCCESS] Response dictionary formatted completely.")
    return parsed_report