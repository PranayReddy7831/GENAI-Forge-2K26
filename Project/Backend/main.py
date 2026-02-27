import json
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import AsyncClient

# --- CONFIGURATION ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("CRITICAL: GROQ_API_KEY is not set in the .env file.")

client = AsyncClient(api_key=GROQ_API_KEY)
app = FastAPI(title="CodeRefine AI Engine")

# --- CORS SETUP ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODELS ---
class CodeRequest(BaseModel):
    code: str

class CodeResponse(BaseModel):
    original_score: int
    refined_score: int
    bugs: str
    performance: str
    improvements: list[str]
    optimized_code: str
    explanation: str
    time_complexity: str
    space_complexity: str

class TranslateRequest(BaseModel):
    code: str
    target_language: str

# --- CORE LOGIC ---
@app.post("/review", response_model=CodeResponse)
async def review_code(request: CodeRequest):
    """
    Analyzes code logic using a weighted 5-pillar scoring system.
    """
    system_prompt = (
        "You are a strict Senior Software Architect. Analyze code using this exact 100-point rubric:\n"
        "1. Logic & Correctness (30 pts): Does it work? Are there edge-case bugs?\n"
        "2. Time/Space Efficiency (25 pts): Is the algorithm optimal (Big O)?\n"
        "3. Security & Safety (15 pts): Are there vulnerabilities or crash risks?\n"
        "4. Readability & Standards (15 pts): Variable naming, clean code patterns?\n"
        "5. Maintainability (15 pts): Is the architecture scalable?\n\n"
        "STRICT RULES:\n"
        "- Calculate the 'original_score' by summing the points above for the USER'S code.\n"
        "- Calculate the 'refined_score' for YOUR optimized version (this should be 95-100).\n"
        "- If the user's code is already optimal, 'original_score' must be 100.\n"
        "- 'optimized_code' must be a single string. Use '\\n' for new lines and escape all quotes.\n"
        "- Return ONLY a valid JSON object. No markdown, no triple quotes.\n\n"
        "JSON Schema:\n"
        "{\n"
        "    \"original_score\": int,\n"
        "    \"refined_score\": int,\n"
        "    \"bugs\": \"string\",\n"
        "    \"performance\": \"string\",\n"
        "    \"improvements\": [\"list\"],\n"
        "    \"optimized_code\": \"string\",\n"
        "    \"explanation\": \"string\",\n"
        "    \"time_complexity\": \"string\",\n"
        "    \"space_complexity\": \"string\"\n"
        "}"
    )

    try:
        response = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Perform a rubric-based review of this code:\n\n{request.code}"},
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        
        ai_data = json.loads(response.choices[0].message.content)
        
        # Explicitly cast to int to ensure frontend animation works
        ai_data["original_score"] = int(ai_data.get("original_score", 0))
        ai_data["refined_score"] = int(ai_data.get("refined_score", 0))
        
        return ai_data

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Heuristic analysis engine failed.")

@app.post("/translate")
async def translate_code(request: TranslateRequest):
    system_prompt = (
        f"You are a polyglot expert. Translate this code to {request.target_language}.\n"
        "Rules: Return ONLY JSON: {\"translated_code\": \"string\"}. "
        "Use '\\n' and escape double quotes. No markdown tags."
    )
    try:
        response = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.code},
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"translated_code": f"// Translation error: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)