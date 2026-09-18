import os
import json
from dotenv import load_dotenv
from groq import Groq
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import pdfplumber

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()

class NotesRequest(BaseModel):
    text: str

class ChatRequest(BaseModel):
    question: str
    context: str = ""

@app.get("/")
def read_root():
    return {"message": "Exam Prep Platform is alive"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        return {"error": "Only PDF files are allowed"}

    with pdfplumber.open(file.file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()

    return {"filename": file.filename, "extracted_text": text}

@app.post("/generate-notes")
def generate_notes(request: NotesRequest):
    prompt = f"""You are an expert exam coach. Break the study material below into separate topics, and for each topic give structured notes.

Study material:
{request.text}

Respond ONLY with valid JSON in exactly this structure, no extra text:
{{
  "topics": [
    {{
      "topic_name": "name of the topic",
      "simple_explanation": "short simple explanation",
      "detailed_explanation": "detailed explanation",
      "definitions": ["definition 1", "definition 2"],
      "key_concepts": ["concept 1", "concept 2"],
      "examples": ["example 1", "example 2"],
      "exam_focused_points": ["point 1", "point 2"]
    }}
  ]
}}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    notes_data = json.loads(response.choices[0].message.content)
    return notes_data

@app.post("/chat")
def chat(request: ChatRequest):
    if request.context:
        prompt = f"""You are a helpful study assistant. Answer using the study material below if it's relevant. If the question is unrelated to the material, answer it normally using your own knowledge. Keep answers short and clear (2-6 sentences) unless more detail is truly needed.

Study material:
{request.context}

Question: {request.question}
"""
    else:
        prompt = f"""Answer the following question briefly, clearly, and accurately. Keep it short (2-5 sentences) unless the question needs more detail.

Question: {request.question}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"answer": response.choices[0].message.content}