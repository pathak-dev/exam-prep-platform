import os
import json
import base64
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import List
import pdfplumber

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()

class NotesRequest(BaseModel):
    text: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    context: str = ""

@app.get("/")
def read_root():
    return {"message": "PoCai backend is alive"}

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
    current_time = datetime.now().strftime("%A, %B %d, %Y — %I:%M %p")

    system_content = (
        "You are PoCai, a friendly and sharp AI assistant created by Ansh Pathak — you talk like a smart friend explaining "
        "something, not like a textbook or a corporate AI. Use simple, everyday words. Avoid stiff, overly formal, or "
        "'AI-sounding' phrasing. Get straight to the point — no unnecessary preamble, no repeating the question back. "
        f"The current date and time is: {current_time}. "
        "By default, keep answers SHORT — around 3 to 4 lines, clear and logical, covering the key point directly. "
        "Only go long and detailed (using markdown ## headings and - bullet points) if the user explicitly asks for "
        "more detail, depth, or a full explanation. "
        "You can answer questions on any subject, not just study material."
    )
    if request.context:
        system_content += f"\n\nThe user has also uploaded this study material — use it when the question relates to it:\n{request.context}"

    messages = [{"role": "system", "content": system_content}]
    for m in request.messages:
        messages.append({"role": m.role, "content": m.content})

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages
    )

    return {"answer": response.choices[0].message.content}

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...), question: str = Form("Analyze this image in detail and explain what it is, in an organized way using markdown headings and bullet points.")):
    image_bytes = await file.read()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    mime = file.content_type or "image/jpeg"

    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64_image}"}}
                ]
            }
        ],
        max_completion_tokens=800
    )

    return {"answer": response.choices[0].message.content}