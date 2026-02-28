# 🚀 CodeRefine AI Engine

### Generative AI–Powered Code Review, Optimization & Translation Platform

CodeRefine is an advanced AI-powered code analysis engine that reviews, scores, optimizes, and translates source code using a structured architectural evaluation model. It helps developers identify bugs, improve performance, and generate production-ready optimized code instantly.

---

# 📌 Overview

CodeRefine uses Large Language Models (LLMs) to perform deep code analysis based on a strict 100-point rubric system covering:

* Logic correctness
* Performance efficiency
* Security and safety
* Code readability
* Maintainability

The system automatically:

* Detects bugs
* Improves algorithm efficiency
* Optimizes architecture
* Calculates time & space complexity
* Translates code between multiple programming languages

---

# 🧠 Core Features

## 1. AI Code Review Engine

* Scores code using a 5-pillar rubric system
* Detects logical bugs and runtime risks
* Provides detailed diagnostics and explanations

## 2. Automated Code Optimization

* Generates refined, production-ready code
* Improves algorithm efficiency
* Reduces time and space complexity

## 3. Performance Analysis

* Calculates time complexity
* Calculates space complexity
* Shows before vs after comparison

## 4. Multi-Language Code Translation

Supports translation to:

* Java
* Python
* Go
* Rust
* C++

## 5. Interactive Visual Dashboard

* Animated performance scores
* Real-time AI feedback
* Confetti animation for high optimization scores
* Modern glassmorphism UI

---

# 🏗️ System Architecture

```
Frontend (React)
    │
    ▼
FastAPI Backend (Python)
    │
    ▼
Groq LLM API (LLaMA 3.3 70B)
    │
    ▼
AI Analysis + Optimization + Translation
```

---

# 🧰 Technology Stack

## Frontend

* React.js
* JavaScript
* CSS (Glassmorphism UI)
* Confetti animation library

## Backend

* Python
* FastAPI
* Uvicorn

## AI Integration

* Groq API
* LLaMA-3.3-70B-Versatile model

## Other Tools

* dotenv
* Pydantic
* CORS Middleware

---

# 📂 Project Structure

```
CodeRefine/
│
├── backend/
│   ├── main.py
│   ├── .env
│   └── requirements.txt
│
├── frontend/
│   ├── Editor.jsx
│   ├── api.js
│   └── components/
│
└── README.md
```

---

# ⚙️ Installation Guide

## 1. Clone Repository

```
git clone https://github.com/yourusername/coderefine.git
cd coderefine
```

---

## 2. Backend Setup

```
cd backend
pip install -r requirements.txt
```

Create `.env`

```
GROQ_API_KEY=your_api_key_here
```

Run server

```
python main.py
```

Backend runs at:

```
http://127.0.0.1:8000
```

---

## 3. Frontend Setup

```
cd frontend
npm install
npm start
```

Runs at:

```
http://localhost:3000
```

---

# 🔌 API Endpoints

## POST /review

Analyzes and optimizes code

Request:

```
{
  "code": "your source code"
}
```

Response:

```
{
  original_score: int,
  refined_score: int,
  bugs: string,
  optimized_code: string,
  explanation: string,
  time_complexity: string,
  space_complexity: string
}
```

---

## POST /translate

Translates optimized code

Request:

```
{
  "code": "optimized code",
  "target_language": "Python"
}
```

Response:

```
{
  translated_code: string
}
```

---

# 🎯 Scoring Rubric

| Category                | Points |
| ----------------------- | ------ |
| Logic & Correctness     | 30     |
| Time & Space Efficiency | 25     |
| Security & Safety       | 15     |
| Readability             | 15     |
| Maintainability         | 15     |
| Total                   | 100    |

---

# 💡 Example Workflow

1. User pastes code
2. AI analyzes logic and performance
3. System calculates score
4. Optimized code is generated
5. Complexity is reduced
6. Code can be translated to another language

---

# 📊 Use Cases

* Code optimization
* Hackathon projects
* Learning data structures & algorithms
* Interview preparation
* Automated code review systems
* Developer productivity tools

---

# 🔐 Environment Variables

```
GROQ_API_KEY=your_key
```

---

# 🚀 Future Enhancements

* Support for more programming languages
* Real-time collaborative editing
* GitHub integration
* Automated pull request review
* Security vulnerability scanner

---

# 👨‍💻 Author

CodeRefine AI Engine
Hackathon Project — 2026

---

# 📄 License

MIT License
