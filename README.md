# Face Recognition System

A production-ready face recognition system with a **Python FastAPI** engine and a **Node.js Express** API gateway.

---

## Architecture

```
Client → Node.js Gateway (port 3000) → Python FastAPI Engine (port 8000)
```

- **Python backend** handles all face encoding and matching using `face_recognition` (dlib).
- **Node.js gateway** handles rate limiting, file validation, and proxying to Python.

---

## Quick Start (Docker — Recommended)

```bash
docker-compose up --build
```

The API will be available at `http://localhost:3000`.

---

## Manual Setup

### Python Backend

```bash
cd backend
cp .env.example .env
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Node.js Gateway

```bash
cd api
cp .env.example .env
npm install
npm run dev
```

---

## API Reference

All requests go through the Node.js gateway at `http://localhost:3000`.

### Register a Reference Face

```
POST /api/setup
Content-Type: multipart/form-data

Fields:
  label  (string)  — unique ID for this face, e.g. "user_123"
  image  (file)    — reference face image (JPG/PNG/WEBP, max 10MB)
```

**Response:**
```json
{ "success": true, "message": "Face registered successfully for label 'user_123'." }
```

---

### Match a Face

```
POST /api/match
Content-Type: multipart/form-data

Fields:
  label      (string)  — label of the registered reference face
  image      (file)    — image to verify
  tolerance  (number)  — optional, default 0.5 (0.4=strict, 0.6=lenient)
```

**Response (match):**
```json
{
  "match": true,
  "confidence": 87.34,
  "distance": 0.1266,
  "message": "Face matched successfully."
}
```

**Response (no match):**
```json
{
  "match": false,
  "confidence": 41.20,
  "distance": 0.5880,
  "message": "Face does not match."
}
```

---

### List Registered Labels

```
GET /api/labels
```

**Response:**
```json
{ "labels": ["user_123", "admin"], "count": 2 }
```

---

### Delete a Label

```
DELETE /api/labels/:label
```

**Response:**
```json
{ "success": true, "message": "Label 'user_123' deleted." }
```

---

### Health Check

```
GET /health
```

---

## Testing with curl

```bash
# 1. Register a reference face
curl -X POST http://localhost:3000/api/setup \
  -F "label=user_123" \
  -F "image=@/path/to/reference.jpg"

# 2. Match an uploaded face
curl -X POST http://localhost:3000/api/match \
  -F "label=user_123" \
  -F "image=@/path/to/test.jpg"

# 3. List all labels
curl http://localhost:3000/api/labels

# 4. Delete a label
curl -X DELETE http://localhost:3000/api/labels/user_123
```

---

## Environment Variables

### backend/.env

| Variable          | Default | Description                        |
|-------------------|---------|------------------------------------|
| PORT              | 8000    | Python server port                 |
| ALLOWED_ORIGINS   | *       | Comma-separated CORS origins       |
| MAX_FILE_SIZE_MB  | 10      | Max upload size in MB              |

### api/.env

| Variable          | Default                    | Description                  |
|-------------------|----------------------------|------------------------------|
| PORT              | 3000                       | Node.js gateway port         |
| PYTHON_API_URL    | http://localhost:8000      | Python backend URL           |
| ALLOWED_ORIGINS   | *                          | Comma-separated CORS origins |

---

## Notes

- Face encodings are stored as a pickle file in `backend/models/encodings.pkl`.
- Uploaded images are deleted immediately after processing — nothing is stored permanently.
- The `tolerance` parameter controls strictness: `0.4` is very strict, `0.6` is lenient. Default `0.5` works well for most cases.
