# Face Recognition System

A production-ready face recognition system with a **Python FastAPI** engine and a **Node.js Express** API gateway.

---

## Architecture

```
Client → Node.js Gateway (port 1005) → Python FastAPI Engine (port 1001)
```

- **Python backend** handles all face encoding and matching using `face_recognition` (dlib).
- **Node.js gateway** handles rate limiting, file validation, and proxying to Python.

---

## Quick Start (Docker — Recommended)

```bash
docker-compose up --build
```

The API will be available at `http://localhost:1005`.

---

## Manual Setup

### Python Backend

```bash
cd backend
cp .env.example .env
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 1001 --reload
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

All requests go through the Node.js gateway at `http://localhost:1005`.

### Register a Reference Face

```
POST /api/setup
Content-Type: multipart/form-data OR application/json

Fields:
  label  (string)  — unique ID for this face, e.g. "user_123"
  image  (file)    — reference face image (JPG/PNG/WEBP, max 10MB)
  imageBase64 (string) — optional alternative to image file (data URL or raw base64)
```

**Response:**
```json
{ "success": true, "message": "Face registered successfully for label 'user_123'." }
```

---

### Match a Face

```
POST /api/match
Content-Type: multipart/form-data OR application/json

Fields:
  label      (string)  — label of the registered reference face
  image      (file)    — image to verify
  imageBase64 (string) — optional alternative to image file (data URL or raw base64)
  tolerance  (number)  — optional, default 0.5 (0.4=strict, 0.6=lenient)
```

**Response (match):**
```json
{
  "match": true,
  "confidence": 87.34,
  "distance": 0.1266,
  "matched_image_url": "/api/images/user_123",
  "message": "Face matched successfully."
}
```

### Match Client Face (Single API with Token)

Use this when you only want to send one image from frontend. The reference image is auto-fetched from CMS profile using token.

```
POST /api/match-client
Content-Type: multipart/form-data OR application/json

Fields:
  image       (file)    — client selfie image to verify (or use imageBase64)
  imageBase64 (string)  — optional alternative to image file
  tolerance   (number)  — optional, default 0.5
  token       (string)  — optional if Authorization header is used
```

Header (recommended):
```
Authorization: Bearer <sessionToken>
```

Flow:
- Gateway calls `https://cms.ezwealth.in/api/auth-client/profile` with token.
- Reads Aadhaar image from `digioDetails.actions[].details.aadhaar.image`.
- Matches uploaded image against Aadhaar image.
- On successful match, saves the uploaded selfie to client profile as `selfieEkyc` using the same bearer token.

Transparent PNG/WEBP images are accepted in both multipart and base64 input. They are normalized internally for processing.

**Response (match):**
```json
{
  "match": true,
  "confidence": 90.12,
  "distance": 0.0988,
  "client_id": "69afe3b420902d7b7ee7c424",
  "matched_image_base64": "<base64-from-cms>",
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

### Validate Signature (Plan Page)

Validates if an image contains a proper signature for document signing.
Automatically rejects:
- Simple starting lines (not a valid signature)
- Random curly/scribble lines (invalid)
- Empty or minimal signatures

If the signature is valid and an Authorization header or token is provided, the gateway saves it to client profile as `wetSignature`.

```
POST /api/validate-signature
Content-Type: multipart/form-data OR application/json

Fields:
  image       (file)    — signature image to validate
  imageBase64 (string)  — optional alternative: signature as base64 data URL or raw base64
  token       (string)  — optional if Authorization header is used for saving `wetSignature`
```

**Response (valid signature):**
```json
{
  "valid": true,
  "confidence": 78.5,
  "message": "Valid signature detected.",
  "metrics": {
    "solidity": 0.512,
    "vertices": 24,
    "aspect_ratio": 1.45,
    "area_ratio": 0.0234
  }
}
```

**Response (rejected - simple line):**
```json
{
  "valid": false,
  "confidence": 15.0,
  "message": "Signature appears to be a simple line (not a valid signature)."
}
```

**Response (rejected - random scribbles):**
```json
{
  "valid": false,
  "confidence": 20.0,
  "message": "Signature appears to be random scribbles (not a valid signature)."
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

### Get Matched Reference Image

```
GET /api/images/:label
```

Returns the registered reference image for that label.

---

## Testing with curl

```bash
# 1. Register a reference face
curl -X POST http://localhost:1005/api/setup \
  -F "label=user_123" \
  -F "image=@/path/to/reference.jpg"

# 2. Match an uploaded face
curl -X POST http://localhost:1005/api/match \
  -F "label=user_123" \
  -F "image=@/path/to/test.jpg"

# 2b. Match with CMS token (single image from client)
curl -X POST http://localhost:1005/api/match-client \
  -H "Authorization: Bearer <sessionToken>" \
  -F "image=@/path/to/test.jpg"

# 3. Validate a signature for plan page
curl -X POST http://localhost:1005/api/validate-signature \
  -F "image=@/path/to/signature.jpg"

# 4. List all labels
curl http://localhost:1005/api/labels

# 5. Delete a label
curl -X DELETE http://localhost:1005/api/labels/user_123
```

---

## Environment Variables

### backend/.env

| Variable          | Default | Description                        |
|-------------------|---------|------------------------------------|
| PORT              | 1001    | Python server port                 |
| ALLOWED_ORIGINS   | *       | Comma-separated CORS origins       |
| MAX_FILE_SIZE_MB  | 10      | Max upload size in MB              |

### api/.env

| Variable          | Default                    | Description                  |
|-------------------|----------------------------|------------------------------|
| PORT              | 1005                       | Node.js gateway port         |
| PYTHON_API_URL    | http://localhost:1001      | Python backend URL           |
| ALLOWED_ORIGINS   | *                          | Comma-separated CORS origins |
| CMS_PROFILE_URL   | https://cms.ezwealth.in/api/auth-client/profile | CMS profile lookup URL |
| CMS_UPDATE_PROFILE_URL | http://192.168.1.22:8000/api/clients/update-client-profile | Saves `selfieEkyc` and `wetSignature` on successful validation |

---

## Notes

- Face encodings are stored as a pickle file in `backend/models/encodings.pkl`.
- Uploaded images are deleted immediately after processing — nothing is stored permanently.
- The `tolerance` parameter controls strictness: `0.4` is very strict, `0.6` is lenient. Default `0.5` works well for most cases.
