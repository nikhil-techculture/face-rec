require("dotenv").config();
const express = require("express");
const multer = require("multer");
const axios = require("axios");
const FormData = require("form-data");
const helmet = require("helmet");
const morgan = require("morgan");
const cors = require("cors");
const errorHandler = require("./middleware/errorHandler");

const app = express();
const PORT = process.env.PORT || 1005;
const PYTHON_API = process.env.PYTHON_API_URL || "http://localhost:1001";
const CMS_PROFILE_URL = process.env.CMS_PROFILE_URL || "https://cms.ezwealth.in/api/auth-client/profile";

// Security & logging middleware
app.use(helmet());
app.use(morgan("combined"));
app.use(cors({
  origin: process.env.ALLOWED_ORIGINS ? process.env.ALLOWED_ORIGINS.split(",") : "*"
}));
app.use(express.json());
app.use(express.static("public"));

// Multer v2 — memory storage, 10MB limit
const { memoryStorage } = multer;
const upload = multer({
  storage: memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const allowed = ["image/jpeg", "image/png", "image/webp"];
    if (!allowed.includes(file.mimetype)) {
      return cb(new multer.MulterError("LIMIT_UNEXPECTED_FILE", file.fieldname));
    }
    cb(null, true);
  }
});


async function forwardToPython(endpoint, fields, fileBuffer, fileName, mimeType) {
  const form = new FormData();
  for (const [key, value] of Object.entries(fields)) {
    form.append(key, String(value));
  }
  if (fileBuffer) {
    form.append("image", fileBuffer, { filename: fileName, contentType: mimeType });
  }

  const response = await axios.post(`${PYTHON_API}${endpoint}`, form, {
    headers: form.getHeaders(),
    timeout: 10050
  });
  return response.data;
}

function parseBase64Image(input) {
  if (!input || typeof input !== "string") return null;

  const trimmed = input.trim();
  let mimeType = "image/jpeg";
  let payload = trimmed;

  const match = trimmed.match(/^data:(image\/[a-zA-Z0-9.+-]+);base64,(.+)$/);
  if (match) {
    mimeType = match[1].toLowerCase();
    payload = match[2];
  }

  const allowed = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp"
  };
  if (!allowed[mimeType]) return null;

  try {
    const buffer = Buffer.from(payload, "base64");
    if (!buffer.length) return null;
    return {
      buffer,
      mimeType,
      fileName: `upload${allowed[mimeType]}`
    };
  } catch {
    return null;
  }
}

function extractClientProfile(payload) {
  if (!payload || typeof payload !== "object") return null;
  if (payload.digioDetails) return payload;
  if (payload.data && typeof payload.data === "object") {
    if (payload.data.digioDetails) return payload.data;
    if (payload.data.profile && payload.data.profile.digioDetails) return payload.data.profile;
    if (Array.isArray(payload.data.items)) {
      const match = payload.data.items.find((item) => item && item.digioDetails);
      if (match) return match;
    }
  }
  return null;
}

function extractAadhaarImage(profile) {
  const actions = profile?.digioDetails?.actions;
  if (!Array.isArray(actions)) return null;

  for (const action of actions) {
    const image = action?.details?.aadhaar?.image;
    if (typeof image === "string" && image.trim()) {
      return image.trim();
    }
  }
  return null;
}

function extractClientId(profile) {
  if (!profile || typeof profile !== "object") return null;
  if (typeof profile.clientId === "string" && profile.clientId.trim()) return profile.clientId;
  if (profile._id && typeof profile._id === "object" && typeof profile._id.$oid === "string") return profile._id.$oid;
  if (typeof profile._id === "string" && profile._id.trim()) return profile._id;
  if (typeof profile.id === "string" && profile.id.trim()) return profile.id;
  if (typeof profile.mobileNo === "string" && profile.mobileNo.trim()) return profile.mobileNo;
  return null;
}

async function fetchProfileByToken(rawToken, authorizationHeader) {
  const token = (rawToken || "").trim();
  const bearer = authorizationHeader && authorizationHeader.trim()
    ? authorizationHeader.trim()
    : (token ? `Bearer ${token}` : "");

  if (!bearer) {
    console.error("[CMS_PROFILE] Missing token for profile fetch.");
    throw new Error("Missing token. Provide Authorization header or token field.");
  }

  try {
    const { data } = await axios.get(CMS_PROFILE_URL, {
      headers: { Authorization: bearer },
      timeout: 12000
    });


    const profile = extractClientProfile(data.data);
    if (!profile) {
      console.error("[CMS_PROFILE] Profile extraction failed. Unexpected response shape from CMS.");
      throw new Error("Unable to extract profile from CMS response.");
    }
    return profile;
  } catch (err) {
    const status = err.response?.status;
    const body = err.response?.data;
    let bodyPreview = "";
    try {
      bodyPreview = typeof body === "string"
        ? body.slice(0, 500)
        : JSON.stringify(body || {}).slice(0, 500);
    } catch {
      bodyPreview = "<unserializable-response-body>";
    }

    console.error("[CMS_PROFILE] Failed to fetch profile.", {
      url: CMS_PROFILE_URL,
      status: status || "no-response",
      message: err.message,
      bodyPreview
    });
    throw err;
  }
}


// ─── Routes ────────────────────────────────────────────────────────────────────

app.get("/health", async (req, res) => {
  try {
    const { data } = await axios.get(`${PYTHON_API}/health`, { timeout: 5000 });
    res.json({ gateway: "ok", python_service: data });
  } catch {
    res.status(503).json({ gateway: "ok", python_service: "unreachable" });
  }
});


/**
 * POST /api/setup
 * Register a reference face.
 * Body (multipart/form-data):
 *   - label: string  (unique identifier)
 *   - image: file    (reference face image)
 */
app.post("/api/setup", upload.single("image"), async (req, res, next) => {
  try {
    const { label, imageBase64 } = req.body;
    if (!label) return res.status(400).json({ error: "Field 'label' is required." });

    const base64Image = req.file ? null : parseBase64Image(imageBase64);
    if (!req.file && !base64Image) {
      return res.status(400).json({ error: "Provide either 'image' (file) or valid 'imageBase64'." });
    }

    const fileBuffer = req.file ? req.file.buffer : base64Image.buffer;
    const fileName = req.file ? req.file.originalname : base64Image.fileName;
    const mimeType = req.file ? req.file.mimetype : base64Image.mimeType;

    const result = await forwardToPython(
      "/setup",
      { label },
      fileBuffer,
      fileName,
      mimeType
    );
    res.status(201).json(result);
  } catch (err) {
    if (err.response) {
      return res.status(err.response.status).json(err.response.data);
    }
    next(err);
  }
});


/**
 * POST /api/match
 * Match an uploaded face against a registered reference.
 * Body (multipart/form-data):
 *   - label:     string  (registered label to compare against)
 *   - image:     file    (face image to verify)
 *   - tolerance: number  (optional, default 0.5)
 */
app.post("/api/match", upload.single("image"), async (req, res, next) => {
  try {
    const { label, tolerance = "0.5", imageBase64 } = req.body;
    if (!label) return res.status(400).json({ error: "Field 'label' is required." });

    const base64Image = req.file ? null : parseBase64Image(imageBase64);
    if (!req.file && !base64Image) {
      return res.status(400).json({ error: "Provide either 'image' (file) or valid 'imageBase64'." });
    }

    const fileBuffer = req.file ? req.file.buffer : base64Image.buffer;
    const fileName = req.file ? req.file.originalname : base64Image.fileName;
    const mimeType = req.file ? req.file.mimetype : base64Image.mimeType;

    const result = await forwardToPython(
      "/match",
      { label, tolerance },
      fileBuffer,
      fileName,
      mimeType
    );
    res.status(200).json(result);
  } catch (err) {
    if (err.response) {
      return res.status(err.response.status).json(err.response.data);
    }
    next(err);
  }
});


/**
 * POST /api/match-client
 * Single-image match flow:
 * 1) Pull reference Aadhaar image from CMS profile using token
 * 2) Match uploaded image against pulled reference image
 * 3) Return match result + client_id + matched image base64 (on success)
 */
app.post("/api/match-client", upload.single("image"), async (req, res, next) => {
  try {
    const { tolerance = "0.5", imageBase64, token, sessionToken } = req.body;

    const base64Image = req.file ? null : parseBase64Image(imageBase64);
    if (!req.file && !base64Image) {
      return res.status(400).json({ error: "Provide either 'image' (file) or valid 'imageBase64'." });
    }

    const fileBuffer = req.file ? req.file.buffer : base64Image.buffer;
    const fileName = req.file ? req.file.originalname : base64Image.fileName;
    const mimeType = req.file ? req.file.mimetype : base64Image.mimeType;

    const profile = await fetchProfileByToken(token || sessionToken || "", req.headers.authorization || "");
    const clientId = extractClientId(profile);
    const aadhaarImageBase64 = extractAadhaarImage(profile);

    if (!aadhaarImageBase64) {
      return res.status(422).json({ error: "Aadhaar image not found in CMS profile." });
    }

    const result = await forwardToPython(
      "/match-direct",
      {
        tolerance,
        referenceImageBase64: aadhaarImageBase64
      },
      fileBuffer,
      fileName,
      mimeType
    );

    result.client_id = clientId;
    if (result.match) {
      result.matched_image_base64 = aadhaarImageBase64;
    }

    return res.status(200).json(result);
  } catch (err) {
    console.error("[MATCH_CLIENT] Request failed.", {
      status: err.response?.status || "internal",
      message: err.message
    });
    if (err.response) {
      return res.status(err.response.status).json(err.response.data);
    }
    if (err.message && err.message.toLowerCase().includes("missing token")) {
      return res.status(401).json({ error: err.message });
    }
    next(err);
  }
});


/**
 * GET /api/labels
 * List all registered face labels.
 */
app.get("/api/labels", async (req, res, next) => {
  try {
    const { data } = await axios.get(`${PYTHON_API}/labels`, { timeout: 10000 });
    res.json(data);
  } catch (err) {
    if (err.response) return res.status(err.response.status).json(err.response.data);
    next(err);
  }
});


/**
 * DELETE /api/labels/:label
 * Remove a registered face label.
 */
app.delete("/api/labels/:label", async (req, res, next) => {
  try {
    const { data } = await axios.delete(`${PYTHON_API}/labels/${encodeURIComponent(req.params.label)}`, { timeout: 10000 });
    res.json(data);
  } catch (err) {
    if (err.response) return res.status(err.response.status).json(err.response.data);
    next(err);
  }
});


/**
 * POST /api/validate-signature
 * Validate if an image contains a proper signature.
 * Rejects simple lines, random scribbles, and empty signatures.
 * Body (multipart/form-data):
 *   - image: file    (signature image to validate, optional if imageBase64 provided)
 *   - imageBase64: string  (signature as base64, optional if image file provided)
 */
app.post("/api/validate-signature", upload.single("image"), async (req, res, next) => {
  try {
    const { imageBase64 } = req.body;
    const base64Image = req.file ? null : parseBase64Image(imageBase64);
    
    if (!req.file && !base64Image) {
      return res.status(400).json({ error: "Provide either 'image' (file) or valid 'imageBase64'." });
    }

    const fileBuffer = req.file ? req.file.buffer : base64Image.buffer;
    const fileName = req.file ? req.file.originalname : base64Image.fileName;
    const mimeType = req.file ? req.file.mimetype : base64Image.mimeType;

    const result = await forwardToPython(
      "/validate-signature",
      {},
      fileBuffer,
      fileName,
      mimeType
    );
    res.status(200).json(result);
  } catch (err) {
    if (err.response) {
      return res.status(err.response.status).json(err.response.data);
    }
    next(err);
  }
});


/**
 * GET /api/images/:label
 * Proxy registered reference image for a label from Python backend.
 */
app.get("/api/images/:label", async (req, res, next) => {
  try {
    const response = await axios.get(`${PYTHON_API}/images/${encodeURIComponent(req.params.label)}`, {
      responseType: "arraybuffer",
      timeout: 10000
    });

    res.setHeader("Content-Type", response.headers["content-type"] || "image/jpeg");
    res.send(Buffer.from(response.data));
  } catch (err) {
    if (err.response) {
      res.status(err.response.status);
      if (err.response.data) {
        try {
          const body = JSON.parse(Buffer.from(err.response.data).toString("utf-8"));
          return res.json(body);
        } catch {
          return res.json({ error: "Unable to fetch image." });
        }
      }
    }
    next(err);
  }
});


app.use(errorHandler);

app.listen(PORT, () => {
  console.log(`[Gateway] Running on http://localhost:${PORT}`);
  console.log(`[Gateway] Forwarding to Python API: ${PYTHON_API}`);
});
