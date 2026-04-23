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
const CMS_DIGIO_SELFIE_URL = process.env.CMS_DIGIO_SELFIE_URL || "https://cms.ezwealth.in/api/auth-client/digio-selfie";
const CMS_UPDATE_PROFILE_URL = process.env.CMS_UPDATE_PROFILE_URL || "http://192.168.1.22:8000/api/clients/update-client-profile";
const CMS_BANK_STATEMENT_URL = process.env.CMS_BANK_STATEMENT_URL || "https://cms.ezwealth.in/api/auth-client/bank-statement-pdf";

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

// Multer for document+face uploads — accepts images AND PDFs
const uploadWithDoc = multer({
  storage: memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const allowedImages = ["image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"];
    const allowedDocs = ["application/pdf"];
    if ([...allowedImages, ...allowedDocs].includes(file.mimetype)) {
      return cb(null, true);
    }
    cb(new Error(`Unsupported file type '${file.mimetype}'. Use JPG, PNG, WEBP, BMP, TIFF, or PDF.`));
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

async function forwardMultiFileToPython(endpoint, fields, files) {
  const form = new FormData();
  for (const [key, value] of Object.entries(fields)) {
    form.append(key, String(value));
  }
  for (const { fieldName, buffer, fileName, mimeType } of files) {
    if (buffer) {
      form.append(fieldName, buffer, { filename: fileName, contentType: mimeType });
    }
  }
  const response = await axios.post(`${PYTHON_API}${endpoint}`, form, {
    headers: form.getHeaders(),
    timeout: 30000
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

    if (!match) {
      if (buffer.subarray(0, 8).equals(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]))) {
        mimeType = "image/png";
      } else if (buffer.subarray(0, 3).equals(Buffer.from([0xff, 0xd8, 0xff]))) {
        mimeType = "image/jpeg";
      } else if (buffer.subarray(0, 4).toString("ascii") === "RIFF" && buffer.subarray(8, 12).toString("ascii") === "WEBP") {
        mimeType = "image/webp";
      }
      if (!allowed[mimeType]) return null;
    }

    return {
      buffer,
      mimeType,
      fileName: `upload${allowed[mimeType]}`
    };
  } catch {
    return null;
  }
}

function buildBearerToken(rawToken, authorizationHeader) {
  const token = (rawToken || "").trim();
  const authHeader = (authorizationHeader || "").trim();

  if (authHeader) return authHeader;
  if (token) return token.toLowerCase().startsWith("bearer ") ? token : `Bearer ${token}`;
  return "";
}

function bufferToBase64(buffer) {
  return buffer.toString("base64");
}

function getImageInput(req) {
  const base64Image = req.file ? null : parseBase64Image(req.body.imageBase64);
  if (!req.file && !base64Image) {
    return null;
  }

  const fileBuffer = req.file ? req.file.buffer : base64Image.buffer;
  const fileName = req.file ? req.file.originalname : base64Image.fileName;
  const mimeType = req.file ? req.file.mimetype : base64Image.mimeType;

  return {
    fileBuffer,
    fileName,
    mimeType,
    base64: bufferToBase64(fileBuffer)
  };
}

async function submitCmsJson(url, payload, rawToken, authorizationHeader) {
  const bearer = buildBearerToken(rawToken, authorizationHeader);
  if (!bearer) {
    return {
      updated: false,
      skipped: true,
      message: "Skipped CMS save because token was not provided."
    };
  }

  const { data } = await axios.post(url, payload, {
    headers: {
      "Content-Type": "application/json",
      Authorization: bearer
    },
    timeout: 20000
  });

  return {
    updated: true,
    skipped: false,
    message: data?.message || "CMS save completed successfully.",
    data
  };
}

async function updateClientProfileImage(fieldName, imageValue, rawToken, authorizationHeader) {
  const bearer = buildBearerToken(rawToken, authorizationHeader);
  if (!bearer) {
    return {
      updated: false,
      skipped: true,
      message: "Skipped profile update because token was not provided."
    };
  }

  const payload = { [fieldName]: imageValue };
  const { data } = await axios.post(CMS_UPDATE_PROFILE_URL, payload, {
    headers: {
      "Content-Type": "application/json",
      Authorization: bearer
    },
    timeout: 15000
  });

  return {
    updated: true,
    skipped: false,
    message: `${fieldName} saved successfully.`,
    data
  };
}

async function saveBankStatementPdf(pdfBase64, rawToken, authorizationHeader) {
  return submitCmsJson(
    CMS_BANK_STATEMENT_URL,
    { pdfBase64 },
    rawToken,
    authorizationHeader
  );
}

function extractDigioSelfieImage(payload) {
  if (!payload || typeof payload !== "object") return null;

  const candidates = [
    payload?.selfie,
    payload?.data?.selfie,
    payload?.data?.data?.selfie
  ];

  for (const image of candidates) {
    if (typeof image === "string" && image.trim()) {
      return image.trim();
    }
  }
  return null;
}

async function fetchDigioSelfieByToken(authorizationHeader) {
  const authHeader = (authorizationHeader || "").trim();
  if (!authHeader) {
    console.error("[CMS_DIGIO_SELFIE] Missing Authorization header for selfie fetch.");
    throw new Error("Missing Authorization header.");
  }

  try {
    const { data } = await axios.get(CMS_DIGIO_SELFIE_URL, {
      // Token is forwarded exactly as received in request header.
      headers: { Authorization: authHeader },
      timeout: 12000
    });

    const selfieBase64 = extractDigioSelfieImage(data);
    if (!selfieBase64) {
      console.error("[CMS_DIGIO_SELFIE] Selfie extraction failed. Unexpected response shape from CMS.");
      throw new Error("Unable to extract selfie image from CMS response.");
    }
    return selfieBase64;
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

    console.error("[CMS_DIGIO_SELFIE] Failed to fetch selfie.", {
      url: CMS_DIGIO_SELFIE_URL,
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
 * 1) Pull reference selfie image from digio-selfie API using Authorization header as-is
 * 2) Match uploaded image against pulled reference image
 * 3) Return match result + matched image base64 (on success)
 */
app.post("/api/match-client", upload.single("image"), async (req, res, next) => {
  try {
    const { tolerance = "0.5", imageBase64, token, sessionToken } = req.body;

    const imageInput = getImageInput(req);
    if (!imageInput) {
      return res.status(400).json({ error: "Provide either 'image' (file) or valid 'imageBase64'." });
    }

    const digioSelfieBase64 = await fetchDigioSelfieByToken(req.headers.authorization || "");

    if (!digioSelfieBase64) {
      return res.status(422).json({ error: "Selfie image not found in digio-selfie API response." });
    }

    const result = await forwardToPython(
      "/match-direct",
      {
        tolerance,
        referenceImageBase64: digioSelfieBase64
      },
      imageInput.fileBuffer,
      imageInput.fileName,
      imageInput.mimeType
    );

    if (result.match) {
      result.matched_image_base64 = digioSelfieBase64;
      try {
        const profileUpdate = await updateClientProfileImage(
          "selfieEkyc",
          imageInput.base64,
          token || sessionToken || "",
          req.headers.authorization || ""
        );
        result.profile_updated = profileUpdate.updated;
        result.profile_update_message = profileUpdate.message;
      } catch (saveError) {
        console.error("[PROFILE_UPDATE] Failed to save selfieEkyc.", {
          status: saveError.response?.status || "internal",
          message: saveError.message
        });
        result.profile_updated = false;
        result.profile_update_message = "Face matched but selfieEkyc save failed.";
        result.profile_update_error = saveError.response?.data || saveError.message;
      }
    } else {
      result.profile_updated = false;
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
    if (err.message && err.message.toLowerCase().includes("missing authorization header")) {
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
    const { token, sessionToken } = req.body;
    const imageInput = getImageInput(req);

    if (!imageInput) {
      return res.status(400).json({ error: "Provide either 'image' (file) or valid 'imageBase64'." });
    }

    const result = await forwardToPython(
      "/validate-signature",
      {},
      imageInput.fileBuffer,
      imageInput.fileName,
      imageInput.mimeType
    );

    if (result.valid) {
      try {
        const signatureBase64ForSave = result.normalized_base64 || imageInput.base64;
        const profileUpdate = await updateClientProfileImage(
          "wetSignature",
          signatureBase64ForSave,
          token || sessionToken || "",
          req.headers.authorization || ""
        );
        result.profile_updated = profileUpdate.updated;
        result.profile_update_message = profileUpdate.message;
      } catch (saveError) {
        console.error("[PROFILE_UPDATE] Failed to save wetSignature.", {
          status: saveError.response?.status || "internal",
          message: saveError.message
        });
        result.profile_updated = false;
        result.profile_update_message = "Signature validated but wetSignature save failed.";
        result.profile_update_error = saveError.response?.data || saveError.message;
      }
    } else {
      result.profile_updated = false;
    }

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


/**
 * POST /api/verify-document
 * Check if username appears in a bank statement document.
 * Body (multipart/form-data):
 *   - username:  string  (full name to search in document)
 *   - document:  file    (bank statement PDF or image)
 */
app.post(
  "/api/verify-document",
  uploadWithDoc.single("document"),
  async (req, res, next) => {
    try {
      const { username, token, sessionToken } = req.body;
      if (!username) return res.status(400).json({ error: "Field 'username' is required." });
      if (!req.file)  return res.status(400).json({ error: "Field 'document' (file) is required." });

      const result = await forwardMultiFileToPython(
        "/verify-document",
        { username },
        [{ fieldName: "document", buffer: req.file.buffer, fileName: req.file.originalname, mimeType: req.file.mimetype }]
      );

      if (result.verified) {
        try {
          const bankStatementSave = await saveBankStatementPdf(
            bufferToBase64(req.file.buffer),
            token || sessionToken || "",
            req.headers.authorization || ""
          );
          result.bank_statement_saved = bankStatementSave.updated;
          result.bank_statement_save_message = bankStatementSave.message;
          if (bankStatementSave.data) {
            result.bank_statement = bankStatementSave.data;
          }
        } catch (saveError) {
          console.error("[PROFILE_UPDATE] Failed to save bank statement PDF.", {
            status: saveError.response?.status || "internal",
            message: saveError.message
          });
          result.bank_statement_saved = false;
          result.bank_statement_save_message = "Document verified but bank statement save failed.";
          result.bank_statement_save_error = saveError.response?.data || saveError.message;
        }
      } else {
        result.bank_statement_saved = false;
      }

      res.status(200).json(result);
    } catch (err) {
      if (err.response) return res.status(err.response.status).json(err.response.data);
      next(err);
    }
  }
);


app.use(errorHandler);

app.listen(PORT, () => {
  console.log(`[Gateway] Running on http://localhost:${PORT}`);
  console.log(`[Gateway] Forwarding to Python API: ${PYTHON_API}`);
});
