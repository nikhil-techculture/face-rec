function errorHandler(err, req, res, next) {
  if (err.code === "LIMIT_FILE_SIZE") {
    return res.status(413).json({ error: "File too large. Max 10MB allowed." });
  }
  if (err.code === "LIMIT_UNEXPECTED_FILE") {
    return res.status(400).json({ error: "Unexpected file field. Use 'image' as the field name." });
  }

  const status = err.status || err.statusCode || 500;
  const message = err.message || "Internal Server Error";
  console.error(`[ERROR] ${status} - ${message}`);
  res.status(status).json({ error: message });
}

module.exports = errorHandler;
