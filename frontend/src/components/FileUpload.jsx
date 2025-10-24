// src/components/FileUpload.jsx

"use client";

import React, { useState } from "react";

export default function FileUpload({ onUploadComplete }) {
  /**
   * Props: 
   *   onUploadComplete: (dataset_id) => void
   *     A callback so parent can know which dataset_id to pass along in future chat/analysis calls.
   */

  const [selectedFile, setSelectedFile] = useState(null);
  const [previewRows, setPreviewRows] = useState([]);
  const [columns, setColumns] = useState([]);
  const [metadata, setMetadata] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [filename, setFilename] = useState("");

  const handleFileChange = (e) => {
    setError(null);
    if (e.target.files && e.target.files.length > 0) {
      setSelectedFile(e.target.files[0]);
      setFilename(e.target.files[0].name);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError("Please select a file first.");
      return;
    }
    setUploading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const resp = await fetch("http://localhost:8000/api/upload", {
        method: "POST",
        // Add auth headers here if needed, e.g. Authorization: `Bearer ${token}`
        body: formData,
      });

      if (!resp.ok) {
        const errBody = await resp.json();
        throw new Error(errBody.detail || "Upload failed");
      }

      const data = await resp.json();
      // data: { dataset_id, filename, preview_rows, metadata }
      setPreviewRows(data.preview_rows || []);
      setColumns(data.preview_rows.length > 0 ? Object.keys(data.preview_rows[0]) : []);
      setMetadata(data.metadata || null);

      // Tell parent which dataset_id to use for subsequent chat calls:
      onUploadComplete && onUploadComplete(data.dataset_id);
    } catch (err) {
      console.error(err);
      setError(err.message || "Upload error");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div
      style={{
        border: "1px solid #ccc",
        borderRadius: 8,
        padding: 16,
        marginBottom: 24,
      }}
    >
      <h2>Upload Data File</h2>
      <p>You may upload an SPSS (<code>.sav</code>), CSV, or Excel file.</p>
      <input
        type="file"
        accept=".sav,.csv,.xlsx,.xls"
        onChange={handleFileChange}
      />{" "}
      <button
        onClick={handleUpload}
        disabled={uploading || !selectedFile}
        style={{
          marginLeft: 8,
          padding: "4px 12px",
          backgroundColor: "#0066CC",
          color: "#fff",
          border: "none",
          borderRadius: 4,
          cursor: uploading ? "not-allowed" : "pointer",
        }}
      >
        {uploading ? "Uploading…" : "Upload File"}
      </button>
      {error && (
        <div style={{ color: "red", marginTop: 8 }}>Error: {error}</div>
      )}

      {previewRows.length > 0 && (
        <div style={{ marginTop: 24 }}>
          <h3>Preview of first 10 rows ({filename})</h3>
          <div style={{ overflowX: "auto", border: "1px solid #ddd" }}>
            <table
              style={{
                borderCollapse: "collapse",
                width: "100%",
                background: "#fafafa",
              }}
            >
              <thead>
                <tr>
                  {columns.map((col) => (
                    <th
                      key={col}
                      style={{
                        border: "1px solid #ddd",
                        padding: "8px",
                        backgroundColor: "#f0f0f0",
                        textAlign: "left",
                      }}
                    >
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {previewRows.map((row, idx) => (
                  <tr key={idx}>
                    {columns.map((col) => (
                      <td
                        key={col}
                        style={{
                          border: "1px solid #ddd",
                          padding: "8px",
                        }}
                      >
                        {row[col] !== null && row[col] !== undefined
                          ? String(row[col])
                          : ""}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {metadata && (
        <div style={{ marginTop: 24 }}>
          <h3>SPSS Metadata (JSON)</h3>
          <pre
            style={{
              maxHeight: 300,
              overflowY: "auto",
              backgroundColor: "#272822",
              color: "#f8f8f2",
              padding: 12,
              borderRadius: 4,
            }}
          >
            {JSON.stringify(metadata, null, 2)}
          </pre>
          <p style={{ fontSize: 12, color: "#666" }}>
            (All SPSS metadata fields have been captured above.)
          </p>
        </div>
      )}
    </div>
  );
}
