import { useState, useCallback } from "react";
import { parseLog } from "../loadLog.js";

export default function DropZone({ onLoad }) {
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFile = useCallback(
    async (file) => {
      if (!file) return;
      try {
        const text = await file.text();
        const games = parseLog(text);
        setError(null);
        onLoad(games, file.name);
      } catch (e) {
        setError(e.message);
      }
    },
    [onLoad]
  );

  return (
    <div
      className={`drop-zone ${dragOver ? "drag-over" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        handleFile(e.dataTransfer.files[0]);
      }}
    >
      <p className="drop-zone-headline">Drop a games.json log here</p>
      <p className="drop-zone-sub">
        Produced by <code>harness.evaluate(..., log_path=...)</code>
      </p>
      <label className="browse-button">
        Browse for a file
        <input
          type="file"
          accept="application/json,.json"
          onChange={(e) => handleFile(e.target.files[0])}
          hidden
        />
      </label>
      {error && <p className="drop-zone-error">{error}</p>}
    </div>
  );
}
