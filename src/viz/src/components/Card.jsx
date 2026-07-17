import { useState } from "react";
import { cardInfo } from "../cardData.js";

export default function Card({ name, tapped = false, size = "", width = null, onHover = null }) {
  const [artFailed, setArtFailed] = useState(false);
  const { artFile } = cardInfo(name);
  const sizeClass = size ? `card-${size}` : "";
  const style = width != null ? { width: `${width}px` } : undefined;

  return (
    <div
      className={`card ${sizeClass} ${tapped ? "card-tapped" : ""}`}
      style={style}
      title={name}
      onMouseEnter={() => onHover?.(name)}
      onMouseLeave={() => onHover?.(null)}
    >
      {artFailed ? (
        <div className="card-fallback">
          <span>{name}</span>
        </div>
      ) : (
        <img
          className="card-art"
          src={`/card_art/${artFile}`}
          alt={name}
          onError={() => setArtFailed(true)}
        />
      )}
    </div>
  );
}
