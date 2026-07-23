import { useState } from "react";
import { cardInfo } from "../cardData.js";

// slot/enchanting/enchantedBy/power/toughness/basePower/baseToughness/
// keywords: only ever set by a battlefield permanent (see GameView.jsx's
// computeBattlefieldLinks/BattlefieldRow) -- undefined/null for every
// other zone (hand, graveyard, exile), which never carries this
// information at all, and power/toughness/etc. are additionally null for
// a non-creature permanent even on the battlefield (see harness.
// _snapshot_player_state -- meaningless for a land/artifact/enchantment).
// onHover always receives the same full shape regardless of zone, so the
// side preview panel (App-level hoveredCard state) never needs to
// special-case which zone or permanent type a card came from.
export default function Card({
  name, tapped = false, size = "", width = null, slot = null,
  enchanting = null, enchantedBy = null,
  power = null, toughness = null, basePower = null, baseToughness = null, keywords = null,
  onHover = null,
}) {
  const [artFailed, setArtFailed] = useState(false);
  const { artFile } = cardInfo(name);
  const sizeClass = size ? `card-${size}` : "";
  const style = width != null ? { width: `${width}px` } : undefined;
  const isEnchanted = Boolean(enchanting) || (enchantedBy && enchantedBy.length > 0);
  // Always-visible caption naming the actual linked card(s), not just a
  // generic "something's enchanted here" flag (the small dot badge this
  // replaced -- report: it didn't say what was enchanting what, so it
  // read as decoration rather than information). "→" (an Aura,
  // pointing at its target) vs. "✦" (a creature, listing what's on
  // it) so the two directions read differently at a glance even before
  // the text itself is legible at battlefield-card size; the full text
  // is always in the title tooltip and the hover side panel too, for
  // whenever this caption itself has to truncate on a narrow card.
  const enchantCaption = enchanting ? `→ ${enchanting}` : isEnchanted ? `✦ ${enchantedBy.join(", ")}` : null;
  const enchantTitle = enchanting
    ? `Enchanting ${enchanting}`
    : isEnchanted
    ? `Enchanted by: ${enchantedBy.join(", ")}`
    : null;

  return (
    <div
      className={`card ${sizeClass} ${tapped ? "card-tapped" : ""}`}
      style={style}
      title={name}
      onMouseEnter={() => onHover?.({ name, enchanting, enchantedBy, power, toughness, basePower, baseToughness, keywords })}
      onMouseLeave={() => onHover?.(null)}
    >
      {slot != null && <span className="card-slot-badge">{slot}</span>}
      {enchantCaption && (
        <span className="card-enchant-caption" title={enchantTitle}>
          {enchantCaption}
        </span>
      )}
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
