// Shared between cardData.js and scripts/fetch-card-art.mjs -- both need
// the exact same filename per card, or art will fail to load silently.
export function slug(name) {
  return name
    .toLowerCase()
    .replace(/'/g, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/(^-|-$)/g, "");
}
