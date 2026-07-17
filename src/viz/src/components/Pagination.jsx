const MAX_VISIBLE_PAGES = 20;

// Pure, exported for direct unit testing: given the current page and total
// page count, which page numbers should be shown as buttons -- a window of
// at most MAX_VISIBLE_PAGES, centered on the current page where possible.
export function visiblePageWindow(page, pageCount) {
  if (pageCount <= MAX_VISIBLE_PAGES) {
    return Array.from({ length: pageCount }, (_, i) => i + 1);
  }
  let start = Math.max(1, page - Math.floor(MAX_VISIBLE_PAGES / 2));
  let end = start + MAX_VISIBLE_PAGES - 1;
  if (end > pageCount) {
    end = pageCount;
    start = end - MAX_VISIBLE_PAGES + 1;
  }
  return Array.from({ length: end - start + 1 }, (_, i) => start + i);
}

export default function Pagination({ page, pageCount, onPageChange }) {
  if (pageCount <= 1) return null;
  const pages = visiblePageWindow(page, pageCount);

  return (
    <div className="pagination">
      <button disabled={page === 1} onClick={() => onPageChange(1)}>
        &laquo; First
      </button>
      <button disabled={page === 1} onClick={() => onPageChange(page - 1)}>
        &lsaquo; Prev
      </button>
      {pages[0] > 1 && <span className="pagination-ellipsis">&hellip;</span>}
      {pages.map((p) => (
        <button key={p} className={p === page ? "active" : ""} onClick={() => onPageChange(p)}>
          {p}
        </button>
      ))}
      {pages[pages.length - 1] < pageCount && <span className="pagination-ellipsis">&hellip;</span>}
      <button disabled={page === pageCount} onClick={() => onPageChange(page + 1)}>
        Next &rsaquo;
      </button>
      <button disabled={page === pageCount} onClick={() => onPageChange(pageCount)}>
        Last &raquo;
      </button>
    </div>
  );
}
