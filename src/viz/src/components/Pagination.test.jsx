import { describe, it, expect } from "vitest";
import { visiblePageWindow } from "./Pagination.jsx";

describe("visiblePageWindow", () => {
  it("shows every page when there are 20 or fewer", () => {
    expect(visiblePageWindow(1, 5)).toEqual([1, 2, 3, 4, 5]);
    expect(visiblePageWindow(10, 20)).toEqual(Array.from({ length: 20 }, (_, i) => i + 1));
  });

  it("caps at 20 visible pages, centered on the current page, for larger totals", () => {
    const window = visiblePageWindow(50, 100);
    expect(window).toHaveLength(20);
    expect(window).toContain(50);
    expect(window[0]).toBeGreaterThan(1);
    expect(window[window.length - 1]).toBeLessThan(100);
  });

  it("clamps the window at the start", () => {
    const window = visiblePageWindow(1, 100);
    expect(window).toHaveLength(20);
    expect(window[0]).toBe(1);
  });

  it("clamps the window at the end", () => {
    const window = visiblePageWindow(100, 100);
    expect(window).toHaveLength(20);
    expect(window[window.length - 1]).toBe(100);
  });
});
