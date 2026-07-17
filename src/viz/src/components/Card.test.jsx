import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import Card from "./Card.jsx";
import { cardInfo } from "../cardData.js";

describe("Card", () => {
  it("resolves a known card name to its expected art file", () => {
    const { artFile } = cardInfo("Urza's Mine");
    expect(artFile).toBe("urzas-mine.png");

    render(<Card name="Urza's Mine" />);
    const img = screen.getByAltText("Urza's Mine");
    expect(img.getAttribute("src")).toBe("/card_art/urzas-mine.png");
    expect(img.closest(".card").className).not.toMatch(/card-tapped/);
  });

  it("applies the tapped rotation class when tapped", () => {
    render(<Card name="Forest" tapped />);
    const img = screen.getByAltText("Forest");
    expect(img.closest(".card").className).toMatch(/card-tapped/);
  });

  it("falls back to a text label if art fails to load", () => {
    render(<Card name="Forest" />);
    const img = screen.getByAltText("Forest");
    fireEvent.error(img);
    expect(screen.getByText("Forest")).toBeTruthy();
  });
});
