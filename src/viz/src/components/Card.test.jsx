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

  it("shows a slot badge only when a slot is given", () => {
    const { rerender } = render(<Card name="Forest" />);
    expect(screen.getByAltText("Forest").closest(".card").querySelector(".card-slot-badge")).toBeNull();

    rerender(<Card name="Forest" slot={2} />);
    const badge = screen.getByAltText("Forest").closest(".card").querySelector(".card-slot-badge");
    expect(badge).toBeTruthy();
    expect(badge.textContent).toBe("2");
  });

  it("shows an enchant caption naming the actual linked card, not just a generic flag", () => {
    const { rerender } = render(<Card name="Rancor" />);
    expect(screen.getByAltText("Rancor").closest(".card").querySelector(".card-enchant-caption")).toBeNull();

    rerender(<Card name="Rancor" enchanting="Slippery Bogle (slot 1)" />);
    const auraCaption = screen.getByAltText("Rancor").closest(".card").querySelector(".card-enchant-caption");
    expect(auraCaption.textContent).toBe("→ Slippery Bogle (slot 1)"); // the AURA names its own target

    rerender(<Card name="Slippery Bogle" enchantedBy={["Rancor", "Ancestral Mask"]} />);
    const creatureCaption = screen
      .getByAltText("Slippery Bogle")
      .closest(".card")
      .querySelector(".card-enchant-caption");
    expect(creatureCaption.textContent).toBe("✦ Rancor, Ancestral Mask"); // the CREATURE lists what's on it
  });

  it("reports the same full shape to onHover regardless of zone", () => {
    const seen = [];
    render(
      <Card
        name="Rancor"
        slot={1}
        enchanting="Slippery Bogle (slot 1)"
        onHover={(detail) => seen.push(detail)}
      />
    );
    fireEvent.mouseEnter(screen.getByAltText("Rancor").closest(".card"));
    expect(seen[0]).toEqual({
      name: "Rancor", enchanting: "Slippery Bogle (slot 1)", enchantedBy: null,
      power: null, toughness: null, basePower: null, baseToughness: null, keywords: null,
    });
    fireEvent.mouseLeave(screen.getByAltText("Rancor").closest(".card"));
    expect(seen[1]).toBeNull();
  });

  it("reports power/toughness/keywords for a creature permanent", () => {
    const seen = [];
    render(
      <Card
        name="Slippery Bogle"
        power={3}
        toughness={1}
        basePower={1}
        baseToughness={1}
        keywords={["hexproof"]}
        onHover={(detail) => seen.push(detail)}
      />
    );
    fireEvent.mouseEnter(screen.getByAltText("Slippery Bogle").closest(".card"));
    expect(seen[0]).toMatchObject({ power: 3, toughness: 1, basePower: 1, baseToughness: 1, keywords: ["hexproof"] });
  });
});
