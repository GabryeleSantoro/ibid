export type FontSize = "s" | "m" | "l";

const STORAGE_KEY = "ibid.fontSize";
const PX: Record<FontSize, string> = { s: "14px", m: "16px", l: "18px" };

export function parseFontSize(value: string | null): FontSize {
  return value === "s" || value === "l" ? value : "m";
}

export function getFontSize(): FontSize {
  try {
    return parseFontSize(localStorage.getItem(STORAGE_KEY));
  } catch {
    // Private windows and blocked site data both throw here.
    return "m";
  }
}

export function applyFontSize(size: FontSize) {
  document.documentElement.style.fontSize = PX[size];
}

export function setFontSize(size: FontSize) {
  applyFontSize(size);
  try {
    localStorage.setItem(STORAGE_KEY, size);
  } catch {
    // Not being able to remember the choice is not worth failing over.
  }
}
