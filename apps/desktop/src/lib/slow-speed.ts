export const SLOW_TOKENS_PER_S = 5;
export const SLOW_STREAK = 2;
const KEY = "localSlowState";

export type SlowState = { streak: number; dismissed: boolean };

export function recordAnswer(state: SlowState, tokensPerS: number | null): SlowState {
  if (tokensPerS === null) return state;
  return { ...state, streak: tokensPerS < SLOW_TOKENS_PER_S ? state.streak + 1 : 0 };
}

export const shouldSuggest = (state: SlowState) => !state.dismissed && state.streak >= SLOW_STREAK;

/** Only under a finished answer that was itself timed: cloud and stub answers carry no speed. */
export const shouldShowBanner = (state: SlowState, tokensPerS: number | null, done: boolean) =>
  done && tokensPerS !== null && shouldSuggest(state);

export function loadSlowState(): SlowState {
  try {
    const raw = JSON.parse(localStorage.getItem(KEY) ?? "null");
    if (raw && typeof raw.streak === "number" && typeof raw.dismissed === "boolean") {
      return { streak: raw.streak, dismissed: raw.dismissed };
    }
  } catch {
    // unavailable or corrupt storage: start fresh
  }
  return { streak: 0, dismissed: false };
}

export function saveSlowState(state: SlowState): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(state));
  } catch {
    // per-viewer convenience only
  }
}
