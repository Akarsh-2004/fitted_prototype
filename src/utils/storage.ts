const KEY = 'vestir.saved.state.v1'

export const loadState = <T>(fallback: T): T => {
  try {
    const raw = localStorage.getItem(KEY)
    return raw ? (JSON.parse(raw) as T) : fallback
  } catch {
    return fallback
  }
}

export const saveState = <T>(value: T) => {
  try {
    localStorage.setItem(KEY, JSON.stringify(value))
  } catch {
    // Quota can be exceeded by large base64 image payloads.
    // Fallback to a minimal snapshot instead of throwing in render effects.
    try {
      const minimal = {
        ...(value as Record<string, unknown>),
        savedLooks: [],
        garments: [],
      }
      localStorage.setItem(KEY, JSON.stringify(minimal))
    } catch {
      // ignore hard failure, app should remain usable
    }
  }
}
