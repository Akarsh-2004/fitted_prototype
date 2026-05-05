/** Simple hex helpers for building garment palettes from a few seed colors */

function clamp(n: number, min: number, max: number) {
  return Math.min(max, Math.max(min, n))
}

export function hexToRgb(hex: string): { r: number; g: number; b: number } {
  const h = hex.replace('#', '').trim()
  const full = h.length === 3 ? h.split('').map((c) => c + c).join('') : h
  const n = parseInt(full, 16)
  return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 }
}

export function rgbToHex(r: number, g: number, b: number) {
  const to = (v: number) => clamp(Math.round(v), 0, 255).toString(16).padStart(2, '0')
  return `#${to(r)}${to(g)}${to(b)}`
}

/** amount: -1 (darker) … 1 (lighter), roughly perceptual */
export function adjustHex(hex: string, amount: number) {
  const { r, g, b } = hexToRgb(hex)
  const t = amount
  const f = (c: number) => (t >= 0 ? c + (255 - c) * t : c * (1 + t))
  return rgbToHex(f(r), f(g), f(b))
}

export function mixHex(a: string, b: string, t: number) {
  const A = hexToRgb(a)
  const B = hexToRgb(b)
  const lerp = (x: number, y: number) => x + (y - x) * clamp(t, 0, 1)
  return rgbToHex(lerp(A.r, B.r), lerp(A.g, B.g), lerp(A.b, B.b))
}
