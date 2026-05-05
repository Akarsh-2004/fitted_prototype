export const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value))

export const clampDropPoint = (x: number, y: number, width: number, height: number) => ({
  x: clamp(x, 0, Math.max(0, width - 180)),
  y: clamp(y, 0, Math.max(0, height - 220)),
})
