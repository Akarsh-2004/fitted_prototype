type Props = {
  label: string
  min: number
  max: number
  step?: number
  value: number
  onChange: (value: number) => void
}

export function Slider({ label, min, max, step = 1, value, onChange }: Props) {
  return (
    <label className="field">
      <span>{label}</span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  )
}
