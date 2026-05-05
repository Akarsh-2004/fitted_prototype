import type { OutfitPreset } from '../../data/outfits'

export function OutfitPresets({
  presets,
  onLoad,
}: {
  presets: OutfitPreset[]
  onLoad: (id: string) => void
}) {
  return (
    <div className="outfit-row">
      {presets.map((p) => (
        <button key={p.id} type="button" className="preset-btn" onClick={() => onLoad(p.id)}>
          {p.name}
        </button>
      ))}
    </div>
  )
}
