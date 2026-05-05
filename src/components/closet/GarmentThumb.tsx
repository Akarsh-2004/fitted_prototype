import { memo } from 'react'
import type { Garment } from '../../data/garments'
import { GarmentSVG } from '../garments/GarmentSVG'
import { Badge } from '../ui/Badge'

type Props = {
  garment: Garment
  onAdd: (garmentId: string) => void
}

export const GarmentThumb = memo(function GarmentThumb({ garment, onAdd }: Props) {
  return (
    <button type="button" className="garment-thumb" onClick={() => onAdd(garment.id)} draggable>
      <div className="thumb-art" draggable onDragStart={(e) => e.dataTransfer.setData('text/garment', garment.id)}>
        {garment.kind === 'svg' && garment.svgKey ? <GarmentSVG type={garment.svgKey} color={garment.palette[0]} /> : null}
        {garment.kind === 'upload' && garment.imageSrc ? <img src={garment.imageSrc} alt={garment.name} className="upload-thumb-img" /> : null}
        {garment.kind === 'ai' ? (
          <div className="ai-placeholder">
            <span className="ai-shimmer" />
            <Badge>✨</Badge>
          </div>
        ) : null}
      </div>
      <div className="thumb-meta">
        <strong>{garment.name}</strong>
        {garment.kind === 'upload' && garment.processingState ? (
          <small className="upload-state">
            {garment.processingState}
            {garment.processingMeta ? ` · q${garment.processingMeta.maskQuality.toFixed(2)}` : ''}
          </small>
        ) : null}
        <div className="thumb-swatches">
          {garment.palette.map((p) => (
            <span key={p} className="swatch" style={{ background: p }} />
          ))}
        </div>
      </div>
    </button>
  )
})
