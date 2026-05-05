import { motion, AnimatePresence } from 'framer-motion'
import type { BoardPiece } from '../../store/wardrobeStore'
import type { Garment } from '../../data/garments'
import { ColorSwatch } from '../ui/ColorSwatch'
import { Slider } from '../ui/Slider'
import { AI_REMIX_COLORS } from '../../data/palettes'
import { useState } from 'react'

type Props = {
  piece?: BoardPiece
  garment?: Garment
  onChange: (patch: Partial<BoardPiece>) => void
  onDelete: () => void
  onDuplicate: () => void
  onBack: () => void
  onFront: () => void
  onRefine: (params: { edgeFeather: number; morphKernel: number; smoothness: number }) => void | Promise<void>
}

export function InspectorPanel({ piece, garment, onChange, onDelete, onDuplicate, onBack, onFront, onRefine }: Props) {
  const [edgeFeather, setEdgeFeather] = useState(3)
  const [morphKernel, setMorphKernel] = useState(3)
  const [smoothness, setSmoothness] = useState(0.5)
  return (
    <AnimatePresence>
      {piece ? (
        <motion.aside
          className="inspector"
          initial={{ x: 340, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          exit={{ x: 340, opacity: 0 }}
          transition={{ duration: 0.25 }}
        >
          <h3>{garment?.name ?? 'Piece'}</h3>
          <small>{garment?.category ?? 'Uploaded'}</small>
          <div className="inspector-swatches">
            {(garment?.palette ?? ['#d7c0a2']).map((p) => (
              <ColorSwatch key={p} color={p} active={piece.colorOverride === p} onClick={() => onChange({ colorOverride: p })} />
            ))}
          </div>
          <div className="grid-fields">
            <label>X<input type="number" value={Math.round(piece.x)} onChange={(e) => onChange({ x: Number(e.target.value) })} /></label>
            <label>Y<input type="number" value={Math.round(piece.y)} onChange={(e) => onChange({ y: Number(e.target.value) })} /></label>
            <label>W<input type="number" value={Math.round(piece.width)} onChange={(e) => onChange({ width: Number(e.target.value) })} /></label>
            <label>H<input type="number" value={Math.round(piece.height)} onChange={(e) => onChange({ height: Number(e.target.value) })} /></label>
          </div>
          <Slider label="Rotation" min={-180} max={180} value={piece.rotation} onChange={(rotation) => onChange({ rotation })} />
          <Slider label="Opacity" min={0.1} max={1} step={0.05} value={piece.opacity} onChange={(opacity) => onChange({ opacity })} />
          {garment?.kind === 'upload' ? (
            <div className="refine-box">
              <Slider label="Edge Feather" min={2} max={4} step={1} value={edgeFeather} onChange={setEdgeFeather} />
              <Slider label="Morph Kernel" min={3} max={9} step={2} value={morphKernel} onChange={setMorphKernel} />
              <Slider label="Smoothness" min={0} max={1} step={0.1} value={smoothness} onChange={setSmoothness} />
              <button type="button" onClick={() => onRefine({ edgeFeather, morphKernel, smoothness })}>
                Refine Cutout
              </button>
            </div>
          ) : null}
          <div className="inspector-actions">
            <button type="button" onClick={onDuplicate}>Duplicate</button>
            <button type="button" onClick={onDelete}>Delete</button>
            <button type="button" onClick={onBack}>Send Back</button>
            <button type="button" onClick={onFront}>Bring Front</button>
            <button
              type="button"
              onClick={() => {
                const i = AI_REMIX_COLORS.indexOf(piece.colorOverride)
                const next = AI_REMIX_COLORS[(i + 1) % AI_REMIX_COLORS.length] ?? AI_REMIX_COLORS[0]
                if (next) onChange({ colorOverride: next })
              }}
            >
              AI Remix
            </button>
          </div>
        </motion.aside>
      ) : null}
    </AnimatePresence>
  )
}
