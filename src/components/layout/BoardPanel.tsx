import type { BackgroundPreset, BoardPiece } from '../../store/wardrobeStore'
import type { Garment } from '../../data/garments'
import { BoardCanvas } from '../board/BoardCanvas'
import { LayerStrip } from './LayerStrip'

type Props = {
  pieces: BoardPiece[]
  garments: Garment[]
  selectedPieceId: string | null
  backgroundPreset: BackgroundPreset
  onSelect: (id: string | null) => void
  onChangePiece: (id: string, patch: Partial<BoardPiece>) => void
  onDelete: (id: string) => void
  onReorder: (id: string, z: number) => void
  onDropGarment: (garmentId: string, point: { x: number; y: number }) => void
}

export function BoardPanel(props: Props) {
  return (
    <section className="board-panel">
      <BoardCanvas {...props} />
      <LayerStrip
        pieces={props.pieces}
        selectedPieceId={props.selectedPieceId}
        onSelect={(id) => props.onSelect(id)}
        onMove={props.onReorder}
      />
    </section>
  )
}
