import { Save, Download, Eraser, Image } from 'lucide-react'

type Props = {
  boardName: string
  onBoardName: (name: string) => void
  onClear: () => void
  onExport: () => void
  onSave: () => void
  onCycleBackground: () => void
  savedLooks: { id: string; name: string }[]
  onLoadLook: (id: string) => void
}

export function Toolbar({
  boardName,
  onBoardName,
  onClear,
  onExport,
  onSave,
  onCycleBackground,
  savedLooks,
  onLoadLook,
}: Props) {
  return (
    <header className="toolbar">
      <h1>vestir</h1>
      <input
        className="board-name"
        value={boardName}
        onChange={(e) => onBoardName(e.target.value)}
        title="Double click to edit"
      />
      <div className="toolbar-actions">
        <button type="button" onClick={onClear}><Eraser size={14} /> Clear</button>
        <button type="button" onClick={onExport}><Download size={14} /> Export PNG</button>
        <button type="button" onClick={onSave}><Save size={14} /> Save Look</button>
        <button type="button" onClick={onCycleBackground}><Image size={14} /> Background</button>
        <select onChange={(e) => onLoadLook(e.target.value)} value="">
          <option value="" disabled>
            Load Saved
          </option>
          {savedLooks.map((l) => (
            <option key={l.id} value={l.id}>
              {l.name}
            </option>
          ))}
        </select>
      </div>
    </header>
  )
}
