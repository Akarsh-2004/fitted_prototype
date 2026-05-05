import { create } from 'zustand'
import { GARMENTS, type Category, type Garment } from '../data/garments'
import { OUTFIT_PRESETS } from '../data/outfits'

export type BackgroundPreset =
  | 'studio'
  | 'night'
  | 'linen'
  | 'marble'
  | 'outdoors'
  | 'grid'

export type BoardPiece = {
  id: string
  garmentId: string
  x: number
  y: number
  width: number
  height: number
  rotation: number
  opacity: number
  flipX: boolean
  flipY: boolean
  zIndex: number
  colorOverride: string
}

type SavedLook = {
  id: string
  name: string
  boardPieces: BoardPiece[]
  backgroundPreset: BackgroundPreset
  garments?: Garment[]
}

type Snapshot = Pick<WardrobeState, 'boardPieces' | 'backgroundPreset'>

type WardrobeState = {
  garments: Garment[]
  boardPieces: BoardPiece[]
  selectedPieceId: string | null
  activeCategory: Category
  activePalette: string
  backgroundPreset: BackgroundPreset
  savedLooks: SavedLook[]
  boardName: string
  processingState: 'idle' | 'uploading' | 'processing' | 'refining' | 'done' | 'failed'
  history: Snapshot[]
  historyIndex: number
  addPiece: (garmentId: string, point?: { x: number; y: number }) => void
  updatePiece: (id: string, patch: Partial<BoardPiece>) => void
  removePiece: (id: string) => void
  duplicatePiece: (id: string) => void
  reorderPiece: (id: string, toZ: number) => void
  setSelectedPieceId: (id: string | null) => void
  setActiveCategory: (category: Category) => void
  setActivePalette: (color: string) => void
  setBackgroundPreset: (preset: BackgroundPreset) => void
  cycleBackgroundPreset: () => void
  clearBoard: () => void
  loadOutfitPreset: (outfitId: string) => void
  saveLook: (name: string) => void
  loadLook: (id: string) => void
  setBoardName: (name: string) => void
  addUploadedGarment: (name: string, imageSrc: string) => string
  updateGarment: (id: string, patch: Partial<Garment>) => void
  hydrateUploadedGarments: (garments: Garment[]) => void
  setSavedLooks: (looks: SavedLook[]) => void
  setProcessingState: (state: WardrobeState['processingState']) => void
  pushHistory: () => void
  undo: () => void
}

const BACKGROUND_ORDER: BackgroundPreset[] = ['studio', 'night', 'linen', 'marble', 'outdoors', 'grid']

const mkId = () => `${Date.now()}-${Math.random().toString(16).slice(2, 8)}`

const toSnapshot = (s: WardrobeState): Snapshot => ({
  boardPieces: s.boardPieces,
  backgroundPreset: s.backgroundPreset,
})

const dedupeGarmentsById = (garments: Garment[]) => {
  const byId = new Map<string, Garment>()
  for (const garment of garments) byId.set(garment.id, garment)
  return Array.from(byId.values())
}

export const useWardrobeStore = create<WardrobeState>((set, get) => ({
  garments: GARMENTS,
  boardPieces: [],
  selectedPieceId: null,
  activeCategory: 'all',
  activePalette: '#d7c0a2',
  backgroundPreset: 'studio',
  savedLooks: [],
  boardName: 'Untitled Board',
  history: [],
  historyIndex: -1,
  processingState: 'idle',
  addPiece: (garmentId, point) => {
    const garment = get().garments.find((g) => g.id === garmentId)
    if (!garment) return
    const z = get().boardPieces.length + 1
    const width = Math.max(100, 180 * garment.aspectRatio)
    const height = Math.max(120, 180)
    const next: BoardPiece = {
      id: mkId(),
      garmentId,
      x: point?.x ?? 160 + Math.random() * 80,
      y: point?.y ?? 120 + Math.random() * 120,
      width,
      height,
      rotation: 0,
      opacity: 1,
      flipX: false,
      flipY: false,
      zIndex: z,
      colorOverride: garment.palette[0] ?? '#d7c0a2',
    }
    set((s) => ({ boardPieces: [...s.boardPieces, next], selectedPieceId: next.id }))
    get().pushHistory()
  },
  updatePiece: (id, patch) => {
    set((s) => ({
      boardPieces: s.boardPieces.map((p) => (p.id === id ? { ...p, ...patch } : p)),
    }))
  },
  removePiece: (id) => {
    set((s) => ({
      boardPieces: s.boardPieces.filter((p) => p.id !== id),
      selectedPieceId: s.selectedPieceId === id ? null : s.selectedPieceId,
    }))
    get().pushHistory()
  },
  duplicatePiece: (id) => {
    const piece = get().boardPieces.find((p) => p.id === id)
    if (!piece) return
    const maxZ = Math.max(1, ...get().boardPieces.map((p) => p.zIndex))
    const clone: BoardPiece = { ...piece, id: mkId(), x: piece.x + 18, y: piece.y + 18, zIndex: maxZ + 1 }
    set((s) => ({ boardPieces: [...s.boardPieces, clone], selectedPieceId: clone.id }))
    get().pushHistory()
  },
  reorderPiece: (id, toZ) => {
    set((s) => ({
      boardPieces: s.boardPieces.map((p) => (p.id === id ? { ...p, zIndex: Math.max(1, toZ) } : p)),
    }))
    get().pushHistory()
  },
  setSelectedPieceId: (id) => set({ selectedPieceId: id }),
  setActiveCategory: (activeCategory) => set({ activeCategory }),
  setActivePalette: (activePalette) => set({ activePalette }),
  setBackgroundPreset: (backgroundPreset) => {
    set({ backgroundPreset })
    get().pushHistory()
  },
  cycleBackgroundPreset: () => {
    const current = get().backgroundPreset
    const index = BACKGROUND_ORDER.indexOf(current)
    const next = BACKGROUND_ORDER[(index + 1) % BACKGROUND_ORDER.length] ?? 'studio'
    set({ backgroundPreset: next })
    get().pushHistory()
  },
  clearBoard: () => {
    set({ boardPieces: [], selectedPieceId: null })
    get().pushHistory()
  },
  loadOutfitPreset: (outfitId) => {
    const preset = OUTFIT_PRESETS.find((o) => o.id === outfitId)
    if (!preset) return
    let z = 1
    const next = preset.garmentIds.map((garmentId, index) => {
      const garment = get().garments.find((g) => g.id === garmentId)
      const width = Math.max(100, 180 * (garment?.aspectRatio ?? 1))
      return {
        id: mkId(),
        garmentId,
        x: 140 + index * 36,
        y: 100 + index * 24,
        width,
        height: 180,
        rotation: 0,
        opacity: 1,
        flipX: false,
        flipY: false,
        zIndex: z++,
        colorOverride: garment?.palette[0] ?? '#d7c0a2',
      } satisfies BoardPiece
    })
    set({ boardPieces: next, selectedPieceId: next.at(-1)?.id ?? null })
    get().pushHistory()
  },
  saveLook: (name) => {
    const look: SavedLook = {
      id: mkId(),
      name,
      boardPieces: get().boardPieces,
      backgroundPreset: get().backgroundPreset,
    }
    set((s) => ({ savedLooks: [look, ...s.savedLooks] }))
  },
  loadLook: (id) => {
    const look = get().savedLooks.find((s) => s.id === id)
    if (!look) return
    const baseGarments = get().garments.filter((g) => g.kind !== 'upload')
    const lookUploads = (look.garments ?? []).filter((g) => g.kind === 'upload')
    set({
      boardPieces: look.boardPieces,
      backgroundPreset: look.backgroundPreset,
      garments: dedupeGarmentsById([...baseGarments, ...lookUploads]),
      selectedPieceId: look.boardPieces.at(-1)?.id ?? null,
    })
    get().pushHistory()
  },
  setBoardName: (boardName) => set({ boardName }),
  addUploadedGarment: (name, imageSrc) => {
    const id = `upload-${mkId()}`
    set((s) => ({
      garments: [
        ...s.garments,
        { id, name, category: 'accessories', kind: 'upload', palette: ['#f5f5f5'], aspectRatio: 1, imageSrc },
      ],
    }))
    return id
  },
  updateGarment: (id, patch) =>
    set((s) => ({
      garments: s.garments.map((g) => (g.id === id ? { ...g, ...patch } : g)),
    })),
  hydrateUploadedGarments: (garments) =>
    set((s) => {
      const base = s.garments.filter((g) => g.kind !== 'upload')
      const uploads = garments.filter((g) => g.kind === 'upload')
      return { garments: dedupeGarmentsById([...base, ...uploads]) }
    }),
  setSavedLooks: (savedLooks) => set({ savedLooks }),
  setProcessingState: (processingState) => set({ processingState }),
  pushHistory: () => {
    const state = get()
    const snap = toSnapshot(state)
    const base = state.history.slice(0, state.historyIndex + 1)
    const next = [...base, snap].slice(-20)
    set({ history: next, historyIndex: next.length - 1 })
  },
  undo: () => {
    const s = get()
    if (s.historyIndex <= 0) return
    const nextIndex = s.historyIndex - 1
    const snap = s.history[nextIndex]
    if (!snap) return
    set({
      historyIndex: nextIndex,
      boardPieces: snap.boardPieces,
      backgroundPreset: snap.backgroundPreset,
      selectedPieceId: null,
    })
  },
}))
