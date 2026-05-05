import { useEffect, useMemo, useRef } from 'react'
import { ClosetPanel } from './components/layout/ClosetPanel'
import { BoardPanel } from './components/layout/BoardPanel'
import { InspectorPanel } from './components/layout/InspectorPanel'
import { Toolbar } from './components/layout/Toolbar'
import { useWardrobeStore } from './store/wardrobeStore'
import { OUTFIT_PRESETS } from './data/outfits'
import { exportBoardPng } from './utils/export'
import { loadState, saveState } from './utils/storage'
import { processSingle, refineSingle } from './utils/tryonApi'

export default function App() {
  const boardRef = useRef<HTMLDivElement | null>(null)
  const store = useWardrobeStore()
  const selectedPiece = useMemo(
    () => store.boardPieces.find((p) => p.id === store.selectedPieceId),
    [store.boardPieces, store.selectedPieceId],
  )
  const selectedGarment = useMemo(
    () => store.garments.find((g) => g.id === selectedPiece?.garmentId),
    [store.garments, selectedPiece?.garmentId],
  )

  useEffect(() => {
    const loaded = loadState<{ savedLooks: typeof store.savedLooks; boardName: string; garments?: typeof store.garments }>({
      savedLooks: [],
      boardName: 'Untitled Board',
      garments: undefined,
    })
    if (loaded.boardName) store.setBoardName(loaded.boardName)
    store.setSavedLooks(loaded.savedLooks)
    if (loaded.garments) {
      store.hydrateUploadedGarments(loaded.garments)
    }
    // intentionally restore lightweight state only
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    const t = setTimeout(() => {
      const trimDataUrl = (value?: string) => (value && value.length <= 180000 ? value : undefined)
      const sanitizedLooks = store.savedLooks.map((look) => ({
        id: look.id,
        name: look.name,
        boardPieces: look.boardPieces,
        backgroundPreset: look.backgroundPreset,
      }))
      const lightweightUploads = store.garments
        .filter((g) => g.kind === 'upload')
        .map((g) => ({
          ...g,
          originalImageSrc: trimDataUrl(g.originalImageSrc),
          imageSrc: trimDataUrl(g.imageSrc),
          maskSrc: trimDataUrl(g.maskSrc),
          compositeSrc: trimDataUrl(g.compositeSrc),
        }))
      try {
        saveState({ savedLooks: sanitizedLooks, boardName: store.boardName, garments: lightweightUploads })
      } catch {
        // storage helper already handles fallback; this prevents effect crashes in edge browsers
      }
    }, 500)
    return () => clearTimeout(t)
  }, [store.savedLooks, store.boardName, store.garments])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') store.setSelectedPieceId(null)
      if (e.key === 'Delete' || e.key === 'Backspace') {
        if (store.selectedPieceId) store.removePiece(store.selectedPieceId)
      }
      if (e.key.toLowerCase() === 'b') store.cycleBackgroundPreset()
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'd') {
        e.preventDefault()
        if (store.selectedPieceId) store.duplicatePiece(store.selectedPieceId)
      }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
        e.preventDefault()
        store.undo()
      }
      if (e.key === '[' && store.selectedPieceId) {
        const p = store.boardPieces.find((x) => x.id === store.selectedPieceId)
        if (p) store.reorderPiece(p.id, p.zIndex - 1)
      }
      if (e.key === ']' && store.selectedPieceId) {
        const p = store.boardPieces.find((x) => x.id === store.selectedPieceId)
        if (p) store.reorderPiece(p.id, p.zIndex + 1)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [store])

  return (
    <div className="app">
      <Toolbar
        boardName={store.boardName}
        onBoardName={store.setBoardName}
        onClear={store.clearBoard}
        onExport={() => {
          if (boardRef.current) exportBoardPng(boardRef.current, store.boardName)
        }}
        onSave={() => store.saveLook(`${store.boardName} ${new Date().toLocaleTimeString()}`)}
        onCycleBackground={store.cycleBackgroundPreset}
        savedLooks={store.savedLooks}
        onLoadLook={store.loadLook}
      />
      <main className="workspace" ref={boardRef}>
        <ClosetPanel
          garments={store.garments}
          activeCategory={store.activeCategory}
          onCategoryChange={store.setActiveCategory}
          onAddGarment={(id) => store.addPiece(id)}
          onUpload={(files) => {
            store.setProcessingState('uploading')
            files.forEach((f) => {
              void (async () => {
                const id = store.addUploadedGarment(f.name, '')
                try {
                  store.setProcessingState('processing')
                  const result = await processSingle(f)
                  const imageSrc = result.cutoutSrc || URL.createObjectURL(f)
                  store.updateGarment(id, {
                    imageSrc,
                    originalImageSrc: URL.createObjectURL(f),
                    maskSrc: result.maskSrc,
                    compositeSrc: result.compositeSrc,
                    processingState: 'done',
                    processingMeta: {
                      maskQuality: result.meta.mask_quality,
                      processingTime: result.meta.processing_time,
                      usedFallback: result.meta.used_fallback,
                      warpMode: result.meta.warp_mode,
                      requestKey: result.meta.request_key,
                    },
                    refineParams: { edgeFeather: 3, morphKernel: 3, smoothness: 0.5 },
                  })
                  store.addPiece(id)
                  store.setProcessingState('done')
                } catch {
                  store.updateGarment(id, {
                    processingState: 'failed',
                    imageSrc: URL.createObjectURL(f),
                    originalImageSrc: URL.createObjectURL(f),
                  })
                  store.addPiece(id)
                  store.setProcessingState('failed')
                }
              })()
            })
          }}
          outfits={OUTFIT_PRESETS}
          onLoadOutfit={store.loadOutfitPreset}
          processingState={store.processingState}
        />
        <BoardPanel
          pieces={store.boardPieces}
          garments={store.garments}
          selectedPieceId={store.selectedPieceId}
          backgroundPreset={store.backgroundPreset}
          onSelect={store.setSelectedPieceId}
          onChangePiece={store.updatePiece}
          onDelete={store.removePiece}
          onReorder={store.reorderPiece}
          onDropGarment={store.addPiece}
        />
        <InspectorPanel
          piece={selectedPiece}
          garment={selectedGarment}
          onChange={(patch) => selectedPiece && store.updatePiece(selectedPiece.id, patch)}
          onDelete={() => selectedPiece && store.removePiece(selectedPiece.id)}
          onDuplicate={() => selectedPiece && store.duplicatePiece(selectedPiece.id)}
          onBack={() => selectedPiece && store.reorderPiece(selectedPiece.id, 1)}
          onFront={() => selectedPiece && store.reorderPiece(selectedPiece.id, store.boardPieces.length + 1)}
          onRefine={async (params) => {
            if (!selectedGarment?.processingMeta?.requestKey) return
            store.setProcessingState('refining')
            try {
              const result = await refineSingle({
                requestKey: selectedGarment.processingMeta.requestKey,
                edgeFeather: params.edgeFeather,
                morphKernel: params.morphKernel,
                smoothness: params.smoothness,
              })
              store.updateGarment(selectedGarment.id, {
                imageSrc: result.cutoutSrc,
                maskSrc: result.maskSrc,
                compositeSrc: result.compositeSrc,
                refineParams: params,
                processingMeta: {
                  maskQuality: result.meta.mask_quality,
                  processingTime: result.meta.processing_time,
                  usedFallback: result.meta.used_fallback,
                  warpMode: result.meta.warp_mode,
                  requestKey: result.meta.request_key,
                },
              })
              store.setProcessingState('done')
            } catch {
              store.setProcessingState('failed')
            }
          }}
        />
      </main>
    </div>
  )
}
