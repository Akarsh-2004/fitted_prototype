import { motion } from 'framer-motion'
import { Bookmark, Link2, MoreHorizontal, Plus, Share2, Square } from 'lucide-react'
import { useMemo, useState, type ComponentType, type ReactNode } from 'react'
import { COLOR_PALETTE_LIST } from '../data/colorPalettes'
import type { OutfitDefinition, WardrobeSlot } from '../types/wardrobe'
import { GARMENT_LISTS } from './clothing/garments'
import { LottiePlaceholder } from './LottiePlaceholder'

type TabId = 'all' | 'head' | 'tops' | 'bottoms' | 'shoes'

const TABS: { id: TabId; label: string }[] = [
  { id: 'all', label: 'All' },
  { id: 'head', label: 'Head' },
  { id: 'tops', label: 'Tops' },
  { id: 'bottoms', label: 'Bottoms' },
  { id: 'shoes', label: 'Shoes' },
]

const SLOT_LABELS = ['Head', 'Top', 'Bottom', 'Shoes'] as const

const fade = {
  initial: { opacity: 0, y: 10 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.4, ease: [0.22, 1, 0.36, 1] as const },
}

function slotVisible(tab: TabId, slot: WardrobeSlot): boolean {
  if (tab === 'all') return true
  if (tab === 'head') return slot === 0
  if (tab === 'tops') return slot === 1
  if (tab === 'bottoms') return slot === 2
  if (tab === 'shoes') return slot === 3
  return true
}

function slotForTab(tab: TabId): WardrobeSlot | null {
  if (tab === 'head') return 0
  if (tab === 'tops') return 1
  if (tab === 'bottoms') return 2
  if (tab === 'shoes') return 3
  return null
}

function paletteIndexForOutfit(outfit: OutfitDefinition): number {
  const byId =
    outfit.id === 'evening-minimal'
      ? 'mono'
      : outfit.id === 'casual-smart'
        ? 'terracotta'
        : 'classic-street'
  const hit = COLOR_PALETTE_LIST.findIndex((p) => p.id === byId)
  return hit >= 0 ? hit : 0
}

type Props = {
  outfit: OutfitDefinition
  outfitIndex: number
  outfitIds: { id: string; name: string }[]
  onSelectOutfit: (index: number) => void
  activeCollectionId: string
  onSelectCollection: (id: string) => void
  onApplyTheme?: (presetIndex: number) => void
}

export function WardrobePhone({
  outfit,
  outfitIndex,
  outfitIds,
  onSelectOutfit,
  activeCollectionId,
  onSelectCollection,
  onApplyTheme,
}: Props) {
  const [tab, setTab] = useState<TabId>('all')
  /** false = palette swatches only change the focused piece (default). true = one palette for the whole fit. */
  const [syncColors, setSyncColors] = useState(false)
  const [focusedSlot, setFocusedSlot] = useState<WardrobeSlot>(1)

  const [paletteBySlot, setPaletteBySlot] = useState<[number, number, number, number]>(() => {
    const i = paletteIndexForOutfit(outfit)
    return [i, i, i, i]
  })

  const [garmentIdx, setGarmentIdx] = useState<[number, number, number, number]>([0, 0, 0, 0])

  const palettes = useMemo(() => {
    const n = COLOR_PALETTE_LIST.length
    const m = (i: number) => ((i % n) + n) % n
    const idx = paletteBySlot
    return {
      hat: COLOR_PALETTE_LIST[m(idx[0])]!.palette.hat,
      top: COLOR_PALETTE_LIST[m(idx[1])]!.palette.top,
      bottom: COLOR_PALETTE_LIST[m(idx[2])]!.palette.bottom,
      shoes: COLOR_PALETTE_LIST[m(idx[3])]!.palette.shoes,
    }
  }, [paletteBySlot])

  function applyPaletteListIndex(listIndex: number) {
    const i = ((listIndex % COLOR_PALETTE_LIST.length) + COLOR_PALETTE_LIST.length) % COLOR_PALETTE_LIST.length
    setPaletteBySlot((prev) => {
      if (syncColors) return [i, i, i, i]
      const next = [...prev] as [number, number, number, number]
      next[focusedSlot] = i
      return next
    })
    onApplyTheme?.(i)
  }

  function matchAllToFocused() {
    const v = paletteBySlot[focusedSlot]
    setPaletteBySlot([v, v, v, v])
  }

  function setGarmentForSlot(slot: WardrobeSlot, index: number) {
    const list = GARMENT_LISTS[slot]
    const len = list.length
    const i = ((index % len) + len) % len
    setGarmentIdx((prev) => {
      const next = [...prev] as [number, number, number, number]
      next[slot] = i
      return next
    })
  }

  const baseEase = [0.22, 1, 0.36, 1] as const
  const titleLines = outfit.panelTitle.split('\n')

  const HatG = GARMENT_LISTS[0][garmentIdx[0] % GARMENT_LISTS[0].length]!.Graphic
  const TopG = GARMENT_LISTS[1][garmentIdx[1] % GARMENT_LISTS[1].length]!.Graphic
  const BotG = GARMENT_LISTS[2][garmentIdx[2] % GARMENT_LISTS[2].length]!.Graphic
  const ShoeG = GARMENT_LISTS[3][garmentIdx[3] % GARMENT_LISTS[3].length]!.Graphic

  const focusedGarmentList = GARMENT_LISTS[focusedSlot]
  const focusedGarmentMeta = focusedGarmentList[garmentIdx[focusedSlot] % focusedGarmentList.length]!
  const focused3dGarmentList = focusedGarmentList
    .map((g, index) => ({ ...g, index }))
    .filter((g) => g.is3d === true)

  const closet3dBySlot = useMemo(
    () =>
      GARMENT_LISTS.map((list) => list.map((g, index) => ({ ...g, index })).filter((g) => g.is3d === true)) as [
        Array<{ id: string; label: string; Graphic: unknown; is3d?: boolean; index: number }>,
        Array<{ id: string; label: string; Graphic: unknown; is3d?: boolean; index: number }>,
        Array<{ id: string; label: string; Graphic: unknown; is3d?: boolean; index: number }>,
        Array<{ id: string; label: string; Graphic: unknown; is3d?: boolean; index: number }>,
      ],
    [],
  )

  const hatGarment = GARMENT_LISTS[0][garmentIdx[0] % GARMENT_LISTS[0].length]!
  const hatIs3d = hatGarment.is3d === true

  const topGarment = GARMENT_LISTS[1][garmentIdx[1] % GARMENT_LISTS[1].length]!
  const topIs3d = topGarment.is3d === true

  const bottomGarment = GARMENT_LISTS[2][garmentIdx[2] % GARMENT_LISTS[2].length]!
  const bottomIs3d = bottomGarment.is3d === true

  const shoesGarment = GARMENT_LISTS[3][garmentIdx[3] % GARMENT_LISTS[3].length]!
  const shoesIs3d = shoesGarment.is3d === true

  return (
    <div className="device-wrap">
      <div className="phone">
        <div className="phone-header">
          <div className="status-bar">
            <span>9:41</span>
            <div className="status-notch" aria-hidden />
            <div className="status-icons">
              {[5, 7, 9, 11].map((h) => (
                <div key={h} className="sig-bar" style={{ height: h }} />
              ))}
              <div className="battery" aria-hidden />
            </div>
          </div>

          <div className="nav">
            <button type="button" className="nav-back" aria-label="Back">
              ←
            </button>
            <span className="nav-logo">vestir</span>
            <div className="nav-right">
              <div className="nav-avatar" aria-hidden>
                YO
              </div>
              <button type="button" className="nav-more" aria-label="More">
                <MoreHorizontal size={18} strokeWidth={2} />
              </button>
            </div>
          </div>

          <div className="search-wrap">
            <label className="search-bar">
              <span className="search-icon" aria-hidden>
                ⌕
              </span>
              <input placeholder="Search your wardrobe…" autoComplete="off" />
            </label>
          </div>

          <div className="filter-tabs" role="tablist" aria-label="Wardrobe categories">
            {TABS.map((t) => (
              <button
                key={t.id}
                type="button"
                role="tab"
                aria-selected={tab === t.id}
                className={`ftab ${tab === t.id ? 'active' : ''}`}
                onClick={() => {
                  setTab(t.id)
                  const s = slotForTab(t.id)
                  if (s !== null) setFocusedSlot(s)
                }}
              >
                {t.label}
              </button>
            ))}
          </div>
        </div>

        <div className="phone-scroll">
          <div className="canvas-area">
            <div className="fabs" aria-label="Quick actions">
              <button type="button" className="fab primary" title="Add piece">
                <Plus size={18} strokeWidth={2.2} />
              </button>
              <button type="button" className="fab" title="Save outfit">
                <Bookmark size={17} strokeWidth={2} />
              </button>
              <button type="button" className="fab" title="Lock item">
                <Square size={16} strokeWidth={2} />
              </button>
              <button type="button" className="fab" title="Share">
                <Share2 size={16} strokeWidth={2} />
              </button>
            </div>

            <div className="canvas-row canvas-row--stacked">
              <div className="outfit-card outfit-card--wide">
                <div className="outfit-stack">
                  <Slot
                    label={SLOT_LABELS[0]}
                    slotKey="hat"
                    visible={slotVisible(tab, 0)}
                    focused={focusedSlot === 0}
                    animDelay={0.05}
                    baseEase={baseEase}
                    onFocus={() => setFocusedSlot(0)}
                    pieceExtraClass={hatIs3d ? 'piece-hat--3d' : undefined}
                  >
                    <HatG className={hatIs3d ? undefined : 'piece-img'} palette={palettes.hat} />
                  </Slot>
                  <Slot
                    label={SLOT_LABELS[1]}
                    slotKey="top"
                    visible={slotVisible(tab, 1)}
                    focused={focusedSlot === 1}
                    animDelay={0.12}
                    baseEase={baseEase}
                    onFocus={() => setFocusedSlot(1)}
                    pieceExtraClass={topIs3d ? 'piece-top--3d' : undefined}
                  >
                    <TopG className={topIs3d ? undefined : 'piece-img'} palette={palettes.top} />
                  </Slot>
                  <Slot
                    label={SLOT_LABELS[2]}
                    slotKey="bottom"
                    visible={slotVisible(tab, 2)}
                    focused={focusedSlot === 2}
                    animDelay={0.19}
                    baseEase={baseEase}
                    onFocus={() => setFocusedSlot(2)}
                    pieceExtraClass={bottomIs3d ? 'piece-bottom--3d' : undefined}
                  >
                    <BotG className={bottomIs3d ? undefined : 'piece-img'} palette={palettes.bottom} />
                  </Slot>
                  <Slot
                    label={SLOT_LABELS[3]}
                    slotKey="shoes"
                    visible={slotVisible(tab, 3)}
                    focused={focusedSlot === 3}
                    animDelay={0.26}
                    baseEase={baseEase}
                    onFocus={() => setFocusedSlot(3)}
                    pieceExtraClass={shoesIs3d ? 'piece-shoes--3d' : undefined}
                  >
                    <ShoeG className={shoesIs3d ? undefined : 'piece-img'} palette={palettes.shoes} />
                  </Slot>
                </div>
                <div className="outfit-card-footer">
                  <div className="outfit-name">{outfit.name}</div>
                  <div className="outfit-tag">{outfit.tag}</div>
                </div>

                <div className="picker-section">
                  <div className="picker-row picker-row--colors">
                    <div className="picker-row-head">
                      <span className="picker-label">Colors</span>
                      <div className="picker-head-actions">
                        <button
                          type="button"
                          className={`sync-toggle ${syncColors ? 'on' : ''}`}
                          onClick={() => setSyncColors((s) => !s)}
                          aria-pressed={syncColors}
                          title={
                            syncColors
                              ? 'Swatches apply the same palette to head, top, bottoms, and shoes'
                              : 'Swatches change only the highlighted piece (tap stack or a category tab)'
                          }
                        >
                          <Link2 size={14} strokeWidth={2} />
                          <span>{syncColors ? 'Whole outfit' : 'This piece'}</span>
                        </button>
                        {!syncColors ? (
                          <button type="button" className="match-all-btn" onClick={matchAllToFocused}>
                            Copy to all
                          </button>
                        ) : null}
                      </div>
                    </div>
                    <p className="picker-hint">
                      {syncColors ? (
                        <>Choosing a swatch updates every layer together.</>
                      ) : (
                        <>
                          Editing <strong>{SLOT_LABELS[focusedSlot]}</strong> — pick a palette for this layer only
                          (e.g. <strong>Black shirt</strong> on Top, then tap Bottoms and choose <strong>White jeans</strong>
                          ).
                        </>
                      )}
                    </p>
                    <div
                      className="palette-rail"
                      role="listbox"
                      aria-label={
                        syncColors
                          ? 'Color palettes for the whole outfit'
                          : `Color palettes for ${SLOT_LABELS[focusedSlot]} only`
                      }
                    >
                      {COLOR_PALETTE_LIST.map((entry, listIndex) => {
                        const n = COLOR_PALETTE_LIST.length
                        const norm = (i: number) => ((i % n) + n) % n
                        const active = syncColors
                          ? paletteBySlot.every((p) => norm(p) === norm(listIndex))
                          : norm(paletteBySlot[focusedSlot]) === norm(listIndex)
                        return (
                          <button
                            key={entry.id}
                            type="button"
                            role="option"
                            aria-selected={active}
                            className={`palette-swatch ${active ? 'active' : ''}`}
                            onClick={() => applyPaletteListIndex(listIndex)}
                            title={entry.name}
                          >
                            <span className="palette-swatch-dot" style={{ background: entry.swatch }} />
                            <span className="palette-swatch-name">{entry.name}</span>
                          </button>
                        )
                      })}
                    </div>
                  </div>

                  <div className="picker-row">
                    <div className="picker-row-head">
                      <span className="picker-label">
                        3D Style · {SLOT_LABELS[focusedSlot]} · {focusedGarmentMeta.label}
                      </span>
                    </div>
                    <div className="garment-rail" role="listbox" aria-label="3D garment shapes">
                      {focused3dGarmentList.map((g) => (
                        <button
                          key={g.id}
                          type="button"
                          role="option"
                          aria-selected={garmentIdx[focusedSlot] % focusedGarmentList.length === g.index}
                          className={`garment-chip ${
                            garmentIdx[focusedSlot] % focusedGarmentList.length === g.index ? 'active' : ''
                          }`}
                          onClick={() => setGarmentForSlot(focusedSlot, g.index)}
                        >
                          {g.label}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="picker-row">
                    <div className="picker-row-head">
                      <span className="picker-label">3D Closet Shelf</span>
                    </div>
                    <p className="picker-hint">
                      Every item below is a 3D closet piece. Tap one to wear it and adjust instantly.
                    </p>
                    <div className="closet-grid" role="list" aria-label="3D closet inventory">
                      {closet3dBySlot.map((slotItems, slot) => {
                        if (!slotVisible(tab, slot as WardrobeSlot)) return null
                        return (
                          <div key={SLOT_LABELS[slot]} className="closet-group">
                            <div className="closet-group-title">{SLOT_LABELS[slot]}</div>
                            <div className="closet-group-items">
                              {slotItems.map((piece) => {
                                const Graphic = piece.Graphic as ComponentType<{
                                  palette: unknown
                                  className?: string
                                }>
                                const active = garmentIdx[slot as WardrobeSlot] === piece.index
                                return (
                                  <button
                                    key={piece.id}
                                    type="button"
                                    role="listitem"
                                    className={`closet-item ${active ? 'active' : ''}`}
                                    onClick={() => {
                                      setFocusedSlot(slot as WardrobeSlot)
                                      setGarmentForSlot(slot as WardrobeSlot, piece.index)
                                    }}
                                  >
                                    <span className="closet-item-preview" aria-hidden>
                                      {slot === 0 ? <Graphic palette={palettes.hat} /> : null}
                                      {slot === 1 ? <Graphic palette={palettes.top} /> : null}
                                      {slot === 2 ? <Graphic palette={palettes.bottom} /> : null}
                                      {slot === 3 ? <Graphic palette={palettes.shoes} /> : null}
                                    </span>
                                    <span className="closet-item-label">{piece.label}</span>
                                  </button>
                                )
                              })}
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="ai-row">
            <div className="ai-dot" aria-hidden />
            <p className="ai-text">
              <strong>Tip:</strong> Each piece has its own color — keep <em>This piece</em> on, tap Top, choose{' '}
              <em>Black shirt</em>, then tap Bottoms and choose <em>White jeans</em>.
            </p>
          </div>

          <div className="phone-feed">
            <motion.div
              className="feed-block"
              {...fade}
              transition={{ ...fade.transition, delay: 0.02 }}
            >
              <div className="panel-toolbar">
                <div className="panel-eyebrow">{outfit.panelEyebrow}</div>
                <label className="outfit-picker">
                  <span className="visually-hidden">Active capsule</span>
                  <select
                    value={outfitIndex}
                    onChange={(e) => onSelectOutfit(Number(e.target.value))}
                    aria-label="Choose capsule"
                  >
                    {outfitIds.map((o, i) => (
                      <option key={o.id} value={i}>
                        {o.name}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <div className="panel-title panel-title--phone">
                {titleLines.map((line, i) => (
                  <span key={line}>
                    {line}
                    {i < titleLines.length - 1 ? <br /> : null}
                  </span>
                ))}
              </div>
              <div className="panel-sub">{outfit.panelSub}</div>
            </motion.div>

            <motion.div
              className="weather-chip weather-chip--feed"
              initial={fade.initial}
              animate={fade.animate}
              transition={{ ...fade.transition, delay: 0.05 }}
            >
              <span aria-hidden>⛅</span>
              <span>{outfit.weatherCity}</span>
              <span className="temp">{outfit.weatherTemp}</span>
              <span className="weather-hint">{outfit.weatherHint}</span>
            </motion.div>

            <motion.div className="feed-block" {...fade} transition={{ ...fade.transition, delay: 0.08 }}>
              <div className="section-label">Wardrobe stats</div>
              <div className="stats-row stats-row--phone">
                <div className="stat-card">
                  <div className="stat-val">{outfit.stats.fits}</div>
                  <div className="stat-label">Fits</div>
                </div>
                <div className="stat-card">
                  <div className="stat-val">{outfit.stats.pieces}</div>
                  <div className="stat-label">Pieces</div>
                </div>
                <div className="stat-card">
                  <div className="stat-val">{outfit.stats.value}</div>
                  <div className="stat-label">Value</div>
                </div>
              </div>
            </motion.div>

            <motion.div className="feed-block" {...fade} transition={{ ...fade.transition, delay: 0.1 }}>
              <div className="section-label">Collections</div>
              <div className="collections">
                {outfit.collections.map((c) => (
                  <button
                    key={c.id}
                    type="button"
                    className={`collection-card collection-card--phone ${
                      activeCollectionId === c.id ? 'active-coll' : ''
                    }`}
                    onClick={() => onSelectCollection(c.id)}
                  >
                    <div className="collection-thumb collection-thumb--phone" aria-hidden>
                      {c.emoji}
                    </div>
                    <div className="collection-info">
                      <div className="collection-name">{c.name}</div>
                      <div className="collection-meta">{c.meta}</div>
                    </div>
                    <span className="collection-arrow" aria-hidden>
                      →
                    </span>
                  </button>
                ))}
              </div>
            </motion.div>

            <motion.div className="feed-block" {...fade} transition={{ ...fade.transition, delay: 0.12 }}>
              <div className="section-label">In this outfit</div>
              <div className="piece-list">
                {outfit.pieces.map((p) => (
                  <div key={p.id} className="piece-row piece-row--phone">
                    <div className="piece-dot on" aria-hidden />
                    <div className="piece-row-text">
                      <span className="piece-row-name">{p.name}</span>
                      <span className="piece-row-brand">{p.brand}</span>
                    </div>
                    <span className="piece-row-price">{p.price}</span>
                  </div>
                ))}
              </div>
            </motion.div>

            <div className="phone-bottom phone-bottom--in-scroll">
              <p className="status-text">{COLOR_PALETTE_LIST.length} palettes · mix silhouettes per slot</p>
              <div className="home-indicator" />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function Slot({
  label,
  slotKey,
  visible,
  focused,
  onFocus,
  animDelay,
  baseEase,
  pieceExtraClass,
  children,
}: {
  label: string
  slotKey: 'hat' | 'top' | 'bottom' | 'shoes'
  visible: boolean
  focused: boolean
  onFocus: () => void
  animDelay: number
  baseEase: readonly [number, number, number, number]
  pieceExtraClass?: string
  children: ReactNode
}) {
  return (
    <motion.div
      className={['outfit-piece', `piece-${slotKey}`, pieceExtraClass, !visible ? 'is-filtered' : '', focused ? 'is-focused' : '']
        .filter(Boolean)
        .join(' ')}
      initial={{ opacity: 0, y: 10, scale: 0.96 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.45, ease: baseEase, delay: animDelay }}
    >
      <button type="button" className="outfit-piece-hit" onClick={onFocus} aria-label={`Focus ${label}`}>
        <LottiePlaceholder slot={slotKey} label={`${label} preview`}>
          {children}
        </LottiePlaceholder>
        <span className="item-label">{label}</span>
      </button>
    </motion.div>
  )
}
