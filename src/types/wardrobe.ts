/** Slot order: hat, top, bottom, shoes */
export type WardrobeSlot = 0 | 1 | 2 | 3

export type ClothingPalette = {
  hat: HatPalette
  top: TopPalette
  bottom: BottomPalette
  shoes: ShoesPalette
}

export type HatPalette = {
  brim: string
  crown: string
  crownInner: string
  band: string
  accentStroke: string
}

export type TopPalette = {
  body: string
  sleeve: string
  hood: string
  hoodShade: string
  pocket: string
  rib: string
}

export type BottomPalette = {
  waist: string
  leg: string
  seam: string
  pocketStroke: string
  loop: string
  fray: string
}

export type ShoesPalette = {
  sole: string
  upper: string
  toe: string
  stripe: string
  lace: string
  tongue: string
  heel: string
}

export type WardrobePieceMeta = {
  id: string
  name: string
  brand: string
  price: string
}

export type OutfitDefinition = {
  id: string
  name: string
  tag: string
  /** Eyebrow + title lines for desktop panel */
  panelEyebrow: string
  panelTitle: string
  panelSub: string
  weatherCity: string
  weatherTemp: string
  weatherHint: string
  stats: { fits: string; pieces: string; value: string }
  collections: Array<{
    id: string
    emoji: string
    name: string
    meta: string
  }>
  pieces: WardrobePieceMeta[]
  /** Visual palette for SVG placeholders (swap with Lottie later) */
  palette: ClothingPalette
}
