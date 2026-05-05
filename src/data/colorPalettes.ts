import type { ClothingPalette } from '../types/wardrobe'
import { adjustHex, mixHex } from '../lib/color'

/** Build a coherent outfit palette from four seed accents */
export function buildPaletteFromSeeds(seeds: {
  hat: string
  top: string
  bottom: string
  shoes: string
}): ClothingPalette {
  const { hat: h, top: t, bottom: b, shoes: s } = seeds
  return {
    hat: {
      brim: mixHex(h, '#E8E4DC', 0.35),
      crown: adjustHex(h, -0.25),
      crownInner: adjustHex(h, -0.12),
      band: adjustHex(h, -0.35),
      accentStroke: adjustHex(h, -0.45),
    },
    top: {
      body: t,
      sleeve: adjustHex(t, -0.18),
      hood: mixHex(t, '#ffffff', 0.08),
      hoodShade: adjustHex(t, -0.28),
      pocket: adjustHex(t, -0.22),
      rib: adjustHex(t, -0.32),
    },
    bottom: {
      waist: mixHex(b, '#F5F4F2', 0.2),
      leg: b,
      seam: adjustHex(b, -0.22),
      pocketStroke: adjustHex(b, -0.28),
      loop: mixHex(b, '#FFFFFF', 0.28),
      fray: adjustHex(b, -0.18),
    },
    shoes: {
      sole: mixHex(s, '#2A2A2A', 0.55),
      upper: mixHex(s, '#FFFFFF', 0.15),
      toe: mixHex(s, '#FFFFFF', 0.32),
      stripe: mixHex(s, '#000000', 0.2),
      lace: adjustHex(s, -0.18),
      tongue: mixHex(s, '#FFFFFF', 0.25),
      heel: adjustHex(s, -0.1),
    },
  }
}

export type NamedPalette = {
  id: string
  name: string
  /** Swatch shown in UI (usually the top / hero color) */
  swatch: string
  palette: ClothingPalette
}

/** Curated + generated colorways — tap in UI to apply (each slot can use a different entry when “This piece” mode is on). */
export const COLOR_PALETTE_LIST: NamedPalette[] = [
  {
    id: 'black-tee',
    name: 'Black shirt',
    swatch: '#141414',
    palette: buildPaletteFromSeeds({
      hat: '#252525',
      top: '#141414',
      bottom: '#6B7280',
      shoes: '#787878',
    }),
  },
  {
    id: 'white-jeans',
    name: 'White jeans',
    swatch: '#EDEDE8',
    palette: buildPaletteFromSeeds({
      hat: '#9CA3AF',
      top: '#D4D4D0',
      bottom: '#EDEDE8',
      shoes: '#DCDCD6',
    }),
  },
  { id: 'classic-street', name: 'Indigo', swatch: '#2563EB', palette: buildPaletteFromSeeds({
      hat: '#1A1A1A',
      top: '#2563EB',
      bottom: '#5A7CA6',
      shoes: '#ECEAE6',
    }) },
  { id: 'terracotta', name: 'Terracotta', swatch: '#C45C3A', palette: buildPaletteFromSeeds({
      hat: '#4A5D3A',
      top: '#C45C3A',
      bottom: '#3D4F73',
      shoes: '#D4B48E',
    }) },
  { id: 'mono', name: 'Concrete', swatch: '#9A9A96', palette: buildPaletteFromSeeds({
      hat: '#2C2C2C',
      top: '#E8E6E1',
      bottom: '#2A2A2A',
      shoes: '#F2F0EC',
    }) },
  { id: 'forest', name: 'Forest', swatch: '#2D6A4F', palette: buildPaletteFromSeeds({
      hat: '#1B4332',
      top: '#40916C',
      bottom: '#344E41',
      shoes: '#D4A373',
    }) },
  { id: 'rose-smoke', name: 'Rose', swatch: '#D87CAC', palette: buildPaletteFromSeeds({
      hat: '#6B4F4E',
      top: '#E8A0BF',
      bottom: '#4A4E69',
      shoes: '#F2E9E4',
    }) },
  { id: 'ocean', name: 'Ocean', swatch: '#0077B6', palette: buildPaletteFromSeeds({
      hat: '#023E8A',
      top: '#48CAE4',
      bottom: '#14213D',
      shoes: '#CAF0F8',
    }) },
  { id: 'sandstorm', name: 'Sand', swatch: '#E9C46A', palette: buildPaletteFromSeeds({
      hat: '#BC6C25',
      top: '#F4A261',
      bottom: '#606C38',
      shoes: '#FAEDCD',
    }) },
  { id: 'lavender', name: 'Lilac', swatch: '#B388EB', palette: buildPaletteFromSeeds({
      hat: '#4A4E69',
      top: '#CDB4DB',
      bottom: '#3D405B',
      shoes: '#F4F1DE',
    }) },
  { id: 'ember', name: 'Ember', swatch: '#D62828', palette: buildPaletteFromSeeds({
      hat: '#1B1B1B',
      top: '#F77F00',
      bottom: '#2B2D42',
      shoes: '#FCBF49',
    }) },
  { id: 'sage', name: 'Sage', swatch: '#95D5B2', palette: buildPaletteFromSeeds({
      hat: '#3A5A40',
      top: '#74C69D',
      bottom: '#2D4739',
      shoes: '#EAF4DE',
    }) },
  { id: 'midnight', name: 'Midnight', swatch: '#14213D', palette: buildPaletteFromSeeds({
      hat: '#03071E',
      top: '#4361EE',
      bottom: '#1B263B',
      shoes: '#E0E1DD',
    }) },
  { id: 'berry', name: 'Berry', swatch: '#7B2CBF', palette: buildPaletteFromSeeds({
      hat: '#240046',
      top: '#9D4EDD',
      bottom: '#3C096C',
      shoes: '#E0AAFF',
    }) },
  { id: 'cream-taupe', name: 'Oat', swatch: '#DDB892', palette: buildPaletteFromSeeds({
      hat: '#7F5539',
      top: '#EDE0D4',
      bottom: '#603808',
      shoes: '#F8F4F0',
    }) },
  { id: 'slate-lime', name: 'Neon lime', swatch: '#CCFF33', palette: buildPaletteFromSeeds({
      hat: '#212529',
      top: '#ADFF02',
      bottom: '#343A40',
      shoes: '#F8F9FA',
    }) },
  { id: 'copper', name: 'Copper', swatch: '#B08968', palette: buildPaletteFromSeeds({
      hat: '#432818',
      top: '#DDB892',
      bottom: '#582F0E',
      shoes: '#EDE0D4',
    }) },
  { id: 'aqua-pop', name: 'Aqua', swatch: '#06FFA5', palette: buildPaletteFromSeeds({
      hat: '#1D3557',
      top: '#2EC4B6',
      bottom: '#023047',
      shoes: '#E0FBFC',
    }) },
]

export function getPaletteByIndex(i: number): ClothingPalette {
  const n = COLOR_PALETTE_LIST.length
  return COLOR_PALETTE_LIST[((i % n) + n) % n]!.palette
}
