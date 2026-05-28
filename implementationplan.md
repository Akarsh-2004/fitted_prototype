Implementation Plan - Premium Mannequin Styling Studio
This plan proposes the creation of a high-fidelity, interactive Mannequin Styling Studio (modular 2D virtual try-on wardrobe) in the React frontend. It allows users to toggle the workspace to a premium clothing styling canvas, where they can layer, swipe, and customize combinations of tops, outerwear, bottoms, shoes, and accessories on a front-facing human mannequin cutout.

Architectural Vision & Design
We will transform the left panel (lg:col-span-7) into a dual-mode workspace switcher:

Active Ingestion Workspace: The existing upload, routing, interactive S4 canvas, and GrabCut segment viewer.
Mannequin Styling Studio (New): A premium interactive wardrobe sandbox showcasing layered 2D compositing.
1. Layered Smart Garment Renderer
Garments extracted during pipeline stages contain alpha channels (RGBA). We will stack them over the base mannequin cutout using strict layer depth values:

Base Mannequin (cutout body)
Inner Top (top_garment)
Lower wear (bottom_garment)
Footwear/Shoes (shoes)
Outer Top (outerwear)
Accessories (accessory)
To handle variations in garment bounding boxes, we will implement a Figma-style Fitting Anchor System that provides:

Category-level sensible defaults (e.g. Tops positioned on chest, Bottoms on waist, Footwear on feet).
Interactive Adjusters: Individual sliders for Scale (%), X-offset (px), and Y-offset (px) for the active layer. This solves fitting constraints elegantly and adds a premium "editor" feel.
2. Multi-Zone Swiping & Cycling
Instead of basic menus, we will build layered swipe controls directly on top of the mannequin body zones:

Head Zone: Cycles accessories (caps, sunglasses).
Chest Zone: Cycles tops (tshirts, shirts).
Torso/Overlay Zone: Cycles outerwear (jackets, coats).
Legs Zone: Cycles bottoms (jeans, shorts, cargos).
Feet Zone: Cycles shoes (sneakers, boots).
Hovering over a zone will display a glowing neon guideline and Left/Right slider buttons to cycle elements smoothly.

3. Style Intelligence Layer
Beneath the styling canvas, we will implement real-time analysis:

OKLCH Color Harmony Score: Calculates the average OKLCH color distance across all active layered garments to present a dynamic match score (e.g. 92% - Harmonious Streetwear).
Archetype & Season Coherence: Synthesizes tags and metadata from active items to generate styling pairing suggestions (e.g., Minimalist Spring Casual).
Detailed File Modifications
1. [MODIFY] 
App.tsx
We will introduce the following states and helper functions:

New Workspace & Garment Indexes States
typescript

const [mannequinMode, setMannequinMode] = useState<'upload' | 'mannequin'>('upload');
// Base Mannequins (Default + Custom Cutouts from parsed Jobs)
const [selectedMannequinUrl, setSelectedMannequinUrl] = useState<string>('/default_mannequin.png');
// Currently active garment index per category
const [activeTopsIdx, setActiveTopsIdx] = useState<number>(-1);
const [activeOuterIdx, setActiveOuterIdx] = useState<number>(-1);
const [activeBottomsIdx, setActiveBottomsIdx] = useState<number>(-1);
const [activeShoesIdx, setActiveShoesIdx] = useState<number>(-1);
const [activeAccessoryIdx, setActiveAccessoryIdx] = useState<number>(-1);
// Layer visibility controls
const [layerVisibility, setLayerVisibility] = useState({
  accessory: true,
  outerwear: true,
  top: true,
  bottom: true,
  shoes: true
});
// Custom Fitting Offsets state (persisted per garment ID locally)
const [fittingOffsets, setFittingOffsets] = useState<Record<string, { scale: number, x: number, y: number }>>({});
Layer Sorting & Filtering
When closetItems are loaded, we will filter them into category arrays:

typescript

const topsList = closetItems.filter(i => i.garment_type === 'top');
const outerwearList = closetItems.filter(i => i.garment_type === 'outerwear');
const bottomsList = closetItems.filter(i => i.garment_type === 'bottom');
const shoesList = closetItems.filter(i => i.garment_type === 'shoes');
const accessoriesList = closetItems.filter(i => i.garment_type === 'accessory');
Dynamic OKLCH Multi-Layer Score
We will implement a lightweight frontend formula to compare OkLCH color weights across the active outfit layers, showing real-time coordination feedback.

Verification Plan
Manual Verification
Navigate to the local studio environment at http://localhost:5188.
Toggle from Ingestion Workspace to Mannequin Studio.
Select the Default Matte Mannequin or import a custom body cutout from S4 parsing results.
Interact with the zone swipers (Tops, Outerwear, Bottoms, Shoes) using arrow buttons and ensure transparent garment layers stack correctly according to z-indexes.
Use the fitting sliders to modify scale/offset parameters of an oversized jacket.
Verify that changing clothes updates the Style Harmony Scorer and Aesthetic Tag Card immediately.