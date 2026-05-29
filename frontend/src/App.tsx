import React, { useState, useRef, useEffect } from 'react';
import { 
  Upload, 
  Sparkles, 
  Layers, 
  Cpu, 
  AlertCircle, 
  Check, 
  ChevronRight,
  ChevronLeft,
  Grid,
  Shirt,
  Search,
  Trash2,
  Compass,
  Palette,
  Heart,
  Eye,
  EyeOff
} from 'lucide-react';

interface ColorProfile {
  rgb: number[];
  weight: number;
  oklch: number[]; // [L, C, H]
}

interface WardrobeItem {
  id: string;
  garment_type: string;
  fit?: string;
  material?: string;
  construction?: string;
  pattern?: string;
  subtype?: string;
  brand?: string;
  style?: string;
  occasion?: string;
  season?: string;
  archetype?: string;
  layering_role?: string;
  pairing_suggestions: string[];
  colors: ColorProfile[];
  tags: string[];
  image_path: string;
  scene_type: string;
  created_at?: string;
}

interface ComposerItem {
  id: string;
  category: 'hat' | 'top' | 'bottom' | 'shoes';
  aligned_url: string;
  image_url?: string;
  name?: string;
  brand?: string;
  subtype?: string;
  style?: string;
  material?: string;
  colors: ColorProfile[];
  tags: string[];
  garment_type?: string;
}

type ColorBearing = { colors: ColorProfile[] };

const GEMINI_STAGE_API_BASE = import.meta.env.VITE_GEMINI_STAGE_API_BASE || '';

interface JobData {
  job_id: string;
  status: string;
  scene_type: string | null;
  original_image_url: string | null;
  detected_items: any[] | null;
  result: WardrobeItem[] | null;
  error: string | null;
  cutout_url?: string;
  parsed_parts?: Array<{
    label: string;
    rgba_crop_path: string;
    mask_path: string;
    bbox: number[];
    pixel_area: number;
    grabcut_refined: boolean;
    confidence: number;
    ingest: boolean;
  }>;
  created_at?: string;
}

interface UploadResponse {
  job_id: string;
  scene_type: string;
  original_image_url: string;
  dimensions: { width: number; height: number };
  counts: { faces: number; people: number; garments: number };
}

type DetectedPerson = {
  id?: number;
  polygon?: number[][];
};

const getPartTheme = (label: string) => {
  switch (label) {
    case 'top_garment':
      return {
        badge: 'Top',
        border: 'border-violet-500/40 hover:border-violet-500',
        text: 'text-violet-400 bg-violet-500/10',
        glow: 'shadow-[0_0_15px_rgba(139,92,246,0.15)] hover:shadow-[0_0_20px_rgba(139,92,246,0.3)]',
        colorName: 'Indigo'
      };
    case 'outerwear':
      return {
        badge: 'Outerwear',
        border: 'border-pink-500/40 hover:border-pink-500',
        text: 'text-pink-400 bg-pink-500/10',
        glow: 'shadow-[0_0_15px_rgba(236,72,153,0.15)] hover:shadow-[0_0_20px_rgba(236,72,153,0.3)]',
        colorName: 'Neon Pink'
      };
    case 'bottom_garment':
      return {
        badge: 'Bottom',
        border: 'border-cyan-500/40 hover:border-cyan-500',
        text: 'text-cyan-400 bg-cyan-500/10',
        glow: 'shadow-[0_0_15px_rgba(6,182,212,0.15)] hover:shadow-[0_0_20px_rgba(6,182,212,0.3)]',
        colorName: 'Cyan'
      };
    case 'left_shoe':
    case 'right_shoe':
    case 'footwear':
      return {
        badge: label === 'left_shoe' ? 'Left Shoe' : (label === 'right_shoe' ? 'Right Shoe' : 'Footwear Pair'),
        border: 'border-orange-500/40 hover:border-orange-500',
        text: 'text-orange-400 bg-orange-500/10',
        glow: 'shadow-[0_0_15px_rgba(249,115,22,0.15)] hover:shadow-[0_0_20px_rgba(249,115,22,0.3)]',
        colorName: 'Safety Orange'
      };
    case 'bag':
      return {
        badge: 'Shoulder Bag',
        border: 'border-amber-500/40 hover:border-amber-500',
        text: 'text-amber-400 bg-amber-500/10',
        glow: 'shadow-[0_0_15px_rgba(245,158,11,0.15)] hover:shadow-[0_0_20px_rgba(245,158,11,0.3)]',
        colorName: 'Amber Gold'
      };
    case 'hat':
      return {
        badge: 'Headwear / Hat',
        border: 'border-emerald-500/40 hover:border-emerald-500',
        text: 'text-emerald-400 bg-emerald-500/10',
        glow: 'shadow-[0_0_15px_rgba(16,185,129,0.15)] hover:shadow-[0_0_20px_rgba(16,185,129,0.3)]',
        colorName: 'Emerald'
      };
    case 'left_arm':
    case 'right_arm':
      return {
        badge: label === 'left_arm' ? 'Left Arm (Skin)' : 'Right Arm (Skin)',
        border: 'border-slate-500/20 hover:border-slate-500/60',
        text: 'text-slate-400 bg-slate-500/10',
        glow: 'shadow-none',
        colorName: 'Slate Neutral'
      };
    case 'wrist_accessory':
    case 'accessory':
      return {
        badge: 'Wrist Accessory',
        border: 'border-fuchsia-500/40 hover:border-fuchsia-500',
        text: 'text-fuchsia-400 bg-fuchsia-500/10',
        glow: 'shadow-[0_0_15px_rgba(217,70,239,0.15)] hover:shadow-[0_0_20px_rgba(217,70,239,0.3)]',
        colorName: 'Fuchsia'
      };
    default:
      return {
        badge: label.replace('_', ' '),
        border: 'border-white/10 hover:border-white/30',
        text: 'text-slate-300 bg-white/5',
        glow: 'shadow-none',
        colorName: 'Classic'
      };
  }
};

interface HumanParsingOverlayProps {
  cutoutUrl: string;
  parts: Array<{
    label: string;
    mask_path: string;
  }>;
  hoveredPartLabel: string | null;
  showMaskOverlay: boolean;
}

const HumanParsingOverlay: React.FC<HumanParsingOverlayProps> = ({
  cutoutUrl,
  parts,
  hoveredPartLabel,
  showMaskOverlay
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [imagesLoaded, setImagesLoaded] = useState<boolean>(false);
  const [loadedImages, setLoadedImages] = useState<{
    cutout: HTMLImageElement | null;
    masks: { [label: string]: HTMLImageElement };
  }>({ cutout: null, masks: {} });

  // 1. Preload cutout image and active garment masks
  useEffect(() => {
    let isMounted = true;
    setImagesLoaded(false);

    const cutoutImg = new Image();
    cutoutImg.crossOrigin = "anonymous";
    cutoutImg.src = `/${cutoutUrl}`;

    const maskPromises = parts.map(part => {
      return new Promise<{ label: string; img: HTMLImageElement }>((resolve) => {
        const maskImg = new Image();
        maskImg.crossOrigin = "anonymous";
        maskImg.src = `/${part.mask_path}`;
        maskImg.onload = () => resolve({ label: part.label, img: maskImg });
        maskImg.onerror = () => resolve({ label: part.label, img: maskImg });
      });
    });

    const cutoutPromise = new Promise<HTMLImageElement | null>((resolve) => {
      cutoutImg.onload = () => resolve(cutoutImg);
      cutoutImg.onerror = () => resolve(null);
    });

    Promise.all([cutoutPromise, ...maskPromises]).then(([cutoutResult, ...maskResults]) => {
      if (!isMounted) return;

      const masksDict: { [label: string]: HTMLImageElement } = {};
      maskResults.forEach((res) => {
        if (res.img.complete && res.img.naturalWidth > 0) {
          masksDict[res.label] = res.img;
        }
      });

      setLoadedImages({
        cutout: cutoutResult,
        masks: masksDict
      });
      setImagesLoaded(true);
    });

    return () => {
      isMounted = false;
    };
  }, [cutoutUrl, parts]);

  // 2. Draw color-coded translucent overlays onto canvas
  useEffect(() => {
    if (!imagesLoaded || !loadedImages.cutout || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const cutoutImg = loadedImages.cutout;
    const width = cutoutImg.naturalWidth;
    const height = cutoutImg.naturalHeight;
    
    // Scale up canvas resolution dynamically by 3x for crisp high-DPI anti-aliasing
    const scale = 3.0;
    const canvasWidth = width * scale;
    const canvasHeight = height * scale;
    
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;

    // Enable high-quality image smoothing
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';

    // Draw main transparent human crop as baseline stretched to high resolution
    ctx.clearRect(0, 0, canvasWidth, canvasHeight);
    ctx.drawImage(cutoutImg, 0, 0, canvasWidth, canvasHeight);

    if (showMaskOverlay) {
      // Temporary canvas to process binary masks pixel-by-pixel at high resolution
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = canvasWidth;
      tempCanvas.height = canvasHeight;
      const tempCtx = tempCanvas.getContext('2d');

      if (tempCtx) {
        parts.forEach(part => {
          const maskImg = loadedImages.masks[part.label];
          if (!maskImg) return;

          let color = { r: 255, g: 255, b: 255 };
          const baseOpacity = 0.45;

          switch (part.label) {
            case 'top_garment':
              color = { r: 239, g: 68, b: 68 }; // Red
              break;
            case 'outerwear':
              color = { r: 236, g: 72, b: 153 }; // Pink
              break;
            case 'bottom_garment':
              color = { r: 6, g: 182, b: 212 }; // Cyan
              break;
            case 'footwear':
            case 'left_shoe':
            case 'right_shoe':
              color = { r: 249, g: 115, b: 22 }; // Orange
              break;
            case 'bag':
              color = { r: 245, g: 158, b: 11 }; // Amber
              break;
            case 'hat':
              color = { r: 16, g: 185, b: 129 }; // Emerald
              break;
            case 'accessory':
            case 'wrist_accessory':
              color = { r: 217, g: 70, b: 239 }; // Fuchsia
              break;
          }

          let opacity = baseOpacity;
          if (hoveredPartLabel !== null) {
            if (hoveredPartLabel === part.label) {
              opacity = 0.75;
            } else {
              opacity = 0.1;
            }
          }

          // Draw binary mask stretched to high resolution with smoothing
          tempCtx.clearRect(0, 0, canvasWidth, canvasHeight);
          tempCtx.imageSmoothingEnabled = true;
          tempCtx.imageSmoothingQuality = 'high';
          tempCtx.drawImage(maskImg, 0, 0, canvasWidth, canvasHeight);

          // Colorize active mask pixels (white) and discard black pixels
          const imgData = tempCtx.getImageData(0, 0, canvasWidth, canvasHeight);
          const data = imgData.data;
          for (let i = 0; i < data.length; i += 4) {
            if (data[i] > 127) {
              data[i] = color.r;
              data[i+1] = color.g;
              data[i+2] = color.b;
              data[i+3] = 255;
            } else {
              data[i+3] = 0;
            }
          }
          tempCtx.putImageData(imgData, 0, 0);

          // Render colorized mask over the transparent human cutout
          ctx.globalAlpha = opacity;
          ctx.drawImage(tempCanvas, 0, 0, canvasWidth, canvasHeight);
        });
      }
    }

    ctx.globalAlpha = 1.0;
  }, [imagesLoaded, loadedImages, parts, hoveredPartLabel, showMaskOverlay]);

  return (
    <div className="relative w-full h-full flex items-center justify-center">
      {!imagesLoaded && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-slate-950/20 backdrop-blur-xs rounded-xl z-10">
          <div className="h-6 w-6 rounded-full border-2 border-violet-500/20 border-t-violet-500 animate-spin"></div>
          <span className="text-[10px] text-violet-400 font-mono tracking-wider">Generating color overlay map...</span>
        </div>
      )}
      <canvas 
        ref={canvasRef} 
        className="max-h-full max-w-full object-contain filter drop-shadow-[0_8px_30px_rgba(139,92,246,0.25)] hover:scale-101 transition-transform duration-300 rounded-lg"
      />
    </div>
  );
};

export default function App() {
  // Global Upload & Job status state
  const [uploadData, setUploadData] = useState<UploadResponse | null>(null);
  const [jobData, setJobData] = useState<JobData | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Segmented mask overlay states
  const [hoveredPartLabel, setHoveredPartLabel] = useState<string | null>(null);
  const [showMaskOverlay, setShowMaskOverlay] = useState<boolean>(true);
  
  // Interactive group canvas S4 state
  const [hoveredPersonId, setHoveredPersonId] = useState<number | null>(null);
  const [matchingClick, setMatchingClick] = useState<boolean>(false);
  const [pendingPersonSelection, setPendingPersonSelection] = useState<{
    x: number;
    y: number;
    personId: number | null;
  } | null>(null);

  // S2 Multi flat-lay confirmation state
  const [confirmedIndices, setConfirmedIndices] = useState<number[]>([]);

  // Batch/Bulk Ingestion states
  const [activeMode, setActiveMode] = useState<'upload' | 'batch_queue'>('upload');
  const [batchJobs, setBatchJobs] = useState<JobData[]>([]);

  // Outfit Composer state (mannequin-less, 1024x1024 canvas with 4 stacked layers)
  const [mannequinMode, setMannequinMode] = useState<'upload' | 'mannequin'>('upload');
  const [composerBg, setComposerBg] = useState<'white' | 'black'>('white');
  const [composerItems, setComposerItems] = useState<{
    hats: ComposerItem[];
    tops: ComposerItem[];
    bottoms: ComposerItem[];
    shoes: ComposerItem[];
  }>({ hats: [], tops: [], bottoms: [], shoes: [] });
  const [composerLoading, setComposerLoading] = useState<boolean>(false);
  const [composerAssetVersion, setComposerAssetVersion] = useState<number>(0);
  const [activeHatIdx, setActiveHatIdx] = useState<number>(-1);
  const [activeTopIdx, setActiveTopIdx] = useState<number>(-1);
  const [activeBottomIdx, setActiveBottomIdx] = useState<number>(-1);
  const [activeShoesIdx, setActiveShoesIdx] = useState<number>(-1);
  const [layerVisibility, setLayerVisibility] = useState<Record<'hat' | 'top' | 'bottom' | 'shoes', boolean>>({
    hat: true,
    top: true,
    bottom: true,
    shoes: true,
  });
  const [stylePrompt, setStylePrompt] = useState<string>('');
  const [stylingBrief, setStylingBrief] = useState<{ name: string; explanation: string } | null>(null);
  const [stylingLoading, setStylingLoading] = useState<boolean>(false);

  // Gemini seed / upload-mode controls
  const [seedLoading, setSeedLoading] = useState<boolean>(false);
  const [seedSummary, setSeedSummary] = useState<{ created: number; errors: number; elapsed: number } | null>(null);
  const [uploadMode, setUploadMode] = useState<'gemini' | 'legacy'>('gemini');
  const [geminiCategoryHint, setGeminiCategoryHint] = useState<'auto' | 'hat' | 'top' | 'bottom' | 'shoes'>('auto');

  // Digital Closet Database state
  const [closetItems, setClosetItems] = useState<WardrobeItem[]>([]);
  const [selectedClosetItem, setSelectedClosetItem] = useState<WardrobeItem | null>(null);
  const [harmonyMatches, setHarmonyMatches] = useState<any[]>([]);
  const [similarMatches, setSimilarMatches] = useState<any[]>([]);
  const [activeCategory, setActiveCategory] = useState<string>('all');
  const [searchText, setSearchText] = useState<string>('');

  const composerItemsCount =
    composerItems.hats.length +
    composerItems.tops.length +
    composerItems.bottoms.length +
    composerItems.shoes.length;

  // Color harmony formula using OKLCH parameters extracted from Gemini
  const calculateOklchHarmony = (
    topItem: ColorBearing | null,
    bottomItem: ColorBearing | null,
    shoesItem: ColorBearing | null,
    hatItem: ColorBearing | null
  ) => {
    const activeItems = [topItem, bottomItem, shoesItem, hatItem].filter(Boolean) as ColorBearing[];
    if (activeItems.length < 2) {
      return { score: 100, label: 'Single Layer Aesthetic', text: 'Select and layer more garments from your digital closet to calculate real-time OKLCH color science harmony metrics.' };
    }
    
    const primaryColors = activeItems.map(item => {
      if (!item.colors || item.colors.length === 0) return null;
      return item.colors.reduce((max, c) => c.weight > max.weight ? c : max, item.colors[0]);
    }).filter(Boolean) as ColorProfile[];

    if (primaryColors.length < 2) {
      return { score: 90, label: 'Standard Contrast Pairing', text: 'Minimal color weights found. Using default matching.' };
    }
    
    let totalDiff = 0;
    let comparisons = 0;

    for (let i = 0; i < primaryColors.length; i++) {
      for (let j = i + 1; j < primaryColors.length; j++) {
        const c1 = primaryColors[i].oklch;
        const c2 = primaryColors[j].oklch;
        
        if (!c1 || !c2 || c1.length < 3 || c2.length < 3) continue;

        let hueDiff = Math.abs(c1[2] - c2[2]);
        if (hueDiff > 180) hueDiff = 360 - hueDiff;

        const lightDiff = Math.abs(c1[0] - c2[0]);

        const isNeutral1 = c1[1] < 0.04;
        const isNeutral2 = c2[1] < 0.04;

        let pairHarmony = 0;
        if (isNeutral1 || isNeutral2) {
          pairHarmony = 96; // Neutral anchors
        } else {
          if (hueDiff < 30) {
            pairHarmony = 94 - (hueDiff * 0.4); // Highly fluid Analogous
          } else if (hueDiff > 150 && hueDiff < 210) {
            const distFromComp = Math.abs(hueDiff - 180);
            pairHarmony = 90 - (distFromComp * 0.3); // High-synergy Complementary
          } else {
            pairHarmony = 72 + (20 - (hueDiff / 10)); // Contrasting/Triadic baseline
          }
        }

        if (lightDiff > 0.35) {
          pairHarmony = Math.min(100, pairHarmony + 6); // Add tone brightness contrast bonus
        }

        totalDiff += pairHarmony;
        comparisons++;
      }
    }

    const finalScore = comparisons > 0 ? Math.round(totalDiff / comparisons) : 88;
    
    let label = 'Classic Contrast';
    let text = 'Bold, distinct color blocking styled intentionally to establish a strong structural frame.';
    
    if (finalScore >= 92) {
      label = 'Elite Complementary Harmony';
      text = 'Excellent color synergy utilizing balanced neutrals and high-contrast complementary hue anchors.';
    } else if (finalScore >= 82) {
      label = 'Sleek Analogous Transition';
      text = 'Smooth, monochromatic fluid transitions across adjacent hue zones that feel extremely premium.';
    } else if (finalScore >= 72) {
      label = 'Balanced Slate Neutrals';
      text = 'Uses understated, classic monochrome weights (grey, black, off-white) to ground the garment layouts.';
    }

    return { score: finalScore, label, text };
  };
  
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load Closet Database + Outfit Composer items on mount
  useEffect(() => {
    fetchCloset();
    fetchComposerItems();
  }, []);

  const fetchCloset = async (category = activeCategory, search = searchText) => {
    try {
      let url = '/api/wardrobe';
      const params: string[] = [];
      if (category && category !== 'all') params.push(`category=${category}`);
      if (search) params.push(`search=${encodeURIComponent(search)}`);
      if (params.length > 0) url += `?${params.join('&')}`;

      const res = await fetch(url);
      if (res.ok) {
        const data = await res.json();
        setClosetItems(data.items || []);
      }
    } catch (err) {
      console.error("Failed to fetch wardrobe:", err);
    }
  };

  // Trigger filters refresh
  const handleCategoryChange = (cat: string) => {
    setActiveCategory(cat);
    fetchCloset(cat, searchText);
  };

  const handleSearchChange = (val: string) => {
    setSearchText(val);
    fetchCloset(activeCategory, val);
  };

  const uploadOneViaLegacy = async (file: File): Promise<{ upload: UploadResponse; job: JobData | null }> => {
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch('/api/upload', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) {
      const detail = await res.json().catch(() => null);
      throw new Error(detail?.detail || "Backend classification routing failed.");
    }

    const upload: UploadResponse = await res.json();
    let job: JobData | null = null;
    const jobRes = await fetch(`/api/job/${upload.job_id}`);
    if (jobRes.ok) {
      job = await jobRes.json();
    }
    return { upload, job };
  };

  // Initializing file uploading and classification routing
  const handleFileUpload = async (file: File) => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setUploadData(null);
    setJobData(null);
    setConfirmedIndices([]);
    setHoveredPersonId(null);
    setPendingPersonSelection(null);

    if (uploadMode === 'gemini') {
      try {
        const result = await uploadOneViaGemini(file);
        if (result.mode === 'select_person' && result.upload && result.job) {
          setUploadData(result.upload);
          setJobData(result.job);
          setStylingBrief({
            name: 'Select Person for Gemini',
            explanation: 'Hover over the uploaded image and click a person. Gemini will extract top, bottom, and shoes from that selected crop.',
          });
        } else {
          await fetchComposerItems();
          fetchCloset();
          setStylingBrief({
            name: 'Gemini Cutout Ready',
            explanation: `Extracted and aligned ${result.category || 'garment'} in one Gemini round-trip. Switch to the Outfit Composer tab to slide it into your layered look.`,
          });
        }
      } catch (err: any) {
        console.error(err);
        try {
          const fallback = await uploadOneViaLegacy(file);
          setUploadData(fallback.upload);
          if (fallback.job) {
            setJobData(fallback.job);
          }
          setStylingBrief({
            name: 'Gemini Unavailable, Using Local Pipeline',
            explanation: `${err.message || 'Gemini upload failed.'} The upload was routed through the local YOLO/SAM/SCHP/SegFormer pipeline instead.`,
          });
        } catch (fallbackErr: any) {
          console.error(fallbackErr);
          setError(`${err.message || 'Gemini upload failed.'} Local fallback also failed: ${fallbackErr.message || 'unknown error'}`);
        }
      } finally {
        setLoading(false);
      }
      return;
    }

    try {
      const result = await uploadOneViaLegacy(file);
      setUploadData(result.upload);
      if (result.job) {
        setJobData(result.job);
      }
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Failed to establish pipeline connection.");
    } finally {
      setLoading(false);
    }
  };

  const fetchBatchJobs = async () => {
    try {
      const res = await fetch('/api/jobs');
      if (res.ok) {
        const jobs = await res.json();
        setBatchJobs(jobs);
      }
    } catch (err) {
      console.error("Failed to fetch batch jobs:", err);
    }
  };

  const handleBulkFileUpload = async (files: File[]) => {
    if (!files || files.length === 0) return;
    
    // If only one file is selected, run standard single file upload for direct preview
    if (files.length === 1) {
      handleFileUpload(files[0]);
      return;
    }

    // Gemini fast-path: process each file sequentially through /api/upload/gemini.
    // Slower than the parallel legacy queue, but avoids hammering the Gemini
    // rate limit and lets every item flow straight into the Composer.
    if (uploadMode === 'gemini') {
      setLoading(true);
      setError(null);
      let succeeded = 0;
      const errors: string[] = [];
      for (const file of files) {
        try {
          const result = await uploadOneViaGemini(file, false);
          if (result.mode === 'direct') succeeded += 1;
        } catch (err: any) {
          errors.push(`${file.name}: ${err.message || 'failed'}`);
        }
      }
      await fetchComposerItems();
      fetchCloset();
      setLoading(false);
      setStylingBrief({
        name: `Gemini Cutouts Done (${succeeded}/${files.length})`,
        explanation: errors.length
          ? `Some uploads failed: ${errors.slice(0, 3).join(' | ')}${errors.length > 3 ? '...' : ''}`
          : 'All photos extracted, keyed and aligned. Composer is ready.',
      });
      if (errors.length === files.length) {
        setError('All Gemini uploads failed. Try Legacy mode or check the API key.');
      }
      return;
    }

    setLoading(true);
    setError(null);
    setActiveMode('batch_queue'); // Switch to batch queue dashboard mode!
    
    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file); // FastAPI accepts list of files with the same name "files"
    });
    
    try {
      const res = await fetch('/api/upload/bulk', {
        method: 'POST',
        body: formData
      });
      
      if (!res.ok) {
        throw new Error("Bulk upload failed.");
      }
      
      const queuedJobs = await res.json();
      console.log("Bulk upload success, queued:", queuedJobs);
      
      // Instantly trigger database reload to fetch active jobs list
      fetchBatchJobs();
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Failed to batch upload photos.");
    } finally {
      setLoading(false);
    }
  };

  const handleLoadJobIntoWorkspace = (job: JobData) => {
    setError(null);
    setConfirmedIndices([]);
    setHoveredPersonId(null);
    setPendingPersonSelection(null);
    
    // Set JobData
    setJobData(job);
    
    // Construct uploadData so the preview renders
    setUploadData({
      job_id: job.job_id,
      scene_type: job.scene_type || 'flat_single',
      original_image_url: job.original_image_url || '',
      dimensions: { width: 800, height: 600 }, // Default, canvas will override with natural size if loaded
      counts: { faces: 0, people: 0, garments: 0 }
    });
    
    // Switch activeMode back to individual workspace preview
    setActiveMode('upload');
  };

  // Batch Queue Polling Controller
  useEffect(() => {
    const hasRunningJobs = batchJobs.some(
      (job) => job.status === 'queued' || job.status === 'processing'
    );
    
    if (activeMode !== 'batch_queue' && !hasRunningJobs) return;
    
    // Poll initially
    fetchBatchJobs();
    
    const interval = setInterval(() => {
      fetchBatchJobs();
    }, 2000);
    
    return () => clearInterval(interval);
  }, [activeMode, batchJobs.length]);

  // Job Polling Controller for S1, S2 (batch ingesting), S3, S4 background tasks
  useEffect(() => {
    if (!jobData?.job_id) return;
    
    // Stop polling if completed, requires user actions, or failed
    if (
      jobData.status === 'completed' || 
      jobData.status === 'failed' || 
      jobData.status === 'requires_confirmation' || 
      jobData.status === 'requires_selection'
    ) {
      // If completed, reload closet database + composer items
      if (jobData.status === 'completed') {
        fetchCloset();
        fetchComposerItems();
      }
      return;
    }

    const interval = setInterval(async () => {
      try {
        const res = await fetch(`/api/job/${jobData.job_id}`);
        if (res.ok) {
          const updatedJob: JobData = await res.json();
          setJobData(updatedJob);
          
          if (updatedJob.status === 'completed' || updatedJob.status === 'failed') {
            clearInterval(interval);
            fetchCloset();
            fetchComposerItems();
          }
        }
      } catch (err) {
        console.error("Job status polling error:", err);
      }
    }, 1500);

    return () => clearInterval(interval);
  }, [jobData?.status, jobData?.job_id]);

  // S2: Multi flat-lay confirmation triggers
  const handleConfirmMultiItems = async () => {
    if (!jobData?.job_id || confirmedIndices.length === 0) return;
    
    setLoading(true);
    try {
      const res = await fetch('/api/confirm-items', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          job_id: jobData.job_id,
          confirmed_indices: confirmedIndices
        })
      });

      if (res.ok) {
        await res.json();
        setJobData(prev => prev ? { ...prev, status: 'processing' } : null);
      } else {
        throw new Error("Item selection confirmation rejected.");
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const toggleConfirmIndex = (index: number) => {
    setConfirmedIndices(prev => 
      prev.includes(index) ? prev.filter(i => i !== index) : [...prev, index]
    );
  };

  // S4: Group Photo interactive click controller
  const handleCanvasClick = async (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || !uploadData || jobData?.status !== 'requires_selection') return;
    
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    
    // Scale client click coordinate to original image size
    const width = uploadData.dimensions?.width || canvas.width;
    const height = uploadData.dimensions?.height || canvas.height;
    const clickX = (e.clientX - rect.left) * (width / rect.width);
    const clickY = (e.clientY - rect.top) * (height / rect.height);

    let matchedPerson: DetectedPerson | null = null;
    for (let i = (jobData.detected_items?.length || 0) - 1; i >= 0; i--) {
      const person = jobData.detected_items?.[i];
      if (Array.isArray(person?.polygon) && isPointInPolygon(clickX, clickY, person.polygon)) {
        matchedPerson = person;
        break;
      }
    }

    if (!matchedPerson) {
      alert("Click missed! Try clicking in the torso center of an individual.");
      setPendingPersonSelection(null);
      return;
    }

    setHoveredPersonId(matchedPerson.id ?? null);
    setPendingPersonSelection({
      x: clickX,
      y: clickY,
      personId: matchedPerson.id ?? null,
    });
  };

  const startSelectedPersonParsing = async (method: 'gemini' | 'segformer') => {
    if (!uploadData || !pendingPersonSelection) return;

    setMatchingClick(true);
    try {
      const apiBase = uploadMode === 'gemini' ? GEMINI_STAGE_API_BASE : '';
      const selectionEndpoint = `${apiBase}/api/select-person/${method}`;
      const res = await fetch(selectionEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          job_id: uploadData.job_id,
          x: pendingPersonSelection.x,
          y: pendingPersonSelection.y
        })
      });

      if (res.ok) {
        const data = await res.json();
        if (data.matched) {
          // Transition job to processing state
          setJobData(prev => prev ? { ...prev, status: 'processing' } : null);
          setPendingPersonSelection(null);
          setStylingBrief(method === 'gemini'
            ? {
                name: 'Gemini Extracting Selected Person',
                explanation: 'Gemini is isolating top, bottom, and shoes from the person you clicked, then aligning them for the Composer.',
              }
            : {
                name: 'SegFormer Parsing Selected Person',
                explanation: 'SegFormer is parsing clothes labels for the selected person, creating top, bottom, shoes, and accessory layers for the Composer.',
              }
          );
        } else {
          alert(data.message || "Click missed! Try clicking in the torso center of an individual.");
        }
      } else {
        const detail = await res.json().catch(() => null);
        throw new Error(detail?.detail || `${method} selection failed (${res.status})`);
      }
    } catch (err: unknown) {
      console.error(err);
      alert(err instanceof Error ? err.message : "Error identifying clicked person.");
    } finally {
      setMatchingClick(false);
    }
  };

  // S4: Hover point-in-polygon checking (Ray-Casting locally)
  const isPointInPolygon = (x: number, y: number, polygon: number[][]): boolean => {
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
      const xi = polygon[i][0], yi = polygon[i][1];
      const xj = polygon[j][0], yj = polygon[j][1];
      
      const intersect = ((yi > y) !== (yj > y))
        && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
      if (intersect) inside = !inside;
    }
    return inside;
  };

  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || !uploadData || !jobData?.detected_items) return;
    
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    
    const width = uploadData.dimensions?.width || canvas.width;
    const height = uploadData.dimensions?.height || canvas.height;
    const scaleX = (e.clientX - rect.left) * (width / rect.width);
    const scaleY = (e.clientY - rect.top) * (height / rect.height);
    
    let matchedId: number | null = null;
    
    // Backward check for foreground overlaps
    for (let i = jobData.detected_items.length - 1; i >= 0; i--) {
      const p = jobData.detected_items[i];
      if (isPointInPolygon(scaleX, scaleY, p.polygon)) {
        matchedId = p.id;
        break;
      }
    }
    
    if (matchedId !== hoveredPersonId) {
      setHoveredPersonId(matchedId);
    }
  };

  // S4: Canvas rendering loop
  useEffect(() => {
    if (!canvasRef.current || !uploadData || !jobData?.detected_items) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const imgElement = new Image();
    imgElement.src = `/${uploadData.original_image_url}`;
    imgElement.onload = () => {
      const width = uploadData.dimensions?.width || imgElement.naturalWidth || 800;
      const height = uploadData.dimensions?.height || imgElement.naturalHeight || 600;
      canvas.width = width;
      canvas.height = height;

      // Draw original face-blurred image
      ctx.drawImage(imgElement, 0, 0, width, height);

      // Draw human selection overlays
      const detectedItems = jobData.detected_items ?? [];
      detectedItems.forEach(person => {
        const isHovered = person.id === hoveredPersonId;
        if (!isHovered) return;

        ctx.save();
        ctx.beginPath();
        
        if (person.polygon.length > 0) {
          ctx.moveTo(person.polygon[0][0], person.polygon[0][1]);
          for (let p = 1; p < person.polygon.length; p++) {
            ctx.lineTo(person.polygon[p][0], person.polygon[p][1]);
          }
        }
        ctx.closePath();

        // Neon indigo overlay glow
        ctx.strokeStyle = '#a78bfa';
        ctx.lineWidth = Math.max(3, width / 250);
        ctx.fillStyle = 'rgba(139, 92, 246, 0.16)';
        
        ctx.shadowBlur = 20;
        ctx.shadowColor = '#8b5cf6';

        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.stroke();
        ctx.restore();
      });
    };
  }, [uploadData, jobData?.detected_items, hoveredPersonId]);

  // Wardrobe Closet selection triggers harmony recommendations
  const handleSelectClosetItem = async (item: WardrobeItem) => {
    setSelectedClosetItem(item);
    setHarmonyMatches([]);
    setSimilarMatches([]);

    try {
      // 1. Fetch color harmony recommendations
      const harmonyRes = await fetch(`/api/wardrobe/${item.id}/harmony`);
      if (harmonyRes.ok) {
        const harmonyData = await harmonyRes.json();
        setHarmonyMatches(harmonyData.matches || []);
      }
      
      // 2. Fetch visual similarity recommendations
      const similarRes = await fetch(`/api/wardrobe/${item.id}/similar`);
      if (similarRes.ok) {
        const similarData = await similarRes.json();
        setSimilarMatches(similarData.matches || []);
      }
    } catch (err) {
      console.error(err);
    }
  };

  // Loads the 4 composer category lists with their pre-aligned 1024x1024 PNGs.
  const fetchComposerItems = async () => {
    setComposerLoading(true);
    try {
      const res = await fetch('/api/composer/all');
      if (!res.ok) throw new Error('Failed to load composer items');
      const data = await res.json();
      setComposerItems({
        hats: data.hats || [],
        tops: data.tops || [],
        bottoms: data.bottoms || [],
        shoes: data.shoes || [],
      });
      setComposerAssetVersion(Date.now());
    } catch (err) {
      console.error('Composer fetch failed:', err);
    } finally {
      setComposerLoading(false);
    }
  };

  const handleRealignAll = async () => {
    setComposerLoading(true);
    try {
      await fetch('/api/composer/realign-all', { method: 'POST' });
      await fetchComposerItems();
    } catch (err) {
      console.error('Realign-all failed:', err);
    } finally {
      setComposerLoading(false);
    }
  };

  // Wipes the closet and regenerates a balanced 12-item starter set via Gemini.
  const handleSeedClosetGemini = async () => {
    if (!confirm(
      'Wipe every closet item and regenerate a 12-item Gemini starter set?\n\n' +
      'This deletes all current crops + aligned PNGs and takes ~4 minutes.'
    )) {
      return;
    }
    setSeedLoading(true);
    setSeedSummary(null);
    setError(null);
    try {
      const res = await fetch('/api/composer/seed', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ per_category: 3, wipe: true }),
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => null);
        throw new Error(detail?.detail || `Seed request failed (${res.status})`);
      }
      const summary = await res.json();
      setSeedSummary({
        created: summary.total_succeeded ?? 0,
        errors: (summary.errors || []).length,
        elapsed: summary.elapsed_seconds ?? 0,
      });
      setActiveHatIdx(-1);
      setActiveTopIdx(-1);
      setActiveBottomIdx(-1);
      setActiveShoesIdx(-1);
      await fetchComposerItems();
      fetchCloset();
    } catch (err: any) {
      console.error('Gemini seed failed:', err);
      setError(err.message || 'Failed to seed closet via Gemini.');
    } finally {
      setSeedLoading(false);
    }
  };

  // Gemini fast-path upload: sends one image to /api/upload/gemini and returns
  // a fully aligned wardrobe item in a single round-trip.
  const uploadOneViaGemini = async (
    file: File,
    staged = true
  ): Promise<{ mode: 'direct' | 'select_person'; id?: string; category?: string; upload?: UploadResponse; job?: JobData }> => {
    const formData = new FormData();
    formData.append('file', file);
    if (geminiCategoryHint !== 'auto') {
      formData.append('category_hint', geminiCategoryHint);
    }
    const endpoint = staged
      ? `${GEMINI_STAGE_API_BASE}/api/upload/gemini-stage`
      : `${GEMINI_STAGE_API_BASE}/api/upload/gemini`;
    const res = await fetch(endpoint, { method: 'POST', body: formData });
    if (!res.ok) {
      const detail = await res.json().catch(() => null);
      throw new Error(detail?.detail || `Gemini upload failed (${res.status})`);
    }
    const data = await res.json();
    if (data.mode === 'select_person') {
      return {
        mode: 'select_person',
        upload: {
          job_id: data.job_id,
          scene_type: data.scene_type,
          original_image_url: data.original_image_url,
          dimensions: data.dimensions,
          counts: data.counts,
        },
        job: {
          job_id: data.job_id,
          status: 'requires_selection',
          scene_type: data.scene_type,
          original_image_url: data.original_image_url,
          detected_items: data.detected_items || [],
          result: null,
          error: null,
        },
      };
    }
    return { mode: 'direct', id: data.id, category: data.composer_category };
  };

  // Coordinates a 4-layer outfit via the Gemini stylist (supports optional text prompt).
  const handleAutoStyleOutfit = async (promptText?: string) => {
    setStylingLoading(true);
    setError(null);
    setStylingBrief(null);

    try {
      const res = await fetch('/api/composer/generate-style', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: promptText || stylePrompt || null }),
      });

      if (!res.ok) {
        throw new Error('Gemini auto-styling API request failed.');
      }

      const styling = await res.json();
      setStylingBrief({
        name: styling.style_name || 'Gemini Balanced Coordinate',
        explanation: styling.explanation || 'Perfectly harmonious outfit composed by Gemini.',
      });

      const layers = styling.layers || {};

      const pickIndex = (
        list: ComposerItem[],
        layer: { id?: string } | null
      ): number => (layer && layer.id ? list.findIndex(it => it.id === layer.id) : -1);

      setActiveHatIdx(pickIndex(composerItems.hats, layers.hat));
      setActiveTopIdx(pickIndex(composerItems.tops, layers.top));
      setActiveBottomIdx(pickIndex(composerItems.bottoms, layers.bottom));
      setActiveShoesIdx(pickIndex(composerItems.shoes, layers.shoes));
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'Failed to style wardrobe using Gemini.');
    } finally {
      setStylingLoading(false);
    }
  };

  // Delete wardrobe garment
  const handleDeleteItem = async (itemId: string) => {
    if (!confirm("Are you sure you want to delete this clothing item from your digital closet?")) return;
    
    try {
      const res = await fetch(`/api/wardrobe/${itemId}`, { method: 'DELETE' });
      if (res.ok) {
        setSelectedClosetItem(null);
        fetchCloset();
        fetchComposerItems();
      }
    } catch (err) {
      console.error(err);
      alert("Failed to delete item.");
    }
  };

  // Helper: map scenario types to human readable subtitles
  const getSceneBadge = (type: string | null) => {
    if (!type) return "Scanning Scene...";
    switch(type) {
      case 'flat_single': return "Single Flat-Lay Garment (S1)";
      case 'flat_multi': return "Multi-Item Flat-Lay (S2)";
      case 'single_person': return "Single Worn Outfit (S3)";
      case 'group_photo': return "Group Photo Selection (S4)";
      default: return type;
    }
  };

  return (
    <div className="min-h-screen text-slate-200 flex flex-col antialiased">
      {/* Header */}
      <header className="border-b border-white/10 backdrop-blur-xl bg-[#090b14]/75 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-violet-600 to-pink-500 flex items-center justify-center shadow-lg shadow-violet-500/20">
              <Layers className="h-5 w-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight flex items-center gap-2">
                Vestir AI <span className="text-gradient-neon text-sm font-semibold py-0.5 px-2.5 rounded-full bg-violet-500/10 border border-violet-500/20">Pipeline v2</span>
              </h1>
              <p className="text-xs text-slate-400">Intelligent Multi-Scenario Wardrobe Ingestion</p>
            </div>
          </div>
          
          <div className="hidden md:flex items-center gap-4">
            <span className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-semibold">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
              FastAPI Pipeline Online
            </span>
            <div className="flex items-center gap-1 bg-white/5 border border-white/10 rounded-lg px-2 py-1 text-xs text-slate-300 font-mono">
              <Cpu className="h-3.5 w-3.5 text-violet-400" />
              <span>YOLO11-seg + SAM2 + OKLCH</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-6 py-8 flex flex-col gap-10">
        
        {/* TOP: Ingestion Console & Workspace */}
        <section className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* LEFT 7 cols: Interactive Stage Console */}
          <div className="lg:col-span-7 flex flex-col gap-6">
            <div className="glass-panel p-6 flex flex-col gap-6 relative overflow-hidden min-h-[480px]">
              
              <div className="flex items-center justify-between border-b border-white/10 pb-4 flex-wrap gap-3">
                <div className="flex items-center gap-2 bg-slate-900/60 p-1 rounded-xl border border-white/5">
                  <button
                    onClick={() => setMannequinMode('upload')}
                    className={`px-3 py-1.5 rounded-lg text-xs font-bold flex items-center gap-2 transition-all cursor-pointer ${
                      mannequinMode === 'upload'
                        ? 'bg-gradient-to-r from-violet-600 to-pink-500 text-white shadow-[0_0_12px_rgba(139,92,246,0.3)]'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    <Upload className="h-3.5 w-3.5" />
                    Ingestion Console
                  </button>
                  <button
                    onClick={() => setMannequinMode('mannequin')}
                    className={`px-3 py-1.5 rounded-lg text-xs font-bold flex items-center gap-2 transition-all cursor-pointer ${
                      mannequinMode === 'mannequin'
                        ? 'bg-gradient-to-r from-violet-600 to-pink-500 text-white shadow-[0_0_12px_rgba(139,92,246,0.3)]'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    <Sparkles className="h-3.5 w-3.5 animate-pulse" />
                    Outfit Composer
                  </button>
                </div>
                
                <div className="flex items-center gap-2">
                  {mannequinMode === 'upload' ? (
                    <>
                      <button
                        onClick={() => {
                          fetchBatchJobs();
                          setActiveMode(activeMode === 'batch_queue' ? 'upload' : 'batch_queue');
                        }}
                        className={`px-3 py-1 rounded-full border text-xs font-semibold tracking-wider transition cursor-pointer ${
                          activeMode === 'batch_queue' 
                            ? 'bg-violet-500/20 border-violet-400 text-violet-300 shadow-[0_0_10px_rgba(139,92,246,0.2)]'
                            : 'bg-white/5 border-white/10 text-slate-400 hover:bg-white/10 hover:border-white/20'
                        }`}
                      >
                        Queue ({batchJobs.length})
                      </button>
                      {jobData?.scene_type && activeMode === 'upload' && (
                        <span className="px-3 py-1 rounded-full bg-violet-500/15 border border-violet-500/30 text-violet-300 text-xs font-semibold">
                          {getSceneBadge(jobData.scene_type)}
                        </span>
                      )}
                    </>
                  ) : (
                    <div className="flex items-center gap-2">
                      <div className="flex items-center gap-1 bg-slate-900/60 p-0.5 rounded-lg border border-white/5">
                        <button
                          onClick={() => setComposerBg('white')}
                          className={`px-2.5 py-1 rounded-md text-[10px] font-extrabold uppercase tracking-widest transition cursor-pointer ${
                            composerBg === 'white' ? 'bg-white text-slate-900' : 'text-slate-400 hover:text-slate-200'
                          }`}
                        >
                          White
                        </button>
                        <button
                          onClick={() => setComposerBg('black')}
                          className={`px-2.5 py-1 rounded-md text-[10px] font-extrabold uppercase tracking-widest transition cursor-pointer ${
                            composerBg === 'black' ? 'bg-black text-white border border-white/20' : 'text-slate-400 hover:text-slate-200'
                          }`}
                        >
                          Black
                        </button>
                      </div>
                      <button
                        onClick={handleRealignAll}
                        disabled={composerLoading || seedLoading}
                        className="px-3 py-1 rounded-full border border-white/10 hover:border-violet-500/50 text-[10px] font-extrabold uppercase tracking-widest text-slate-400 hover:text-violet-300 transition cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
                        title="Regenerate the 1024x1024 aligned PNGs for every closet item"
                      >
                        {composerLoading ? 'Aligning...' : 'Realign'}
                      </button>
                      <button
                        onClick={handleSeedClosetGemini}
                        disabled={seedLoading || composerLoading}
                        className="px-3 py-1 rounded-full border border-emerald-500/30 hover:border-emerald-400 bg-emerald-500/10 hover:bg-emerald-500/20 text-[10px] font-extrabold uppercase tracking-widest text-emerald-300 hover:text-emerald-200 transition cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1.5"
                        title="Wipe the closet and regenerate 12 fresh apparel cutouts using the Gemini image API. Takes ~4 minutes."
                      >
                        {seedLoading ? (
                          <>
                            <span className="h-2.5 w-2.5 rounded-full border-2 border-emerald-300/40 border-t-emerald-300 animate-spin" />
                            Seeding...
                          </>
                        ) : (
                          <>
                            <Sparkles className="h-3 w-3" />
                            Generate Seed (12)
                          </>
                        )}
                      </button>
                    </div>
                  )}
                </div>
              </div>

              {mannequinMode === 'upload' ? (
                <>

              {/* Batch Ingestion Queue Dashboard */}
              {activeMode === 'batch_queue' && !loading && (
                <div className="flex-grow flex flex-col gap-4 pt-3 overflow-hidden">
                  <div className="flex items-center justify-between border-b border-white/5 pb-2">
                    <span className="text-[10px] font-bold tracking-wider text-slate-400 uppercase">Live Ingestion List</span>
                    <button
                      onClick={() => {
                        setUploadData(null);
                        setJobData(null);
                        setActiveMode('upload');
                        setPendingPersonSelection(null);
                      }}
                      className="px-2 py-0.5 rounded border border-white/10 hover:border-violet-500/50 text-[8px] font-extrabold uppercase tracking-widest text-slate-400 hover:text-violet-300 transition"
                    >
                      New Upload
                    </button>
                  </div>

                  <div className="flex-1 overflow-y-auto pr-1 flex flex-col gap-3 max-h-[350px]">
                    {batchJobs.length === 0 ? (
                      <div className="flex-1 flex flex-col items-center justify-center py-16 gap-2 text-slate-500 text-xs">
                        <Shirt className="h-10 w-10 text-slate-600 mb-1" />
                        <span>No active or historic ingestion jobs found</span>
                      </div>
                    ) : (
                      batchJobs.map((job) => {
                        const dateStr = job.created_at ? new Date(job.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';
                        const isInteractive = job.status === 'requires_confirmation' || job.status === 'requires_selection';
                        const badgeColor = 
                          job.status === 'completed' ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400' :
                          job.status === 'failed' ? 'bg-rose-500/10 border-rose-500/30 text-rose-400' :
                          job.status === 'processing' ? 'bg-blue-500/10 border-blue-500/30 text-blue-400 font-bold animate-pulse' :
                          isInteractive ? 'bg-amber-500/15 border-amber-500/30 text-amber-400 font-bold animate-bounce' :
                          'bg-slate-500/10 border-slate-500/30 text-slate-400';

                        return (
                          <div 
                            key={job.job_id} 
                            className={`p-3 rounded-xl border flex items-center justify-between gap-4 transition bg-slate-950/40 ${
                              isInteractive 
                                ? 'border-violet-500/30 bg-violet-500/5 hover:border-violet-400 hover:shadow-[0_0_12px_rgba(139,92,246,0.15)] cursor-pointer' 
                                : 'border-white/5 hover:border-white/10 hover:bg-slate-900/40'
                            }`}
                            onClick={() => isInteractive && handleLoadJobIntoWorkspace(job)}
                          >
                            <div className="flex items-center gap-3 min-w-0">
                              <div className="h-12 w-12 rounded-lg bg-slate-900 border border-white/5 flex items-center justify-center p-1 shrink-0 overflow-hidden">
                                {job.original_image_url ? (
                                  <img src={`/${job.original_image_url}`} alt="Job crop" className="max-h-full max-w-full object-contain rounded" />
                                ) : (
                                  <Shirt className="h-5 w-5 text-slate-700" />
                                )}
                              </div>
                              
                              <div className="min-w-0">
                                <div className="flex items-center gap-2">
                                  <h5 className="font-bold text-xs text-slate-200 truncate font-mono">
                                    {job.job_id.substring(4, 12).toUpperCase()}
                                  </h5>
                                  <span className="text-[8px] text-slate-500 font-mono shrink-0">
                                    {dateStr}
                                  </span>
                                </div>
                                <p className="text-[10px] text-slate-400 truncate mt-0.5 capitalize">
                                  {job.scene_type ? getSceneBadge(job.scene_type) : 'Pending classification'}
                                </p>
                              </div>
                            </div>

                            <div className="flex items-center gap-2 shrink-0">
                              <span className={`px-2 py-0.5 rounded-full border text-[8px] font-extrabold uppercase tracking-wider ${badgeColor}`}>
                                {job.status === 'requires_selection' ? 'Select Person' :
                                 job.status === 'requires_confirmation' ? 'Confirm Items' :
                                 job.status}
                              </span>
                              
                              {isInteractive && (
                                <button
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    handleLoadJobIntoWorkspace(job);
                                  }}
                                  className="px-2 py-0.5 rounded bg-violet-500 hover:bg-violet-600 text-[9px] font-bold text-white transition flex items-center gap-1 cursor-pointer"
                                >
                                  <Sparkles className="h-2.5 w-2.5 animate-pulse" />
                                  Open
                                </button>
                              )}
                            </div>
                          </div>
                        );
                      })
                    )}
                  </div>
                </div>
              )}

              {/* Upload Selector (Idle state) */}
              {!uploadData && !loading && activeMode === 'upload' && (
                <div className="flex-1 flex flex-col gap-3">
                  {/* Pipeline mode toggle: Gemini fast-path vs legacy YOLO+SAM2 */}
                  <div className="rounded-2xl border border-white/10 bg-slate-950/40 p-3 flex flex-col gap-2.5">
                    <div className="flex items-center justify-between gap-3">
                      <div className="flex flex-col">
                        <span className="text-[10px] font-extrabold tracking-widest text-slate-400 uppercase">Ingestion Mode</span>
                        <span className="text-[11px] text-slate-500 mt-0.5">
                          {uploadMode === 'gemini'
                            ? 'Gemini extracts the cutout in one round-trip. Fast and tolerant of any photo background.'
                            : 'Legacy YOLO11 + SAM2 + SCHP local pipeline. Slower but fully offline.'}
                        </span>
                      </div>
                      <div className="flex items-center gap-1 bg-slate-900/60 p-0.5 rounded-lg border border-white/5 shrink-0">
                        <button
                          onClick={() => setUploadMode('gemini')}
                          className={`px-2.5 py-1 rounded-md text-[10px] font-extrabold uppercase tracking-widest transition cursor-pointer flex items-center gap-1.5 ${
                            uploadMode === 'gemini'
                              ? 'bg-gradient-to-r from-emerald-500/80 to-emerald-400 text-slate-950'
                              : 'text-slate-400 hover:text-slate-200'
                          }`}
                        >
                          <Sparkles className="h-3 w-3" />
                          Gemini (Fast)
                        </button>
                        <button
                          onClick={() => setUploadMode('legacy')}
                          className={`px-2.5 py-1 rounded-md text-[10px] font-extrabold uppercase tracking-widest transition cursor-pointer ${
                            uploadMode === 'legacy'
                              ? 'bg-violet-500/80 text-white'
                              : 'text-slate-400 hover:text-slate-200'
                          }`}
                        >
                          Legacy
                        </button>
                      </div>
                    </div>
                    {uploadMode === 'gemini' && (
                      <div className="flex items-center gap-2 pt-1.5 border-t border-white/5">
                        <span className="text-[9px] font-extrabold uppercase tracking-widest text-slate-500 shrink-0">Category Hint</span>
                        <div className="flex items-center gap-1 flex-wrap">
                          {(['auto', 'hat', 'top', 'bottom', 'shoes'] as const).map(cat => (
                            <button
                              key={cat}
                              onClick={() => setGeminiCategoryHint(cat)}
                              className={`px-2 py-0.5 rounded-md text-[9px] font-bold uppercase tracking-widest transition cursor-pointer ${
                                geminiCategoryHint === cat
                                  ? 'bg-emerald-500/20 border border-emerald-400/50 text-emerald-300'
                                  : 'border border-white/10 text-slate-500 hover:text-slate-300 hover:border-white/20'
                              }`}
                            >
                              {cat}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  <div
                    onClick={() => fileInputRef.current?.click()}
                    className="flex-1 border-2 border-dashed border-white/10 hover:border-violet-500/50 rounded-2xl flex flex-col items-center justify-center p-8 text-center cursor-pointer hover:bg-white/[0.01] transition-all group"
                  >
                    <input
                      type="file"
                      ref={fileInputRef}
                      onChange={(e) => e.target.files && handleBulkFileUpload(Array.from(e.target.files))}
                      className="hidden"
                      accept="image/*"
                      multiple
                    />
                    <div className={`h-16 w-16 rounded-2xl border flex items-center justify-center mb-4 group-hover:scale-110 transition-all duration-300 ${
                      uploadMode === 'gemini'
                        ? 'bg-emerald-500/5 border-emerald-500/20 group-hover:border-emerald-500/40 group-hover:bg-emerald-500/10'
                        : 'bg-white/5 border-white/10 group-hover:border-violet-500/30 group-hover:bg-violet-500/10'
                    }`}>
                      <Upload className={`h-8 w-8 ${uploadMode === 'gemini' ? 'text-emerald-400 group-hover:text-emerald-300' : 'text-slate-400 group-hover:text-violet-400'}`} />
                    </div>
                    <h3 className="font-bold text-lg text-white">
                      {uploadMode === 'gemini' ? 'Drop Photos for Gemini Cutout' : 'Drag & Drop Clothing Photos'}
                    </h3>
                    <p className="text-xs text-slate-400 mt-1 max-w-sm">
                      {uploadMode === 'gemini'
                        ? 'Worn outfits, flat lays, mirror selfies - Gemini will redraw the garment on a clean white background and add it straight to your Composer.'
                        : 'Supports single flat lays, multi-flat lays, worn outfits, or group photos. Upload multiple images to ingest in bulk parallel streams.'}
                    </p>
                    <span className="btn-neon px-6 py-2.5 mt-5 text-xs">
                      <Sparkles className="h-4 w-4" />
                      Browse Photos
                    </span>
                  </div>
                </div>
              )}

              {/* Loader Spinner */}
              {loading && (
                <div className="flex-1 flex flex-col items-center justify-center gap-4">
                  <div className="h-12 w-12 rounded-full border-4 border-violet-500/20 border-t-violet-500 animate-spin"></div>
                  <p className="text-slate-300 font-semibold animate-pulse">Running Ingestion Classifier...</p>
                  <p className="text-xs text-slate-500">Detecting human contours, garments, and analyzing scene type</p>
                </div>
              )}

              {/* Active Workspace States */}
              {uploadData && jobData && activeMode === 'upload' && (
                <div className="flex-grow flex flex-col gap-6">
                  
                  {/* Job In Progress (S1, S3, S2 confirmation progress) */}
                  {jobData.status === 'processing' && (
                    <div className="flex-1 flex flex-col items-center justify-center py-16 gap-4 text-center">
                      <div className="h-14 w-14 rounded-full border-4 border-violet-500/20 border-t-pink-500 animate-spin"></div>
                      <div>
                        <h4 className="font-bold text-white text-base animate-pulse">
                          {stylingBrief?.name || 'Executing Local AI Segmentation...'}
                        </h4>
                        <p className="text-xs text-slate-400 mt-1 max-w-xs mx-auto">
                          {stylingBrief?.explanation || 'Running segmentation, extracting transparent crops, and synthesizing fashion parameters.'}
                        </p>
                      </div>
                    </div>
                  )}

                  {jobData.status === 'failed' && (
                    <div className="flex-1 flex flex-col items-center justify-center py-16 gap-4 text-center">
                      <div className="h-14 w-14 rounded-full border border-rose-500/30 bg-rose-500/10 flex items-center justify-center">
                        <AlertCircle className="h-7 w-7 text-rose-300" />
                      </div>
                      <div>
                        <h4 className="font-bold text-white text-base">Pipeline Failed</h4>
                        <p className="text-xs text-rose-200/90 mt-1 max-w-md mx-auto">
                          {jobData.error || 'The backend rejected this job. Try again or choose the other parser.'}
                        </p>
                      </div>
                    </div>
                  )}

                  {/* S1: Single Flat Completed View */}
                  {jobData.scene_type === 'flat_single' && jobData.status === 'completed' && jobData.result && (
                    <div className="flex flex-col md:flex-row gap-6 items-start">
                      <div className="w-full md:w-1/2 flex flex-col gap-3">
                        <div className="h-64 rounded-xl border border-white/10 bg-slate-950 flex items-center justify-center p-4 relative overflow-hidden"
                             style={{
                               backgroundImage: 'linear-gradient(45deg, #111422 25%, transparent 25%), linear-gradient(-45deg, #111422 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #111422 75%), linear-gradient(-45deg, transparent 75%, #111422 75%)',
                               backgroundSize: '16px 16px',
                               backgroundPosition: '0 0, 0 8px, 8px -8px, -8px 0px'
                             }}>
                          <img src={`/${jobData.result[0].image_path}`} alt="Crop" className="max-h-full max-w-full object-contain filter drop-shadow-2xl" />
                        </div>
                      </div>
                      <div className="w-full md:w-1/2 flex flex-col gap-4">
                        <span className="text-xs font-bold text-slate-400 tracking-wider block uppercase">Ingestion Report</span>
                        <h3 className="text-xl font-bold text-white leading-tight capitalize">
                          {jobData.result[0].brand} {jobData.result[0].subtype}
                        </h3>
                        <div className="flex flex-wrap gap-1.5">
                          {jobData.result[0].tags.slice(0, 5).map(t => (
                            <span key={t} className="px-2 py-0.5 rounded bg-white/5 border border-white/10 text-slate-300 text-[10px] font-semibold">
                              #{t}
                            </span>
                          ))}
                        </div>
                        
                        <div className="grid grid-cols-2 gap-2 text-xs pt-2">
                          <div className="p-2.5 rounded-lg bg-white/5 border border-white/5">
                            <span className="text-slate-400 block text-[10px]">Occasion</span>
                            <span className="font-semibold mt-0.5 block text-slate-200 capitalize">{jobData.result[0].occasion}</span>
                          </div>
                          <div className="p-2.5 rounded-lg bg-white/5 border border-white/5">
                            <span className="text-slate-400 block text-[10px]">Season</span>
                            <span className="font-semibold mt-0.5 block text-slate-200 capitalize">{jobData.result[0].season}</span>
                          </div>
                        </div>

                        {/* Colors */}
                        <div className="flex flex-col gap-1.5 pt-2">
                          <span className="text-xs font-semibold text-slate-400">Dominant OKLCH Colors</span>
                          <div className="flex items-center gap-2">
                            {jobData.result[0].colors.map((c, i) => (
                              <div key={i} className="flex items-center gap-1.5 bg-white/5 px-2 py-1 rounded-full border border-white/5">
                                <span className="h-3.5 w-3.5 rounded-full border border-white/20 shadow-inner" style={{ backgroundColor: `rgb(${c.rgb.join(',')})` }} />
                                <span className="text-[10px] font-mono text-slate-300 font-bold">{Math.round(c.weight * 100)}%</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* S2: Multi Flat lay checklist confirmation */}
                  {jobData.scene_type === 'flat_multi' && jobData.status === 'requires_confirmation' && jobData.detected_items && (
                    <div className="flex flex-col gap-5 flex-grow">
                      <div>
                        <h4 className="font-bold text-white text-sm">Select flat-lay garments to confirm:</h4>
                        <p className="text-xs text-slate-400 mt-0.5">We detected {jobData.detected_items.length} garments. Choose which pieces to ingest into your closet.</p>
                      </div>
                      
                      <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 max-h-[300px] overflow-y-auto pr-1">
                        {jobData.detected_items.map((item) => (
                          <div 
                            key={item.index}
                            onClick={() => toggleConfirmIndex(item.index)}
                            className={`glass-card p-3 flex flex-col gap-3 relative cursor-pointer group border ${
                              confirmedIndices.includes(item.index) 
                                ? 'border-violet-500 bg-violet-500/5' 
                                : 'border-white/5'
                            }`}
                          >
                            <div className="h-28 rounded-lg bg-slate-950 flex items-center justify-center p-2 relative overflow-hidden">
                              <img src={`/${item.image_url}`} alt="Garment Preview" className="max-h-full max-w-full object-contain filter drop-shadow-md" />
                              <div className="absolute top-1.5 right-1.5 h-5 w-5 rounded bg-black/75 border border-white/20 flex items-center justify-center">
                                {confirmedIndices.includes(item.index) && <Check className="h-3.5 w-3.5 text-violet-400" />}
                              </div>
                            </div>
                            <div className="flex justify-between items-center text-[10px] font-semibold text-slate-400">
                              <span>Piece #{item.index + 1}</span>
                              <span className="font-mono bg-white/5 px-1.5 py-0.5 rounded">
                                score: {item.salience_score}
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>

                      <div className="mt-auto pt-4 border-t border-white/10 flex items-center justify-between gap-4">
                        <span className="text-xs text-slate-400">
                          {confirmedIndices.length} of {jobData.detected_items.length} items selected
                        </span>
                        <button
                          onClick={handleConfirmMultiItems}
                          disabled={confirmedIndices.length === 0}
                          className="btn-neon px-6 py-2.5 text-xs"
                        >
                          <Sparkles className="h-4 w-4" />
                          Confirm & Extract Selected
                        </button>
                      </div>
                    </div>
                  )}

                  {/* S2 Multi Complete View */}
                  {jobData.scene_type === 'flat_multi' && jobData.status === 'completed' && jobData.result && (
                    <div className="flex flex-col gap-4">
                      <div className="p-3 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 rounded-xl text-xs flex items-center gap-2">
                        <Check className="h-4.5 w-4.5" />
                        <span>Successfully ingested {jobData.result.length} flat-lay garments into your digital closet!</span>
                      </div>
                      
                      <div className="grid grid-cols-3 gap-3">
                        {jobData.result.map(item => (
                          <div key={item.id} className="p-2.5 rounded-lg border border-white/5 bg-white/[0.01] flex flex-col gap-2">
                            <div className="h-20 rounded-md bg-slate-950 flex items-center justify-center p-1.5">
                              <img src={`/${item.image_path}`} alt="Ingested Crop" className="max-h-full object-contain filter drop-shadow-md" />
                            </div>
                            <span className="text-[10px] font-bold text-center block text-slate-300 capitalize truncate">
                              {item.brand} {item.subtype}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* S3: Worn outfit layer details completed */}
                  {jobData.scene_type === 'single_person' && jobData.status === 'completed' && jobData.result && (
                    <div className="flex flex-col gap-5">
                      <div>
                        <h4 className="font-bold text-white text-sm">Parsed Wardrobe Layer Cutouts:</h4>
                        <p className="text-xs text-slate-400 mt-0.5">We segmented the single-worn outfit and parsed upper-body layers, lower garments, and shoes.</p>
                      </div>
                      
                      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                        {jobData.result.map((item) => (
                          <div key={item.id} className="p-3.5 rounded-xl border border-white/5 bg-white/[0.01] flex flex-col gap-3">
                            <div className="h-32 rounded-lg bg-slate-950 flex items-center justify-center p-2 relative overflow-hidden"
                                 style={{
                                   backgroundImage: 'linear-gradient(45deg, #111422 25%, transparent 25%), linear-gradient(-45deg, #111422 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #111422 75%), linear-gradient(-45deg, transparent 75%, #111422 75%)',
                                   backgroundSize: '12px 12px',
                                   backgroundPosition: '0 0, 0 6px, 6px -6px, -6px 0px'
                                 }}>
                              <img src={`/${item.image_path}`} alt="Ingested layer" className="max-h-full max-w-full object-contain filter drop-shadow-md" />
                            </div>
                            <div>
                              <span className="px-1.5 py-0.5 rounded bg-violet-500/10 border border-violet-500/20 text-violet-300 text-[9px] font-bold uppercase tracking-wide inline-block">
                                {item.layering_role === 'outer' ? 'Outerwear' : (item.garment_type === 'bottom' ? 'Bottom' : (item.garment_type === 'shoes' ? 'Shoes' : 'Top'))}
                              </span>
                              <h5 className="font-bold text-xs text-slate-200 mt-1 capitalize truncate">{item.brand} {item.subtype}</h5>
                              <span className="text-[10px] text-slate-500 font-medium block truncate mt-0.5 capitalize">{item.material} · {item.style}</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* S4: Group photo selection canvas */}
                  {jobData.scene_type === 'group_photo' && (jobData.status === 'requires_selection' || jobData.status === 'completed') && (
                    <div className="flex flex-col gap-4 flex-grow">
                      
                      {jobData.status === 'requires_selection' && (
                        <div className="flex flex-col gap-3 flex-grow">
                          <div className="flex items-start justify-between">
                            <div>
                              <h4 className="font-bold text-white text-sm">Interactive Selection Viewport</h4>
                              <p className="text-xs text-slate-400 mt-0.5">Move your cursor to highlight a person, and click them to extract their clothes layers.</p>
                            </div>
                            <span className="text-[10px] font-mono bg-white/5 border border-white/10 px-2 py-0.5 rounded text-slate-300">
                              {jobData.detected_items?.length} people detected
                            </span>
                          </div>

                          <div className="flex-1 flex justify-center bg-black/35 rounded-xl border border-white/5 p-4 relative overflow-hidden select-none">
                            {matchingClick && (
                              <div className="absolute inset-0 bg-black/60 backdrop-blur-xs z-20 flex items-center justify-center gap-2">
                                <div className="h-6 w-6 rounded-full border-2 border-white/20 border-t-white animate-spin"></div>
                                <span className="text-xs font-semibold">Starting selected parser...</span>
                              </div>
                            )}
                            <canvas
                              ref={canvasRef}
                              onClick={handleCanvasClick}
                              onMouseMove={handleCanvasMouseMove}
                              onMouseLeave={() => setHoveredPersonId(null)}
                              className="max-h-[350px] max-w-full object-contain rounded-lg border border-white/10 cursor-crosshair"
                            />
                          </div>

                          {pendingPersonSelection && (
                            <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/10 p-3 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                              <div>
                                <h5 className="text-xs font-bold text-emerald-300">
                                  Person {pendingPersonSelection.personId !== null ? pendingPersonSelection.personId + 1 : ''} selected
                                </h5>
                                <p className="text-[11px] text-slate-400 mt-0.5">
                                  Choose how to parse this person before continuing the wardrobe pipeline.
                                </p>
                              </div>
                              <div className="flex items-center gap-2">
                                <button
                                  onClick={() => startSelectedPersonParsing('gemini')}
                                  disabled={matchingClick}
                                  className="px-3 py-2 rounded-lg bg-emerald-500 text-slate-950 text-[10px] font-extrabold uppercase tracking-widest flex items-center gap-1.5 hover:bg-emerald-400 disabled:opacity-50 disabled:cursor-not-allowed transition"
                                >
                                  <Sparkles className="h-3.5 w-3.5" />
                                  Gemini
                                </button>
                                <button
                                  onClick={() => startSelectedPersonParsing('segformer')}
                                  disabled={matchingClick}
                                  className="px-3 py-2 rounded-lg bg-violet-500 text-white text-[10px] font-extrabold uppercase tracking-widest flex items-center gap-1.5 hover:bg-violet-400 disabled:opacity-50 disabled:cursor-not-allowed transition"
                                >
                                  <Cpu className="h-3.5 w-3.5" />
                                  SegFormer
                                </button>
                              </div>
                            </div>
                          )}
                        </div>
                      )}

                      {jobData.status === 'completed' && (() => {
                        const filteredParts = jobData.parsed_parts 
                          ? jobData.parsed_parts.filter(part => part.label !== 'left_arm' && part.label !== 'right_arm')
                          : [];
                        return (
                          <div className="flex flex-col gap-5 flex-grow">
                            <div className="p-3 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 rounded-xl text-xs flex items-center gap-2">
                              <Check className="h-4.5 w-4.5" />
                              <span>Successfully extracted person cutout and parsed layered clothing segments in premium visual themes!</span>
                            </div>

                            <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-stretch flex-grow">
                              
                              {/* LEFT: Main Human Cutout Entity Card (5 cols) */}
                              <div className="md:col-span-5 flex flex-col gap-3">
                                <div className="flex items-center justify-between">
                                  <span className="text-[10px] font-bold tracking-wider text-slate-400 block uppercase">Selected Human Cutout</span>
                                  <span className="text-[10px] text-violet-400 font-mono font-bold tracking-widest uppercase">Floating Card Entity</span>
                                </div>
                                <div className="h-80 rounded-2xl border-2 border-violet-500/30 bg-slate-950 flex items-center justify-center p-6 relative overflow-hidden pulse-border-glow"
                                     style={{
                                       background: 'radial-gradient(circle at 50% 50%, #202440 0%, #090a12 100%)'
                                     }}>
                                  
                                  {/* Toggle Overlay Map Button */}
                                  <div className="absolute top-3 right-3 z-10">
                                    <button
                                      onClick={() => setShowMaskOverlay(!showMaskOverlay)}
                                      className={`px-2 py-0.5 rounded border text-[8px] font-extrabold uppercase tracking-widest transition-all flex items-center gap-1 backdrop-blur-md cursor-pointer ${
                                        showMaskOverlay 
                                          ? 'bg-violet-500/20 border-violet-400 text-violet-300 shadow-[0_0_10px_rgba(139,92,246,0.2)]' 
                                          : 'bg-black/50 border-white/10 text-slate-400 hover:border-white/20'
                                      }`}
                                    >
                                      {showMaskOverlay ? <Eye className="h-2.5 w-2.5" /> : <EyeOff className="h-2.5 w-2.5" />}
                                      {showMaskOverlay ? 'Map On' : 'Map Off'}
                                    </button>
                                  </div>

                                  {jobData.cutout_url ? (
                                    <HumanParsingOverlay 
                                      cutoutUrl={jobData.cutout_url}
                                      parts={filteredParts}
                                      hoveredPartLabel={hoveredPartLabel}
                                      showMaskOverlay={showMaskOverlay}
                                    />
                                  ) : (
                                    <div className="text-slate-500 text-xs flex flex-col items-center gap-2">
                                      <span className="h-12 w-12 rounded-full bg-white/5 flex items-center justify-center border border-white/10">
                                        <Shirt className="h-6 w-6 text-slate-400" />
                                      </span>
                                      <span>Cutout unavailable</span>
                                    </div>
                                  )}

                                  {/* Color Legend for Parsed Map */}
                                  {showMaskOverlay && filteredParts.length > 0 && (
                                    <div className="absolute bottom-3 left-3 flex flex-wrap gap-x-2 gap-y-1 max-w-[90%] bg-black/85 backdrop-blur-md px-2 py-1.5 border border-white/10 rounded-xl">
                                      <div className="flex items-center gap-1 shrink-0">
                                        <span className="h-1.5 w-1.5 rounded-full bg-[#ef4444]" />
                                        <span className="text-[7.5px] text-slate-400 font-mono font-bold uppercase">Top</span>
                                      </div>
                                      <div className="flex items-center gap-1 shrink-0">
                                        <span className="h-1.5 w-1.5 rounded-full bg-[#ec4899]" />
                                        <span className="text-[7.5px] text-slate-400 font-mono font-bold uppercase">Outer</span>
                                      </div>
                                      <div className="flex items-center gap-1 shrink-0">
                                        <span className="h-1.5 w-1.5 rounded-full bg-[#06b6d4]" />
                                        <span className="text-[7.5px] text-slate-400 font-mono font-bold uppercase">Bottom</span>
                                      </div>
                                      <div className="flex items-center gap-1 shrink-0">
                                        <span className="h-1.5 w-1.5 rounded-full bg-[#f97316]" />
                                        <span className="text-[7.5px] text-slate-400 font-mono font-bold uppercase">Shoes</span>
                                      </div>
                                      <div className="flex items-center gap-1 shrink-0">
                                        <span className="h-1.5 w-1.5 rounded-full bg-[#f59e0b]" />
                                        <span className="text-[7.5px] text-slate-400 font-mono font-bold uppercase">Bag</span>
                                      </div>
                                      <div className="flex items-center gap-1 shrink-0">
                                        <span className="h-1.5 w-1.5 rounded-full bg-[#10b981]" />
                                        <span className="text-[7.5px] text-slate-400 font-mono font-bold uppercase">Hat</span>
                                      </div>
                                      <div className="flex items-center gap-1 shrink-0">
                                        <span className="h-1.5 w-1.5 rounded-full bg-[#d946ef]" />
                                        <span className="text-[7.5px] text-slate-400 font-mono font-bold uppercase">Acc</span>
                                      </div>
                                    </div>
                                  )}
                                </div>
                              </div>

                              {/* RIGHT: Segmented Parts Explorer in Distinct Colors (7 cols) */}
                              <div className="md:col-span-7 flex flex-col gap-3">
                                <div className="flex items-center justify-between">
                                  <span className="text-[10px] font-bold tracking-wider text-slate-400 block uppercase">Parsed Clothing Segments</span>
                                  <span className="text-[10px] text-slate-500 font-mono">
                                    {filteredParts.length} garment{filteredParts.length === 1 ? '' : 's'} parsed
                                  </span>
                                </div>
                                
                                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-h-[310px] overflow-y-auto pr-1">
                                  {filteredParts.map((part, index) => {
                                    const theme = getPartTheme(part.label);
                                    const isHovered = hoveredPartLabel === part.label;
                                    return (
                                      <div 
                                        key={index} 
                                        className={`glass-card p-3 flex items-center gap-3 border transition-all duration-200 cursor-default ${
                                          isHovered 
                                            ? 'border-violet-400/80 bg-violet-500/5 ring-1 ring-violet-500/30 scale-[1.02] shadow-[0_4px_20px_rgba(139,92,246,0.15)]' 
                                            : `${theme.border} ${theme.glow}`
                                        }`}
                                        onMouseEnter={() => setHoveredPartLabel(part.label)}
                                        onMouseLeave={() => setHoveredPartLabel(null)}
                                      >
                                        {/* Color-Coded Part Crop Thumbnail */}
                                        <div className={`h-16 w-16 shrink-0 rounded-lg bg-slate-950 flex items-center justify-center p-1 border-2 transition-all ${
                                          isHovered ? 'border-violet-400 shadow-[0_0_10px_rgba(139,92,246,0.2)]' : theme.border
                                        }`}
                                             style={{
                                               backgroundImage: 'linear-gradient(45deg, #111422 25%, transparent 25%), linear-gradient(-45deg, #111422 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #111422 75%), linear-gradient(-45deg, transparent 75%, #111422 75%)',
                                               backgroundSize: '8px 8px',
                                               backgroundPosition: '0 0, 0 4px, 4px -4px, -4px 0px'
                                             }}>
                                          <img src={`/${part.rgba_crop_path}`} alt="Part crop" className="max-h-full max-w-full object-contain filter drop-shadow" />
                                        </div>

                                        {/* Part details */}
                                        <div className="min-w-0 flex-1">
                                          <span className={`text-[8px] font-extrabold uppercase px-1.5 py-0.5 rounded tracking-wider inline-block transition-all ${
                                            isHovered ? 'text-violet-300 bg-violet-500/20' : theme.text
                                          }`}>
                                            {theme.badge}
                                          </span>
                                          <h5 className="font-bold text-xs text-slate-200 mt-1.5 capitalize truncate">
                                            {part.label.replace('_', ' ')}
                                          </h5>
                                          <div className="flex items-center gap-2 mt-1 text-[9px] text-slate-400">
                                            <span>{part.pixel_area.toLocaleString()} px²</span>
                                            <span>·</span>
                                            <span className="font-semibold text-slate-300">
                                              {part.ingest ? 'Ingested' : 'Display Skin'}
                                            </span>
                                          </div>
                                        </div>
                                      </div>
                                    );
                                  })}
                                </div>
                              </div>

                            </div>
                          </div>
                        );
                      })()}
                    </div>
                  )}

                  {/* Reset/Clean button */}
                  <div className="mt-auto pt-4 border-t border-white/10 flex justify-between items-center gap-3">
                    {batchJobs.length > 0 && (
                      <button
                        onClick={() => {
                          fetchBatchJobs();
                          setActiveMode('batch_queue');
                        }}
                        className="px-4 py-2 rounded-xl bg-violet-500/10 border border-violet-500/20 text-xs font-bold text-violet-300 hover:bg-violet-500/20 transition flex items-center gap-1.5 cursor-pointer"
                      >
                        <Sparkles className="h-4 w-4" />
                        Back to Batch Queue
                      </button>
                    )}
                    <button
                      onClick={() => {
                        setUploadData(null);
                        setJobData(null);
                        setActiveMode('upload');
                      }}
                      className="px-4 py-2 rounded-xl bg-white/5 border border-white/10 text-xs font-bold hover:bg-white/10 transition ml-auto cursor-pointer"
                    >
                      Clear / Reset Pipeline
                    </button>
                  </div>
                  
                </div>
              )}
            </>) : (
                <div className="flex-grow flex flex-col gap-6 pt-2">

                  {/* AI Stylist: text-to-outfit prompt + Auto-Style button */}
                  <div className="flex items-center gap-2 flex-wrap">
                    <div className="flex-1 min-w-[200px] relative">
                      <Sparkles className="absolute left-3 top-2.5 h-3.5 w-3.5 text-violet-400 pointer-events-none" />
                      <input
                        type="text"
                        value={stylePrompt}
                        onChange={(e) => setStylePrompt(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' && !stylingLoading && composerItemsCount > 0) {
                            handleAutoStyleOutfit(stylePrompt);
                          }
                        }}
                        placeholder="Describe an aesthetic: cyberpunk streetwear, old money, gorpcore..."
                        className="w-full bg-slate-950/60 border border-white/10 rounded-xl pl-9 pr-3 py-2 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-violet-500/50 transition-colors"
                      />
                    </div>
                    <button
                      onClick={() => handleAutoStyleOutfit(stylePrompt)}
                      disabled={stylingLoading || composerItemsCount === 0}
                      className="py-2 px-4 rounded-xl bg-gradient-to-r from-violet-600 to-pink-500 hover:from-violet-500 hover:to-pink-400 text-xs font-bold text-white flex items-center justify-center gap-2 transition-all shadow-[0_0_15px_rgba(139,92,246,0.25)] hover:shadow-[0_0_20px_rgba(139,92,246,0.45)] cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed disabled:shadow-none shrink-0"
                    >
                      {stylingLoading ? (
                        <>
                          <div className="h-3.5 w-3.5 rounded-full border-2 border-white/20 border-t-white animate-spin"></div>
                          Styling...
                        </>
                      ) : (
                        <>
                          <Sparkles className="h-3.5 w-3.5 animate-pulse" />
                          Generate Style
                        </>
                      )}
                    </button>
                    <button
                      onClick={() => {
                        setActiveHatIdx(-1);
                        setActiveTopIdx(-1);
                        setActiveBottomIdx(-1);
                        setActiveShoesIdx(-1);
                        setStylingBrief(null);
                        setStylePrompt('');
                      }}
                      className="py-2 px-3 rounded-xl border border-white/10 hover:border-red-500/50 text-[10px] font-extrabold uppercase tracking-widest text-slate-400 hover:text-red-300 transition cursor-pointer"
                    >
                      Reset
                    </button>
                  </div>

                  {/* Gemini Stylist Brief */}
                  {stylingBrief && (
                    <div className="p-3 bg-gradient-to-r from-violet-500/10 to-pink-500/10 border border-violet-500/20 rounded-xl text-xs flex flex-col gap-1.5 animate-fade-in">
                      <div className="flex items-center gap-1.5">
                        <Sparkles className="h-3.5 w-3.5 text-violet-400 animate-pulse" />
                        <span className="font-extrabold text-[10px] text-violet-300 uppercase tracking-wider">Gemini Stylist Brief</span>
                      </div>
                      <h4 className="font-bold text-xs text-white capitalize">{stylingBrief.name}</h4>
                      <p className="text-[10px] text-slate-400 leading-relaxed italic">{stylingBrief.explanation}</p>
                    </div>
                  )}

                  {/* Gemini Seed Toast */}
                  {seedSummary && (
                    <div className="p-3 bg-emerald-500/10 border border-emerald-500/30 rounded-xl text-xs flex items-center justify-between gap-3 animate-fade-in">
                      <div className="flex items-center gap-2">
                        <Sparkles className="h-3.5 w-3.5 text-emerald-400" />
                        <span className="text-[10px] font-extrabold uppercase tracking-wider text-emerald-300">
                          Gemini Seed Complete
                        </span>
                        <span className="text-[10px] text-slate-300">
                          {seedSummary.created} items in {seedSummary.elapsed.toFixed(1)}s
                          {seedSummary.errors > 0 && (
                            <span className="text-rose-300 ml-2">({seedSummary.errors} failed)</span>
                          )}
                        </span>
                      </div>
                      <button
                        onClick={() => setSeedSummary(null)}
                        className="text-[10px] font-bold uppercase tracking-widest text-emerald-300/70 hover:text-emerald-200 cursor-pointer"
                      >
                        Dismiss
                      </button>
                    </div>
                  )}

                  {/* Canvas (center) + Category Sliders Sidebar (right) */}
                  <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-stretch flex-grow">

                    {/* LEFT: 1024x1024 Outfit Canvas (7 cols) */}
                    <div className="md:col-span-7 flex flex-col gap-3 relative min-h-[440px]">

                      <div className="flex items-center justify-between">
                        <span className="text-[10px] font-bold tracking-wider text-slate-400 block uppercase">Outfit Composer Canvas</span>
                        <span className="text-[10px] text-violet-400 font-mono font-bold tracking-widest uppercase">Hat &middot; Top &middot; Bottom &middot; Shoes</span>
                      </div>
                      
                      {/* 1024x1024 Outfit Canvas - mannequin-less, fixed-anchor stacked layers */}
                      <div className="flex-1 flex justify-center items-start">
                        <div
                          className="aspect-square w-full max-w-[540px] relative rounded-2xl overflow-hidden border border-white/10 shadow-[0_18px_60px_rgba(0,0,0,0.4)] transition-colors duration-300"
                          style={{
                            backgroundColor: composerBg === 'white' ? '#ffffff' : '#000000',
                          }}
                        >
                          {/* Layer 1 (z10): Shoes */}
                          {layerVisibility.shoes && activeShoesIdx !== -1 && composerItems.shoes[activeShoesIdx] && (
                            <img
                              src={`/${composerItems.shoes[activeShoesIdx].aligned_url}?v=${composerAssetVersion}`}
                              alt="Shoes layer"
                              draggable={false}
                              style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'contain', zIndex: 10, pointerEvents: 'none' }}
                            />
                          )}
                          {/* Layer 2 (z20): Bottom */}
                          {layerVisibility.bottom && activeBottomIdx !== -1 && composerItems.bottoms[activeBottomIdx] && (
                            <img
                              src={`/${composerItems.bottoms[activeBottomIdx].aligned_url}?v=${composerAssetVersion}`}
                              alt="Bottom layer"
                              draggable={false}
                              style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'contain', zIndex: 20, pointerEvents: 'none' }}
                            />
                          )}
                          {/* Layer 3 (z30): Top */}
                          {layerVisibility.top && activeTopIdx !== -1 && composerItems.tops[activeTopIdx] && (
                            <img
                              src={`/${composerItems.tops[activeTopIdx].aligned_url}?v=${composerAssetVersion}`}
                              alt="Top layer"
                              draggable={false}
                              style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'contain', zIndex: 30, pointerEvents: 'none' }}
                            />
                          )}
                          {/* Layer 4 (z40): Hat */}
                          {layerVisibility.hat && activeHatIdx !== -1 && composerItems.hats[activeHatIdx] && (
                            <img
                              src={`/${composerItems.hats[activeHatIdx].aligned_url}?v=${composerAssetVersion}`}
                              alt="Hat layer"
                              draggable={false}
                              style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'contain', zIndex: 40, pointerEvents: 'none' }}
                            />
                          )}

                          {composerLoading && (
                            <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-white/30 backdrop-blur-sm z-50">
                              <div className="h-6 w-6 rounded-full border-2 border-violet-500/20 border-t-violet-500 animate-spin"></div>
                              <span className="text-[10px] text-violet-500 font-mono tracking-wider font-extrabold">Aligning 1024x1024 canvas...</span>
                            </div>
                          )}

                          {!composerLoading && activeHatIdx === -1 && activeTopIdx === -1 && activeBottomIdx === -1 && activeShoesIdx === -1 && (
                            <div className="absolute inset-0 flex flex-col items-center justify-center text-center px-6 pointer-events-none">
                              <Shirt className={`h-12 w-12 mb-3 ${composerBg === 'white' ? 'text-slate-300' : 'text-slate-700'}`} />
                              <span className={`text-[11px] font-extrabold uppercase tracking-widest ${composerBg === 'white' ? 'text-slate-400' : 'text-slate-500'}`}>
                                Pick a hat, top, bottom &amp; shoes
                              </span>
                              <span className={`text-[10px] mt-1 ${composerBg === 'white' ? 'text-slate-400' : 'text-slate-600'}`}>
                                Use the sliders on the right or hit "Generate Style"
                              </span>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>

                    {/* RIGHT: 4 Category Sliders (5 cols) */}
                    <div className="md:col-span-5 flex flex-col gap-4">

                      <div className="flex items-center justify-between">
                        <span className="text-[10px] font-bold tracking-wider text-slate-400 block uppercase">Layer Architecture</span>
                        <span className="text-[10px] text-slate-500 font-mono">
                          {composerItemsCount} composable item{composerItemsCount === 1 ? '' : 's'}
                        </span>
                      </div>

                      <div className="flex-1 flex flex-col gap-2.5 overflow-y-auto pr-1">
                        {([
                          { category: 'hat', title: 'Hats / Headwear', list: composerItems.hats, activeIdx: activeHatIdx, setIdx: setActiveHatIdx, accent: 'violet' },
                          { category: 'top', title: 'Tops / Inners', list: composerItems.tops, activeIdx: activeTopIdx, setIdx: setActiveTopIdx, accent: 'pink' },
                          { category: 'bottom', title: 'Lowers / Pants', list: composerItems.bottoms, activeIdx: activeBottomIdx, setIdx: setActiveBottomIdx, accent: 'cyan' },
                          { category: 'shoes', title: 'Footwear', list: composerItems.shoes, activeIdx: activeShoesIdx, setIdx: setActiveShoesIdx, accent: 'emerald' },
                        ] as const).map((layer) => {
                          const activeItem = layer.activeIdx !== -1 ? layer.list[layer.activeIdx] : null;
                          const isVis = layerVisibility[layer.category];
                          const total = layer.list.length;
                          const accentBg: Record<string, string> = {
                            violet: 'hover:border-violet-500/40',
                            pink: 'hover:border-pink-500/40',
                            cyan: 'hover:border-cyan-500/40',
                            emerald: 'hover:border-emerald-500/40',
                          };
                          const accentText: Record<string, string> = {
                            violet: 'text-violet-400',
                            pink: 'text-pink-400',
                            cyan: 'text-cyan-400',
                            emerald: 'text-emerald-400',
                          };

                          const advance = (delta: number) => {
                            if (total === 0) return;
                            const next = layer.activeIdx === -1
                              ? (delta > 0 ? 0 : total - 1)
                              : (layer.activeIdx + delta + total) % total;
                            layer.setIdx(next);
                          };

                          return (
                            <div
                              key={layer.category}
                              className={`p-3 rounded-xl border bg-slate-950/40 flex flex-col gap-2 transition-all duration-200 border-white/5 ${accentBg[layer.accent]}`}
                            >
                              <div className="flex items-center justify-between gap-2">
                                <div className="flex items-center gap-2 min-w-0">
                                  <div className="h-11 w-11 shrink-0 rounded-lg bg-slate-900 border border-white/5 flex items-center justify-center p-1 overflow-hidden"
                                       style={{
                                         backgroundImage: 'linear-gradient(45deg, #111422 25%, transparent 25%), linear-gradient(-45deg, #111422 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #111422 75%), linear-gradient(-45deg, transparent 75%, #111422 75%)',
                                         backgroundSize: '6px 6px',
                                         backgroundPosition: '0 0, 0 3px, 3px -3px, -3px 0px'
                                       }}>
                                    {activeItem ? (
                                      <img src={`/${activeItem.aligned_url}?v=${composerAssetVersion}`} alt="layer thumb" className="max-h-full max-w-full object-contain filter drop-shadow-sm" />
                                    ) : (
                                      <span className={`h-2 w-2 rounded-full ${activeItem ? '' : 'bg-slate-700'}`} />
                                    )}
                                  </div>

                                  <div className="min-w-0">
                                    <h5 className={`font-extrabold text-[10px] uppercase tracking-wider ${accentText[layer.accent]}`}>{layer.title}</h5>
                                    <p className="text-[10px] font-bold text-slate-200 truncate mt-0.5 capitalize">
                                      {activeItem ? (activeItem.name || `${activeItem.brand || 'Classic'} ${activeItem.subtype || ''}`) : (total === 0 ? 'No items yet' : 'Empty slot')}
                                    </p>
                                  </div>
                                </div>

                                <div className="flex items-center gap-1.5 shrink-0">
                                  <button
                                    onClick={() => setLayerVisibility(prev => ({ ...prev, [layer.category]: !prev[layer.category] }))}
                                    className="p-1 rounded hover:bg-white/5 text-slate-400 hover:text-slate-200 transition cursor-pointer"
                                    title="Toggle layer visibility"
                                  >
                                    {isVis ? <Eye className="h-3.5 w-3.5" /> : <EyeOff className="h-3.5 w-3.5 text-red-500" />}
                                  </button>
                                  {activeItem && (
                                    <button
                                      onClick={() => layer.setIdx(-1)}
                                      className="p-1 rounded hover:bg-white/5 text-slate-500 hover:text-red-400 transition cursor-pointer"
                                      title="Remove layer"
                                    >
                                      <Trash2 className="h-3.5 w-3.5" />
                                    </button>
                                  )}
                                </div>
                              </div>

                              <div className="flex items-center justify-between gap-2 pt-1.5 border-t border-white/5">
                                <button
                                  onClick={() => advance(-1)}
                                  disabled={total === 0}
                                  className="h-7 w-7 rounded-full bg-slate-900 hover:bg-white/10 text-slate-300 flex items-center justify-center transition border border-white/5 cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed"
                                >
                                  <ChevronLeft className="h-4 w-4" />
                                </button>
                                <span className="text-[10px] font-mono text-slate-400 tabular-nums">
                                  {total === 0 ? '0 / 0' : `${(layer.activeIdx === -1 ? 0 : layer.activeIdx + 1)} / ${total}`}
                                </span>
                                <button
                                  onClick={() => advance(1)}
                                  disabled={total === 0}
                                  className="h-7 w-7 rounded-full bg-slate-900 hover:bg-white/10 text-slate-300 flex items-center justify-center transition border border-white/5 cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed"
                                >
                                  <ChevronRight className="h-4 w-4" />
                                </button>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>

                  </div>

                  {/* BOTTOM: Style Intelligence Layer (OKLCH color harmony & recommendations) */}
                  {(() => {
                    const topItem = activeTopIdx !== -1 ? composerItems.tops[activeTopIdx] : null;
                    const bottomItem = activeBottomIdx !== -1 ? composerItems.bottoms[activeBottomIdx] : null;
                    const shoesItem = activeShoesIdx !== -1 ? composerItems.shoes[activeShoesIdx] : null;
                    const hatItem = activeHatIdx !== -1 ? composerItems.hats[activeHatIdx] : null;
                    
                    const harmony = calculateOklchHarmony(topItem, bottomItem, shoesItem, hatItem);
                    const activeCount = [topItem, bottomItem, shoesItem, hatItem].filter(Boolean).length;
                    
                    const allTags = [
                      ...(topItem?.tags || []),
                      ...(bottomItem?.tags || []),
                      ...(shoesItem?.tags || []),
                      ...(hatItem?.tags || [])
                    ];
                    
                    let archetype = 'Casual Minimalist';
                    let archetypeDesc = 'A well-balanced, clean aesthetic grounded in neutral fits.';
                    
                    if (allTags.includes('denim') || allTags.includes('baggy') || allTags.includes('oversized') || allTags.includes('streetwear')) {
                      archetype = 'Modern Streetwear';
                      archetypeDesc = 'Relaxed proportions and industrial layering elements designed for comfort and presence.';
                    } else if (allTags.includes('formal') || allTags.includes('classic') || allTags.includes('tailored') || allTags.includes('suit')) {
                      archetype = 'Elevated Classic / Smart-Casual';
                      archetypeDesc = 'Sleek structural tailoring meets highly sophisticated everyday color block coordination.';
                    } else if (allTags.includes('vintage') || allTags.includes('knitwear') || allTags.includes('wool') || allTags.includes('sweater')) {
                      archetype = 'Heritage Cozy Textured';
                      archetypeDesc = 'Rich, weighted material profiles (wool, knits, corduroy) structured beautifully.';
                    }

                    return (
                      <div className="pt-4 border-t border-white/10 grid grid-cols-1 md:grid-cols-12 gap-5 items-stretch">
                        {/* Gauge / Score circle */}
                        <div className="md:col-span-4 flex items-center gap-4 bg-slate-950/40 p-3.5 rounded-2xl border border-white/5">
                          <div className="relative h-14 w-14 flex items-center justify-center shrink-0">
                            <svg className="absolute inset-0 w-full h-full transform -rotate-90">
                              <circle 
                                cx="28" 
                                cy="28" 
                                r="24" 
                                className="stroke-slate-800 fill-none" 
                                strokeWidth="4" 
                              />
                              <circle 
                                cx="28" 
                                cy="28" 
                                r="24" 
                                className="fill-none transition-all duration-500" 
                                strokeWidth="4" 
                                strokeDasharray={150.7}
                                strokeDashoffset={150.7 - (150.7 * harmony.score) / 100}
                                strokeLinecap="round"
                                style={{ 
                                  stroke: harmony.score >= 90 ? '#10b981' : (harmony.score >= 75 ? '#8b5cf6' : '#ec4899') 
                                }}
                              />
                            </svg>
                            <span className="font-mono font-bold text-sm text-white absolute">{harmony.score}%</span>
                          </div>
                          <div>
                            <span className="text-[8px] font-extrabold uppercase tracking-widest text-slate-500">Color Science score</span>
                            <h4 className="font-bold text-xs text-slate-200 mt-0.5 truncate max-w-[170px]">{harmony.label}</h4>
                          </div>
                        </div>

                        {/* Recommendation description */}
                        <div className="md:col-span-8 bg-slate-950/40 p-3.5 rounded-2xl border border-white/5 h-full flex flex-col justify-center">
                          <div className="flex items-center gap-2">
                            <span className="px-2 py-0.5 rounded bg-violet-500/10 border border-violet-500/20 text-violet-300 text-[8px] font-extrabold uppercase tracking-wider">
                              Archetype: {archetype}
                            </span>
                            <span className="text-[10px] text-slate-500 font-medium">
                              {activeCount} Layer{activeCount === 1 ? '' : 's'} Equipped
                            </span>
                          </div>
                          <p className="text-[10px] text-slate-400 mt-1.5 leading-relaxed">
                            {harmony.text} {activeCount >= 2 && archetypeDesc}
                          </p>
                        </div>
                      </div>
                    );
                  })()}

                </div>
              )}

              {/* Error banner */}
              {error && (
                <div className="mt-4 p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 flex items-start gap-3">
                  <AlertCircle className="h-5 w-5 shrink-0 mt-0.5" />
                  <div>
                    <h4 className="font-bold text-sm">Pipeline Failure</h4>
                    <p className="text-xs mt-0.5 text-red-300">{error}</p>
                    <button onClick={() => setError(null)} className="text-[10px] text-red-300 underline font-bold mt-1.5 block">Dismiss</button>
                  </div>
                </div>
              )}

            </div>
          </div>
          
          {/* RIGHT 5 cols: Wardrobe Explorer Panel */}
          <div className="lg:col-span-5 flex flex-col gap-6">
            <div className="glass-panel p-6 flex flex-col gap-5 flex-1 min-h-[480px]">
              
              <div className="flex items-center justify-between border-b border-white/10 pb-4">
                <div className="flex items-center gap-2">
                  <Compass className="h-5 w-5 text-emerald-400" />
                  <h2 className="font-bold text-base text-white">Closet Explorer</h2>
                </div>
                <span className="font-mono text-xs bg-emerald-500/15 border border-emerald-500/30 px-2 py-0.5 rounded text-emerald-300">
                  {closetItems.length} items
                </span>
              </div>

              {/* Closet Category Filters */}
              <div className="flex flex-wrap gap-1.5">
                {['all', 'hat', 'outerwear', 'top', 'bottom', 'shoes'].map(cat => (
                  <button
                    key={cat}
                    onClick={() => handleCategoryChange(cat)}
                    className={`px-3 py-1 rounded-lg text-xs font-semibold border capitalize transition-all ${
                      activeCategory === cat 
                        ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400' 
                        : 'bg-white/5 border-white/5 hover:border-white/10 text-slate-300'
                    }`}
                  >
                    {cat}
                  </button>
                ))}
              </div>

              {/* Text Search bar */}
              <div className="relative">
                <input
                  type="text"
                  placeholder="Search tags, brand, subtype..."
                  value={searchText}
                  onChange={(e) => handleSearchChange(e.target.value)}
                  className="w-full bg-slate-950/60 border border-white/10 rounded-xl px-3 py-2 pl-9 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-emerald-500/50 transition-colors"
                />
                <Search className="absolute left-3 top-2.5 h-3.5 w-3.5 text-slate-500" />
              </div>

              {/* Items Grid View */}
              {closetItems.length === 0 ? (
                <div className="flex-1 flex flex-col items-center justify-center text-center p-8 text-slate-500">
                  <Shirt className="h-10 w-10 mb-3 opacity-25" />
                  <h4 className="font-bold text-sm">Closet is Empty</h4>
                  <p className="text-[10px] mt-1 max-w-[200px]">Upload outfit images above to start extracting wardrobe garments!</p>
                </div>
              ) : (
                <div className="grid grid-cols-3 gap-3 overflow-y-auto max-h-[300px] pr-1">
                  {closetItems.map(item => (
                    <div
                      key={item.id}
                      onClick={() => handleSelectClosetItem(item)}
                      className={`glass-card p-2 flex flex-col gap-2 cursor-pointer border relative group ${
                        selectedClosetItem?.id === item.id 
                          ? 'border-emerald-500 bg-emerald-500/5' 
                          : 'border-white/5'
                      }`}
                    >
                      <div className="h-20 rounded-lg bg-slate-950 flex items-center justify-center p-1.5"
                           style={{
                             backgroundImage: 'linear-gradient(45deg, #111422 25%, transparent 25%), linear-gradient(-45deg, #111422 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #111422 75%), linear-gradient(-45deg, transparent 75%, #111422 75%)',
                             backgroundSize: '8px 8px',
                             backgroundPosition: '0 0, 0 4px, 4px -4px, -4px 0px'
                           }}>
                        <img src={`/${item.image_path}`} alt="Closet Garment" className="max-h-full max-w-full object-contain filter drop-shadow-md group-hover:scale-105 transition-transform" />
                      </div>
                      <span className="text-[9px] font-bold text-slate-300 truncate capitalize text-center">
                        {item.brand} {item.subtype}
                      </span>
                    </div>
                  ))}
                </div>
              )}

            </div>
          </div>

        </section>

        {/* BOTTOM: Closet Item Details & OKLCH Color Harmony recommendations */}
        {selectedClosetItem && (
          <section className="glass-panel p-6 grid grid-cols-1 lg:grid-cols-12 gap-8 relative overflow-hidden animate-fade-in">
            
            {/* 1. Item Visuals & Primary Attributes (Left: 4 cols) */}
            <div className="lg:col-span-4 flex flex-col gap-4 border-r border-white/10 pr-0 lg:pr-8">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-bold tracking-wider text-slate-400 block uppercase">Garment Profile</span>
                <button
                  onClick={() => handleDeleteItem(selectedClosetItem.id)}
                  className="p-1.5 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 hover:bg-red-500/25 transition-colors cursor-pointer"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </div>

              <div className="h-64 rounded-2xl bg-slate-950 flex items-center justify-center p-6 border border-white/5"
                   style={{
                     backgroundImage: 'linear-gradient(45deg, #111422 25%, transparent 25%), linear-gradient(-45deg, #111422 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #111422 75%), linear-gradient(-45deg, transparent 75%, #111422 75%)',
                     backgroundSize: '16px 16px',
                     backgroundPosition: '0 0, 0 8px, 8px -8px, -8px 0px'
                   }}>
                <img src={`/${selectedClosetItem.image_path}`} alt="Selected item crop" className="max-h-full max-w-full object-contain filter drop-shadow-2xl" />
              </div>

              <div className="flex flex-col gap-1.5">
                <h3 className="text-xl font-extrabold text-white leading-tight capitalize">
                  {selectedClosetItem.brand} {selectedClosetItem.subtype}
                </h3>
                <span className="text-xs text-slate-400 font-medium capitalize">
                  {selectedClosetItem.material} · {selectedClosetItem.style} · {selectedClosetItem.occasion} Occasions
                </span>
              </div>

              <div className="flex flex-wrap gap-1.5 pt-1">
                {selectedClosetItem.tags.map(t => (
                  <span key={t} className="px-2.5 py-0.5 rounded bg-white/5 border border-white/10 text-slate-300 text-[10px] font-semibold">
                    #{t}
                  </span>
                ))}
              </div>
            </div>

            {/* 2. Color Science & Pairings (Middle: 4 cols) */}
            <div className="lg:col-span-4 flex flex-col gap-5 border-r border-white/10 pr-0 lg:pr-8">
              <h4 className="font-bold text-sm flex items-center gap-2 text-white border-b border-white/10 pb-2">
                <Palette className="h-4 w-4 text-violet-400" />
                Dominant OKLCH Colors
              </h4>
              
              <div className="flex flex-col gap-3">
                {selectedClosetItem.colors.map((c, i) => (
                  <div key={i} className="flex items-center justify-between bg-white/5 border border-white/5 rounded-xl p-2.5">
                    <div className="flex items-center gap-2">
                      <span className="h-6 w-6 rounded-full border border-white/20 shadow" style={{ backgroundColor: `rgb(${c.rgb.join(',')})` }} />
                      <div className="text-xs">
                        <span className="font-semibold text-slate-200">RGB: {c.rgb.join(',')}</span>
                        <span className="text-[10px] text-slate-400 block font-mono">
                          L: {c.oklch[0].toFixed(2)} · C: {c.oklch[1].toFixed(2)} · H: {c.oklch[2].toFixed(0)}°
                        </span>
                      </div>
                    </div>
                    <span className="text-xs font-mono font-bold text-violet-300">{Math.round(c.weight * 100)}%</span>
                  </div>
                ))}
              </div>

              <h4 className="font-bold text-sm flex items-center gap-2 text-white border-b border-white/10 pb-2 pt-2">
                <Heart className="h-4 w-4 text-pink-400" />
                Layering & Pairings
              </h4>
              <ul className="flex flex-col gap-2 max-h-[140px] overflow-y-auto pr-1">
                {selectedClosetItem.pairing_suggestions.map((p, idx) => (
                  <li key={idx} className="text-xs text-slate-300 flex items-start gap-2 leading-normal">
                    <ChevronRight className="h-3.5 w-3.5 text-pink-400 shrink-0 mt-0.5" />
                    <span>{p}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* 3. Color Harmony & Similarity Recommendations (Right: 4 cols) */}
            <div className="lg:col-span-4 flex flex-col gap-5">
              <h4 className="font-bold text-sm flex items-center justify-between text-white border-b border-white/10 pb-2">
                <span className="flex items-center gap-2">
                  <Sparkles className="h-4 w-4 text-emerald-400" />
                  Color Harmony Recommendations
                </span>
                <span className="text-[10px] text-slate-500 uppercase font-bold tracking-wide">OKLCH Distance</span>
              </h4>

              {harmonyMatches.length === 0 ? (
                <div className="py-6 text-center text-slate-500 text-xs">
                  No other items in closet to harmonise colors with.
                </div>
              ) : (
                <div className="flex flex-col gap-2 max-h-[180px] overflow-y-auto pr-1">
                  {harmonyMatches.slice(0, 4).map(match => (
                    <div 
                      key={match.item.id} 
                      onClick={() => handleSelectClosetItem(match.item)}
                      className="p-2 rounded-xl border border-white/5 hover:border-emerald-500/30 bg-white/[0.01] hover:bg-emerald-500/[0.02] flex items-center gap-3 transition-colors cursor-pointer"
                    >
                      <div className="h-10 w-10 rounded-md bg-slate-950 flex items-center justify-center p-1 relative overflow-hidden"
                           style={{
                             backgroundImage: 'linear-gradient(45deg, #111422 25%, transparent 25%), linear-gradient(-45deg, #111422 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #111422 75%), linear-gradient(-45deg, transparent 75%, #111422 75%)',
                             backgroundSize: '6px 6px',
                             backgroundPosition: '0 0, 0 3px, 3px -3px, -3px 0px'
                           }}>
                        <img src={`/${match.item.image_path}`} alt="Match crop" className="max-h-full object-contain filter drop-shadow-md" />
                      </div>
                      <div className="min-w-0 flex-1">
                        <h5 className="font-bold text-xs text-slate-200 capitalize truncate">{match.item.brand} {match.item.subtype}</h5>
                        <span className={`text-[10px] block truncate capitalize mt-0.5 ${
                          match.match_status === 'highly_harmonious' ? 'text-emerald-400 font-semibold' : 'text-slate-400 font-medium'
                        }`}>
                          {match.match_status === 'highly_harmonious' ? 'Highly Harmonious Complementary' : 'Compatible Color Match'}
                        </span>
                      </div>
                      <span className="font-mono text-xs text-slate-300 font-bold bg-black/45 px-2 py-0.5 rounded">
                        {Math.round(match.harmony_score * 100)}%
                      </span>
                    </div>
                  ))}
                </div>
              )}

              {/* Visual Similarity (SigLIP Vector space) */}
              <h4 className="font-bold text-sm flex items-center justify-between text-white border-b border-white/10 pb-2 pt-2">
                <span className="flex items-center gap-2">
                  <Grid className="h-4 w-4 text-cyan-400" />
                  Visually Similar Pieces
                </span>
                <span className="text-[10px] text-slate-500 uppercase font-bold tracking-wide">SigLIP Embedding</span>
              </h4>

              {similarMatches.length === 0 ? (
                <div className="py-6 text-center text-slate-500 text-xs">
                  No similar items found.
                </div>
              ) : (
                <div className="flex flex-col gap-2 max-h-[180px] overflow-y-auto pr-1">
                  {similarMatches.slice(0, 4).map(match => (
                    <div 
                      key={match.item.id} 
                      onClick={() => handleSelectClosetItem(match.item)}
                      className="p-2 rounded-xl border border-white/5 hover:border-cyan-500/30 bg-white/[0.01] hover:bg-cyan-500/[0.02] flex items-center gap-3 transition-colors cursor-pointer"
                    >
                      <div className="h-10 w-10 rounded-md bg-slate-950 flex items-center justify-center p-1 relative overflow-hidden"
                           style={{
                             backgroundImage: 'linear-gradient(45deg, #111422 25%, transparent 25%), linear-gradient(-45deg, #111422 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #111422 75%), linear-gradient(-45deg, transparent 75%, #111422 75%)',
                             backgroundSize: '6px 6px',
                             backgroundPosition: '0 0, 0 3px, 3px -3px, -3px 0px'
                           }}>
                        <img src={`/${match.item.image_path}`} alt="Match crop" className="max-h-full object-contain filter drop-shadow-md" />
                      </div>
                      <div className="min-w-0 flex-1">
                        <h5 className="font-bold text-xs text-slate-200 capitalize truncate">{match.item.brand} {match.item.subtype}</h5>
                        <span className="text-[10px] text-slate-400 block truncate capitalize mt-0.5">{match.item.style} · {match.item.material}</span>
                      </div>
                      <span className="font-mono text-xs text-slate-300 font-bold bg-black/45 px-2 py-0.5 rounded">
                        {Math.round(match.similarity_score * 100)}%
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>

          </section>
        )}

      </main>
    </div>
  );
}
