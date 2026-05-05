import type { Category } from '../../data/garments'

const CATEGORIES: { id: Category; label: string }[] = [
  { id: 'all', label: 'All' },
  { id: 'head', label: 'Head' },
  { id: 'tops', label: 'Tops' },
  { id: 'bottoms', label: 'Bottoms' },
  { id: 'shoes', label: 'Shoes' },
  { id: 'accessories', label: 'Accessories' },
]

export function CategoryRail({
  activeCategory,
  onChange,
}: {
  activeCategory: Category
  onChange: (category: Category) => void
}) {
  return (
    <div className="category-rail">
      {CATEGORIES.map((c) => (
        <button
          key={c.id}
          type="button"
          className={`category-pill ${activeCategory === c.id ? 'is-active' : ''}`}
          onClick={() => onChange(c.id)}
        >
          {c.label}
        </button>
      ))}
    </div>
  )
}
