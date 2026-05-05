export function UploadZone({
  onFiles,
  processingState,
}: {
  onFiles: (files: File[]) => void
  processingState: 'idle' | 'uploading' | 'processing' | 'refining' | 'done' | 'failed'
}) {
  const statusText =
    processingState === 'uploading'
      ? 'Uploading...'
      : processingState === 'processing'
        ? 'Processing mask...'
        : processingState === 'refining'
          ? 'Refining cutout...'
          : processingState === 'failed'
            ? 'Processing failed, fallback applied'
            : 'Drop JPG/PNG/WEBP or click to upload'
  return (
    <label className="upload-zone">
      <input
        type="file"
        hidden
        accept="image/png,image/jpeg,image/webp"
        multiple
        onChange={(e) => onFiles(Array.from(e.target.files ?? []))}
      />
      {statusText}
    </label>
  )
}
