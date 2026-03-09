import { useState, useRef, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { uploadVideo } from '../api/client'

export function Record() {
  const navigate = useNavigate()
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [notes, setNotes] = useState('')
  const [shootingSide, setShootingSide] = useState<'left' | 'right'>('left')
  const [uploading, setUploading] = useState(false)
  const cameraInputRef = useRef<HTMLInputElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setSelectedFile(file)
    setPreviewUrl(URL.createObjectURL(file))
  }, [])

  const handleUpload = useCallback(async () => {
    if (!selectedFile) return
    setUploading(true)
    const meta = await uploadVideo(selectedFile, notes, shootingSide)
    setUploading(false)
    if (meta) {
      navigate(`/analysis/${meta.filename}`)
    }
  }, [selectedFile, notes, shootingSide, navigate])

  const handleClear = useCallback(() => {
    setSelectedFile(null)
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setPreviewUrl(null)
    setNotes('')
    if (fileInputRef.current) fileInputRef.current.value = ''
    if (cameraInputRef.current) cameraInputRef.current.value = ''
  }, [previewUrl])

  return (
    <div className="space-y-4 animate-slide-up">
      <h1 className="text-2xl font-bold text-white">Upload Shot Video</h1>

      {/* Slow-mo tip */}
      <div className="bg-cyan-950 border border-cyan-800 rounded-lg p-3 text-sm">
        <div className="font-bold text-cyan-300 mb-1">Best results with Slow-Mo</div>
        <div className="text-cyan-200/80">
          Record in <span className="font-semibold text-white">slow motion</span> using
          your Camera app (swipe to "Slo-Mo"), then upload here.
          This gives 240fps for much better shot analysis.
        </div>
      </div>

      {/* Shooting side toggle */}
      <div className="flex gap-2">
        <button
          onClick={() => setShootingSide('left')}
          className={`flex-1 py-2 rounded-lg font-bold min-h-[48px] ${
            shootingSide === 'left' ? 'bg-cyan-600 text-white' : 'bg-gray-800 text-gray-400'
          }`}
        >
          Left Shot
        </button>
        <button
          onClick={() => setShootingSide('right')}
          className={`flex-1 py-2 rounded-lg font-bold min-h-[48px] ${
            shootingSide === 'right' ? 'bg-cyan-600 text-white' : 'bg-gray-800 text-gray-400'
          }`}
        >
          Right Shot
        </button>
      </div>

      {!selectedFile && (
        <>
          {/* Primary: choose from files (for slow-mo uploads) */}
          <button
            onClick={() => fileInputRef.current?.click()}
            className="w-full bg-green-600 hover:bg-green-500 text-white font-bold py-6 rounded-xl text-xl min-h-[48px] shadow-lg shadow-green-500/20"
          >
            Upload Slow-Mo Video
          </button>
          <input
            type="file"
            accept="video/*"
            onChange={handleFileSelect}
            className="hidden"
            ref={fileInputRef}
          />

          {/* Secondary: record with standard camera */}
          <div className="text-center text-gray-500 text-sm">or record at standard speed</div>

          <button
            onClick={() => cameraInputRef.current?.click()}
            className="w-full bg-gray-800 hover:bg-gray-700 text-gray-300 font-bold py-4 rounded-xl text-center min-h-[48px] border border-gray-700 border-dashed"
          >
            Record Video
          </button>
          <input
            type="file"
            accept="video/*"
            capture="environment"
            onChange={handleFileSelect}
            className="hidden"
            ref={cameraInputRef}
          />
        </>
      )}

      {selectedFile && previewUrl && (
        <>
          <video
            src={previewUrl}
            controls
            className="w-full rounded-lg bg-black aspect-video"
          />

          <div className="bg-gray-900 rounded-lg p-3 text-sm text-gray-400">
            <div>{selectedFile.name}</div>
            <div>{(selectedFile.size / (1024 * 1024)).toFixed(1)} MB</div>
          </div>

          <textarea
            placeholder="Notes (optional) - e.g. 'working on weight transfer'"
            value={notes}
            onChange={e => setNotes(e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white min-h-[60px]"
          />

          <div className="flex gap-3">
            <button
              onClick={handleClear}
              className="flex-1 bg-gray-700 text-white py-3 rounded-lg min-h-[48px] font-bold"
            >
              Clear
            </button>
            <button
              onClick={handleUpload}
              disabled={uploading}
              className="flex-1 bg-green-600 hover:bg-green-500 disabled:bg-green-800 text-white py-3 rounded-lg min-h-[48px] font-bold"
            >
              {uploading ? 'Uploading...' : 'Save & Analyze'}
            </button>
          </div>
        </>
      )}
    </div>
  )
}
