import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import type { VideoMeta } from '../types'
import { getVideos, deleteVideo, getVideoUrl } from '../api/client'

export function Videos() {
  const navigate = useNavigate()
  const [videos, setVideos] = useState<VideoMeta[]>([])
  const [loading, setLoading] = useState(true)

  const loadVideos = () => {
    setLoading(true)
    getVideos().then(v => { setVideos(v); setLoading(false) })
  }

  useEffect(() => { loadVideos() }, [])

  const handleDelete = async (filename: string) => {
    if (!confirm('Delete this video and its analysis?')) return
    await deleteVideo(filename)
    setVideos(prev => prev.filter(v => v.filename !== filename))
  }

  const statusBadge = (v: VideoMeta) => {
    switch (v.analysisStatus) {
      case 'complete':
        return <span className="text-xs bg-green-900 text-green-300 px-2 py-0.5 rounded-full">Analyzed</span>
      case 'running':
        return <span className="text-xs bg-yellow-900 text-yellow-300 px-2 py-0.5 rounded-full animate-pulse">Analyzing...</span>
      case 'error':
        return <span className="text-xs bg-red-900 text-red-300 px-2 py-0.5 rounded-full">Error</span>
      default:
        return <span className="text-xs bg-gray-700 text-gray-300 px-2 py-0.5 rounded-full">Pending</span>
    }
  }

  if (loading) return <div className="text-center py-8 text-gray-400">Loading...</div>

  return (
    <div className="space-y-4 animate-slide-up">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-white">Shot Videos</h1>
        <button
          onClick={() => navigate('/record')}
          className="bg-red-600 hover:bg-red-500 text-white px-4 py-2 rounded-lg font-bold min-h-[44px]"
        >
          Record
        </button>
      </div>

      {videos.length === 0 && (
        <div className="text-center py-16 text-gray-500">
          <div className="text-5xl mb-4">🏒</div>
          <p className="text-lg mb-2">No videos yet</p>
          <p className="text-sm">Record a shot to get started with analysis</p>
          <button
            onClick={() => navigate('/record')}
            className="mt-4 bg-cyan-600 hover:bg-cyan-500 text-white px-6 py-3 rounded-lg font-bold min-h-[48px]"
          >
            Record Your First Shot
          </button>
        </div>
      )}

      <div className="space-y-3">
        {videos.map(video => (
          <div
            key={video.filename}
            className="bg-gray-900 rounded-xl overflow-hidden border border-gray-800"
          >
            <div
              className="cursor-pointer"
              onClick={() => navigate(`/analysis/${video.filename}`)}
            >
              <video
                src={getVideoUrl(video.filename)}
                className="w-full aspect-video bg-black"
                preload="metadata"
              />
            </div>
            <div className="p-3">
              <div className="flex justify-between items-start mb-1">
                <div>
                  <div className="text-sm text-white font-medium">
                    {new Date(video.date).toLocaleDateString()} {new Date(video.date).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </div>
                  {video.shootingSide && (
                    <span className="text-xs text-gray-400">{video.shootingSide} shot</span>
                  )}
                </div>
                {statusBadge(video)}
              </div>
              {video.notes && (
                <p className="text-xs text-gray-400 mt-1">{video.notes}</p>
              )}
              <div className="flex justify-between items-center mt-2">
                <button
                  onClick={() => navigate(`/analysis/${video.filename}`)}
                  className="text-sm text-cyan-400 font-medium"
                >
                  {video.analysisStatus === 'complete' ? 'View Analysis →' : 'Details →'}
                </button>
                <button
                  onClick={() => handleDelete(video.filename)}
                  className="text-xs text-red-400"
                >
                  Delete
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
