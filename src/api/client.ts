import type { VideoMeta, AnalysisResult } from '../types'

const API_BASE = `http://${window.location.hostname}:3003/api`

// Video endpoints
export async function uploadVideo(file: Blob, notes?: string, shootingSide?: string): Promise<VideoMeta | null> {
  try {
    const formData = new FormData()
    const ext = file.type === 'video/mp4' ? '.mp4' : '.webm'
    formData.append('video', file, `recording${ext}`)
    if (notes) formData.append('notes', notes)
    if (shootingSide) formData.append('shootingSide', shootingSide)
    const response = await fetch(`${API_BASE}/videos/upload`, {
      method: 'POST',
      body: formData,
    })
    if (!response.ok) throw new Error('Failed')
    return response.json()
  } catch (error) {
    console.error('Upload error:', error)
    return null
  }
}

export async function getVideos(): Promise<VideoMeta[]> {
  try {
    const response = await fetch(`${API_BASE}/videos`)
    if (!response.ok) return []
    return response.json()
  } catch (error) {
    console.error('API error:', error)
    return []
  }
}

export function getVideoUrl(filename: string): string {
  return `${API_BASE}/videos/${filename}`
}

export async function deleteVideo(filename: string): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE}/videos/${filename}`, { method: 'DELETE' })
    return response.ok
  } catch (error) {
    console.error('API error:', error)
    return false
  }
}

export async function updateVideoNotes(filename: string, notes: string): Promise<VideoMeta | null> {
  try {
    const response = await fetch(`${API_BASE}/videos/${filename}/notes`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ notes }),
    })
    if (!response.ok) throw new Error('Failed')
    return response.json()
  } catch (error) {
    console.error('API error:', error)
    return null
  }
}

// Analysis endpoints
export async function startAnalysis(filename: string, shootingSide?: string): Promise<{
  analysisId: string
  status: string
} | null> {
  try {
    const response = await fetch(`${API_BASE}/analyze/${filename}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ shootingSide }),
    })
    if (!response.ok) throw new Error('Failed')
    return response.json()
  } catch (error) {
    console.error('API error:', error)
    return null
  }
}

export async function getAnalysis(id: string): Promise<AnalysisResult | null> {
  try {
    const response = await fetch(`${API_BASE}/analyses/${id}`)
    if (!response.ok) return null
    return response.json()
  } catch (error) {
    console.error('API error:', error)
    return null
  }
}

export function getAnnotatedVideoUrl(analysisId: string): string {
  return `${API_BASE}/analyses/${analysisId}/video`
}

export async function getVideoMeta(filename: string): Promise<VideoMeta | null> {
  try {
    const videos = await getVideos()
    return videos.find(v => v.filename === filename) || null
  } catch (error) {
    console.error('API error:', error)
    return null
  }
}
