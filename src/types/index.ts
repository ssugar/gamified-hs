export interface VideoMeta {
  filename: string
  date: string
  notes: string
  size: number
  shootingSide: string | null
  analysisStatus: 'pending' | 'running' | 'complete' | 'error'
  analysisId: string | null
  analysisError?: string
  analysisLog?: string
}

export interface AnalysisResult {
  video: string
  fps: number
  total_frames: number
  frame_size: number[]
  shooting_side: string
  shot_info: {
    release_frame: number
    shot_start_frame: number
    confidence: number
  }
  score: {
    total_score: number
    breakdown: Record<string, {
      points: number
      max_points: number
      rating: string
      detail: string
    }>
    feedback: string[]
  }
  mechanics: Record<string, {
    score: number
    rating: string
    detail: string
    [key: string]: unknown
  }>
}
