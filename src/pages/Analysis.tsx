import { useState, useEffect, useCallback } from 'react'
import { useParams } from 'react-router-dom'
import type { VideoMeta, AnalysisResult } from '../types'
import {
  getVideoMeta, getVideoUrl, startAnalysis,
  getAnalysis, getAnnotatedVideoUrl
} from '../api/client'

const ratingColors: Record<string, string> = {
  excellent: 'text-green-400',
  good: 'text-green-300',
  moderate: 'text-yellow-400',
  weak: 'text-red-400',
  unknown: 'text-gray-400',
}

const ratingBg: Record<string, string> = {
  excellent: 'bg-green-900/30 border-green-500/30',
  good: 'bg-green-900/20 border-green-500/20',
  moderate: 'bg-yellow-900/20 border-yellow-500/20',
  weak: 'bg-red-900/20 border-red-500/20',
  unknown: 'bg-gray-800 border-gray-700',
}

export function Analysis() {
  const { filename } = useParams<{ filename: string }>()
  const [meta, setMeta] = useState<VideoMeta | null>(null)
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null)
  const [loading, setLoading] = useState(true)
  const [showAnnotated, setShowAnnotated] = useState(false)

  const loadData = useCallback(async () => {
    if (!filename) return
    const m = await getVideoMeta(filename)
    setMeta(m)
    if (m?.analysisId && m.analysisStatus === 'complete') {
      const a = await getAnalysis(m.analysisId)
      setAnalysis(a)
    }
    setLoading(false)
  }, [filename])

  useEffect(() => { loadData() }, [loadData])

  // Poll for analysis completion
  useEffect(() => {
    if (!meta || meta.analysisStatus !== 'running') return
    const interval = setInterval(async () => {
      const m = await getVideoMeta(filename!)
      if (m) {
        setMeta(m)
        if (m.analysisStatus === 'complete' && m.analysisId) {
          const a = await getAnalysis(m.analysisId)
          setAnalysis(a)
          clearInterval(interval)
        } else if (m.analysisStatus === 'error') {
          clearInterval(interval)
        }
      }
    }, 3000)
    return () => clearInterval(interval)
  }, [meta?.analysisStatus, filename])

  const handleAnalyze = async () => {
    if (!filename || !meta) return
    await startAnalysis(filename, meta.shootingSide || undefined)
    setMeta(prev => prev ? { ...prev, analysisStatus: 'running' } : null)
  }

  if (loading) return <div className="text-center py-8 text-gray-400">Loading...</div>
  if (!meta || !filename) return <div className="text-center py-8 text-gray-400">Video not found</div>

  const scoreColor = (score: number) => {
    if (score >= 80) return 'text-green-400'
    if (score >= 60) return 'text-yellow-300'
    if (score >= 40) return 'text-orange-400'
    return 'text-red-400'
  }

  return (
    <div className="space-y-4 animate-slide-up">
      {/* Video player - toggle between original and annotated */}
      <div className="relative">
        {showAnnotated && meta.analysisId ? (
          <video
            key="annotated"
            src={getAnnotatedVideoUrl(meta.analysisId)}
            controls
            className="w-full rounded-lg bg-black aspect-video"
          />
        ) : (
          <video
            key="original"
            src={getVideoUrl(filename)}
            controls
            className="w-full rounded-lg bg-black aspect-video"
          />
        )}
      </div>

      {/* Toggle original/annotated */}
      {meta.analysisStatus === 'complete' && meta.analysisId && (
        <div className="flex gap-2">
          <button
            onClick={() => setShowAnnotated(false)}
            className={`flex-1 py-2 rounded-lg font-bold min-h-[44px] text-sm ${
              !showAnnotated ? 'bg-cyan-600 text-white' : 'bg-gray-800 text-gray-400'
            }`}
          >
            Original
          </button>
          <button
            onClick={() => setShowAnnotated(true)}
            className={`flex-1 py-2 rounded-lg font-bold min-h-[44px] text-sm ${
              showAnnotated ? 'bg-cyan-600 text-white' : 'bg-gray-800 text-gray-400'
            }`}
          >
            With Pose Overlay
          </button>
        </div>
      )}

      {/* Video info */}
      <div className="flex justify-between items-center text-sm text-gray-400">
        <span>{new Date(meta.date).toLocaleDateString()} {new Date(meta.date).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
        {meta.shootingSide && <span>{meta.shootingSide} shot</span>}
      </div>
      {meta.notes && <p className="text-sm text-gray-300">{meta.notes}</p>}

      {/* Analysis status / trigger */}
      {meta.analysisStatus === 'pending' && (
        <button
          onClick={handleAnalyze}
          className="w-full bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-bold py-4 rounded-xl text-lg min-h-[48px] shadow-lg shadow-cyan-500/20"
        >
          Analyze Shot Mechanics
        </button>
      )}

      {meta.analysisStatus === 'running' && (
        <div className="bg-gray-900 rounded-xl p-6 text-center">
          <div className="text-3xl mb-3 animate-pulse">⚙️</div>
          <div className="font-bold text-white mb-1">Analyzing...</div>
          <div className="text-sm text-gray-400">Detecting poses and evaluating mechanics</div>
          <div className="mt-3 w-full bg-gray-800 rounded-full h-2 overflow-hidden">
            <div className="bg-cyan-500 h-full rounded-full animate-pulse" style={{ width: '60%' }} />
          </div>
        </div>
      )}

      {meta.analysisStatus === 'error' && (
        <div className="bg-red-900/20 border border-red-500/30 rounded-xl p-4">
          <div className="font-bold text-red-400 mb-1">Analysis Failed</div>
          <div className="text-sm text-gray-400">{meta.analysisError || 'Unknown error'}</div>
          <button
            onClick={handleAnalyze}
            className="mt-3 bg-red-600 text-white px-4 py-2 rounded-lg font-bold min-h-[44px]"
          >
            Retry
          </button>
        </div>
      )}

      {/* Analysis results */}
      {analysis && (
        <>
          {/* Overall score */}
          <div className="bg-gray-900 rounded-xl p-6 text-center">
            <div className="text-sm text-gray-400 mb-1">Shot Mechanics Score</div>
            <div className={`text-6xl font-bold ${scoreColor(analysis.score.total_score)}`}>
              {analysis.score.total_score.toFixed(0)}
            </div>
            <div className="text-sm text-gray-500">out of 100</div>
          </div>

          {/* Breakdown */}
          <div className="space-y-2">
            {Object.entries(analysis.score.breakdown).map(([key, data]) => {
              const name = key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
              const pct = (data.points / data.max_points) * 100
              return (
                <div
                  key={key}
                  className={`rounded-xl p-4 border ${ratingBg[data.rating] || ratingBg.unknown}`}
                >
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-bold text-white">{name}</span>
                    <span className={`font-bold ${ratingColors[data.rating] || 'text-gray-400'}`}>
                      {data.rating.toUpperCase()}
                    </span>
                  </div>
                  <div className="w-full bg-gray-800 rounded-full h-2 mb-2">
                    <div
                      className={`h-full rounded-full transition-all ${
                        data.rating === 'excellent' || data.rating === 'good'
                          ? 'bg-green-500'
                          : data.rating === 'moderate'
                            ? 'bg-yellow-500'
                            : 'bg-red-500'
                      }`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-400">{data.detail}</span>
                    <span className="text-gray-500">{data.points.toFixed(0)}/{data.max_points}</span>
                  </div>
                </div>
              )
            })}
          </div>

          {/* Feedback */}
          {analysis.score.feedback.length > 0 && (
            <div className="bg-gray-900 rounded-xl p-4">
              <h3 className="font-bold text-white mb-2">Coaching Notes</h3>
              {analysis.score.feedback.map((line, i) => (
                <p key={i} className={`text-sm ${line.startsWith('  -') ? 'text-cyan-300 ml-2' : 'text-gray-300'} ${!line ? 'h-2' : ''}`}>
                  {line}
                </p>
              ))}
            </div>
          )}

          {/* Shot info */}
          <div className="bg-gray-900 rounded-xl p-4">
            <h3 className="font-bold text-white mb-2">Technical Details</h3>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div>
                <span className="text-gray-400">Shooting Side:</span>
                <span className="text-white ml-1">{analysis.shooting_side}</span>
              </div>
              <div>
                <span className="text-gray-400">Confidence:</span>
                <span className="text-white ml-1">{(analysis.shot_info.confidence * 100).toFixed(0)}%</span>
              </div>
              <div>
                <span className="text-gray-400">Release Frame:</span>
                <span className="text-white ml-1">{analysis.shot_info.release_frame}</span>
              </div>
              <div>
                <span className="text-gray-400">Total Frames:</span>
                <span className="text-white ml-1">{analysis.total_frames}</span>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
