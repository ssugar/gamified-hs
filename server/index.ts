import express from 'express'
import cors from 'cors'
import multer from 'multer'
import fs from 'fs/promises'
import path from 'path'
import { fileURLToPath } from 'url'
import { createReadStream, statSync, existsSync, readSync, openSync, closeSync } from 'fs'
import { spawn, execSync } from 'child_process'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const DATA_DIR = path.join(__dirname, '..', 'data')
const VIDEOS_DIR = path.join(DATA_DIR, 'videos')
const ANALYSES_DIR = path.join(DATA_DIR, 'analyses')
const ANALYZER_DIR = path.join(__dirname, '..', 'analyzer')
const VENV_PYTHON = path.join(ANALYZER_DIR, 'venv', 'bin', 'python3')

const app = express()
app.use(cors())
app.use(express.json())

// Multer config for video uploads
const storage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, VIDEOS_DIR),
  filename: (_req, file, cb) => {
    const timestamp = Date.now()
    const ext = path.extname(file.originalname) || '.webm'
    cb(null, `${timestamp}${ext}`)
  },
})
const upload = multer({ storage, limits: { fileSize: 500 * 1024 * 1024 } })

// Ensure data directories exist
async function ensureDataDirs() {
  const dirs = ['players', 'sessions', 'config', 'videos', 'analyses']
  for (const dir of dirs) {
    await fs.mkdir(path.join(DATA_DIR, dir), { recursive: true })
  }
}

// Path traversal guard
function safePath(folder: string, file: string): string | null {
  const resolved = path.resolve(DATA_DIR, folder, `${file}.json`)
  if (!resolved.startsWith(path.resolve(DATA_DIR))) return null
  return resolved
}

// Atomic write helper
async function atomicWriteFile(filePath: string, data: string) {
  const tempPath = `${filePath}.${Date.now()}.tmp`
  await fs.writeFile(tempPath, data)
  await fs.rename(tempPath, filePath)
}

// Generic CRUD endpoints
app.get('/api/data/:folder/:file', async (req, res) => {
  try {
    const filePath = safePath(req.params.folder, req.params.file)
    if (!filePath) return res.status(400).json({ error: 'Invalid path' })
    const data = await fs.readFile(filePath, 'utf-8')
    res.json(JSON.parse(data))
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      res.status(404).json({ error: 'Not found' })
    } else {
      res.status(500).json({ error: 'Server error' })
    }
  }
})

app.put('/api/data/:folder/:file', async (req, res) => {
  try {
    const filePath = safePath(req.params.folder, req.params.file)
    if (!filePath) return res.status(400).json({ error: 'Invalid path' })
    await atomicWriteFile(filePath, JSON.stringify(req.body, null, 2))
    res.json({ success: true })
  } catch (error) {
    res.status(500).json({ error: 'Server error' })
  }
})

app.delete('/api/data/:folder/:file', async (req, res) => {
  try {
    const filePath = safePath(req.params.folder, req.params.file)
    if (!filePath) return res.status(400).json({ error: 'Invalid path' })
    await fs.unlink(filePath)
    res.json({ success: true })
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      res.status(404).json({ error: 'Not found' })
    } else {
      res.status(500).json({ error: 'Server error' })
    }
  }
})

app.get('/api/data/:folder', async (req, res) => {
  try {
    const folderPath = path.join(DATA_DIR, req.params.folder)
    const resolved = path.resolve(folderPath)
    if (!resolved.startsWith(path.resolve(DATA_DIR))) return res.status(400).json({ error: 'Invalid path' })
    const files = await fs.readdir(folderPath)
    const jsonFiles = files.filter(f => f.endsWith('.json')).map(f => f.replace('.json', ''))
    res.json(jsonFiles)
  } catch (error) {
    res.json([])
  }
})

// ==========================================
// Video endpoints
// ==========================================

app.post('/api/videos/upload', upload.single('video'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'No file uploaded' })
    const { notes, shootingSide } = req.body
    const meta = {
      filename: req.file.filename,
      date: new Date().toISOString(),
      notes: notes || '',
      size: req.file.size,
      shootingSide: shootingSide || null,
      analysisStatus: 'pending' as string,
      analysisId: null as string | null,
    }
    await atomicWriteFile(
      path.join(VIDEOS_DIR, `${req.file.filename}.meta.json`),
      JSON.stringify(meta, null, 2)
    )
    res.json(meta)
  } catch (error) {
    console.error('Upload error:', error)
    res.status(500).json({ error: 'Upload failed' })
  }
})

app.get('/api/videos', async (_req, res) => {
  try {
    const files = await fs.readdir(VIDEOS_DIR)
    const metaFiles = files.filter(f => f.endsWith('.meta.json'))
    const metas = await Promise.all(
      metaFiles.map(async f => {
        const data = await fs.readFile(path.join(VIDEOS_DIR, f), 'utf-8')
        return JSON.parse(data)
      })
    )
    metas.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())
    res.json(metas)
  } catch (error) {
    res.json([])
  }
})

app.get('/api/videos/:filename', (req, res) => {
  try {
    const filename = path.basename(req.params.filename)
    const filePath = path.join(VIDEOS_DIR, filename)
    const stat = statSync(filePath)
    const range = req.headers.range

    // Detect content type from file magic bytes, not extension.
    // iOS recordings saved as .webm are often actually MP4/QuickTime containers.
    let contentType = 'video/mp4'
    try {
      const fd = openSync(filePath, 'r')
      const buf = Buffer.alloc(12)
      readSync(fd, buf, 0, 12, 0)
      closeSync(fd)
      // WebM starts with 0x1A45DFA3 (EBML header)
      if (buf[0] === 0x1a && buf[1] === 0x45 && buf[2] === 0xdf && buf[3] === 0xa3) {
        contentType = 'video/webm'
      }
    } catch (_e) {
      // Fall back to mp4
    }

    if (range) {
      const parts = range.replace(/bytes=/, '').split('-')
      const start = parseInt(parts[0], 10)
      const end = parts[1] ? parseInt(parts[1], 10) : stat.size - 1
      res.writeHead(206, {
        'Content-Range': `bytes ${start}-${end}/${stat.size}`,
        'Accept-Ranges': 'bytes',
        'Content-Length': end - start + 1,
        'Content-Type': contentType,
      })
      createReadStream(filePath, { start, end }).pipe(res)
    } else {
      res.writeHead(200, {
        'Content-Length': stat.size,
        'Content-Type': contentType,
      })
      createReadStream(filePath).pipe(res)
    }
  } catch (error) {
    res.status(404).json({ error: 'Video not found' })
  }
})

app.delete('/api/videos/:filename', async (req, res) => {
  try {
    const filename = path.basename(req.params.filename)
    await fs.unlink(path.join(VIDEOS_DIR, filename))
    try { await fs.unlink(path.join(VIDEOS_DIR, `${filename}.meta.json`)) } catch (_e) { /* ok */ }
    // Delete associated analysis
    try {
      const metaPath = path.join(VIDEOS_DIR, `${filename}.meta.json`)
      if (existsSync(metaPath)) {
        const meta = JSON.parse(await fs.readFile(metaPath, 'utf-8'))
        if (meta.analysisId) {
          await fs.unlink(path.join(ANALYSES_DIR, `${meta.analysisId}.json`))
          // Delete annotated video
          try { await fs.unlink(path.join(ANALYSES_DIR, `${meta.analysisId}.mp4`)) } catch (_e) { /* ok */ }
        }
      }
    } catch (_e) { /* ok */ }
    res.json({ success: true })
  } catch (error) {
    res.status(404).json({ error: 'Video not found' })
  }
})

app.put('/api/videos/:filename/notes', async (req, res) => {
  try {
    const filename = path.basename(req.params.filename)
    const metaPath = path.join(VIDEOS_DIR, `${filename}.meta.json`)
    const meta = JSON.parse(await fs.readFile(metaPath, 'utf-8'))
    meta.notes = req.body.notes || ''
    await atomicWriteFile(metaPath, JSON.stringify(meta, null, 2))
    res.json(meta)
  } catch (error) {
    res.status(404).json({ error: 'Video metadata not found' })
  }
})

// ==========================================
// Analysis endpoints
// ==========================================

// Trigger analysis on a video
app.post('/api/analyze/:filename', async (req, res) => {
  try {
    const filename = path.basename(req.params.filename)
    const videoPath = path.join(VIDEOS_DIR, filename)
    const metaPath = path.join(VIDEOS_DIR, `${filename}.meta.json`)

    if (!existsSync(videoPath)) {
      return res.status(404).json({ error: 'Video not found' })
    }

    // Read meta
    let meta: any = {}
    try {
      meta = JSON.parse(await fs.readFile(metaPath, 'utf-8'))
    } catch (_e) {
      meta = { filename, date: new Date().toISOString(), notes: '' }
    }

    const analysisId = `analysis-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`
    const outputVideoPath = path.join(ANALYSES_DIR, `${analysisId}.mp4`)
    const jsonPath = path.join(ANALYSES_DIR, `${analysisId}.json`)

    // Update meta to show analysis in progress
    meta.analysisStatus = 'running'
    meta.analysisId = analysisId
    await atomicWriteFile(metaPath, JSON.stringify(meta, null, 2))

    // Build command args
    const args = [
      path.join(ANALYZER_DIR, 'main.py'),
      '--video', videoPath,
      '--output', outputVideoPath,
      '--json', jsonPath,
    ]
    if (req.body.shootingSide) {
      args.push('--side', req.body.shootingSide)
    } else if (meta.shootingSide) {
      args.push('--side', meta.shootingSide)
    }

    // Auto-detect high-fps video and add --skip-frames
    // For 240fps slow-mo, skip every 4 frames -> ~60fps analysis
    // For 120fps, skip every 2 -> ~60fps
    try {
      const ffprobeOut = execSync(
        `ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "${videoPath}"`,
        { timeout: 5000 }
      ).toString().trim()
      const [num, den] = ffprobeOut.split('/')
      const detectedFps = den ? parseInt(num) / parseInt(den) : parseFloat(num)
      if (detectedFps > 60) {
        const skip = Math.round(detectedFps / 60)
        args.push('--skip-frames', String(skip))
        console.log(`High-fps video detected (${detectedFps.toFixed(0)}fps), using --skip-frames ${skip}`)
      }
    } catch (_e) {
      // ffprobe not available or failed - proceed without skip-frames
    }

    // Respond immediately, run analysis in background
    res.json({ analysisId, status: 'running' })

    // Spawn analyzer process
    console.log(`Starting analysis ${analysisId} for ${filename}`)
    const proc = spawn(VENV_PYTHON, args, {
      cwd: ANALYZER_DIR,
      env: { ...process.env, PYTHONPATH: ANALYZER_DIR },
    })

    let stdout = ''
    let stderr = ''
    proc.stdout.on('data', (data) => { stdout += data.toString() })
    proc.stderr.on('data', (data) => { stderr += data.toString() })

    proc.on('close', async (code) => {
      try {
        if (code === 0 && existsSync(jsonPath)) {
          meta.analysisStatus = 'complete'
          console.log(`Analysis ${analysisId} complete`)
        } else {
          meta.analysisStatus = 'error'
          meta.analysisError = stderr || `Process exited with code ${code}`
          console.error(`Analysis ${analysisId} failed:`, stderr)
        }
        // Save stdout/stderr for debugging
        meta.analysisLog = stdout.slice(-2000)
        await atomicWriteFile(metaPath, JSON.stringify(meta, null, 2))
      } catch (e) {
        console.error('Failed to update meta after analysis:', e)
      }
    })
  } catch (error) {
    console.error('Analyze error:', error)
    res.status(500).json({ error: 'Failed to start analysis' })
  }
})

// Get analysis results
app.get('/api/analyses/:id', async (req, res) => {
  try {
    const id = path.basename(req.params.id)
    const jsonPath = path.join(ANALYSES_DIR, `${id}.json`)
    const data = await fs.readFile(jsonPath, 'utf-8')
    res.json(JSON.parse(data))
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      res.status(404).json({ error: 'Analysis not found' })
    } else {
      res.status(500).json({ error: 'Server error' })
    }
  }
})

// Stream annotated video
app.get('/api/analyses/:id/video', (req, res) => {
  try {
    const id = path.basename(req.params.id)
    const filePath = path.join(ANALYSES_DIR, `${id}.mp4`)
    const stat = statSync(filePath)
    const range = req.headers.range

    if (range) {
      const parts = range.replace(/bytes=/, '').split('-')
      const start = parseInt(parts[0], 10)
      const end = parts[1] ? parseInt(parts[1], 10) : stat.size - 1
      res.writeHead(206, {
        'Content-Range': `bytes ${start}-${end}/${stat.size}`,
        'Accept-Ranges': 'bytes',
        'Content-Length': end - start + 1,
        'Content-Type': 'video/mp4',
      })
      createReadStream(filePath, { start, end }).pipe(res)
    } else {
      res.writeHead(200, {
        'Content-Length': stat.size,
        'Content-Type': 'video/mp4',
      })
      createReadStream(filePath).pipe(res)
    }
  } catch (error) {
    res.status(404).json({ error: 'Annotated video not found' })
  }
})

// List all analyses
app.get('/api/analyses', async (_req, res) => {
  try {
    const files = await fs.readdir(ANALYSES_DIR)
    const jsonFiles = files.filter(f => f.endsWith('.json'))
    const analyses = await Promise.all(
      jsonFiles.map(async f => {
        const data = await fs.readFile(path.join(ANALYSES_DIR, f), 'utf-8')
        return JSON.parse(data)
      })
    )
    res.json(analyses)
  } catch (error) {
    res.json([])
  }
})

const PORT = 3003
const HOST = '0.0.0.0'

ensureDataDirs().then(() => {
  app.listen(PORT, HOST, () => {
    console.log(`Server running on http://${HOST}:${PORT}`)
  })
})
