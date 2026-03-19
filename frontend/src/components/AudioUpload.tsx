import { useState, useRef, useEffect } from 'react'
import type { DragEvent, ChangeEvent } from 'react'

interface WorkflowEntry { id: string; name: string }
interface Workflows { image: WorkflowEntry[]; video: WorkflowEntry[] }

const FALLBACK_WORKFLOWS: Workflows = {
  image: [{ id: 'zit', name: 'ZIT with Reactor' }, { id: 'qie', name: 'Qwen Image Edit' }],
  video: [{ id: 'ltx_humo', name: 'LTX with HuMo' }, { id: 'ltx', name: 'LTX' }],
}

interface Props {
  onBack:         () => void
  onSessionReady: (sessionId: string, filename: string, savePath?: string, orientation?: string) => void
}

export function AudioUpload({ onBack, onSessionReady }: Props) {
  const [dragging,     setDragging]     = useState(false)
  const [refDragging,  setRefDragging]  = useState(false)
  const [uploading,    setUploading]    = useState(false)
  const [error,        setError]        = useState<string | null>(null)
  const [file,         setFile]         = useState<File | null>(null)
  const [refFile,      setRefFile]      = useState<File | null>(null)
  const [projectName,  setProjectName]  = useState('')
  const [lyrics,       setLyrics]       = useState('')
  const [savePath,     setSavePath]     = useState<string | null>(null)
  const [orientation,  setOrientation]  = useState<'landscape' | 'portrait'>('landscape')
  const [imageWorkflow,  setImageWorkflow]  = useState('zit')
  const [videoWorkflow,  setVideoWorkflow]  = useState('ltx_humo')
  const [humoResolution, setHumoResolution] = useState<1280 | 1536 | 1920>(1280)
  const [workflows,      setWorkflows]      = useState<Workflows>(FALLBACK_WORKFLOWS)

  useEffect(() => {
    fetch('/workflows').then(r => r.json()).then(setWorkflows).catch(() => {})
  }, [])

  const audioInputRef = useRef<HTMLInputElement>(null)
  const refInputRef   = useRef<HTMLInputElement>(null)

  async function handleStart() {
    if (!file || uploading) return
    if (!lyrics.trim()) {
      setError('Paste the song lyrics — the forced aligner needs them to timestamp every word.')
      return
    }
    setError(null)
    setUploading(true)
    setSavePath(null)

    try {
      const form = new FormData()
      form.append('audio', file)
      form.append('lyrics', lyrics.trim())
      if (projectName.trim()) form.append('project_name', projectName.trim())
      form.append('orientation', orientation)
      form.append('image_workflow', imageWorkflow)
      form.append('video_workflow', videoWorkflow)
      if (videoWorkflow === 'ltx_humo' || videoWorkflow === 'humo')
        form.append('humo_resolution', String(humoResolution))
      if (refFile) form.append('reference', refFile)

      const r = await fetch('/sessions', { method: 'POST', body: form })
      if (!r.ok) {
        const body = await r.json().catch(() => ({ detail: r.statusText }))
        throw new Error((body as { detail?: string }).detail || `Error ${r.status}`)
      }
      const { session_id, save_path } = await r.json()
      if (save_path) setSavePath(save_path)
      onSessionReady(session_id, file.name, save_path, orientation)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
      setUploading(false)
    }
  }

  function pickAudio(f: File) {
    setFile(f)
    setError(null)
    if (!projectName.trim()) {
      setProjectName(f.name.replace(/\.[^.]+$/, '').replace(/[_-]+/g, ' ').trim())
    }
  }

  function onAudioDrop(e: DragEvent) {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f && /\.(mp3|wav)$/i.test(f.name)) pickAudio(f)
  }

  function onLyricsDrop(e: DragEvent) {
    e.preventDefault()
    const f = e.dataTransfer.files[0]
    if (!f || !/\.txt$/i.test(f.name)) return
    const reader = new FileReader()
    reader.onload = ev => setLyrics((ev.target?.result as string) ?? '')
    reader.readAsText(f)
  }

  function onAudioChange(e: ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]
    if (f) pickAudio(f)
  }

  return (
    <div className="upload-screen">

      <button className="back-btn" onClick={onBack} disabled={uploading}>
        ← Back
      </button>

      <h2 className="setup-heading">New project</h2>

      {/* Audio drop zone */}
      <div
        className={`drop-zone ${dragging ? 'drop-zone--active' : ''} ${file ? 'drop-zone--file' : ''} ${uploading ? 'drop-zone--uploading' : ''}`}
        onClick={() => !uploading && audioInputRef.current?.click()}
        onDragOver={e => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onAudioDrop}
      >
        <input
          ref={audioInputRef}
          type="file"
          accept=".mp3,.wav"
          style={{ display: 'none' }}
          onChange={onAudioChange}
        />
        {file ? (
          <>
            <div className="drop-icon">♪</div>
            <p className="drop-filename">{file.name}</p>
            <p className="drop-hint">Click to change</p>
          </>
        ) : (
          <>
            <div className="drop-icon">♪</div>
            <p>Drop audio here or click to browse</p>
            <p className="drop-hint">MP3 · WAV</p>
          </>
        )}
      </div>

      {/* Project name */}
      <div className="lyrics-field">
        <label className="lyrics-label" htmlFor="project-name">Project name</label>
        <input
          id="project-name"
          className="project-name-input"
          type="text"
          placeholder="my-song"
          value={projectName}
          onChange={e => setProjectName(e.target.value)}
          disabled={uploading}
        />
        {savePath && <span className="save-path-hint">Saving to: {savePath}</span>}
      </div>

      {/* Orientation */}
      <div className="lyrics-field">
        <label className="lyrics-label">Orientation</label>
        <div className="orientation-toggle">
          <button
            type="button"
            className={`orientation-btn ${orientation === 'landscape' ? 'orientation-btn--active' : ''}`}
            onClick={() => setOrientation('landscape')}
            disabled={uploading}
          >
            Landscape
          </button>
          <button
            type="button"
            className={`orientation-btn ${orientation === 'portrait' ? 'orientation-btn--active' : ''}`}
            onClick={() => setOrientation('portrait')}
            disabled={uploading}
          >
            Portrait
          </button>
        </div>
      </div>

      {/* Image workflow */}
      <div className="lyrics-field">
        <label className="lyrics-label">Image Workflow</label>
        <div className="orientation-toggle">
          {workflows.image.map(wf => (
            <button
              key={wf.id}
              type="button"
              className={`orientation-btn ${imageWorkflow === wf.id ? 'orientation-btn--active' : ''}`}
              onClick={() => setImageWorkflow(wf.id)}
              disabled={uploading}
            >
              {wf.name}
            </button>
          ))}
        </div>
      </div>

      {/* Video workflow */}
      <div className="lyrics-field">
        <label className="lyrics-label">Video Workflow</label>
        <div className="orientation-toggle">
          {workflows.video.map(wf => (
            <button
              key={wf.id}
              type="button"
              className={`orientation-btn ${videoWorkflow === wf.id ? 'orientation-btn--active' : ''}`}
              onClick={() => setVideoWorkflow(wf.id)}
              disabled={uploading}
            >
              {wf.name}
            </button>
          ))}
        </div>
      </div>

      {/* Final resolution (only for HuMo-involved workflows) */}
      {(videoWorkflow === 'ltx_humo' || videoWorkflow === 'humo') && (
        <div className="lyrics-field">
          <label className="lyrics-label">Final Resolution</label>
          <div className="orientation-toggle">
            {([1280, 1536, 1920] as const).map(res => (
              <button
                key={res}
                type="button"
                className={`orientation-btn ${humoResolution === res ? 'orientation-btn--active' : ''}`}
                onClick={() => setHumoResolution(res)}
                disabled={uploading}
              >
                {res}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Lyrics */}
      <div className="lyrics-field">
        <label className="lyrics-label" htmlFor="lyrics-input">
          Lyrics
          <span className="lyrics-label-hint"> — paste the full song text</span>
        </label>
        <textarea
          id="lyrics-input"
          className="lyrics-textarea"
          placeholder={"[Verse 1]\nIt started with a look...\n\n[Chorus]\nNever gonna let you go..."}
          value={lyrics}
          onChange={e => setLyrics(e.target.value)}
          onDrop={onLyricsDrop}
          onDragOver={e => e.preventDefault()}
          disabled={uploading}
          rows={6}
        />
      </div>

      {/* Reference image (optional) */}
      <div className="lyrics-field">
        <label className="lyrics-label">
          Reference image
          <span className="lyrics-label-hint"> — optional character or style reference</span>
        </label>
        <div
          className={`ref-zone ${refFile ? 'ref-zone--file' : ''} ${refDragging ? 'ref-zone--active' : ''}`}
          onClick={() => !uploading && refInputRef.current?.click()}
          onDragOver={e => { e.preventDefault(); setRefDragging(true) }}
          onDragLeave={() => setRefDragging(false)}
          onDrop={e => {
            e.preventDefault()
            setRefDragging(false)
            const f = e.dataTransfer.files[0]
            if (f && /\.(jpe?g|png|webp)$/i.test(f.name)) setRefFile(f)
          }}
        >
          <input
            ref={refInputRef}
            type="file"
            accept=".jpg,.jpeg,.png,.webp"
            style={{ display: 'none' }}
            onChange={e => { const f = e.target.files?.[0]; if (f) setRefFile(f) }}
          />
          {refFile
            ? <span className="ref-zone-name">📷 {refFile.name}</span>
            : <span className="ref-zone-hint">Drop image here or click to browse</span>
          }
        </div>
      </div>

      {error && <p className="upload-error">{error}</p>}

      <button
        className="start-btn"
        onClick={handleStart}
        disabled={!file || !lyrics.trim() || uploading}
      >
        {uploading
          ? <><span className="spinner-inline" /> Uploading…</>
          : 'Start →'
        }
      </button>

    </div>
  )
}
