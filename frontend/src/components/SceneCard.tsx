import { useState, useEffect } from 'react'
import type { SceneData } from '../types'

interface Props {
  index:       number
  scene:       SceneData
  sessionId:   string
  orientation: string
  onApproveImage:   (index: number) => void
  onApproveVideo:   (index: number) => void
  onUnapproveImage: (index: number) => void
  onUnapproveVideo: (index: number) => void
}

function fmtTime(s: number) {
  if (s == null || isNaN(s)) return '--:--'
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
  return `${m}:${sec.toString().padStart(2, '0')}`
}

function fmtDur(start: number, end: number) {
  if (start == null || end == null || isNaN(start - end)) return '--'
  return `${(end - start).toFixed(1)}s`
}

export function SceneCard({ index, scene, sessionId, orientation, onApproveImage, onApproveVideo, onUnapproveImage, onUnapproveVideo }: Props) {
  const aspectRatio = orientation === 'portrait' ? '9 / 16' : '16 / 9'
  const hasPrompts = !!(scene.image_prompt || scene.video_prompt)
  const imgDone    = scene.image_status === 'done' || scene.image_status === 'approved'
  const vidDone    = scene.video_status === 'done' || scene.video_status === 'approved'

  // Prompts are editable until images are locked
  const promptsLocked = imgDone

  return (
    <div className={`scene-card scene-card--${scene.image_status}`}>
      {/* Header */}
      <div className="scene-card-header">
        <span className="scene-num">Scene {index}</span>
        <span className="scene-time">
          {fmtTime(scene.start_s)}–{fmtTime(scene.end_s)}
          <span className="scene-dur"> {fmtDur(scene.start_s, scene.end_s)}</span>
        </span>
        {scene.label && <span className="scene-label">{scene.label}</span>}
      </div>

      {/* Lyrics excerpt */}
      {scene.lyrics_full && (
        <p className="scene-lyrics">"{scene.lyrics_full.slice(0, 80)}{scene.lyrics_full.length > 80 ? '…' : ''}"</p>
      )}

      {/* Image slot */}
      <div className="scene-media-slot" style={{ aspectRatio }}>
        {scene.image_status === 'generating' && (
          <div className="scene-generating"><div className="spinner" /><span>Generating image…</span></div>
        )}
        {imgDone && scene.image_path && (
          <ImageSlot sessionId={sessionId} imagePath={scene.image_path} imageStatus={scene.image_status} />
        )}
        {!imgDone && scene.image_status !== 'generating' && (
          <div className="scene-placeholder">
            {hasPrompts ? 'Image pending' : 'Prompts pending'}
          </div>
        )}
      </div>

      {/* Editable prompts */}
      <PromptField
        label="Image"
        value={scene.image_prompt ?? ''}
        locked={promptsLocked}
        sessionId={sessionId}
        sceneIndex={index}
        field="image_prompt"
      />
      <PromptField
        label="Video"
        value={scene.video_prompt ?? ''}
        locked={promptsLocked}
        sessionId={sessionId}
        sceneIndex={index}
        field="video_prompt"
      />

      {/* Video slot */}
      {vidDone && scene.video_path && (
        <div className="scene-media-slot" style={{ aspectRatio }}>
          <VideoSlot sessionId={sessionId} videoPath={scene.video_path} videoStatus={scene.video_status} />
        </div>
      )}
      {scene.video_status === 'generating' && (
        <div className="scene-generating"><div className="spinner" /><span>Generating video…</span></div>
      )}

      {/* Action buttons */}
      <div className="scene-actions">
        {scene.image_status === 'done' && (
          <button className="btn btn--approve" onClick={() => onApproveImage(index)}>
            Approve Image
          </button>
        )}
        {scene.image_status === 'approved' && (
          <button className="btn btn--unapprove" onClick={() => onUnapproveImage(index)}>
            Image ✓
          </button>
        )}
        {scene.video_status === 'done' && (
          <button className="btn btn--approve" onClick={() => onApproveVideo(index)}>
            Approve Video
          </button>
        )}
        {scene.video_status === 'approved' && (
          <button className="btn btn--unapprove" onClick={() => onUnapproveVideo(index)}>
            Video ✓
          </button>
        )}
      </div>
    </div>
  )
}

// ── Editable prompt field ──────────────────────────────────────────────────────

interface PromptFieldProps {
  label:      string
  value:      string
  locked:     boolean
  sessionId:  string
  sceneIndex: number
  field:      'image_prompt' | 'video_prompt'
}

function PromptField({ label, value, locked, sessionId, sceneIndex, field }: PromptFieldProps) {
  const [draft,   setDraft]   = useState<string | null>(null)  // null = not editing
  const [saving,  setSaving]  = useState(false)

  const current = draft !== null ? draft : value

  async function save(text: string) {
    if (text === value) { setDraft(null); return }
    setSaving(true)
    try {
      await fetch(`/sessions/${sessionId}/scenes/${sceneIndex}`, {
        method:  'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ [field]: text }),
      })
    } finally {
      setSaving(false)
      setDraft(null)
    }
  }

  if (locked) {
    // After image gen: show read-only (image is already generated from this prompt)
    return value ? (
      <div className="scene-prompt scene-prompt--locked">
        <span className="prompt-label">{label}</span>
        <span className="prompt-text">{value}</span>
      </div>
    ) : null
  }

  return (
    <div className={`scene-prompt scene-prompt--edit ${saving ? 'scene-prompt--saving' : ''}`}>
      <span className="prompt-label">{label}</span>
      <textarea
        className="prompt-textarea"
        placeholder={`${label} prompt…`}
        value={current}
        rows={3}
        onChange={e => setDraft(e.target.value)}
        onBlur={e => save(e.target.value)}
      />
    </div>
  )
}

// ── Video display ─────────────────────────────────────────────────────────────

function VideoSlot({ sessionId, videoPath, videoStatus }: { sessionId: string; videoPath: string; videoStatus: string }) {
  const [ts, setTs] = useState(() => Date.now())
  useEffect(() => { setTs(Date.now()) }, [videoPath, videoStatus])
  const filename = videoPath.split(/[\\/]/).pop() ?? ''
  return (
    <video
      key={ts}
      src={`/sessions/${sessionId}/files/videos/${filename}?t=${ts}`}
      controls
      className="scene-video"
    />
  )
}

// ── Image display ──────────────────────────────────────────────────────────────

function ImageSlot({ sessionId, imagePath, imageStatus }: { sessionId: string; imagePath: string; imageStatus: string }) {
  const [ts, setTs] = useState(() => Date.now())
  // Update timestamp whenever imagePath or imageStatus changes (regen complete)
  useEffect(() => { setTs(Date.now()) }, [imagePath, imageStatus])
  const filename = imagePath.split(/[\\/]/).pop() ?? ''
  const src = `/sessions/${sessionId}/files/images/${filename}?t=${ts}`
  return (
    <a href={src} target="_blank" rel="noreferrer">
      <img
        src={src}
        alt="Generated scene"
        className="scene-image"
        onError={(e) => { (e.target as HTMLImageElement).style.display = 'none' }}
      />
    </a>
  )
}
