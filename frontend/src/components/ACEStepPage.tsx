import { useState, useEffect, useRef } from 'react'

// ── Types ────────────────────────────────────────────────────────────────────

export interface ACEStepPrefill {
  caption:  string
  bpm:      string
  key:      string
  timeSig:  string
  duration: string
  lyrics:   string
}

interface Take {
  take_n:    number
  audio_url: string
  metadata:  { bpm?: number; keyscale?: string; duration?: number; genres?: string }
  status:    'done' | 'generating' | 'error'
  errorMsg?: string
}

interface Props {
  onBack:         () => void
  onSendToFade?:  (videoSessionId: string) => void
  prefill?:       Partial<ACEStepPrefill>
  sessionId?:     string
}

// ── Component ─────────────────────────────────────────────────────────────────

export function ACEStepPage({ onBack, onSendToFade, prefill, sessionId: resumeId }: Props) {
  const [caption,  setCaption]  = useState(prefill?.caption  ?? '')
  const [bpm,      setBpm]      = useState(prefill?.bpm      ?? '')
  const [key,      setKey]      = useState(prefill?.key      ?? '')
  const [timeSig,  setTimeSig]  = useState(prefill?.timeSig  ?? '')
  const [duration, setDuration] = useState(prefill?.duration ?? '90')
  const [lyrics,   setLyrics]   = useState(prefill?.lyrics   ?? '')

  const [takes,            setTakes]           = useState<Take[]>([])
  const [selectedTake,     setSelectedTake]     = useState<number | null>(null)
  const [generating,       setGenerating]       = useState(false)
  const [sending,          setSending]          = useState(false)
  const [serverStatus,     setServerStatus]     = useState<'starting' | 'ready' | 'error'>('starting')
  const [wsStatus,         setWsStatus]         = useState<'connecting' | 'connected' | 'error'>('connecting')
  const [aceStepSessionId, setAceStepSessionId] = useState<string | null>(null)

  const wsRef       = useRef<WebSocket | null>(null)
  const healthTimer = useRef<ReturnType<typeof setInterval> | null>(null)

  // ── Lifecycle ──────────────────────────────────────────────────────────────

  useEffect(() => {
    let ws: WebSocket | null = null
    let cancelled = false

    async function init() {
      try {
        let session_id: string
        if (resumeId) {
          // Resume existing session — load params + takes from disk
          const r = await fetch(`/acestep/sessions/${resumeId}/resume`, { method: 'POST' })
          if (!r.ok) throw new Error(`Resume failed: ${r.status}`)
          if (cancelled) return
          const data = await r.json()
          session_id = data.session_id
          setAceStepSessionId(session_id)
          const p = data.params ?? {}
          if (p.caption)        setCaption(p.caption)
          if (p.bpm)            setBpm(p.bpm)
          if (p.key)            setKey(p.key)
          if (p.time_signature) setTimeSig(p.time_signature)
          if (p.duration)       setDuration(p.duration)
          if (p.lyrics)         setLyrics(p.lyrics)
          if (data.takes?.length) setTakes(data.takes)
        } else {
          const r = await fetch('/acestep/sessions', { method: 'POST' })
          if (!r.ok) throw new Error(`Session failed: ${r.status}`)
          if (cancelled) return
          session_id = (await r.json()).session_id
          setAceStepSessionId(session_id)
        }
        if (cancelled) return

        const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
        const host  = import.meta.env.DEV ? '127.0.0.1:8001' : window.location.host
        ws = new WebSocket(`${proto}://${host}/acestep/${session_id}/ws`)
        if (cancelled) { ws.close(); return }
        wsRef.current = ws

        ws.onopen  = () => setWsStatus('connected')
        ws.onerror = () => setWsStatus('error')
        ws.onclose = () => setWsStatus('error')
        ws.onmessage = (e) => {
          try { handleServerEvent(JSON.parse(e.data)) } catch { /* ignore */ }
        }
      } catch (err) {
        if (!cancelled) {
          setWsStatus('error')
        }
      }

      // Start ACEStep subprocess
      try { await fetch('/acestep/start', { method: 'POST' }) } catch { /* ignore */ }

      // Poll health
      if (!cancelled) {
        let elapsed = 0
        healthTimer.current = setInterval(async () => {
          elapsed += 2000
          try {
            const r = await fetch('/acestep/health')
            const { ok } = await r.json()
            if (ok) {
              setServerStatus('ready')
              clearInterval(healthTimer.current!)
            } else if (elapsed >= 600_000) {
              setServerStatus('error')
              clearInterval(healthTimer.current!)
            }
          } catch {
            if (elapsed >= 600_000) {
              setServerStatus('error')
              clearInterval(healthTimer.current!)
            }
          }
        }, 2000)
      }
    }

    init()
    return () => {
      cancelled = true
      ws?.close()
      wsRef.current = null
      if (healthTimer.current) clearInterval(healthTimer.current)
      fetch('/acestep/stop', { method: 'POST' }).catch(() => {})
    }
  }, [])

  // ── Server events ──────────────────────────────────────────────────────────

  function handleServerEvent(data: {
    event: string; message?: string;
    take_n?: number; audio_url?: string; metadata?: Take['metadata']
  }) {
    switch (data.event) {
      case 'gen_start': {
        const n = data.take_n!
        setTakes(prev => [...prev, { take_n: n, audio_url: '', metadata: {}, status: 'generating' }])
        break
      }
      case 'gen_done': {
        const n = data.take_n!
        setTakes(prev => prev.map(t =>
          t.take_n === n
            ? { ...t, audio_url: data.audio_url!, metadata: data.metadata ?? {}, status: 'done' }
            : t
        ))
        // Auto-select the latest completed take
        setSelectedTake(n)
        setGenerating(false)
        break
      }
      case 'gen_error': {
        const n = data.take_n!
        setTakes(prev => prev.map(t =>
          t.take_n === n ? { ...t, status: 'error', errorMsg: data.message } : t
        ))
        setGenerating(false)
        break
      }
    }
  }

  // ── Generate ───────────────────────────────────────────────────────────────

  function generate() {
    if (generating || serverStatus !== 'ready' || wsRef.current?.readyState !== WebSocket.OPEN) return
    setGenerating(true)
    wsRef.current.send(JSON.stringify({
      type: 'generate',
      params: { caption, lyrics, bpm, key, time_signature: timeSig, duration },
    }))
  }

  // ── Send to FADE ───────────────────────────────────────────────────────────

  async function sendToFade() {
    if (!aceStepSessionId || selectedTake === null || sending) return
    setSending(true)
    try {
      const r = await fetch('/sessions/from_acestep', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ acestep_session_id: aceStepSessionId, take_n: selectedTake }),
      })
      if (!r.ok) throw new Error(`${r.status}`)
      const data = await r.json()
      onSendToFade?.(data.session_id)
    } catch (err) {
      console.error('Send to FADE failed', err)
      setSending(false)
    }
  }

  // ── Render ────────────────────────────────────────────────────────────────

  const canGenerate    = !generating && serverStatus === 'ready' && !!caption.trim()
  const selectedIsDone = selectedTake !== null && takes.some(t => t.take_n === selectedTake && t.status === 'done')
  const canSendToFade  = !!onSendToFade && selectedIsDone && !sending

  return (
    <div className="acestep-page">

      {/* Header */}
      <div className="acestep-header">
        <button className="back-btn" onClick={onBack}>← Back</button>
        <div className="acestep-header-title">
          <span className="acestep-title">Make a Song</span>
          <span className="acestep-subtitle">ACE-Step 1.5 · local generation</span>
        </div>
        <div className="acestep-status-row">
          <span className={`acestep-server-dot acestep-server-dot--${serverStatus}`} title={`ACEStep: ${serverStatus}`} />
          <span className="acestep-server-label">
            {serverStatus === 'starting' ? 'Starting…' : serverStatus === 'ready' ? 'Ready' : 'Error'}
          </span>
          <span className={`ws-dot ws-dot--${wsStatus}`} title={`WS: ${wsStatus}`} />
        </div>
      </div>

      {/* Prompt panel */}
      <div className="acestep-panel">

        <div className="panel-field">
          <label className="panel-label">CAPTION</label>
          <textarea
            className="panel-textarea"
            rows={3}
            placeholder="genre, mood, instrumentation, vocal style…"
            value={caption}
            onChange={e => setCaption(e.target.value)}
          />
        </div>

        <div className="panel-meta-row">
          <div className="panel-chip">
            <label className="panel-chip-label">BPM</label>
            <input className="panel-chip-input" placeholder="auto" value={bpm} onChange={e => setBpm(e.target.value)} />
          </div>
          <div className="panel-chip">
            <label className="panel-chip-label">KEY</label>
            <input className="panel-chip-input panel-chip-input--wide" placeholder="auto" value={key} onChange={e => setKey(e.target.value)} />
          </div>
          <div className="panel-chip">
            <label className="panel-chip-label">TIME</label>
            <input className="panel-chip-input" placeholder="4/4" value={timeSig} onChange={e => setTimeSig(e.target.value)} />
          </div>
          <div className="panel-chip">
            <label className="panel-chip-label">DUR (s)</label>
            <input className="panel-chip-input" placeholder="90" value={duration} onChange={e => setDuration(e.target.value)} />
          </div>
        </div>

        <div className="panel-field panel-field--grow">
          <label className="panel-label">LYRICS</label>
          <textarea
            className="panel-textarea panel-textarea--lyrics"
            placeholder={"[verse]\nyour lyrics here\n\n[chorus]\nhook line"}
            value={lyrics}
            onChange={e => setLyrics(e.target.value)}
          />
        </div>

        {takes.length > 0 && (
          <div className="takes-section">
            <div className="panel-label">TAKES</div>
            <div className="takes-list">
              {takes.map(t => (
                <TakeRow
                  key={t.take_n}
                  take={t}
                  selected={t.take_n === selectedTake}
                  onSelect={() => t.status === 'done' && setSelectedTake(t.take_n)}
                />
              ))}
            </div>
          </div>
        )}

        <div className="panel-actions">
          <button
            className={`btn btn--primary${canGenerate ? '' : ' btn--disabled'}`}
            disabled={!canGenerate}
            onClick={generate}
          >
            {generating ? 'Generating…' : serverStatus === 'starting' ? 'Starting ACE-Step…' : 'Generate'}
          </button>
          {onSendToFade && (
            <button
              className={`btn btn--ghost${canSendToFade ? '' : ' btn--disabled'}`}
              disabled={!canSendToFade}
              onClick={sendToFade}
              title={selectedIsDone ? 'Send selected take to video director' : 'Select a completed take first'}
            >
              {sending ? 'Sending…' : 'Use in FADE'}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

// ── TakeRow ───────────────────────────────────────────────────────────────────

interface TakeRowProps {
  take:      Take
  selected:  boolean
  onSelect:  () => void
}

function TakeRow({ take, selected, onSelect }: TakeRowProps) {
  const meta = take.metadata

  if (take.status === 'generating') {
    return (
      <div className="take-row take-row--generating">
        <span className="take-label">Take {take.take_n}</span>
        <span className="spinner-inline" />
        <span className="take-status-text">Generating…</span>
      </div>
    )
  }

  if (take.status === 'error') {
    return (
      <div className="take-row take-row--error">
        <span className="take-label">Take {take.take_n}</span>
        <span className="take-error">{take.errorMsg ?? 'Generation failed'}</span>
      </div>
    )
  }

  return (
    <div
      className={`take-row take-row--done${selected ? ' take-row--selected' : ''}`}
      onClick={onSelect}
      role="radio"
      aria-checked={selected}
    >
      <div className="take-row-top">
        <span className={`take-select-dot${selected ? ' take-select-dot--on' : ''}`} />
        <span className="take-label">Take {take.take_n}</span>
        {meta.bpm      && <span className="take-chip">{meta.bpm} BPM</span>}
        {meta.keyscale && <span className="take-chip">{meta.keyscale}</span>}
        {meta.duration && <span className="take-chip">{Math.round(meta.duration)}s</span>}
      </div>
      <audio
        className="take-audio"
        controls
        src={take.audio_url}
        preload="metadata"
        onClick={e => e.stopPropagation()}
      />
    </div>
  )
}
