import { useState, useEffect, useRef, useCallback } from 'react'
import type { KeyboardEvent } from 'react'
import type { ACEStepPrefill } from './ACEStepPage'

type Mode = 'suno' | 'acestep'

interface Message {
  id:      number
  role:    'user' | 'assistant' | 'error'
  content: string
}

interface Props {
  onBack:              () => void
  onSendToGenerator?:  (prefill: Partial<ACEStepPrefill>) => void
}

let msgSeq = 0
function nextId() { return ++msgSeq }

function hasPackage(text: string, mode: Mode): boolean {
  if (mode === 'acestep') return text.includes('CAPTION:') && text.includes('LYRICS:')
  return text.includes('STYLE TAGS:') && text.includes('LYRICS:')
}

const MODE_LABELS: Record<Mode, string> = {
  suno:    'Suno',
  acestep: 'ACE-Step 1.5',
}

const SUBTITLES: Record<Mode, string> = {
  suno:    'Suno v5 prompt engineer',
  acestep: 'ACE-Step 1.5 prompt engineer',
}

const WELCOME: Record<Mode, string> = {
  suno:
    "Describe the song you want to make — genre, vibe, what it's about — and I'll craft a Suno prompt package that actually works.",
  acestep:
    "Tell me about the song you want: genre, mood, instrumentation, what it's about. I'll write a complete ACE-Step 1.5 prompt — caption, lyrics, and generation settings.",
}

function parseAceStepPrefill(text: string): Partial<ACEStepPrefill> {
  // Work inside the code block if present
  const codeM = text.match(/```[^\n]*\n([\s\S]*?)```/)
  const src   = codeM ? codeM[1] : text

  // Same-line values: "KEY: value"
  const sameLine = (key: string) => {
    const m = src.match(new RegExp(`^${key}:[\\t ]+(.+)`, 'm'))
    return m?.[1]?.trim() ?? ''
  }
  // Next-line value: "KEY:\nvalue" — caption sits on the line after its header
  const nextLine = (key: string) => {
    const m = src.match(new RegExp(`^${key}:[\\t ]*\\n(.+)`, 'm'))
    return m?.[1]?.trim() ?? sameLine(key)
  }

  const lyricsStart = src.indexOf('\nLYRICS:\n')
  const notesStart  = src.indexOf('\nNOTES:')
  const lyricsBody  = lyricsStart !== -1
    ? src.slice(lyricsStart + '\nLYRICS:\n'.length, notesStart !== -1 ? notesStart : undefined).trim()
    : ''

  return {
    caption:  nextLine('CAPTION'),
    bpm:      sameLine('BPM'),
    key:      sameLine('KEY'),
    timeSig:  sameLine('TIME SIGNATURE'),
    duration: sameLine('DURATION').replace(/[^0-9]/g, ''),
    lyrics:   lyricsBody,
  }
}

export function SunoPage({ onBack, onSendToGenerator }: Props) {
  const [mode,       setMode]       = useState<Mode>('suno')
  const [messages,   setMessages]   = useState<Message[]>([])
  const [draft,      setDraft]      = useState('')
  const [busy,       setBusy]       = useState(false)
  const [wsStatus,   setWsStatus]   = useState<'connecting' | 'connected' | 'error'>('connecting')
  const [models,     setModels]     = useState<string[]>([])
  const [model,      setModel]      = useState('')

  const wsRef       = useRef<WebSocket | null>(null)
  const streamIdRef = useRef<number | null>(null)
  const bottomRef   = useRef<HTMLDivElement>(null)

  // ── Fetch available models once on mount ──────────────────────────────────

  useEffect(() => {
    fetch('/models')
      .then(r => r.json())
      .then(({ models: list }: { models: string[] }) => {
        if (list.length > 0) {
          setModels(list)
          setModel(prev => prev || list[0])
        }
      })
      .catch(() => {})
  }, [])

  // ── Session + WebSocket — recreate when mode or model changes ─────────────

  useEffect(() => {
    let ws: WebSocket | null = null
    let cancelled = false

    setWsStatus('connecting')

    async function init() {
      try {
        const params = new URLSearchParams({ mode })
        if (model) params.set('model', model)
        const r = await fetch(`/suno/sessions?${params}`, { method: 'POST' })
        if (!r.ok) throw new Error(`Session failed: ${r.status}`)
        if (cancelled) return
        const { session_id } = await r.json()

        const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
        const host  = import.meta.env.DEV ? '127.0.0.1:8001' : window.location.host
        ws = new WebSocket(`${proto}://${host}/suno/${session_id}/ws`)
        if (cancelled) { ws.close(); return }
        wsRef.current = ws

        ws.onopen  = () => setWsStatus('connected')
        ws.onerror = () => setWsStatus('error')
        ws.onclose = () => setWsStatus('error')

        ws.onmessage = (e) => {
          try {
            const data = JSON.parse(e.data)
            handleServerEvent(data)
          } catch { /* ignore */ }
        }
      } catch (err) {
        if (!cancelled) {
          setWsStatus('error')
          setMessages([{ id: nextId(), role: 'error', content: String(err) }])
        }
      }
    }

    init()
    return () => {
      cancelled = true
      ws?.close()
      wsRef.current = null
    }
  }, [mode, model])  // eslint-disable-line react-hooks/exhaustive-deps

  // Auto-scroll
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // ── Server event handler ──────────────────────────────────────────────────

  const handleServerEvent = useCallback((data: { event: string; text?: string; message?: string }) => {
    switch (data.event) {
      case 'token': {
        const text = data.text ?? ''
        setMessages(prev => {
          if (streamIdRef.current === null) {
            const id = nextId()
            streamIdRef.current = id
            return [...prev, { id, role: 'assistant', content: text }]
          }
          return prev.map(m =>
            m.id === streamIdRef.current ? { ...m, content: m.content + text } : m
          )
        })
        break
      }
      case 'assistant_done': {
        streamIdRef.current = null
        setBusy(false)
        break
      }
      case 'error': {
        streamIdRef.current = null
        setBusy(false)
        setMessages(prev => [...prev, { id: nextId(), role: 'error', content: data.message ?? 'Unknown error' }])
        break
      }
    }
  }, [])

  // ── Mode switching — clears conversation, recreates session ───────────────

  function switchMode(m: Mode) {
    if (m === mode) return
    setMessages([])
    setDraft('')
    setBusy(false)
    streamIdRef.current = null
    setMode(m)
  }

  // ── Send ──────────────────────────────────────────────────────────────────

  function send() {
    const text = draft.trim()
    if (!text || busy || wsRef.current?.readyState !== WebSocket.OPEN) return
    streamIdRef.current = null
    setMessages(prev => [...prev, { id: nextId(), role: 'user', content: text }])
    wsRef.current.send(JSON.stringify({ type: 'chat', message: text }))
    setDraft('')
    setBusy(true)
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() }
  }

  function copyText(text: string) {
    navigator.clipboard.writeText(text).catch(() => {})
  }

  // ── Render ────────────────────────────────────────────────────────────────

  const connected    = wsStatus === 'connected'
  const inputDisabled = busy || !connected

  return (
    <div className="suno-page">

      {/* Header */}
      <div className="suno-header">
        <button className="back-btn" onClick={onBack}>← Back</button>
        <div className="suno-header-title">
          <span className="suno-title">Write a Song</span>
          <span className="suno-subtitle">{SUBTITLES[mode]}</span>
        </div>

        {/* Mode toggle */}
        <div className="suno-mode-toggle">
          {(Object.keys(MODE_LABELS) as Mode[]).map(m => (
            <button
              key={m}
              className={`suno-mode-btn${mode === m ? ' suno-mode-btn--active' : ''}`}
              onClick={() => switchMode(m)}
            >
              {MODE_LABELS[m]}
            </button>
          ))}
        </div>

        {models.length > 0 && (
          <select
            className="suno-model-select"
            value={model}
            onChange={e => { setMessages([]); setDraft(''); setBusy(false); streamIdRef.current = null; setModel(e.target.value) }}
            disabled={busy}
          >
            {models.map(m => <option key={m} value={m}>{m}</option>)}
          </select>
        )}

        <span className={`ws-dot ws-dot--${wsStatus}`} title={`WS: ${wsStatus}`} />
        {busy && <span className="chat-busy-dot" title="Thinking" />}
      </div>

      {/* Messages */}
      <div className="suno-messages">
        {messages.length === 0 && (
          <div className="suno-welcome">
            <p>{WELCOME[mode]}</p>
          </div>
        )}

        {messages.map(m => (
          <div key={m.id} className={`suno-msg suno-msg--${m.role}`}>
            {m.role === 'assistant'
              ? <AssistantBubble
                  text={m.content}
                  mode={mode}
                  onCopy={copyText}
                  onSendToGenerator={onSendToGenerator
                    ? (prefill) => onSendToGenerator(prefill)
                    : undefined
                  }
                />
              : <span className="suno-msg-content">{m.content}</span>
            }
          </div>
        ))}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="suno-input-row">
        <textarea
          className="chat-input"
          rows={2}
          placeholder={
            !connected     ? 'Connecting…'
            : busy         ? 'Working — please wait…'
            : 'Describe your song, or ask for changes…'
          }
          disabled={inputDisabled}
          value={draft}
          onChange={e => setDraft(e.target.value)}
          onKeyDown={onKeyDown}
        />
        {busy
          ? <button className="chat-send-btn chat-stop-btn" onClick={() => wsRef.current?.send(JSON.stringify({ type: 'stop' }))}>Stop</button>
          : <button className="chat-send-btn" disabled={inputDisabled || !draft.trim()} onClick={send}>Send</button>
        }
      </div>
    </div>
  )
}

// ── AssistantBubble ───────────────────────────────────────────────────────────

function AssistantBubble({ text, mode, onCopy, onSendToGenerator }: {
  text:                string
  mode:                Mode
  onCopy:              (t: string) => void
  onSendToGenerator?:  (prefill: Partial<ACEStepPrefill>) => void
}) {
  const [copied, setCopied] = useState(false)

  const isPackage = hasPackage(text, mode)

  const codeMatch  = text.match(/```[\s\S]*?```/)
  const copyTarget = codeMatch ? codeMatch[0].replace(/^```\s*\n?/, '').replace(/\n?```$/, '') : text

  function copy() {
    onCopy(copyTarget)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="suno-assistant-bubble">
      {isPackage && (
        <div className="suno-bubble-actions">
          <button className="suno-copy-btn" onClick={copy}>
            {copied ? '✓ Copied' : 'Copy prompt'}
          </button>
          {mode === 'acestep' && onSendToGenerator && (
            <button
              className="suno-copy-btn suno-copy-btn--generate"
              onClick={() => onSendToGenerator(parseAceStepPrefill(text))}
            >
              Send to Generator →
            </button>
          )}
        </div>
      )}
      <pre className="suno-msg-content suno-msg-content--pre">{text}</pre>
    </div>
  )
}
