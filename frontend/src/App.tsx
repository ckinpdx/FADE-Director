import { useState, useCallback, useRef } from 'react'
import type { ChatMessage, SceneData, WsEvent } from './types'
import { useWebSocket } from './hooks/useWebSocket'
import { HomePage } from './components/HomePage'
import { AudioUpload } from './components/AudioUpload'
import { ChatPanel } from './components/ChatPanel'
import { Storyboard } from './components/Storyboard'
import { SunoPage } from './components/SunoPage'
import { ACEStepPage } from './components/ACEStepPage'
import type { ACEStepPrefill } from './components/ACEStepPage'

let msgIdSeq = 0
function nextId() { return ++msgIdSeq }

type Page = 'home' | 'setup' | 'director' | 'suno' | 'acestep'

// ── Generation-button visibility helpers ──────────────────────────────────────

function allHavePrompts(scenes: Map<number, SceneData>): boolean {
  if (scenes.size === 0) return false
  return [...scenes.values()].every(s => s.image_prompt && s.video_prompt)
}

function anyGeneratingImages(scenes: Map<number, SceneData>): boolean {
  return [...scenes.values()].some(s => s.image_status === 'generating')
}

function anyImageDone(scenes: Map<number, SceneData>): boolean {
  return [...scenes.values()].some(s => s.image_status === 'done' || s.image_status === 'approved')
}

function allImagesApproved(scenes: Map<number, SceneData>): boolean {
  if (scenes.size === 0) return false
  return [...scenes.values()].every(s => s.image_status === 'approved')
}

function anyGeneratingVideos(scenes: Map<number, SceneData>): boolean {
  return [...scenes.values()].some(s => s.video_status === 'generating')
}

function anyVideoDone(scenes: Map<number, SceneData>): boolean {
  return [...scenes.values()].some(s => s.video_status === 'done' || s.video_status === 'approved')
}

function allVideosApproved(scenes: Map<number, SceneData>): boolean {
  if (scenes.size === 0) return false
  return [...scenes.values()].every(s => s.video_status === 'approved')
}

// ── App ───────────────────────────────────────────────────────────────────────

export default function App() {
  const [page,             setPage]             = useState<Page>('home')
  const [aceStepPrefill,   setAceStepPrefill]   = useState<Partial<ACEStepPrefill> | undefined>(undefined)
  const [sessionId,        setSessionId]        = useState<string | null>(null)
  const [filename,    setFilename]    = useState('')
  const [savePath,    setSavePath]    = useState('')
  const [resumed,     setResumed]     = useState(false)
  const [orientation, setOrientation] = useState<string>('landscape')
  const [messages,   setMessages]   = useState<ChatMessage[]>([])
  const [scenes,     setScenes]     = useState<Map<number, SceneData>>(new Map())
  const [busy,       setBusy]       = useState(false)
  const [genBusy,    setGenBusy]    = useState(false)
  const [exportPath, setExportPath] = useState<string | null>(null)

  const streamIdRef = useRef<number | null>(null)

  // ── WebSocket event handler ───────────────────────────────────────────────

  const handleEvent = useCallback((e: WsEvent) => {
    switch (e.event) {

      case 'token': {
        setMessages(prev => {
          if (streamIdRef.current === null) {
            const id = nextId()
            streamIdRef.current = id
            return [...prev, { id, role: 'assistant', content: e.text }]
          }
          return prev.map(m =>
            m.id === streamIdRef.current ? { ...m, content: m.content + e.text } : m
          )
        })
        break
      }

      case 'assistant_done': {
        streamIdRef.current = null
        setBusy(false)
        break
      }

      case 'tool_call': {
        const argStr = JSON.stringify(e.args ?? {})
        setMessages(prev => [
          ...prev,
          { id: nextId(), role: 'tool', content: `${e.name}(${argStr.slice(0, 120)})` },
        ])
        break
      }

      case 'status': {
        setMessages(prev => [...prev, { id: nextId(), role: 'status', content: e.message }])
        break
      }

      case 'step_done': {
        setMessages(prev => [...prev, { id: nextId(), role: 'step_done', content: e.message }])
        break
      }

      case 'analysis_result': {
        setMessages(prev => [...prev, { id: nextId(), role: 'analysis_result', title: e.title, content: e.content }])
        break
      }

      case 'scene_update': {
        setScenes(prev => {
          const next = new Map(prev)
          if (e.scene) {
            next.set(e.scene_index, e.scene)
          } else if (e.fields) {
            const existing = next.get(e.scene_index) ?? {} as SceneData
            next.set(e.scene_index, { ...existing, ...e.fields })
          }
          return next
        })
        break
      }

      case 'analysis_done': {
        // New session: analysis just finished → trigger agent auto-start.
        // (Resumed sessions get this via auto_start sent on WS connect.)
        send('auto_start', {})
        setBusy(true)
        break
      }

      case 'prompts_done': {
        setMessages(prev => [
          ...prev,
          { id: nextId(), role: 'step_done', content: `All ${e.scenes} scene prompts written.` },
        ])
        setGenBusy(false)
        break
      }

      case 'gen_done': {
        setGenBusy(false)
        break
      }

      case 'export_done': {
        setExportPath(e.path)
        setGenBusy(false)
        setMessages(prev => [
          ...prev,
          { id: nextId(), role: 'step_done', content: 'Export complete — download ready.' },
        ])
        break
      }

      case 'error': {
        streamIdRef.current = null
        setBusy(false)
        setGenBusy(false)
        setMessages(prev => [...prev, { id: nextId(), role: 'error', content: e.message }])
        break
      }
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const { send, wsStatus } = useWebSocket(sessionId, handleEvent)

  // ── Navigation helpers ────────────────────────────────────────────────────

  function enterDirector(sid: string, fname: string, path: string, isResumed: boolean, orient = 'landscape') {
    setSessionId(sid)
    setFilename(fname)
    setSavePath(path)
    setResumed(isResumed)
    setOrientation(orient)
    setMessages([])
    setScenes(new Map())
    setExportPath(null)
    setBusy(false)
    setGenBusy(false)
    setPage('director')
  }

  // ── Chat ──────────────────────────────────────────────────────────────────

  function onSend(text: string) {
    streamIdRef.current = null
    setMessages(prev => [...prev, { id: nextId(), role: 'user', content: text }])
    const sent = send('chat', { message: text })
    if (sent) {
      setBusy(true)
    } else {
      setMessages(prev => [...prev, {
        id: nextId(),
        role: 'error',
        content: 'Not connected — the server may have restarted. Refresh to reconnect.',
      }])
    }
  }

  // ── Generation controls ───────────────────────────────────────────────────

  async function generatePrompts() {
    if (!sessionId || genBusy) return
    setGenBusy(true)
    try {
      const r = await fetch(`/sessions/${sessionId}/generate/prompts`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: 'null',
      })
      if (!r.ok) throw new Error(`Generate prompts failed: ${r.status}`)
    } catch (e) {
      setMessages(prev => [...prev, { id: nextId(), role: 'error', content: String(e) }])
      setGenBusy(false)
    }
  }

  async function generateImages() {
    if (!sessionId || genBusy) return
    setGenBusy(true)
    try {
      const r = await fetch(`/sessions/${sessionId}/generate/images`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: 'null',
      })
      if (!r.ok) throw new Error(`Generate images failed: ${r.status}`)
    } catch (e) {
      setMessages(prev => [...prev, { id: nextId(), role: 'error', content: String(e) }])
      setGenBusy(false)
    }
  }

  async function generateVideos() {
    if (!sessionId || genBusy) return
    setGenBusy(true)
    try {
      const r = await fetch(`/sessions/${sessionId}/generate/videos`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: 'null',
      })
      if (!r.ok) throw new Error(`Generate videos failed: ${r.status}`)
    } catch (e) {
      setMessages(prev => [...prev, { id: nextId(), role: 'error', content: String(e) }])
      setGenBusy(false)
    }
  }

  async function exportFinal() {
    if (!sessionId || genBusy) return
    setGenBusy(true)
    try {
      const r = await fetch(`/sessions/${sessionId}/export`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}',
      })
      if (!r.ok) throw new Error(`Export failed: ${r.status}`)
    } catch (e) {
      setMessages(prev => [...prev, { id: nextId(), role: 'error', content: String(e) }])
      setGenBusy(false)
    }
  }

  async function onApproveImage(index: number) {
    await fetch(`/sessions/${sessionId}/scenes/${index}/approve/image`, { method: 'POST' })
  }

  async function onApproveVideo(index: number) {
    await fetch(`/sessions/${sessionId}/scenes/${index}/approve/video`, { method: 'POST' })
  }

  async function onUnapproveImage(index: number) {
    await fetch(`/sessions/${sessionId}/scenes/${index}/approve/image`, { method: 'DELETE' })
  }

  async function onUnapproveVideo(index: number) {
    await fetch(`/sessions/${sessionId}/scenes/${index}/approve/video`, { method: 'DELETE' })
  }

  // ── Generation button visibility ──────────────────────────────────────────

  const showGenPrompts   = scenes.size > 0 && !allHavePrompts(scenes) && !anyGeneratingImages(scenes) && !anyGeneratingVideos(scenes)
  const showGenImages    = allHavePrompts(scenes) && !anyImageDone(scenes) && !anyGeneratingImages(scenes)
  const showGenVideos    = allImagesApproved(scenes) && !anyVideoDone(scenes) && !anyGeneratingVideos(scenes)
  const showExport       = allVideosApproved(scenes) && !exportPath
  const generatingImages = anyGeneratingImages(scenes)
  const generatingVideos = anyGeneratingVideos(scenes)

  // ── Render ────────────────────────────────────────────────────────────────

  if (page === 'home') {
    return (
      <HomePage
        onMakeVideo={() => setPage('setup')}
        onWriteSong={() => setPage('suno')}
        onMakeSong={() => setPage('acestep')}
        onResume={(sid, name, path) => enterDirector(sid, name, path, true)}
      />
    )
  }

  if (page === 'suno') {
    return (
      <SunoPage
        onBack={() => setPage('home')}
        onSendToGenerator={(prefill) => {
          setAceStepPrefill(prefill)
          setPage('acestep')
        }}
      />
    )
  }

  if (page === 'acestep') {
    return (
      <ACEStepPage
        onBack={() => setPage('home')}
        prefill={aceStepPrefill}
      />
    )
  }

  if (page === 'setup') {
    return (
      <AudioUpload
        onBack={() => setPage('home')}
        onSessionReady={(sid, fname, path = '', orient = 'landscape') => enterDirector(sid, fname, path, false, orient)}
      />
    )
  }

  // page === 'director'
  return (
    <div className="workspace">
      <ChatPanel
        messages={messages}
        busy={busy}
        filename={filename}
        savePath={savePath}
        wsStatus={wsStatus}
        resumed={resumed}
        onSend={onSend}
      />
      <Storyboard
        scenes={scenes}
        sessionId={sessionId!}
        orientation={orientation}
        genBusy={genBusy || busy}
        showGenPrompts={showGenPrompts}
        showGenImages={showGenImages}
        showGenVideos={showGenVideos}
        showExport={showExport}
        generatingImages={generatingImages}
        generatingVideos={generatingVideos}
        exportPath={exportPath}
        onGeneratePrompts={generatePrompts}
        onGenerateImages={generateImages}
        onGenerateVideos={generateVideos}
        onExport={exportFinal}
        onApproveImage={onApproveImage}
        onApproveVideo={onApproveVideo}
        onUnapproveImage={onUnapproveImage}
        onUnapproveVideo={onUnapproveVideo}
      />
    </div>
  )
}
