import { useState, useEffect, useRef } from 'react'
import type { KeyboardEvent } from 'react'
import type { ChatMessage } from '../types'
import type { WsStatus } from '../hooks/useWebSocket'

interface Props {
  messages:  ChatMessage[]
  busy:      boolean
  filename:  string
  savePath:  string
  wsStatus:  WsStatus
  resumed:   boolean
  onSend:    (msg: string) => void
}

export function ChatPanel({ messages, busy, filename, savePath, wsStatus, resumed, onSend }: Props) {
  const [draft, setDraft] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)

  // Auto-scroll on new content
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  function submit() {
    const text = draft.trim()
    if (!text || busy) return
    onSend(text)
    setDraft('')
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  const connected = wsStatus === 'connected'
  const inputDisabled = busy || !connected

  return (
    <div className="chat-panel">
      <div className="chat-header">
        <div className="chat-header-info">
          <span className="chat-filename">{filename}</span>
          {savePath && <span className="chat-savepath">{savePath}</span>}
        </div>
        <span
          className={`ws-dot ws-dot--${wsStatus}`}
          title={`WebSocket: ${wsStatus}`}
        />
        {busy && <span className="chat-busy-dot" title="Agent working" />}
      </div>

      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-autostart-notice">
            <span className="chat-autostart-spinner" />
            <span>
              {resumed
                ? 'Resuming — the director will catch you up in a moment.'
                : 'Separating vocals, aligning lyrics, and analysing audio — this takes a couple of minutes. The director will brief you when it\u2019s ready.'}
            </span>
          </div>
        )}
        {messages.map((m) => (
          <div key={m.id} className={`chat-msg chat-msg--${m.role}`}>
            {m.role === 'tool' && (
              <span className="chat-tool-label">tool</span>
            )}
            {m.role === 'analysis_result' ? (
              <div className="analysis-card">
                {m.title && <div className="analysis-card-title">{m.title}</div>}
                <pre className="analysis-card-body">{m.content}</pre>
              </div>
            ) : (
              <span className="chat-msg-content">{m.content}</span>
            )}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="chat-input-row">
        <textarea
          className="chat-input"
          rows={2}
          placeholder={
            !connected          ? 'Not connected — refresh to reconnect…'
            : busy              ? 'Working — please wait…'
            : messages.length === 0 ? 'Pipeline running — sit tight…'
                                : 'Message the director…'
          }
          disabled={inputDisabled}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={onKeyDown}
        />
        <button
          className="chat-send-btn"
          disabled={inputDisabled || !draft.trim()}
          onClick={submit}
        >
          Send
        </button>
      </div>
    </div>
  )
}
