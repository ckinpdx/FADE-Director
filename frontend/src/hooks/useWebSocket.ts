import { useEffect, useRef, useCallback, useState } from 'react'
import type { WsEvent } from '../types'

export type WsStatus = 'disconnected' | 'connecting' | 'connected' | 'error'

export function useWebSocket(
  sessionId: string | null,
  onEvent: (e: WsEvent) => void,
) {
  const wsRef      = useRef<WebSocket | null>(null)
  const onEventRef = useRef(onEvent)
  onEventRef.current = onEvent

  const [wsStatus, setWsStatus] = useState<WsStatus>('disconnected')

  useEffect(() => {
    if (!sessionId) return

    setWsStatus('connecting')

    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    // Use 127.0.0.1 explicitly — on Windows, 'localhost' may resolve to ::1
    // (IPv6) first, which fails if uvicorn is bound to 0.0.0.0 (IPv4 only).
    const host = import.meta.env.DEV ? '127.0.0.1:8001' : window.location.host
    const url  = `${protocol}://${host}/sessions/${sessionId}/ws`

    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => {
      setWsStatus('connected')
      onEventRef.current({ event: 'status', message: 'Connected to session.' })
      // Trigger auto-start from the frontend — backend fires separate_vocals()
      // if audio is uploaded and analysis hasn't begun yet.
      ws.send(JSON.stringify({ type: 'auto_start' }))
    }

    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data) as WsEvent
        onEventRef.current(data)
      } catch {
        // ignore unparseable frames
      }
    }

    ws.onerror = () => {
      setWsStatus('error')
      onEventRef.current({ event: 'error', message: 'WebSocket connection error — is the backend running on port 8001?' })
    }

    ws.onclose = (ev) => {
      setWsStatus(ev.code === 1000 ? 'disconnected' : 'error')
      // Code 1000 = normal/intentional close — never show error for it
      if (ev.code !== 1000) {
        onEventRef.current({
          event: 'error',
          message: ev.code === 4004
            ? 'Session not found — the server may have restarted. Refresh to start a new session.'
            : `WebSocket closed (code ${ev.code}). Refresh to reconnect.`,
        })
      }
    }

    // Keepalive ping every 25 s
    const ping = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'ping' }))
      }
    }, 25_000)

    return () => {
      clearInterval(ping)
      ws.close()
      wsRef.current = null
    }
  }, [sessionId])

  const send = useCallback((type: string, payload: Record<string, unknown> = {}): boolean => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type, ...payload }))
      return true
    }
    return false
  }, [])

  return { send, wsStatus }
}
