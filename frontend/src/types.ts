// ── WebSocket events from the server ──────────────────────────────────────────

export type WsEvent =
  | { event: 'token';          text: string }
  | { event: 'assistant_done' }
  | { event: 'tool_call';      name: string; args: Record<string, unknown> }
  | { event: 'tool_result';    name: string; result: unknown }
  | { event: 'status';         message: string }
  | { event: 'step_done';       message: string }
  | { event: 'analysis_result'; title: string; content: string }
  | { event: 'scene_update';   scene_index: number; scene?: SceneData; fields?: Partial<SceneData> }
  | { event: 'analysis_done';  music_data: { bpm: number; key: string; mode: string; duration: number }; word_count: number; auto_k: number }
  | { event: 'prompts_done';   scenes: number }
  | { event: 'gen_done';       phase: 'images' | 'videos' }
  | { event: 'export_done';    path: string }
  | { event: 'error';          message: string }
  | { event: 'pong' }

// ── Scene ─────────────────────────────────────────────────────────────────────

export interface SceneData {
  start_s:           number
  end_s:             number
  frame_count:       number
  label:             string
  lyrics_full:       string
  lyrics_window:     string
  lyric_theme:       string
  intonation_note:   string
  energy_level:      string
  rationale:         string
  location:          string
  establishing_shot: boolean
  image_prompt:      string
  video_prompt:      string
  seed:              number
  image_path:        string | null
  video_path:        string | null
  image_status:      SceneStatus
  video_status:      SceneStatus
}

export type SceneStatus =
  | 'planned'
  | 'prompts_ready'
  | 'generating'
  | 'done'
  | 'approved'
  | 'failed'

// ── Chat message (client-side) ────────────────────────────────────────────────

export type MessageRole = 'user' | 'assistant' | 'tool' | 'status' | 'step_done' | 'error' | 'analysis_result'

export interface ChatMessage {
  id:      number
  role:    MessageRole
  content: string
  title?:  string   // used by analysis_result
}

// ── Session config ────────────────────────────────────────────────────────────

export interface SessionConfig {
  orientation: string
  width:       number
  height:      number
  fps:         number
}
