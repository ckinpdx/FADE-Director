import { useState, useEffect } from 'react'

interface Project {
  dir_name:     string
  project_name: string
  session_id:   string
  phase:        string
  n_scenes:     number
  save_path:    string
}

interface SongSession {
  session_id: string
  created_at: string
  caption:    string
  take_count: number
}

interface Props {
  onMakeVideo:       () => void
  onWriteSong:       () => void
  onMakeSong:        () => void
  onVerifySetup:     () => void
  onManageWorkflows: () => void
  onResume:          (sessionId: string, name: string, savePath: string) => void
  onResumeSong:      (sessionId: string) => void
}

function phaseLabel(phase: string, n: number): string {
  switch (phase) {
    case 'upload':      return 'Uploading'
    case 'analysis':    return 'Analysing'
    case 'planning':    return `Planning`
    case 'style_bible': return 'Style bible'
    case 'prompts':     return 'Writing prompts'
    case 'images':      return `Images (${n} scenes)`
    case 'videos':      return `Videos (${n} scenes)`
    case 'done':        return 'Done'
    default:            return phase
  }
}

export function HomePage({ onMakeVideo, onWriteSong, onMakeSong, onVerifySetup, onManageWorkflows, onResume, onResumeSong }: Props) {
  const [projects,  setProjects]  = useState<Project[]>([])
  const [songs,     setSongs]     = useState<SongSession[]>([])
  const [resuming,  setResuming]  = useState<string | null>(null)
  const [resumeErr, setResumeErr] = useState<string | null>(null)

  useEffect(() => {
    fetch('/projects')
      .then(r => r.json())
      .then(d => setProjects(d.projects ?? []))
      .catch(() => {})
    fetch('/acestep/sessions')
      .then(r => r.json())
      .then(d => setSongs(d.sessions ?? []))
      .catch(() => {})
  }, [])

  async function handleResume(proj: Project) {
    setResuming(proj.dir_name)
    setResumeErr(null)
    try {
      const r = await fetch(`/projects/${proj.dir_name}/resume`, { method: 'POST' })
      if (!r.ok) throw new Error(`Resume failed: ${r.status}`)
      const { session_id, save_path } = await r.json()
      onResume(session_id, proj.project_name, save_path ?? proj.save_path)
    } catch (e: unknown) {
      setResumeErr(e instanceof Error ? e.message : String(e))
    } finally {
      setResuming(null)
    }
  }

  return (
    <div className="home-screen">
      <div className="home-hero">
        <h1 className="app-title">FADE</h1>
        <p className="app-acronym">Film Automated Direction Engine</p>
        <p className="app-subtitle">An AI director for your music.</p>
      </div>

      <div className="mode-cards">
        {/* Write a Song — Suno prompt assistant */}
        <button className="mode-card mode-card--active" onClick={onWriteSong}>
          <div className="mode-card-icon">✍️</div>
          <div className="mode-card-body">
            <div className="mode-card-title">Write a Song</div>
            <div className="mode-card-desc">
              Craft a prompt package for Suno (style tags + metatag lyrics) or
              ACE-Step 1.5 (caption + structured lyrics + generation settings).
            </div>
          </div>
        </button>

        {/* Make a Song — ACEStep local generator */}
        <button className="mode-card mode-card--active" onClick={onMakeSong}>
          <div className="mode-card-icon">🎵</div>
          <div className="mode-card-body">
            <div className="mode-card-title">Make a Song</div>
            <div className="mode-card-desc">
              Generate music locally with ACE-Step 1.5. The agent interviews you,
              crafts the prompt, and runs inference on-device.
            </div>
          </div>
        </button>

        {/* Make a Video */}
        <button className="mode-card mode-card--active" onClick={onMakeVideo}>
          <div className="mode-card-icon">🎬</div>
          <div className="mode-card-body">
            <div className="mode-card-title">Make a Video</div>
            <div className="mode-card-desc">
              Upload a song. The director plans scenes, writes prompts, and generates
              your music video with LTX-2 + HuMo.
            </div>
          </div>
        </button>

        {/* Verify Setup */}
        <button className="mode-card mode-card--active" onClick={onVerifySetup}>
          <div className="mode-card-icon">🔧</div>
          <div className="mode-card-body">
            <div className="mode-card-title">Verify Setup</div>
            <div className="mode-card-desc">
              Check that all models, custom nodes, and LLM endpoints are present
              and reachable.
            </div>
          </div>
        </button>

        {/* Manage Workflows */}
        <button className="mode-card mode-card--active" onClick={onManageWorkflows}>
          <div className="mode-card-icon">⚙️</div>
          <div className="mode-card-body">
            <div className="mode-card-title">Manage Workflows</div>
            <div className="mode-card-desc">
              Install custom ComfyUI workflows for image or video generation.
              FADE auto-generates the node map from tagged nodes.
            </div>
          </div>
        </button>
      </div>

      {/* Recent songs */}
      {songs.length > 0 && (
        <div className="home-projects">
          <p className="projects-list-heading">Recent songs</p>
          <div className="projects-list">
            {songs.map(s => (
              <button
                key={s.session_id}
                className="project-row"
                onClick={() => onResumeSong(s.session_id)}
              >
                <span className="project-row-name">{s.caption}</span>
                <span className="project-row-state">{s.take_count} take{s.take_count !== 1 ? 's' : ''}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Recent projects */}
      {projects.length > 0 && (
        <div className="home-projects">
          <p className="projects-list-heading">Recent projects</p>
          {resumeErr && <p className="upload-error">{resumeErr}</p>}
          <div className="projects-list">
            {projects.map(proj => (
              <button
                key={proj.dir_name}
                className="project-row"
                onClick={() => handleResume(proj)}
                disabled={resuming !== null}
              >
                <span className="project-row-name">{proj.project_name}</span>
                <span className="project-row-state">{phaseLabel(proj.phase, proj.n_scenes)}</span>
                {resuming === proj.dir_name && <span className="spinner-inline" />}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
