import { useState, useEffect, useRef } from 'react'

interface UserWorkflow {
  stem:         string
  display_name: string
  type:         't2i' | 'i2v' | 'unknown'
  has_nodemap:  boolean
}

interface Props {
  onBack: () => void
}

export function WorkflowManager({ onBack }: Props) {
  const [workflows,    setWorkflows]    = useState<UserWorkflow[]>([])
  const [loading,      setLoading]      = useState(true)
  const [uploading,    setUploading]    = useState(false)
  const [deleting,     setDeleting]     = useState<string | null>(null)
  const [error,        setError]        = useState<string | null>(null)
  const [uploadResult, setUploadResult] = useState<string | null>(null)

  const [displayName,  setDisplayName]  = useState('')
  const [wfType,       setWfType]       = useState<'t2i' | 'i2v'>('t2i')
  const [file,         setFile]         = useState<File | null>(null)

  const fileRef = useRef<HTMLInputElement>(null)

  async function loadWorkflows() {
    setLoading(true)
    try {
      const r = await fetch('/workflows/user')
      const d = await r.json()
      setWorkflows(d.workflows ?? [])
    } catch {
      setError('Failed to load workflows.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { loadWorkflows() }, [])

  async function handleUpload() {
    if (!file || !displayName.trim()) return
    setUploading(true)
    setError(null)
    setUploadResult(null)
    try {
      const form = new FormData()
      form.append('workflow', file)
      form.append('display_name', displayName.trim())
      form.append('wf_type', wfType)
      const r = await fetch('/workflows/user', { method: 'POST', body: form })
      if (!r.ok) {
        const body = await r.json().catch(() => ({ detail: r.statusText }))
        throw new Error((body as { detail?: string }).detail || `Error ${r.status}`)
      }
      const d = await r.json()
      setUploadResult(d.stem)
      setFile(null)
      setDisplayName('')
      if (fileRef.current) fileRef.current.value = ''
      await loadWorkflows()
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setUploading(false)
    }
  }

  async function handleDelete(stem: string) {
    setDeleting(stem)
    setError(null)
    try {
      const r = await fetch(`/workflows/user/${encodeURIComponent(stem)}`, { method: 'DELETE' })
      if (!r.ok) throw new Error(`Delete failed: ${r.status}`)
      await loadWorkflows()
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setDeleting(null)
    }
  }

  const typeLabel = (t: string) => t === 't2i' ? 'Image (T2I)' : t === 'i2v' ? 'Video (I2V)' : t

  return (
    <div className="setup-check">
      <div className="setup-check-header">
        <button className="back-btn" onClick={onBack}>← Back</button>
        <h2>Manage Workflows</h2>
      </div>

      <p className="lyrics-label-hint" style={{ marginBottom: '1.2rem' }}>
        Upload ComfyUI API-format workflow JSON files. FADE auto-generates the node map
        from nodes titled <strong>FADE: Positive Prompt</strong>, <strong>FADE: Seed</strong>, etc.
        See the example stubs in <code>backend/comfyui/workflows/user/</code> for required node titles.
      </p>

      {/* Upload form */}
      <div className="setup-section">
        <h3>Add Workflow</h3>

        <div className="lyrics-field">
          <label className="lyrics-label">Display name</label>
          <input
            className="project-name-input"
            type="text"
            placeholder="My Custom Workflow"
            value={displayName}
            onChange={e => setDisplayName(e.target.value)}
            disabled={uploading}
          />
        </div>

        <div className="lyrics-field">
          <label className="lyrics-label">Type</label>
          <div className="orientation-toggle">
            {(['t2i', 'i2v'] as const).map(t => (
              <button
                key={t}
                type="button"
                className={`orientation-btn ${wfType === t ? 'orientation-btn--active' : ''}`}
                onClick={() => setWfType(t)}
                disabled={uploading}
              >
                {typeLabel(t)}
              </button>
            ))}
          </div>
        </div>

        <div className="lyrics-field">
          <label className="lyrics-label">Workflow JSON (API format)</label>
          <div className="setup-dir-row">
            <span className="project-name-input" style={{ opacity: 0.6, cursor: 'default' }}>
              {file ? file.name : 'No file selected'}
            </span>
            <button
              className="btn btn--secondary"
              onClick={() => fileRef.current?.click()}
              disabled={uploading}
            >
              Browse…
            </button>
          </div>
          <input
            ref={fileRef}
            type="file"
            accept=".json"
            style={{ display: 'none' }}
            onChange={e => { const f = e.target.files?.[0]; if (f) setFile(f) }}
          />
          <span className="lyrics-label-hint">Export from ComfyUI: Settings → Enable Dev Mode → Save (API Format)</span>
        </div>

        <div className="setup-actions">
          <button
            className="btn btn--primary"
            onClick={handleUpload}
            disabled={uploading || !file || !displayName.trim()}
          >
            {uploading ? <><span className="spinner-inline" /> Uploading…</> : 'Install'}
          </button>
        </div>

        {error && (
          <pre className="upload-error" style={{ whiteSpace: 'pre-wrap', fontFamily: 'inherit' }}>
            {error}
          </pre>
        )}

        {uploadResult && (
          <div className="setup-banner setup-banner--ok">
            Installed "{uploadResult}" — all required nodes found.
          </div>
        )}
      </div>

      {/* Installed workflows */}
      <div className="setup-section">
        <h3>Installed</h3>
        {loading && <p className="lyrics-label-hint">Loading…</p>}
        {!loading && workflows.length === 0 && (
          <p className="lyrics-label-hint">No user workflows installed.</p>
        )}
        {workflows.map(wf => (
          <div key={wf.stem} className="setup-row">
            <span className="setup-row-icon">{wf.has_nodemap ? '✓' : '⚠'}</span>
            <span className="setup-row-label">{wf.display_name}</span>
            <span className="setup-row-sub">{typeLabel(wf.type)}</span>
            {!wf.has_nodemap && (
              <span className="setup-row-sub" style={{ color: 'var(--c-warn, orange)' }}>no nodemap</span>
            )}
            <button
              className="btn btn--secondary"
              style={{ marginLeft: 'auto', padding: '0.1rem 0.6rem', fontSize: '0.78rem' }}
              onClick={() => handleDelete(wf.stem)}
              disabled={deleting === wf.stem}
            >
              {deleting === wf.stem ? <span className="spinner-inline" /> : 'Remove'}
            </button>
          </div>
        ))}
      </div>
    </div>
  )
}
