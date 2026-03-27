import { useState, useEffect } from 'react'

interface CheckEntry {
  ok:        boolean
  label?:    string
  file?:     string
  workflow?: string
  node?:     string
  package?:  string
  url?:      string
  model?:    string
  bundled?:  boolean
}

interface ValidateResult {
  all_ok:    boolean
  model_dir: string
  models:    Record<string, CheckEntry>
  nodes:     Record<string, CheckEntry>
  llm:       Record<string, CheckEntry>
}

interface Props {
  onBack: () => void
}

function Row({ entry, name, nodeKey, onInstalled }: {
  entry: CheckEntry; name: string; nodeKey?: string; onInstalled?: () => void
}) {
  const [installing, setInstalling] = useState(false)
  const label = entry.label ?? entry.file ?? entry.node ?? name
  const sub   = entry.workflow ?? entry.package ?? entry.model ?? ''

  async function install() {
    if (!nodeKey) return
    setInstalling(true)
    try {
      const r = await fetch(`/setup/install-node/${nodeKey}`, { method: 'POST' })
      if (!r.ok) throw new Error(await r.text())
      onInstalled?.()
    } catch (e) {
      alert(`Install failed: ${e}`)
    } finally {
      setInstalling(false)
    }
  }

  return (
    <div className={`setup-row ${entry.ok ? 'setup-row--ok' : 'setup-row--fail'}`}>
      <span className="setup-row-icon">{entry.ok ? '✓' : '✗'}</span>
      <span className="setup-row-label">{label}</span>
      {sub && <span className="setup-row-sub">{sub}</span>}
      {!entry.ok && entry.bundled && (
        <button className="setup-row-link" onClick={install} disabled={installing}>
          {installing ? 'installing…' : 'install'}
        </button>
      )}
      {!entry.ok && !entry.bundled && entry.url && (
        <a className="setup-row-link" href={entry.url} target="_blank" rel="noreferrer">
          install
        </a>
      )}
    </div>
  )
}

function DirInput({
  label, hint, value, onChange, disabled,
}: {
  label: string; hint?: string; value: string
  onChange: (v: string) => void; disabled: boolean
}) {
  const [browsing, setBrowsing] = useState(false)

  async function browse() {
    setBrowsing(true)
    try {
      const r = await fetch('/setup/browse')
      const d = await r.json()
      if (d.path) onChange(d.path)
    } finally {
      setBrowsing(false)
    }
  }

  return (
    <div className="lyrics-field">
      <label className="lyrics-label">{label}</label>
      <div className="setup-dir-row">
        <input
          className="project-name-input"
          type="text"
          value={value}
          onChange={e => onChange(e.target.value)}
          disabled={disabled}
        />
        <button className="btn btn--secondary" onClick={browse} disabled={disabled || browsing}>
          {browsing ? <span className="spinner-inline" /> : 'Browse…'}
        </button>
      </div>
      {hint && <span className="lyrics-label-hint">{hint}</span>}
    </div>
  )
}

export function SetupCheck({ onBack }: Props) {
  const [comfyuiDir, setComfyuiDir] = useState('')
  const [modelDir,   setModelDir]   = useState('')
  const [result,     setResult]     = useState<ValidateResult | null>(null)
  const [loading,    setLoading]    = useState(false)
  const [saving,     setSaving]     = useState(false)
  const [saveMsg,    setSaveMsg]    = useState<string | null>(null)
  const [error,      setError]      = useState<string | null>(null)

  useEffect(() => {
    fetch('/setup/config')
      .then(r => r.json())
      .then(d => {
        setComfyuiDir(d.comfyui_dir ?? '')
        setModelDir(d.model_dir ?? '')
      })
      .catch(() => {})
  }, [])

  async function runCheck() {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const params = new URLSearchParams()
      if (comfyuiDir.trim()) params.set('comfyui_dir', comfyuiDir.trim())
      if (modelDir.trim())   params.set('model_dir',   modelDir.trim())
      const r = await fetch(`/setup/validate?${params}`)
      if (!r.ok) throw new Error(`Server error ${r.status}`)
      setResult(await r.json())
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  async function savePaths() {
    setSaving(true)
    setSaveMsg(null)
    try {
      await fetch('/setup/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ comfyui_dir: comfyuiDir.trim(), model_dir: modelDir.trim() }),
      })
      setSaveMsg('Saved to .env — restart FADE to apply.')
    } catch {
      setSaveMsg('Save failed.')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="setup-check">
      <div className="setup-check-header">
        <button className="back-btn" onClick={onBack}>← Back</button>
        <h2>Setup Verification</h2>
      </div>

      <DirInput
        label="ComfyUI folder"
        hint="Root ComfyUI installation directory."
        value={comfyuiDir}
        onChange={setComfyuiDir}
        disabled={loading || saving}
      />

      <DirInput
        label="Models folder"
        hint="Where your model files live — may differ from the ComfyUI folder."
        value={modelDir}
        onChange={setModelDir}
        disabled={loading || saving}
      />

      <div className="setup-actions">
        <button className="btn btn--primary" onClick={runCheck} disabled={loading || saving}>
          {loading ? <><span className="spinner-inline" /> Checking…</> : 'Check'}
        </button>
        <button className="btn btn--secondary" onClick={savePaths} disabled={loading || saving}>
          {saving ? <span className="spinner-inline" /> : 'Save to .env'}
        </button>
        {saveMsg && <span className="setup-save-msg">{saveMsg}</span>}
      </div>

      {error && <p className="upload-error">{error}</p>}

      {result && (
        <>
          <div className={`setup-banner ${result.all_ok ? 'setup-banner--ok' : 'setup-banner--fail'}`}>
            {result.all_ok ? "All checks passed — you're good to go." : 'Some checks failed. See below.'}
          </div>

          <section className="setup-section">
            <h3>LLM</h3>
            {Object.entries(result.llm).map(([k, v]) => <Row key={k} name={k} entry={v} />)}
          </section>

          <section className="setup-section">
            <h3>Models</h3>
            {Object.entries(result.models).map(([k, v]) => <Row key={k} name={k} entry={v} />)}
          </section>

          <section className="setup-section">
            <h3>Custom Nodes</h3>
            {Object.entries(result.nodes).map(([k, v]) => (
              <Row key={k} name={k} entry={v}
                nodeKey={v.bundled ? k : undefined}
                onInstalled={runCheck}
              />
            ))}
          </section>
        </>
      )}
    </div>
  )
}
