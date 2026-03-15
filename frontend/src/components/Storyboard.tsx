import type { SceneData } from '../types'
import { SceneCard } from './SceneCard'

interface Props {
  scenes:           Map<number, SceneData>
  sessionId:        string
  orientation:      string
  genBusy:          boolean
  showGenPrompts:   boolean
  showGenImages:    boolean
  showGenVideos:    boolean
  showExport:       boolean
  generatingImages: boolean
  generatingVideos: boolean
  exportPath:       string | null
  onGeneratePrompts: () => void
  onGenerateImages: () => void
  onGenerateVideos: () => void
  onExport:         () => void
  onApproveImage:   (index: number) => void
  onApproveVideo:   (index: number) => void
  onUnapproveImage: (index: number) => void
  onUnapproveVideo: (index: number) => void
}

export function Storyboard({
  scenes, sessionId, orientation, genBusy,
  showGenPrompts, showGenImages, showGenVideos, showExport,
  generatingImages, generatingVideos, exportPath,
  onGeneratePrompts, onGenerateImages, onGenerateVideos, onExport,
  onApproveImage, onApproveVideo, onUnapproveImage, onUnapproveVideo,
}: Props) {
  const hasBanner = showGenPrompts || showGenImages || showGenVideos || showExport || generatingImages || generatingVideos || exportPath

  return (
    <div className="storyboard-panel">

      {/* ── Phase action banner ───────────────────────────────────────────── */}
      {hasBanner && (
        <div className="gen-banner">
          {generatingImages && (
            <div className="gen-banner-running">
              <span className="spinner" />
              <span>Generating images…</span>
            </div>
          )}
          {generatingVideos && (
            <div className="gen-banner-running">
              <span className="spinner" />
              <span>Generating videos…</span>
            </div>
          )}
          {showGenPrompts && (
            <button
              className="gen-btn gen-btn--prompts"
              disabled={genBusy}
              onClick={onGeneratePrompts}
            >
              Generate Prompts →
            </button>
          )}
          {showGenImages && (
            <button
              className="gen-btn gen-btn--images"
              disabled={genBusy}
              onClick={onGenerateImages}
            >
              Generate Images →
            </button>
          )}
          {showGenVideos && (
            <button
              className="gen-btn gen-btn--videos"
              disabled={genBusy}
              onClick={onGenerateVideos}
            >
              Generate Videos →
            </button>
          )}
          {showExport && (
            <button
              className="gen-btn gen-btn--export"
              disabled={genBusy}
              onClick={onExport}
            >
              Export Final Video →
            </button>
          )}
          {exportPath && (
            <a
              href={`/sessions/${sessionId}/export/download`}
              className="gen-btn gen-btn--download"
              download="final.mp4"
            >
              ↓ Download final.mp4
            </a>
          )}
        </div>
      )}

      {/* ── Scene cards ───────────────────────────────────────────────────── */}
      {scenes.size === 0 ? (
        <div className="storyboard-empty">
          <p>Scene cards will appear here as the director plans the video.</p>
        </div>
      ) : (
        <div className="storyboard">
          {[...scenes.entries()].sort(([a], [b]) => a - b).map(([idx, scene]) => (
            <SceneCard
              key={idx}
              index={idx}
              scene={scene}
              sessionId={sessionId}
              orientation={orientation}
              onApproveImage={onApproveImage}
              onApproveVideo={onApproveVideo}
              onUnapproveImage={onUnapproveImage}
              onUnapproveVideo={onUnapproveVideo}
            />
          ))}
        </div>
      )}
    </div>
  )
}
