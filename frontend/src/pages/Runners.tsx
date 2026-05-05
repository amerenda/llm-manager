import { Fragment, useEffect, useState } from 'react'
import { Server, Cpu, HardDrive, MemoryStick, Volume2, RefreshCw, Power, Upload, AlertCircle, ChevronDown, ChevronRight, Play, Loader2, Zap, Settings2, CheckCircle2, Trash2 } from 'lucide-react'
import { useRunners, useUpdateRunner, useAgentTargetVersion, useSetAgentTargetVersion, useRunnerStatus, useTriggerRunnerUpdate, useFlushRunnerVram, useRestartOllama, useOllamaSettings, useUpdateOllamaSettings, useOllamaVersion, useUpgradeOllama, useDeleteRunner, useDeleteStaleRunners } from '../hooks/useBackend'
import { StatusDot } from '../components/StatusDot'
import type { Runner } from '../types'
import { agentVersionsEquivalent } from '../utils/agentVersion'
import { isRunnerInSchedulerPool } from '../utils/runnerActive'

function relativeTime(iso: string | null): string {
  if (!iso) return 'Never'
  const diffMs = Date.now() - new Date(iso).getTime()
  const diffSec = Math.floor(diffMs / 1000)
  if (diffSec < 60) return `${diffSec}s ago`
  const diffMin = Math.floor(diffSec / 60)
  if (diffMin < 60) return `${diffMin}m ago`
  return `${Math.floor(diffMin / 60)}h ago`
}

function isOnline(lastSeen: string | null): boolean {
  if (!lastSeen) return false
  return Date.now() - new Date(lastSeen).getTime() < 2 * 60 * 1000
}

function fmtBytes(n: number): string {
  if (n >= 1024 ** 4) return `${(n / 1024 ** 4).toFixed(1)} TB`
  return `${(n / 1024 ** 3).toFixed(1)} GB`
}

function ProgressBar({ pct, thresholds }: { pct: number; thresholds?: { red: number; yellow: number } }) {
  const t = thresholds ?? { red: 80, yellow: 50 }
  const color = pct > t.red ? 'bg-red-500' : pct > t.yellow ? 'bg-yellow-500' : 'bg-brand-500'
  return (
    <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
      <div className={`h-full rounded-full transition-all ${color}`} style={{ width: `${Math.min(pct, 100)}%` }} />
    </div>
  )
}

function RunnerDetail({ runner, target }: { runner: Runner; target: string }) {
  const status = useRunnerStatus(runner.id)
  const update = useUpdateRunner()
  const delRunner = useDeleteRunner()
  const triggerUpdate = useTriggerRunnerUpdate()
  const flushVram = useFlushRunnerVram()
  const restartOllama = useRestartOllama()
  const ollamaVersion = useOllamaVersion(runner.id)
  const upgradeOllama = useUpgradeOllama()
  const [updateVersion, setUpdateVersion] = useState('')
  const [ollamaUpgradeTag, setOllamaUpgradeTag] = useState('')
  const [flushMsg, setFlushMsg] = useState<string | null>(null)
  const caps = runner.capabilities
  const s = status.data
  const isGpu = !!caps.gpu_vram_total_bytes
  const isUnifiedMem = caps.gpu_vendor === 'unified'
  const unifiedTotalBytes =
    s?.mem_total_gb != null && s.mem_total_gb > 0
      ? s.mem_total_gb * 1e9
      : (caps.gpu_vram_total_bytes ?? 0)
  const unifiedUsedBytes =
    s?.mem_total_gb != null && s.mem_total_gb > 0
      ? (s.mem_used_gb ?? 0) * 1e9
      : (caps.gpu_vram_used_bytes ?? 0)

  const isOutdated = Boolean(
    target && caps.agent_version && !agentVersionsEquivalent(caps.agent_version, target),
  )

  return (
    <div className="border-t border-gray-800 pt-3 mt-3 space-y-4">
      {/* Resource bars from live status */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {/* VRAM or unified memory (same pool as RAM on Apple Silicon) */}
        {isUnifiedMem && caps.gpu_vram_total_bytes ? (
          <div className="space-y-1">
            <div className="flex justify-between text-xs text-gray-500">
              <span className="flex items-center gap-1"><MemoryStick className="w-3 h-3" />Unified memory</span>
              <span>{fmtBytes(unifiedUsedBytes)} / {fmtBytes(unifiedTotalBytes)}</span>
            </div>
            <ProgressBar
              pct={unifiedTotalBytes > 0 ? Math.round((unifiedUsedBytes / unifiedTotalBytes) * 100) : 0}
              thresholds={{ red: 90, yellow: 75 }}
            />
          </div>
        ) : caps.gpu_vram_total_bytes ? (
          <div className="space-y-1">
            <div className="flex justify-between text-xs text-gray-500">
              <span className="flex items-center gap-1"><Cpu className="w-3 h-3" />VRAM</span>
              <span>{fmtBytes(caps.gpu_vram_used_bytes ?? 0)} / {fmtBytes(caps.gpu_vram_total_bytes)}</span>
            </div>
            <ProgressBar pct={Math.round(((caps.gpu_vram_used_bytes ?? 0) / caps.gpu_vram_total_bytes) * 100)} />
          </div>
        ) : null}

        {/* Disk */}
        {caps.disk_total_bytes ? (
          <div className="space-y-1">
            <div className="flex justify-between text-xs text-gray-500">
              <span className="flex items-center gap-1"><HardDrive className="w-3 h-3" />Disk</span>
              <span>{fmtBytes(caps.disk_free_bytes ?? 0)} free / {fmtBytes(caps.disk_total_bytes)}</span>
            </div>
            <ProgressBar
              pct={Math.round(((caps.disk_used_bytes ?? 0) / caps.disk_total_bytes) * 100)}
              thresholds={{ red: 95, yellow: 85 }}
            />
          </div>
        ) : null}

        {/* CPU — from live status */}
        {s?.cpu_pct !== undefined && (
          <div className="space-y-1">
            <div className="flex justify-between text-xs text-gray-500">
              <span className="flex items-center gap-1"><Cpu className="w-3 h-3" />CPU</span>
              <span>{s.cpu_pct.toFixed(0)}%</span>
            </div>
            <ProgressBar pct={s.cpu_pct} thresholds={{ red: 90, yellow: 70 }} />
          </div>
        )}

        {/* System RAM — discrete-GPU runners only (unified pool is shown above) */}
        {!isUnifiedMem && s?.mem_total_gb ? (
          <div className="space-y-1">
            <div className="flex justify-between text-xs text-gray-500">
              <span className="flex items-center gap-1"><MemoryStick className="w-3 h-3" />Memory</span>
              <span>{s.mem_used_gb?.toFixed(1)} / {s.mem_total_gb.toFixed(1)} GB</span>
            </div>
            <ProgressBar
              pct={Math.round(((s.mem_used_gb ?? 0) / s.mem_total_gb) * 100)}
              thresholds={{ red: 90, yellow: 75 }}
            />
          </div>
        ) : null}
      </div>

      {/* Loaded models */}
      {(caps.loaded_models?.length ?? 0) > 0 && (
        <div className="space-y-1">
          <p className="text-xs text-gray-500">Loaded models</p>
          <div className="flex flex-wrap gap-1">
            {(caps.loaded_models ?? []).map((m: string) => (
              <span key={m} className="text-[10px] bg-gray-800 text-gray-400 px-2 py-0.5 rounded-full font-mono">{m}</span>
            ))}
          </div>
        </div>
      )}

      {/* TTS voices */}
      {caps.tts && (caps.voices?.length ?? 0) > 0 && (
        <div className="flex items-center gap-1 text-xs text-gray-500">
          <Volume2 className="w-3 h-3" />
          <span>{caps.voices?.length} voice{caps.voices?.length !== 1 ? 's' : ''}</span>
          {caps.default_voice && <span className="text-gray-600">· default: {caps.default_voice}</span>}
        </div>
      )}

      {/* Settings row */}
      <div className="flex flex-wrap items-center gap-4 pt-1 border-t border-gray-800">
        {/* Auto-update toggle */}
        <label className="flex items-center gap-1.5 cursor-pointer">
          <input
            type="checkbox"
            checked={runner.auto_update ?? false}
            onChange={() => update.mutate({ runnerId: runner.id, auto_update: !runner.auto_update })}
            disabled={update.isPending}
            className="w-3.5 h-3.5 rounded border-gray-600 bg-gray-800 text-brand-500 focus:ring-brand-500 focus:ring-offset-0 cursor-pointer"
          />
          <span className="text-xs text-gray-400">Auto-update</span>
        </label>

        {/* Enable/disable */}
        <label className="flex items-center gap-1.5 cursor-pointer">
          <input
            type="checkbox"
            checked={runner.enabled !== false}
            onChange={() => update.mutate({ runnerId: runner.id, enabled: !runner.enabled })}
            disabled={update.isPending}
            className="w-3.5 h-3.5 rounded border-gray-600 bg-gray-800 text-green-500 focus:ring-green-500 focus:ring-offset-0 cursor-pointer"
          />
          <span className="text-xs text-gray-400">Enabled</span>
        </label>

        {/* Drain — stop scheduling new jobs; let in-flight finish */}
        {isGpu && (
          <label className="flex items-center gap-1.5 cursor-pointer" title="Stop assigning new jobs to this runner. The current in-flight job finishes normally. Use before restarting Ollama or doing maintenance.">
            <input
              type="checkbox"
              checked={runner.draining === true}
              onChange={() => update.mutate({ runnerId: runner.id, draining: !runner.draining })}
              disabled={update.isPending}
              className="w-3.5 h-3.5 rounded border-gray-600 bg-gray-800 text-amber-500 focus:ring-amber-500 focus:ring-offset-0 cursor-pointer"
            />
            <span className="text-xs text-gray-400">
              Drain
              {runner.draining && runner.in_flight_job_id && (
                <span className="ml-1 text-amber-400">(job {runner.in_flight_job_id.slice(0, 8)} running)</span>
              )}
              {runner.draining && !runner.in_flight_job_id && (
                <span className="ml-1 text-amber-400">(drained)</span>
              )}
            </span>
          </label>
        )}

        {/* Flush VRAM + Restart Ollama — GPU runners only */}
        {isGpu && (
          <div className="ml-auto flex items-center gap-1.5">
            <button
              onClick={() => {
                if (window.confirm(`Unload all models from VRAM on ${runner.hostname}?`)) {
                  setFlushMsg(null)
                  flushVram.mutate(runner.id, {
                    onSuccess: (data) => setFlushMsg(data.message),
                    onError: (e) => setFlushMsg(`Error: ${(e as Error).message}`),
                  })
                }
              }}
              disabled={flushVram.isPending}
              className="flex items-center gap-1 text-xs bg-orange-900/30 hover:bg-orange-800/40 text-orange-400 border border-orange-800/50 px-2.5 py-1 rounded-lg transition-colors disabled:opacity-40"
            >
              {flushVram.isPending ? <Loader2 className="w-3 h-3 animate-spin" /> : <Zap className="w-3 h-3" />}
              Flush VRAM
            </button>
            <button
              onClick={() => {
                if (window.confirm(`Restart Ollama on ${runner.hostname}? This will briefly interrupt inference.`)) {
                  setFlushMsg(null)
                  restartOllama.mutate(runner.id, {
                    onSuccess: (data) => setFlushMsg(data.message),
                    onError: (e) => setFlushMsg(`Error: ${(e as Error).message}`),
                  })
                }
              }}
              disabled={restartOllama.isPending}
              className="flex items-center gap-1 text-xs bg-red-900/30 hover:bg-red-800/40 text-red-400 border border-red-800/50 px-2.5 py-1 rounded-lg transition-colors disabled:opacity-40"
            >
              {restartOllama.isPending ? <Loader2 className="w-3 h-3 animate-spin" /> : <Power className="w-3 h-3" />}
              Restart Ollama
            </button>
          </div>
        )}
      </div>
      {flushMsg && (
        <p className="text-xs text-amber-400 mt-1">{flushMsg}</p>
      )}

      {/* Ollama tunables — GPU runners only */}
      {isGpu && <OllamaSettingsPanel runner={runner} />}

      {/* Version management */}
      <div className="border-t border-gray-800 pt-3 space-y-2">
        <p className="text-xs text-gray-500 font-medium uppercase tracking-wide">Agent Version</p>
        <div className="flex items-center gap-2 text-xs">
          <span className="text-gray-400">Current:</span>
          <span className="font-mono text-gray-200">{caps.agent_version || 'unknown'}</span>
          {isOutdated && (
            <>
              <span className="text-gray-600">→</span>
              <span className="font-mono text-amber-400">{target}</span>
              <button
                onClick={() => triggerUpdate.mutate({ runnerId: runner.id })}
                disabled={triggerUpdate.isPending}
                className="flex items-center gap-1 text-xs bg-amber-600 hover:bg-amber-500 disabled:opacity-40 text-white px-2 py-1 rounded-lg transition-colors"
              >
                <Play className="w-3 h-3" />
                Update now
              </button>
            </>
          )}
        </div>
        <div className="flex gap-2 items-center">
          <input
            type="text"
            value={updateVersion}
            onChange={e => setUpdateVersion(e.target.value)}
            placeholder="Specific tag, e.g. sha-abc1234"
            className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-xs text-gray-200 placeholder-gray-600 focus:outline-none focus:border-brand-600"
          />
          <button
            onClick={() => { triggerUpdate.mutate({ runnerId: runner.id, target_version: updateVersion.trim() }); setUpdateVersion('') }}
            disabled={!updateVersion.trim() || triggerUpdate.isPending}
            className="flex items-center gap-1 text-xs bg-brand-600 hover:bg-brand-500 disabled:opacity-40 text-white px-3 py-1.5 rounded-lg transition-colors whitespace-nowrap"
          >
            <Upload className="w-3 h-3" />
            Deploy tag
          </button>
        </div>
        {triggerUpdate.isSuccess && (
          <p className="text-xs text-green-400">Update triggered</p>
        )}
        {triggerUpdate.isError && (
          <p className="text-xs text-red-400">Failed: {(triggerUpdate.error as Error).message}</p>
        )}
      </div>

      {/* GPU stack versions */}
      {isGpu && (
        <div className="border-t border-gray-800 pt-3 space-y-1.5">
          <p className="text-xs text-gray-500 font-medium uppercase tracking-wide">GPU Stack</p>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-gray-400">
            {caps.gpu_vendor === 'nvidia' && (
              <>
                <span>NVIDIA: <span className="font-mono text-gray-200">{caps.nvidia_driver_version || 'unknown'}</span></span>
                <span>CUDA: <span className="font-mono text-gray-200">{caps.cuda_driver_version || 'unknown'}</span></span>
              </>
            )}
            {caps.gpu_vendor === 'amd' && (
              <>
                <span>AMD: <span className="font-mono text-gray-200">{caps.amd_driver_version || 'unknown'}</span></span>
                <span>ROCm: <span className="font-mono text-gray-200">{caps.rocm_version || 'unknown'}</span></span>
              </>
            )}
            {caps.gpu_vendor === 'unified' && (
              <span>Unified memory host (no discrete GPU driver stack)</span>
            )}
          </div>
        </div>
      )}

      {/* Ollama version + upgrade — GPU runners only */}
      {isGpu && (
        <div className="border-t border-gray-800 pt-3 space-y-2">
          <p className="text-xs text-gray-500 font-medium uppercase tracking-wide">Ollama Version</p>
          {ollamaVersion.isLoading && <p className="text-xs text-gray-600">Loading…</p>}
          {ollamaVersion.data && (
            <div className="space-y-1.5">
              <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
                <span className="text-gray-400">Running: <span className="font-mono text-gray-200">{ollamaVersion.data.version ?? 'unknown'}</span></span>
                {ollamaVersion.data.image_tag && (
                  <span className="text-gray-400">Tag: <span className="font-mono text-gray-200">{ollamaVersion.data.image_tag}</span></span>
                )}
                {ollamaVersion.data.commit && (
                  <span className="text-gray-400">Commit: <span className="font-mono text-gray-500">{ollamaVersion.data.commit.slice(0, 12)}</span></span>
                )}
              </div>
              <div className="flex gap-2 items-center">
                <input
                  type="text"
                  value={ollamaUpgradeTag}
                  onChange={e => setOllamaUpgradeTag(e.target.value)}
                  placeholder="e.g. 0.6.5 or 0.6.5-rocm"
                  className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-xs text-gray-200 placeholder-gray-600 focus:outline-none focus:border-brand-600"
                />
                <button
                  onClick={() => {
                    if (!window.confirm(`Upgrade Ollama on ${runner.hostname} to ${ollamaUpgradeTag.trim()}?\n\nThis will pull the image and restart Ollama — ~30s of downtime.`)) return
                    upgradeOllama.mutate(
                      { runnerId: runner.id, tag: ollamaUpgradeTag.trim() },
                      { onSuccess: () => setOllamaUpgradeTag('') }
                    )
                  }}
                  disabled={!ollamaUpgradeTag.trim() || upgradeOllama.isPending}
                  className="flex items-center gap-1 text-xs bg-brand-600 hover:bg-brand-500 disabled:opacity-40 text-white px-3 py-1.5 rounded-lg transition-colors whitespace-nowrap"
                >
                  {upgradeOllama.isPending ? <Loader2 className="w-3 h-3 animate-spin" /> : <Upload className="w-3 h-3" />}
                  Upgrade
                </button>
              </div>
              {upgradeOllama.isSuccess && upgradeOllama.data && (
                <p className="text-xs text-green-400">{upgradeOllama.data.message}{upgradeOllama.data.commit ? ` · ${upgradeOllama.data.commit.slice(0, 12)}` : ''}</p>
              )}
              {upgradeOllama.isError && (
                <p className="text-xs text-red-400">Failed: {(upgradeOllama.error as Error).message}</p>
              )}
            </div>
          )}
        </div>
      )}

      {/* Remove ghost / offline row (same bucket as "stale" in the list header) */}
      {!isRunnerInSchedulerPool(runner) && (
        <div className="border-t border-gray-800 pt-3">
          <button
            type="button"
            onClick={() => {
              if (!window.confirm(
                `Remove runner "${runner.hostname}" (#${runner.id}) from the fleet?\n\n`
                + 'This deletes the database row (disabled, or no heartbeat in ~90s). '
                + 'App allowlists are updated. Cannot be undone.',
              )) return
              delRunner.mutate(runner.id)
            }}
            disabled={delRunner.isPending}
            className="flex items-center gap-1 text-xs bg-red-950/50 hover:bg-red-900/50 text-red-400 border border-red-900/60 px-3 py-1.5 rounded-lg transition-colors disabled:opacity-40"
          >
            {delRunner.isPending ? <Loader2 className="w-3 h-3 animate-spin" /> : <Trash2 className="w-3 h-3" />}
            Remove from fleet
          </button>
          {delRunner.isError && (
            <p className="text-xs text-red-400 mt-1">{(delRunner.error as Error).message}</p>
          )}
        </div>
      )}

      {/* Address */}
      <div className="text-xs text-gray-600 font-mono truncate">{runner.address}</div>
    </div>
  )
}

export function Runners() {
  const runners = useRunners()
  const deleteStale = useDeleteStaleRunners()
  const targetVersion = useAgentTargetVersion()
  const setTarget = useSetAgentTargetVersion()
  const [versionInput, setVersionInput] = useState('')
  const [expandedId, setExpandedId] = useState<number | null>(null)
  const [showStaleRunners, setShowStaleRunners] = useState(false)
  const list = runners.data ?? []
  const target = targetVersion.data?.target_version || ''

  const activeRunners = list.filter(isRunnerInSchedulerPool)
  const staleRunners = list.filter(r => !isRunnerInSchedulerPool(r))
  const bodyRunners = showStaleRunners ? [...activeRunners, ...staleRunners] : activeRunners

  const outdatedRunners = activeRunners.filter((r: Runner) =>
    Boolean(
      target &&
        r.capabilities.agent_version &&
        !agentVersionsEquivalent(r.capabilities.agent_version, target),
    )
  )

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Server className="w-4 h-4 text-brand-400" />
          <h1 className="text-base font-semibold text-gray-200">Runners</h1>
          <span className="text-xs text-gray-500">
            {activeRunners.length} online
            {staleRunners.length > 0 ? ` · ${staleRunners.length} stale` : ''}
          </span>
        </div>
        <div className="flex items-center gap-3">
          {staleRunners.length > 0 && (
            <button
              type="button"
              onClick={() => setShowStaleRunners(v => !v)}
              className="text-xs text-gray-500 hover:text-gray-300 underline-offset-2 hover:underline"
            >
              {showStaleRunners ? 'Hide' : 'Show'} stale runners ({staleRunners.length})
            </button>
          )}
          {staleRunners.length > 0 && (
            <button
              type="button"
              onClick={() => {
                const names = staleRunners.map(r => `${r.hostname} (#${r.id})`).join(', ')
                if (!window.confirm(
                  `Delete all ${staleRunners.length} stale runner(s)?\n\n${names}\n\n`
                  + 'Stale means disabled or no heartbeat in ~90s (same rows as when you expand the stale list). '
                  + 'App allowlists are updated. Cannot be undone.',
                )) return
                deleteStale.mutate()
              }}
              disabled={deleteStale.isPending}
              className="flex items-center gap-1 text-xs text-red-400 hover:text-red-300 border border-red-900/50 hover:border-red-800 rounded-lg px-2 py-1 disabled:opacity-40"
            >
              {deleteStale.isPending ? <Loader2 className="w-3 h-3 animate-spin" /> : <Trash2 className="w-3 h-3" />}
              Delete all stale
            </button>
          )}
          {deleteStale.isError && (
            <span className="text-xs text-red-400 max-w-xs truncate" title={(deleteStale.error as Error).message}>
              {(deleteStale.error as Error).message}
            </span>
          )}
          {runners.isFetching && <RefreshCw className="w-3 h-3 text-gray-600 animate-spin" />}
        </div>
      </div>

      {/* Global agent version control */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 space-y-3">
        <p className="text-xs text-gray-500 font-medium uppercase tracking-wide">Global Target Version</p>
        <div className="flex items-center gap-3">
          <div className="flex-1">
            <p className="text-xs text-gray-400">
              Target: <span className="text-gray-200 font-mono">{target || 'not set'}</span>
            </p>
          </div>
          <div className="flex gap-2">
            <input
              type="text"
              value={versionInput}
              onChange={e => setVersionInput(e.target.value)}
              placeholder="e.g. sha-abc1234"
              className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-xs text-gray-200 placeholder-gray-600 focus:outline-none focus:border-brand-600 w-40"
            />
            <button
              onClick={() => { setTarget.mutate(versionInput.trim()); setVersionInput('') }}
              disabled={!versionInput.trim() || setTarget.isPending}
              className="flex items-center gap-1 text-xs bg-brand-600 hover:bg-brand-500 disabled:opacity-40 text-white px-3 py-1.5 rounded-lg transition-colors"
            >
              <Upload className="w-3 h-3" />
              Set
            </button>
          </div>
        </div>
        {outdatedRunners.length > 0 && (
          <div className="flex items-center gap-1.5 text-xs text-amber-400">
            <AlertCircle className="w-3 h-3" />
            {outdatedRunners.length} runner{outdatedRunners.length > 1 ? 's' : ''} outdated
          </div>
        )}
      </div>

      {runners.isLoading ? (
        <div className="py-12 text-center text-gray-600 text-sm">Loading runners...</div>
      ) : list.length === 0 ? (
        <div className="py-12 text-center text-gray-600 text-sm">No runners registered</div>
      ) : bodyRunners.length === 0 ? (
        <div className="py-12 text-center text-gray-600 text-sm space-y-2">
          <p>No runners with a recent heartbeat (~90s).</p>
          {staleRunners.length > 0 && (
            <p className="text-xs text-gray-500">
              <button type="button" onClick={() => setShowStaleRunners(true)} className="text-brand-400 hover:underline">
                Show {staleRunners.length} stale row{staleRunners.length !== 1 ? 's' : ''}
              </button>
              {' '}from the last 7 days (e.g. old Docker hostnames after RUNNER_HOSTNAME changed).
            </p>
          )}
        </div>
      ) : (
        <div className="space-y-3">
          {bodyRunners.map((runner: Runner, idx: number) => {
            const showStaleHeader =
              showStaleRunners && staleRunners.length > 0 && idx === activeRunners.length
            const online = isOnline(runner.last_seen)
            const enabled = runner.enabled !== false
            const caps = runner.capabilities
            const isGpu = !!caps.gpu_vram_total_bytes
            const isTts = !!caps.tts
            const expanded = expandedId === runner.id
            const isOutdated = Boolean(
    target && caps.agent_version && !agentVersionsEquivalent(caps.agent_version, target),
  )

            return (
              <Fragment key={runner.id}>
                {showStaleHeader && (
                  <p className="text-xs text-gray-500 uppercase tracking-wide pt-2 border-t border-gray-800">
                    Stale / offline (no ~90s heartbeat)
                  </p>
                )}
                <div
                  className={`bg-gray-900 border rounded-xl transition-opacity ${
                    enabled ? 'border-gray-800' : 'border-gray-800/50 opacity-50'
                  } ${!isRunnerInSchedulerPool(runner) ? 'opacity-70' : ''}`}
                >
                {/* Clickable header */}
                <button
                  onClick={() => setExpandedId(expanded ? null : runner.id)}
                  className="w-full p-4 text-left hover:bg-gray-800/30 rounded-xl transition-colors space-y-2"
                >
                  {/* Top row: identity + right-side info */}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 min-w-0">
                      {expanded
                        ? <ChevronDown className="w-3.5 h-3.5 text-gray-500 shrink-0" />
                        : <ChevronRight className="w-3.5 h-3.5 text-gray-500 shrink-0" />
                      }
                      <StatusDot online={online && enabled} />
                      <span className="text-sm font-medium text-gray-200">{runner.hostname}</span>
                      <span className="text-xs text-gray-600 font-mono">#{runner.id}</span>
                      {isOutdated && (
                        <span title={`Agent outdated: ${caps.agent_version} → ${target}`} className="flex items-center">
                          <AlertCircle className="w-3.5 h-3.5 text-amber-400 shrink-0" />
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-3 shrink-0">
                      {!expanded && isGpu && (
                        <span className="text-xs text-gray-500 tabular-nums">
                          {fmtBytes(caps.gpu_vram_used_bytes ?? 0)} / {fmtBytes(caps.gpu_vram_total_bytes ?? 0)}
                        </span>
                      )}
                      <span className="text-xs text-gray-600 tabular-nums">{relativeTime(runner.last_seen)}</span>
                    </div>
                  </div>
                  {/* Bottom row: version + badges */}
                  <div className="flex items-center gap-2 pl-6">
                    {caps.agent_version && (
                      <span className={`text-[10px] font-mono ${isOutdated ? 'text-amber-400' : 'text-gray-500'}`}>
                        {caps.agent_version}
                      </span>
                    )}
                    {isOutdated && (
                      <span className="flex items-center gap-0.5 text-[10px] bg-amber-900/40 text-amber-400 border border-amber-800/50 px-1.5 py-0.5 rounded">
                        <AlertCircle className="w-2.5 h-2.5" />
                        outdated
                      </span>
                    )}
                    <div className="flex gap-1">
                      {isGpu && (
                        <span className="text-[10px] bg-purple-900/50 text-purple-400 border border-purple-800 px-1.5 py-0.5 rounded-full">GPU</span>
                      )}
                      {isTts && (
                        <span className="text-[10px] bg-blue-900/50 text-blue-400 border border-blue-800 px-1.5 py-0.5 rounded-full">TTS</span>
                      )}
                      {!enabled && (
                        <span className="text-[10px] bg-gray-800 text-gray-500 px-1.5 py-0.5 rounded-full">disabled</span>
                      )}
                      {runner.auto_update && (
                        <span className="text-[10px] bg-green-900/50 text-green-400 border border-green-800 px-1.5 py-0.5 rounded-full">auto</span>
                      )}
                    </div>
                  </div>
                </button>

                {/* Expandable detail panel */}
                {expanded && (
                  <div className="px-4 pb-4">
                    <RunnerDetail runner={runner} target={target} />
                  </div>
                )}
              </div>
              </Fragment>
            )
          })}
        </div>
      )}
    </div>
  )
}


// ── Ollama tunables panel ────────────────────────────────────────────────
// Per-runner form that reads /api/llm/runners/{id}/ollama-settings, lets the
// admin edit allow-listed Ollama env vars, and applies them. Applying writes
// ollama.env on the host and recreates the Ollama container — ~5–15s of
// downtime. Strong recommendation to drain the runner first; the panel shows
// a warning when an in-flight job is present.

const OLLAMA_FIELD_HELP: Record<string, { label: string; placeholder: string; hint: string }> = {
  OLLAMA_NUM_CTX:           { label: 'Context window',        placeholder: '2048',             hint: 'Tokens per request. Larger = more VRAM.' },
  OLLAMA_FLASH_ATTENTION:   { label: 'Flash attention',       placeholder: '1 / 0',            hint: 'Usually a large speedup + VRAM reduction on supported GPUs.' },
  OLLAMA_KV_CACHE_TYPE:     { label: 'KV cache quantization', placeholder: 'f16 / q8_0 / q4_0', hint: 'Requires flash attention. q8_0 halves KV VRAM at tiny quality cost.' },
  OLLAMA_NUM_GPU:           { label: 'GPU layers',            placeholder: 'all',              hint: 'Layers offloaded to GPU. 0 = CPU-only. Blank = all that fit.' },
  OLLAMA_NUM_PARALLEL:      { label: 'Parallel slots',        placeholder: '1',                hint: 'Concurrent requests per loaded model.' },
  OLLAMA_MAX_LOADED_MODELS: { label: 'Max loaded models',     placeholder: '1',                hint: 'Keep at 1 — the scheduler does one-model-per-GPU.' },
  OLLAMA_KEEP_ALIVE:        { label: 'Keep-alive',            placeholder: '5m',               hint: 'How long to keep a model in VRAM after last use. -1 = forever.' },
  OLLAMA_MAX_QUEUE:         { label: 'Max queued requests',   placeholder: '512',              hint: 'Past this, Ollama rejects with 503.' },
  OLLAMA_HOST:              { label: 'Listen address',        placeholder: '127.0.0.1:11434',  hint: 'Leave as-is unless you know why.' },
}

function OllamaSettingsPanel({ runner }: { runner: Runner }) {
  const [open, setOpen] = useState(false)
  const { data, isLoading, error, refetch } = useOllamaSettings(runner.id, open)
  const update = useUpdateOllamaSettings()
  const [draft, setDraft] = useState<Record<string, string>>({})
  const [msg, setMsg] = useState<{ kind: 'ok' | 'err'; text: string } | null>(null)

  // Seed the draft from server whenever we open or the server value changes.
  useEffect(() => {
    if (data?.settings) setDraft(data.settings)
  }, [data])

  const dirty = data ? JSON.stringify(draft) !== JSON.stringify(data.settings) : false
  const keys = data ? Object.keys(data.allowlist) : []

  return (
    <div className="border-t border-gray-800 pt-3">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center gap-2 text-xs text-gray-400 hover:text-gray-200"
      >
        {open ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
        <Settings2 className="w-3 h-3" />
        <span className="font-medium uppercase tracking-wide">Ollama Tunables</span>
        {runner.in_flight_job_id && open && (
          <span className="ml-auto text-amber-400 flex items-center gap-1">
            <AlertCircle className="w-3 h-3" /> job running — drain first
          </span>
        )}
      </button>

      {open && (
        <div className="mt-2 space-y-2">
          {isLoading && <p className="text-xs text-gray-500">Loading…</p>}
          {error && (
            <p className="text-xs text-red-400">
              Couldn't load settings: {(error as Error).message}
            </p>
          )}
          {data && (
            <>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                {keys.map(k => {
                  const h = OLLAMA_FIELD_HELP[k] ?? { label: k, placeholder: '', hint: '' }
                  const val = draft[k] ?? ''
                  return (
                    <label key={k} className="flex flex-col gap-0.5">
                      <span className="text-[11px] text-gray-400">
                        {h.label}
                        <span className="ml-2 text-gray-600 font-mono">{k}</span>
                      </span>
                      <input
                        type="text"
                        value={val}
                        placeholder={h.placeholder}
                        onChange={e => setDraft({ ...draft, [k]: e.target.value })}
                        className="bg-gray-950 border border-gray-700 rounded px-2 py-1 text-xs text-gray-200 placeholder-gray-600 focus:outline-none focus:border-brand-600 font-mono"
                      />
                      <span className="text-[10px] text-gray-600">{h.hint}</span>
                    </label>
                  )
                })}
              </div>

              <div className="flex items-center gap-2 pt-1">
                <button
                  onClick={() => {
                    if (runner.in_flight_job_id) {
                      if (!window.confirm(`Job ${runner.in_flight_job_id.slice(0, 8)} is running on ${runner.hostname}. Applying will interrupt it. Continue?`)) return
                    }
                    setMsg(null)
                    update.mutate(
                      { runnerId: runner.id, settings: draft },
                      {
                        onSuccess: (d) => { setMsg({ kind: 'ok', text: d.message }); refetch() },
                        onError: (e) => setMsg({ kind: 'err', text: (e as Error).message }),
                      }
                    )
                  }}
                  disabled={!dirty || update.isPending}
                  className="flex items-center gap-1 text-xs bg-brand-600 hover:bg-brand-500 disabled:opacity-40 text-white px-3 py-1.5 rounded-lg transition-colors"
                >
                  {update.isPending ? <Loader2 className="w-3 h-3 animate-spin" /> : <CheckCircle2 className="w-3 h-3" />}
                  Apply (restart Ollama)
                </button>
                <button
                  onClick={() => data.settings && setDraft(data.settings)}
                  disabled={!dirty || update.isPending}
                  className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-200 disabled:opacity-40 px-2 py-1.5"
                >
                  Reset
                </button>
                {dirty && <span className="text-[11px] text-amber-400">unsaved changes</span>}
              </div>

              {msg && (
                <p className={`text-xs ${msg.kind === 'ok' ? 'text-green-400' : 'text-red-400'}`}>
                  {msg.text}
                </p>
              )}
            </>
          )}
        </div>
      )}
    </div>
  )
}
