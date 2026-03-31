import { useState } from 'react'
import { Server, Cpu, HardDrive, MemoryStick, Volume2, RefreshCw, Power, Upload, AlertCircle, ChevronDown, ChevronRight, Play } from 'lucide-react'
import { useRunners, useUpdateRunner, useAgentTargetVersion, useSetAgentTargetVersion, useRunnerStatus, useTriggerRunnerUpdate } from '../hooks/useBackend'
import { StatusDot } from '../components/StatusDot'
import type { Runner } from '../types'

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
  const triggerUpdate = useTriggerRunnerUpdate()
  const [updateVersion, setUpdateVersion] = useState('')
  const caps = runner.capabilities
  const s = status.data

  const isOutdated = target && caps.agent_version && caps.agent_version !== target

  return (
    <div className="border-t border-gray-800 pt-3 mt-3 space-y-4">
      {/* Resource bars from live status */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {/* VRAM */}
        {caps.gpu_vram_total_bytes ? (
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

        {/* Memory — from live status */}
        {s?.mem_total_gb ? (
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
      </div>

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

      {/* Address */}
      <div className="text-xs text-gray-600 font-mono truncate">{runner.address}</div>
    </div>
  )
}

export function Runners() {
  const runners = useRunners()
  const update = useUpdateRunner()
  const targetVersion = useAgentTargetVersion()
  const setTarget = useSetAgentTargetVersion()
  const [versionInput, setVersionInput] = useState('')
  const [expandedId, setExpandedId] = useState<number | null>(null)
  const list = runners.data ?? []
  const target = targetVersion.data?.target_version || ''

  const outdatedRunners = list.filter((r: Runner) =>
    target && r.capabilities.agent_version && r.capabilities.agent_version !== target
  )

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Server className="w-4 h-4 text-brand-400" />
          <h1 className="text-base font-semibold text-gray-200">Runners</h1>
          <span className="text-xs text-gray-500">{list.length} active</span>
        </div>
        {runners.isFetching && <RefreshCw className="w-3 h-3 text-gray-600 animate-spin" />}
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
        <div className="py-12 text-center text-gray-600 text-sm">No active runners</div>
      ) : (
        <div className="space-y-3">
          {list.map((runner: Runner) => {
            const online = isOnline(runner.last_seen)
            const enabled = runner.enabled !== false
            const caps = runner.capabilities
            const isGpu = !!caps.gpu_vram_total_bytes
            const isTts = !!caps.tts
            const expanded = expandedId === runner.id
            const isOutdated = target && caps.agent_version && caps.agent_version !== target

            return (
              <div
                key={runner.id}
                className={`bg-gray-900 border rounded-xl transition-opacity ${
                  enabled ? 'border-gray-800' : 'border-gray-800/50 opacity-50'
                }`}
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
                      <span className="text-[10px] text-gray-500 font-mono">{caps.agent_version}</span>
                    )}
                    {isOutdated && (
                      <span className="text-[10px] bg-amber-900/40 text-amber-400 px-1.5 py-0.5 rounded">Update pending</span>
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
            )
          })}
        </div>
      )}
    </div>
  )
}
