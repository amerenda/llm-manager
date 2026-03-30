import { useState } from 'react'
import { Server, Cpu, MemoryStick, Volume2, RefreshCw, Power, Upload, AlertCircle } from 'lucide-react'
import { useRunners, useUpdateRunner, useAgentTargetVersion, useSetAgentTargetVersion } from '../hooks/useBackend'
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

function bytes(n: number): string {
  return `${(n / 1024 ** 3).toFixed(1)} GB`
}

export function Runners() {
  const runners = useRunners()
  const update = useUpdateRunner()
  const targetVersion = useAgentTargetVersion()
  const setTarget = useSetAgentTargetVersion()
  const [versionInput, setVersionInput] = useState('')
  const list = runners.data ?? []
  const target = targetVersion.data?.target_version || ''

  const gpuRunners = list.filter((r: Runner) => r.capabilities.gpu_vram_total_bytes)
  const ttsRunners = list.filter((r: Runner) => r.capabilities.tts)

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

      {/* Agent version control */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 space-y-3">
        <p className="text-xs text-gray-500 font-medium uppercase tracking-wide">Agent Version</p>
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
            {outdatedRunners.length} runner{outdatedRunners.length > 1 ? 's' : ''} outdated — will update on next heartbeat
          </div>
        )}
      </div>

      {runners.isLoading ? (
        <div className="py-12 text-center text-gray-600 text-sm">Loading runners…</div>
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
            const vramPct = isGpu
              ? Math.round(((caps.gpu_vram_used_bytes ?? 0) / (caps.gpu_vram_total_bytes ?? 1)) * 100)
              : null

            return (
              <div
                key={runner.id}
                className={`bg-gray-900 border rounded-xl p-4 space-y-3 transition-opacity ${
                  enabled ? 'border-gray-800' : 'border-gray-800/50 opacity-50'
                }`}
              >
                {/* Header */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <StatusDot online={online && enabled} />
                    <span className="text-sm font-medium text-gray-200">{runner.hostname}</span>
                    <span className="text-xs text-gray-600 font-mono">#{runner.id}</span>
                    {caps.agent_version && (
                      <span className="text-[10px] text-gray-500 font-mono">{caps.agent_version}</span>
                    )}
                    {target && caps.agent_version && caps.agent_version !== target && (
                      <span className="text-[10px] bg-amber-900/40 text-amber-400 px-1.5 py-0.5 rounded">Update pending</span>
                    )}
                    {!caps.agent_version && (
                      <span className="text-[10px] bg-gray-800 text-gray-600 px-1.5 py-0.5 rounded">No version</span>
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
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => update.mutate({ runnerId: runner.id, enabled: !enabled })}
                      disabled={update.isPending}
                      title={enabled ? 'Disable runner' : 'Enable runner'}
                      className={`p-1.5 rounded-lg transition-colors ${
                        enabled
                          ? 'bg-green-900/30 hover:bg-green-900/50 text-green-400'
                          : 'bg-gray-800 hover:bg-gray-700 text-gray-500'
                      }`}
                    >
                      <Power className="w-3.5 h-3.5" />
                    </button>
                    <span className="text-xs text-gray-600 tabular-nums">{relativeTime(runner.last_seen)}</span>
                  </div>
                </div>

                {/* GPU VRAM bar */}
                {isGpu && (
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs text-gray-500">
                      <span className="flex items-center gap-1">
                        <Cpu className="w-3 h-3" />VRAM
                      </span>
                      <span>{bytes(caps.gpu_vram_used_bytes ?? 0)} / {bytes(caps.gpu_vram_total_bytes ?? 0)}</span>
                    </div>
                    <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all ${vramPct! > 80 ? 'bg-red-500' : vramPct! > 50 ? 'bg-yellow-500' : 'bg-brand-500'}`}
                        style={{ width: `${vramPct}%` }}
                      />
                    </div>
                  </div>
                )}

                {/* Loaded models */}
                {isGpu && (caps.loaded_models?.length ?? 0) > 0 && (
                  <div className="flex flex-wrap gap-1">
                    {(caps.loaded_models ?? []).map((m: string) => (
                      <span key={m} className="text-[10px] bg-gray-800 text-gray-400 px-2 py-0.5 rounded-full font-mono">
                        {m}
                      </span>
                    ))}
                  </div>
                )}

                {/* TTS voices */}
                {isTts && (caps.voices?.length ?? 0) > 0 && (
                  <div className="space-y-1">
                    <div className="flex items-center gap-1 text-xs text-gray-500">
                      <Volume2 className="w-3 h-3" />
                      <span>{caps.voices?.length} voice{caps.voices?.length !== 1 ? 's' : ''}</span>
                      {caps.default_voice && (
                        <span className="text-gray-600">· default: {caps.default_voice}</span>
                      )}
                    </div>
                  </div>
                )}

                {/* Address */}
                <div className="text-xs text-gray-600 font-mono truncate">{runner.address}</div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
