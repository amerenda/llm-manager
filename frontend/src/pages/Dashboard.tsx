import { useMemo } from 'react'
import { Cpu, MemoryStick, Layers, AppWindow, Trash2, RefreshCw, Server, Pin, Upload } from 'lucide-react'
import { useLlmStatus, useApps, useUnloadFromVram, useModelList, useUpdateModelSettings } from '../hooks/useBackend'
import { StatCard } from '../components/StatCard'
import { StatusDot } from '../components/StatusDot'
import type { RegisteredApp, RunnerStatus } from '../types'

function relativeTime(iso: string | null): string {
  if (!iso) return 'Never'
  const diffMs = Date.now() - new Date(iso).getTime()
  const diffSec = Math.floor(diffMs / 1000)
  if (diffSec < 60) return `${diffSec}s ago`
  const diffMin = Math.floor(diffSec / 60)
  if (diffMin < 60) return `${diffMin}m ago`
  const diffHr = Math.floor(diffMin / 60)
  return `${diffHr}h ago`
}

function isOnline(lastSeen: string | null): boolean {
  if (!lastSeen) return false
  return Date.now() - new Date(lastSeen).getTime() < 2 * 60 * 1000
}

function progressColor(pct: number): string {
  if (pct >= 90) return 'bg-red-500'
  if (pct >= 70) return 'bg-yellow-500'
  return 'bg-brand-500'
}

function RunnerCard({ r }: { r: RunnerStatus }) {
  const vramPct = r.gpu_vram_total_gb > 0
    ? Math.round((r.gpu_vram_used_gb / r.gpu_vram_total_gb) * 100)
    : 0
  const memPct = (r.mem_total_gb ?? 0) > 0
    ? Math.round(((r.mem_used_gb ?? 0) / (r.mem_total_gb ?? 1)) * 100)
    : 0
  const loadedCount = r.loaded_ollama_models?.length ?? 0

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
      <div className="flex items-center gap-2 mb-3">
        <Server className="w-4 h-4 text-brand-400" />
        <span className="text-sm font-medium text-gray-200">{r.runner_hostname}</span>
        {r.error ? (
          <span className="text-[10px] bg-red-900/40 text-red-400 px-1.5 py-0.5 rounded">unreachable</span>
        ) : (
          <span className="text-[10px] bg-green-900/40 text-green-400 px-1.5 py-0.5 rounded">online</span>
        )}
        {r.gpu_vendor && (
          <span className="text-[10px] bg-gray-800 text-gray-500 px-1.5 py-0.5 rounded">{r.gpu_vendor}</span>
        )}
      </div>

      {r.error ? (
        <p className="text-xs text-gray-600">Runner is not responding</p>
      ) : (
        <div className="space-y-2.5">
          {/* GPU VRAM */}
          <div>
            <div className="flex items-center justify-between text-xs mb-1">
              <span className="text-gray-500">GPU VRAM</span>
              <span className="text-gray-300 tabular-nums">
                {r.gpu_vram_used_gb.toFixed(1)} / {r.gpu_vram_total_gb.toFixed(1)} GB
              </span>
            </div>
            <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
              <div className={`h-full rounded-full transition-all ${progressColor(vramPct)}`}
                   style={{ width: `${Math.min(100, vramPct)}%` }} />
            </div>
          </div>

          {/* CPU + Memory inline */}
          <div className="flex gap-4">
            {r.cpu_pct !== undefined && (
              <div className="flex-1">
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className="text-gray-500">CPU</span>
                  <span className="text-gray-400 tabular-nums">{r.cpu_pct.toFixed(0)}%</span>
                </div>
                <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                  <div className={`h-full rounded-full ${progressColor(r.cpu_pct)}`}
                       style={{ width: `${Math.min(100, r.cpu_pct)}%` }} />
                </div>
              </div>
            )}
            {r.mem_total_gb !== undefined && (
              <div className="flex-1">
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className="text-gray-500">Memory</span>
                  <span className="text-gray-400 tabular-nums">
                    {(r.mem_used_gb ?? 0).toFixed(1)} / {r.mem_total_gb.toFixed(1)} GB
                  </span>
                </div>
                <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                  <div className={`h-full rounded-full ${progressColor(memPct)}`}
                       style={{ width: `${Math.min(100, memPct)}%` }} />
                </div>
              </div>
            )}
          </div>

          {/* Loaded models count */}
          <div className="text-xs text-gray-500">
            {loadedCount} model{loadedCount !== 1 ? 's' : ''} loaded
          </div>
        </div>
      )}
    </div>
  )
}

export function Dashboard() {
  const status = useLlmStatus()
  const apps = useApps()
  const unloadModel = useUnloadFromVram()
  const modelList = useModelList()
  const updateSettings = useUpdateModelSettings()

  const s = status.data
  const appList = apps.data ?? []
  const loadedModels = s?.loaded_ollama_models ?? []
  const runners = s?.runners ?? []
  const allModels = modelList.data ?? []

  const aliasMap = useMemo(() => {
    const map: Record<string, string> = {}
    for (const m of allModels) {
      const mAny = m as unknown as { is_alias?: boolean; base_model?: string }
      if (mAny.is_alias && mAny.base_model && !map[mAny.base_model]) map[mAny.base_model] = m.name
    }
    return map
  }, [allModels])
  const modelInfoMap = useMemo(() => Object.fromEntries(allModels.map(m => [m.name, m])), [allModels])

  return (
    <div className="space-y-6">
      {/* Per-runner GPU cards (1-2 runners: individual, 3+: aggregate) */}
      {runners.length <= 2 && runners.length > 0 ? (
        <div className={`grid gap-3 ${runners.length === 2 ? 'grid-cols-1 md:grid-cols-2' : 'grid-cols-1'}`}>
          {runners.map((r: RunnerStatus) => (
            <RunnerCard key={r.runner_id} r={r} />
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          <StatCard
            label="Total GPU VRAM"
            value={s ? `${s.gpu_vram_used_gb.toFixed(1)} / ${s.gpu_vram_total_gb.toFixed(1)} GB` : '—'}
            sub={s ? `${s.gpu_vram_pct.toFixed(0)}% used across ${runners.length} runners` : undefined}
            progress={s?.gpu_vram_pct}
            icon={<Cpu className="w-4 h-4" />}
          />
          <StatCard
            label="CPU Usage"
            value={s ? `${s.cpu_pct.toFixed(0)}%` : '—'}
            sub="Average across runners"
            progress={s?.cpu_pct}
            icon={<Cpu className="w-4 h-4" />}
          />
          <StatCard
            label="Memory"
            value={s ? `${s.mem_used_gb.toFixed(1)} / ${s.mem_total_gb.toFixed(1)} GB` : '—'}
            icon={<MemoryStick className="w-4 h-4" />}
          />
        </div>
      )}

      {/* Summary stats */}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
        <StatCard
          label="Active Models"
          value={loadedModels.length.toString()}
          sub="Ollama loaded"
          icon={<Layers className="w-4 h-4" />}
        />
        <StatCard
          label="Runners"
          value={runners.length.toString()}
          sub={`${runners.filter(r => !r.error).length} online`}
          icon={<Server className="w-4 h-4" />}
        />
        <StatCard
          label="Connected Apps"
          value={appList.length.toString()}
          sub={`${appList.filter((a: RegisteredApp) => isOnline(a.last_seen)).length} online`}
          icon={<AppWindow className="w-4 h-4" />}
        />
      </div>

      {/* Two-column lower section */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Active Models */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-medium text-gray-300 flex items-center gap-2">
              <Layers className="w-4 h-4 text-brand-400" />
              Active Models
            </h2>
            {status.isFetching && (
              <RefreshCw className="w-3 h-3 text-gray-600 animate-spin" />
            )}
          </div>
          {status.isLoading ? (
            <p className="text-xs text-gray-600 py-4 text-center">Loading…</p>
          ) : loadedModels.length === 0 ? (
            <p className="text-xs text-gray-600 py-4 text-center">No models loaded</p>
          ) : (
            <div className="space-y-2">
              {loadedModels.map((m: { name: string; size_gb: number; runner?: string }) => {
                const alias = aliasMap[m.name]
                const info = modelInfoMap[m.name]
                const isPinned = info?.do_not_evict ?? false
                return (
                  <div
                    key={`${m.name}-${m.runner}`}
                    className="flex items-center justify-between bg-gray-950 rounded-lg px-3 py-2"
                  >
                    <div className="min-w-0 flex items-center gap-1.5">
                      {isPinned && <Pin className="w-3 h-3 text-indigo-400 shrink-0" />}
                      <div className="min-w-0">
                        <p className="text-sm text-gray-200 truncate">{alias ?? m.name}</p>
                        {m.runner && <p className="text-xs text-gray-500">on {m.runner}</p>}
                      </div>
                    </div>
                    <div className="ml-2 flex items-center gap-1 flex-shrink-0">
                      <button
                        onClick={() => updateSettings.mutate({ model: m.name, do_not_evict: !isPinned })}
                        disabled={updateSettings.isPending}
                        title={isPinned ? 'Unpin' : 'Pin (keep in VRAM)'}
                        className={`p-1.5 rounded-lg transition-colors disabled:opacity-40 ${isPinned ? 'text-indigo-400 hover:text-indigo-300' : 'text-gray-600 hover:text-gray-400'}`}
                      >
                        <Pin className="w-3.5 h-3.5" />
                      </button>
                      <button
                        onClick={() => unloadModel.mutate({ model: m.name })}
                        disabled={unloadModel.isPending}
                        title="Evict from VRAM"
                        className="p-1.5 rounded-lg bg-gray-800 hover:bg-yellow-900/50 hover:text-yellow-400 text-gray-500 transition-colors"
                      >
                        <Upload className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>

        {/* Connected Apps */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-medium text-gray-300 flex items-center gap-2">
              <AppWindow className="w-4 h-4 text-brand-400" />
              Connected Apps
            </h2>
            {apps.isFetching && (
              <RefreshCw className="w-3 h-3 text-gray-600 animate-spin" />
            )}
          </div>
          {apps.isLoading ? (
            <p className="text-xs text-gray-600 py-4 text-center">Loading…</p>
          ) : appList.length === 0 ? (
            <p className="text-xs text-gray-600 py-4 text-center">No apps registered</p>
          ) : (
            <div className="space-y-2">
              {appList.map((app: RegisteredApp) => (
                <div
                  key={app.id}
                  className="flex items-center gap-3 bg-gray-950 rounded-lg px-3 py-2"
                >
                  <StatusDot online={isOnline(app.last_seen)} className="mt-0.5" />
                  <div className="min-w-0 flex-1">
                    <p className="text-sm text-gray-200 truncate">{app.name}</p>
                    <p className="text-xs text-gray-500 truncate">{app.base_url}</p>
                  </div>
                  <span className="text-xs text-gray-600 flex-shrink-0 tabular-nums">
                    {relativeTime(app.last_seen)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
