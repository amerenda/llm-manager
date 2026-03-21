import { Cpu, MemoryStick, Layers, Image, AppWindow, Trash2, RefreshCw } from 'lucide-react'
import { useLlmStatus, useApps, useDeleteModel } from '../hooks/useBackend'
import { StatCard } from '../components/StatCard'
import { StatusDot } from '../components/StatusDot'
import type { RegisteredApp } from '../types'

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

export function Dashboard() {
  const status = useLlmStatus()
  const apps = useApps()
  const deleteModel = useDeleteModel()

  const s = status.data
  const appList = apps.data ?? []
  const loadedModels = s?.loaded_ollama_models ?? []
  const memPct = s ? Math.round((s.mem_used_gb / s.mem_total_gb) * 100) : 0

  return (
    <div className="space-y-6">
      {/* Stat cards grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
        <StatCard
          label="GPU VRAM"
          value={s ? `${s.gpu_vram_used_gb.toFixed(1)} / ${s.gpu_vram_total_gb.toFixed(1)} GB` : '—'}
          sub={s ? `${s.gpu_vram_pct.toFixed(0)}% used` : undefined}
          progress={s?.gpu_vram_pct}
          icon={<Cpu className="w-4 h-4" />}
        />
        <StatCard
          label="CPU Usage"
          value={s ? `${s.cpu_pct.toFixed(0)}%` : '—'}
          progress={s?.cpu_pct}
          icon={<Cpu className="w-4 h-4" />}
        />
        <StatCard
          label="Memory"
          value={s ? `${s.mem_used_gb.toFixed(1)} / ${s.mem_total_gb.toFixed(1)} GB` : '—'}
          progress={memPct}
          icon={<MemoryStick className="w-4 h-4" />}
        />
        <StatCard
          label="Active Models"
          value={loadedModels.length.toString()}
          sub="Ollama loaded"
          icon={<Layers className="w-4 h-4" />}
        />
        <StatCard
          label="ComfyUI"
          value={
            s == null ? '—' : (
              <span className={`inline-flex items-center gap-1.5 text-sm font-medium px-2 py-0.5 rounded-full ${
                s.comfyui_running
                  ? 'bg-green-900/50 text-green-400'
                  : 'bg-gray-800 text-gray-500'
              }`}>
                <span className={`w-1.5 h-1.5 rounded-full ${s.comfyui_running ? 'bg-green-400' : 'bg-gray-600'}`} />
                {s.comfyui_running ? 'Running' : 'Stopped'}
              </span>
            )
          }
          icon={<Image className="w-4 h-4" />}
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
              {loadedModels.map((m: { name: string; size_gb: number }) => (
                <div
                  key={m.name}
                  className="flex items-center justify-between bg-gray-950 rounded-lg px-3 py-2"
                >
                  <div className="min-w-0">
                    <p className="text-sm text-gray-200 truncate">{m.name}</p>
                    <p className="text-xs text-gray-500">{m.size_gb.toFixed(1)} GB</p>
                  </div>
                  <button
                    onClick={() => deleteModel.mutate(m.name)}
                    disabled={deleteModel.isPending}
                    title="Unload model"
                    className="ml-2 p-1.5 rounded-lg bg-gray-800 hover:bg-red-900/50 hover:text-red-400 text-gray-500 transition-colors flex-shrink-0"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))}
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
