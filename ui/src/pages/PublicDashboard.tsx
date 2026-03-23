import { Cpu, Layers, AppWindow, Server } from 'lucide-react'
import { usePublicStats } from '../hooks/useBackend'
import { StatCard } from '../components/StatCard'

export function PublicDashboard() {
  const stats = usePublicStats()
  const s = stats.data

  const vramPct = s && s.gpu.vram_total_gb > 0
    ? Math.round((s.gpu.vram_used_gb / s.gpu.vram_total_gb) * 100)
    : 0

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <StatCard
          label="GPU VRAM"
          value={s ? `${s.gpu.vram_used_gb.toFixed(1)} / ${s.gpu.vram_total_gb.toFixed(1)} GB` : '—'}
          sub={s ? `${vramPct}% used` : undefined}
          progress={vramPct}
          icon={<Cpu className="w-4 h-4" />}
        />
        <StatCard
          label="Active Models"
          value={s?.active_models.toString() ?? '—'}
          icon={<Layers className="w-4 h-4" />}
        />
        <StatCard
          label="Connected Apps"
          value={s?.connected_apps.toString() ?? '—'}
          icon={<AppWindow className="w-4 h-4" />}
        />
        <StatCard
          label="Runners"
          value={s?.active_runners.toString() ?? '—'}
          icon={<Server className="w-4 h-4" />}
        />
      </div>

      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6 text-center">
        <p className="text-sm text-gray-400">
          Log in as admin to manage models, apps, and profiles.
        </p>
      </div>
    </div>
  )
}
