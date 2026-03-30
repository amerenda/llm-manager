import { useState } from 'react'
import { Download, Trash2, Loader2, CheckCircle2, AlertCircle, Image, Layers, Cpu, Upload, Shield, ShieldOff, Play, Square, Search, RefreshCw, BookOpen, Cloud, Settings2, Power, RefreshCcw, Server } from 'lucide-react'
import { useLlmModels, usePullModel, useDeleteModel, useCheckpoints, useSwitchCheckpoint, useLlmStatus, useLoadModel, useUnloadModel, useStartComfyui, useStopComfyui, useLibrary, useRefreshLibrary, useCloudModels, useCloudStatus, useUpdateCloudModel, useCloudKeys, useStoreCloudKey, useDeleteCloudKey, useRunners, useSyncModels, useOps } from '../hooks/useBackend'
import type { LlmModel, LibraryModel, Runner } from '../types'
import type { CloudModel, StoredApiKey } from '../hooks/useBackend'

function stripExt(filename: string): string {
  return filename.replace(/\.[^.]+$/, '')
}

// ── Runner Tab Bar ──────────────────────────────────────────────────────────

function RunnerTabs({ runners, selected, onSelect }: {
  runners: Runner[]
  selected: number | undefined
  onSelect: (id: number | undefined) => void
}) {
  return (
    <div className="flex gap-1 bg-gray-900 rounded-lg p-1 flex-wrap">
      <button
        onClick={() => onSelect(undefined)}
        className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
          selected === undefined ? 'bg-brand-900 text-brand-300' : 'text-gray-400 hover:text-gray-200'
        }`}
      >
        All Runners
      </button>
      {runners.map(r => (
        <button
          key={r.id}
          onClick={() => onSelect(r.id)}
          className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
            selected === r.id ? 'bg-brand-900 text-brand-300' : 'text-gray-400 hover:text-gray-200'
          }`}
        >
          {r.hostname}
        </button>
      ))}
    </div>
  )
}

// ── All Runners Model Matrix ────────────────────────────────────────────────

function AllRunnersModelView({ runners }: { runners: Runner[] }) {
  const models = useLlmModels()
  const status = useLlmStatus()
  const allLoadedModels = status.data?.loaded_ollama_models ?? []
  const runnerStatuses = status.data?.runners ?? []
  const textModels = (models.data ?? []).filter((m: LlmModel) => m.type === 'text').sort((a, b) => a.id.localeCompare(b.id))

  // Group by base model name to detect same-model-different-weights
  const baseGroups: Record<string, LlmModel[]> = {}
  for (const m of textModels) {
    const base = m.id.split(':')[0]
    ;(baseGroups[base] ??= []).push(m)
  }

  if (models.isLoading) {
    return <div className="py-8 text-center text-gray-600 text-sm">Loading models...</div>
  }

  if (textModels.length === 0) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-xl py-8 text-center text-gray-600 text-sm">
        No models downloaded on any runner
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {/* Per-runner disk bars */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        {runners.map(r => {
          const rs = runnerStatuses.find(s => s.runner_id === r.id)
          const diskTotal = rs?.disk_total_gb ?? 0
          const diskFree = rs?.disk_free_gb ?? 0
          const diskPct = diskTotal > 0 ? Math.round((diskTotal - diskFree) / diskTotal * 100) : 0
          if (!diskTotal) return null
          return (
            <div key={r.id} className="bg-gray-900 border border-gray-800 rounded-xl p-3">
              <div className="flex items-center justify-between text-xs text-gray-400 mb-1.5">
                <span>{r.hostname} disk</span>
                <span>{diskFree.toFixed(1)} GB free</span>
              </div>
              <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                <div className={`h-full rounded-full transition-all ${
                  diskPct > 95 ? 'bg-red-500' : diskPct > 85 ? 'bg-yellow-500' : 'bg-emerald-500'
                }`} style={{ width: `${diskPct}%` }} />
              </div>
              {diskFree < 5 && (
                <p className="text-[10px] text-red-400 mt-1">Low disk space</p>
              )}
            </div>
          )
        })}
      </div>

    <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-800 text-xs text-gray-500 uppercase tracking-wide">
            <th className="text-left px-4 py-3 font-medium">Model</th>
            {runners.map(r => (
              <th key={r.id} className="text-center px-3 py-3 font-medium">{r.hostname}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {textModels.map((m: LlmModel) => {
            const modelRunners = m.runners ?? []
            const runnerIds = new Set(modelRunners.map(r => r.runner_id))
            const allHave = runners.every(r => runnerIds.has(r.id))
            const base = m.id.split(':')[0]
            const hasDifferentWeights = (baseGroups[base]?.length ?? 0) > 1

            return (
              <tr key={m.id} className="border-b border-gray-800 last:border-0 hover:bg-gray-800/40 transition-colors">
                <td className="px-4 py-3 text-gray-200 font-medium truncate max-w-xs">
                  {m.id}
                </td>
                {runners.map(r => {
                  const hasModel = runnerIds.has(r.id)
                  const isLoaded = allLoadedModels.some(lm => lm.name === m.id && lm.runner === r.hostname)

                  return (
                    <td key={r.id} className="text-center px-3 py-3">
                      {hasModel ? (
                        <span className={`inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full ${
                          isLoaded
                            ? 'bg-blue-900/40 text-blue-400'
                            : allHave
                            ? 'bg-green-900/40 text-green-400'
                            : hasDifferentWeights
                            ? 'bg-yellow-900/40 text-yellow-400'
                            : 'bg-green-900/40 text-green-400'
                        }`}>
                          {isLoaded ? (
                            <><Cpu className="w-2.5 h-2.5" /> VRAM</>
                          ) : (
                            <><CheckCircle2 className="w-2.5 h-2.5" /> On disk</>
                          )}
                        </span>
                      ) : (
                        <span className="text-xs text-gray-700">—</span>
                      )}
                    </td>
                  )
                })}
              </tr>
            )
          })}
        </tbody>
      </table>
      <div className="px-4 py-2 border-t border-gray-800 flex gap-4 text-[10px] text-gray-600">
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-green-900" /> On all runners</span>
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-yellow-900" /> Different weights</span>
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-blue-900" /> Loaded in VRAM</span>
      </div>
    </div>
    </div>
  )
}

// ── Per-Runner Model View ───────────────────────────────────────────────────

function RunnerModelView({ runnerId, runnerHostname }: { runnerId: number; runnerHostname: string }) {
  const models = useLlmModels()
  const status = useLlmStatus()
  const pull = usePullModel()
  const del = useDeleteModel()
  const load = useLoadModel()
  const unload = useUnloadModel()
  const [pullInput, setPullInput] = useState('')
  const [pullMsg, setPullMsg] = useState<{ type: 'ok' | 'err'; text: string } | null>(null)

  // Filter models to only those on this runner
  const textModels = (models.data ?? [])
    .filter((m: LlmModel) => m.type === 'text')
    .filter((m: LlmModel) => m.runners?.some(r => r.runner_id === runnerId))
    .sort((a, b) => a.id.localeCompare(b.id))

  const allLoadedModels = status.data?.loaded_ollama_models ?? []
  const loadedModels = allLoadedModels.filter(m => m.runner === runnerHostname)
  const loadedNames = new Set(loadedModels.map(m => m.name))

  // Runner VRAM stats
  const runnerStatus = status.data?.runners?.find(r => r.runner_id === runnerId)
  const vramUsed = runnerStatus?.gpu_vram_used_gb ?? 0
  const vramTotal = runnerStatus?.gpu_vram_total_gb ?? 0
  const vramPct = vramTotal > 0 ? Math.round(vramUsed / vramTotal * 100) : 0

  const diskUsed = runnerStatus?.disk_used_gb ?? 0
  const diskTotal = runnerStatus?.disk_total_gb ?? 0
  const diskFree = runnerStatus?.disk_free_gb ?? 0
  const diskPct = diskTotal > 0 ? Math.round(diskUsed / diskTotal * 100) : 0

  async function handlePull() {
    const name = pullInput.trim()
    if (!name) return
    setPullMsg(null)
    try {
      const result = await pull.mutateAsync({ model: name, runner_id: runnerId })
      if (result.ok) {
        setPullMsg({ type: 'ok', text: `Pulling ${name} on ${runnerHostname}...` })
        setPullInput('')
      }
    } catch (e) {
      setPullMsg({ type: 'err', text: (e as Error).message })
    }
  }

  return (
    <section className="space-y-4">
      {/* VRAM + Disk bars */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-3 space-y-3">
        {vramTotal > 0 && (
          <div>
            <div className="flex items-center justify-between text-xs text-gray-400 mb-1.5">
              <span>VRAM on {runnerHostname}</span>
              <span>{vramUsed.toFixed(1)} / {vramTotal.toFixed(1)} GB ({vramPct}%)</span>
            </div>
            <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all ${
                  vramPct > 90 ? 'bg-red-500' : vramPct > 70 ? 'bg-yellow-500' : 'bg-brand-500'
                }`}
                style={{ width: `${vramPct}%` }}
              />
            </div>
          </div>
        )}
        {diskTotal > 0 && (
          <div>
            <div className="flex items-center justify-between text-xs text-gray-400 mb-1.5">
              <span>Model storage</span>
              <span>
                {diskFree.toFixed(1)} GB free of {diskTotal.toFixed(1)} GB ({diskPct}% used)
              </span>
            </div>
            <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all ${
                  diskPct > 95 ? 'bg-red-500' : diskPct > 85 ? 'bg-yellow-500' : 'bg-emerald-500'
                }`}
                style={{ width: `${diskPct}%` }}
              />
            </div>
            {diskFree < 5 && (
              <p className="text-[10px] text-red-400 mt-1 flex items-center gap-1">
                <AlertCircle className="w-3 h-3" />
                Low disk space — model downloads may fail
              </p>
            )}
          </div>
        )}
      </div>

      {/* Pull input */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
        <p className="text-xs text-gray-500 mb-2 font-medium uppercase tracking-wide">Pull model to {runnerHostname}</p>
        <div className="flex gap-2">
          <input
            type="text"
            value={pullInput}
            onChange={e => setPullInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handlePull()}
            placeholder="e.g. qwen2.5:7b"
            className="flex-1 bg-gray-950 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-brand-600"
          />
          <button
            onClick={handlePull}
            disabled={pull.isPending || !pullInput.trim()}
            className="flex items-center gap-2 bg-brand-600 hover:bg-brand-500 disabled:opacity-40 text-white text-sm font-medium px-4 py-2 rounded-lg transition-colors"
          >
            {pull.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
            Pull
          </button>
        </div>
        {pullMsg && (
          <div className={`mt-2 flex items-center gap-1.5 text-xs ${pullMsg.type === 'ok' ? 'text-green-400' : 'text-red-400'}`}>
            {pullMsg.type === 'ok' ? <CheckCircle2 className="w-3.5 h-3.5" /> : <AlertCircle className="w-3.5 h-3.5" />}
            {pullMsg.text}
          </div>
        )}
      </div>

      {/* Model table */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
        {models.isLoading ? (
          <div className="py-8 text-center text-gray-600 text-sm">Loading models...</div>
        ) : textModels.length === 0 ? (
          <div className="py-8 text-center text-gray-600 text-sm">No models on {runnerHostname}</div>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800 text-xs text-gray-500 uppercase tracking-wide">
                <th className="text-left px-4 py-3 font-medium">Model</th>
                <th className="text-left px-4 py-3 font-medium">Status</th>
                <th className="px-4 py-3" />
              </tr>
            </thead>
            <tbody>
              {textModels.map((m: LlmModel) => {
                const isLoaded = loadedNames.has(m.id)
                const loadedInfo = loadedModels.find(lm => lm.name === m.id)
                return (
                  <tr key={m.id} className="border-b border-gray-800 last:border-0 hover:bg-gray-800/40 transition-colors">
                    <td className="px-4 py-3 text-gray-200 font-medium truncate max-w-xs">{m.id}</td>
                    <td className="px-4 py-3">
                      {isLoaded ? (
                        <span className="flex items-center gap-1.5 text-xs text-blue-400">
                          <Cpu className="w-3 h-3" />
                          In VRAM{loadedInfo ? ` (${loadedInfo.size_gb} GB)` : ''}
                        </span>
                      ) : (
                        <span className="text-xs text-gray-600">On disk</span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <div className="flex items-center justify-end gap-2">
                        {isLoaded ? (
                          <button
                            onClick={() => unload.mutate({ model: m.id, runner_id: runnerId })}
                            disabled={unload.isPending}
                            title="Unload from VRAM"
                            className="flex items-center gap-1 text-xs bg-yellow-900/30 hover:bg-yellow-800/40 text-yellow-400 px-2.5 py-1 rounded-lg transition-colors disabled:opacity-40"
                          >
                            {unload.isPending ? <Loader2 className="w-3 h-3 animate-spin" /> : <Upload className="w-3 h-3" />}
                            Unload
                          </button>
                        ) : (
                          <button
                            onClick={() => load.mutate({ model: m.id, runner_id: runnerId })}
                            disabled={load.isPending}
                            title="Load into VRAM"
                            className="flex items-center gap-1 text-xs bg-brand-900/50 hover:bg-brand-800/50 text-brand-300 px-2.5 py-1 rounded-lg transition-colors disabled:opacity-40"
                          >
                            {load.isPending ? <Loader2 className="w-3 h-3 animate-spin" /> : <Cpu className="w-3 h-3" />}
                            Load
                          </button>
                        )}
                        <button
                          onClick={() => {
                            if (window.confirm(`Delete ${m.id} from ${runnerHostname}? This removes it from disk.`)) {
                              del.mutate(m.id)
                            }
                          }}
                          disabled={del.isPending}
                          title="Delete model"
                          className="p-1.5 rounded-lg bg-gray-800 hover:bg-red-900/50 hover:text-red-400 text-gray-500 transition-colors"
                        >
                          {del.isPending ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Trash2 className="w-3.5 h-3.5" />}
                        </button>
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </div>
    </section>
  )
}

// ── Library Browser ─────────────────────────────────────────────────────────

function LibraryBrowserSection({ selectedRunner, selectedRunnerHostname, allRunners }: {
  selectedRunner?: number
  selectedRunnerHostname?: string
  allRunners: Runner[]
}) {
  const [search, setSearch] = useState('')
  const [safety, setSafety] = useState<string>('safe')
  const [fitsOnly, setFitsOnly] = useState(true)
  const [hideDownloaded, setHideDownloaded] = useState(true)
  const [sort, setSort] = useState<string>('pulls')
  const [expanded, setExpanded] = useState<string | null>(null)
  const pull = usePullModel()
  const refresh = useRefreshLibrary()
  const ops = useOps()
  const pullingOps = (ops.data ?? []).filter(op => op.type === 'pull' && op.status === 'running')
  const completedOps = (ops.data ?? []).filter(op => op.type === 'pull' && op.status === 'completed')
  const failedOps = (ops.data ?? []).filter(op => op.type === 'pull' && op.status === 'failed')
  const [pullMsg, setPullMsg] = useState<{ model: string; type: 'ok' | 'err' | 'pulling'; text: string } | null>(null)

  const library = useLibrary({
    search: search || undefined,
    safety,
    fits: fitsOnly || undefined,
    downloaded: hideDownloaded ? false : undefined,
    hasPulling: pullingOps.length > 0,
  })

  const rawModels = library.data?.models ?? []
  const models = [...rawModels].sort((a, b) => {
    if (sort === 'pulls') {
      const pa = parseFloat(a.pulls?.replace(/[KMB]/i, '') || '0')
      const pb = parseFloat(b.pulls?.replace(/[KMB]/i, '') || '0')
      const ma = a.pulls?.match(/[KMB]/i)?.[0]?.toUpperCase() || ''
      const mb = b.pulls?.match(/[KMB]/i)?.[0]?.toUpperCase() || ''
      const mult: Record<string, number> = { '': 1, 'K': 1e3, 'M': 1e6, 'B': 1e9 }
      return (pb * (mult[mb] || 1)) - (pa * (mult[ma] || 1))
    }
    if (sort === 'vram') return a.vram_estimate_gb - b.vram_estimate_gb
    return a.name.localeCompare(b.name)
  })
  const cacheAge = library.data?.cache_age_hours ?? 0

  function isModelPulling(model: string) {
    return pullingOps.some(op => op.model === model || model.startsWith(op.model + ':') || op.model.startsWith(model + ':'))
  }
  function getModelOpStatus(model: string) {
    const failed = failedOps.find(op => op.model === model || model.startsWith(op.model + ':') || op.model.startsWith(model + ':'))
    if (failed) return { status: 'failed' as const, error: failed.error }
    const completed = completedOps.find(op => op.model === model || model.startsWith(op.model + ':') || op.model.startsWith(model + ':'))
    if (completed) return { status: 'completed' as const }
    if (isModelPulling(model)) return { status: 'running' as const }
    return null
  }

  async function handlePull(model: string, runnerId?: number) {
    const target = runnerId
      ? allRunners.find(r => r.id === runnerId)?.hostname ?? ''
      : 'all runners'
    setPullMsg({ model, type: 'pulling', text: `Pulling to ${target}...` })
    try {
      await pull.mutateAsync({ model, runner_id: runnerId })
      setPullMsg({ model, type: 'ok', text: `Downloading on ${target}` })
    } catch (e) {
      setPullMsg({ model, type: 'err', text: (e as Error).message })
    }
  }

  return (
    <section className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <BookOpen className="w-4 h-4 text-brand-400" />
          <h2 className="text-base font-semibold text-gray-200">Ollama Library</h2>
          <span className="text-xs text-gray-600">{library.data?.total ?? 0} models</span>
        </div>
        <button
          onClick={() => refresh.mutate()}
          disabled={refresh.isPending}
          className="flex items-center gap-1 text-xs text-gray-500 hover:text-gray-300 transition-colors"
          title={`Cache age: ${cacheAge.toFixed(1)}h`}
        >
          {refresh.isPending ? <Loader2 className="w-3 h-3 animate-spin" /> : <RefreshCw className="w-3 h-3" />}
          Refresh
        </button>
      </div>

      {/* Filters */}
      <div className="flex gap-2 flex-wrap">
        <div className="relative flex-1 min-w-48">
          <Search className="absolute left-3 top-2.5 w-3.5 h-3.5 text-gray-600" />
          <input
            type="text"
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search models..."
            className="w-full bg-gray-900 border border-gray-800 rounded-lg pl-8 pr-3 py-2 text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-brand-600"
          />
        </div>
        <button
          onClick={() => setHideDownloaded(!hideDownloaded)}
          className={`flex items-center gap-1 text-xs px-2.5 py-2 rounded-lg transition-colors ${
            hideDownloaded
              ? 'bg-brand-900/40 text-brand-400 border border-brand-800'
              : 'bg-gray-900 text-gray-400 border border-gray-800'
          }`}
        >
          <Download className="w-3 h-3" />
          {hideDownloaded ? 'Not downloaded' : 'All'}
        </button>
        <button
          onClick={() => setFitsOnly(!fitsOnly)}
          className={`flex items-center gap-1 text-xs px-2.5 py-2 rounded-lg transition-colors ${
            fitsOnly
              ? 'bg-brand-900/40 text-brand-400 border border-brand-800'
              : 'bg-gray-900 text-gray-400 border border-gray-800'
          }`}
        >
          <Cpu className="w-3 h-3" />
          {fitsOnly ? 'Fits on GPU' : 'All sizes'}
        </button>
        <button
          onClick={() => setSafety(safety === 'safe' ? 'all' : safety === 'all' ? 'unsafe' : 'safe')}
          className={`flex items-center gap-1 text-xs px-2.5 py-2 rounded-lg transition-colors ${
            safety === 'unsafe'
              ? 'bg-red-900/40 text-red-400 border border-red-800'
              : safety === 'all'
              ? 'bg-gray-900 text-gray-400 border border-gray-800'
              : 'bg-green-900/40 text-green-400 border border-green-800'
          }`}
        >
          {safety === 'safe' ? <Shield className="w-3 h-3" /> : <ShieldOff className="w-3 h-3" />}
          {safety === 'safe' ? 'Safe only' : safety === 'all' ? 'All models' : 'Unsafe only'}
        </button>
        <select
          value={sort}
          onChange={e => setSort(e.target.value)}
          className="text-xs bg-gray-900 border border-gray-800 rounded-lg px-2.5 py-2 text-gray-400 focus:outline-none focus:border-brand-600"
        >
          <option value="pulls">Most popular</option>
          <option value="name">Name</option>
          <option value="vram">VRAM (smallest)</option>
        </select>
      </div>

      {/* Model list */}
      <div className="space-y-2">
        {library.isLoading ? (
          <div className="py-8 text-center text-gray-600 text-sm">Loading library...</div>
        ) : models.length === 0 ? (
          <div className="py-8 text-center text-gray-600 text-sm">No models match your filters</div>
        ) : (
          models.map((m: LibraryModel) => {
            const isExpanded = expanded === m.name
            return (
            <div
              key={m.name}
              className={`bg-gray-900 border rounded-xl transition-colors cursor-pointer ${
                isExpanded ? 'border-brand-700 bg-gray-800/30' : 'border-gray-800 hover:bg-gray-800/40'
              }`}
              onClick={() => setExpanded(isExpanded ? null : m.name)}
            >
              <div className="p-4">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-sm text-gray-200 font-medium">{m.name}</span>
                      {m.safety === 'unsafe' && (
                        <span className="text-[10px] bg-red-900/40 text-red-400 px-1.5 py-0.5 rounded">unsafe</span>
                      )}
                      {m.downloaded && m.downloaded_on?.length > 0 && (
                        <span className="flex items-center gap-0.5 text-[10px] text-green-400" title={`On: ${m.downloaded_on.join(', ')}`}>
                          <CheckCircle2 className="w-2.5 h-2.5" />
                          {m.downloaded_on.length === allRunners.length
                            ? 'All runners'
                            : m.downloaded_on.join(', ')}
                        </span>
                      )}
                    </div>
                    {m.description && (
                      <p className={`text-xs text-gray-500 mt-1 ${isExpanded ? '' : 'line-clamp-1'}`}>{m.description}</p>
                    )}
                    {!isExpanded && (
                      <div className="flex items-center gap-3 mt-2 flex-wrap">
                        {m.parameter_sizes.length > 0 && (
                          <div className="flex gap-1 flex-wrap">
                            {m.parameter_sizes.slice(0, 4).map(s => (
                              <span key={s} className="text-[10px] bg-blue-900/30 text-blue-400 px-1.5 py-0.5 rounded">{s}</span>
                            ))}
                            {m.parameter_sizes.length > 4 && (
                              <span className="text-[10px] text-gray-600">+{m.parameter_sizes.length - 4} more</span>
                            )}
                          </div>
                        )}
                        <span className="text-[10px] text-gray-600">~{m.vram_estimate_gb}GB VRAM</span>
                        {m.pulls && <span className="text-[10px] text-gray-600">{m.pulls} pulls</span>}
                      </div>
                    )}
                  </div>
                  <div className="flex-shrink-0 flex items-center gap-1.5" onClick={e => e.stopPropagation()}>
                    {isModelPulling(m.name) && (
                      <span className="flex items-center gap-1 text-xs text-amber-400">
                        <Loader2 className="w-3 h-3 animate-spin" />
                        Downloading...
                      </span>
                    )}
                    {!isModelPulling(m.name) && !m.downloaded && m.fits && selectedRunner !== undefined && (
                      <button
                        onClick={() => handlePull(m.name, selectedRunner)}
                        disabled={pull.isPending && pullMsg?.model === m.name}
                        title={`Pull to ${selectedRunnerHostname}`}
                        className="flex items-center gap-1 text-xs bg-brand-900/50 hover:bg-brand-800/50 text-brand-300 px-3 py-1.5 rounded-lg transition-colors disabled:opacity-40"
                      >
                        {pull.isPending && pullMsg?.model === m.name
                          ? <Loader2 className="w-3 h-3 animate-spin" />
                          : <Download className="w-3 h-3" />}
                        Pull to {selectedRunnerHostname}
                      </button>
                    )}
                    {!isModelPulling(m.name) && !m.downloaded && m.fits && selectedRunner === undefined && allRunners.length > 0 && (
                      <div className="flex gap-1">
                        {allRunners.map(r => {
                          const fitsOnThis = m.fits_on.some(f => f.runner === r.hostname)
                          if (!fitsOnThis) return null
                          return (
                            <button
                              key={r.id}
                              onClick={() => handlePull(m.name, r.id)}
                              disabled={pull.isPending && pullMsg?.model === m.name}
                              title={`Pull to ${r.hostname}`}
                              className="flex items-center gap-1 text-[10px] bg-brand-900/50 hover:bg-brand-800/50 text-brand-300 px-2 py-1 rounded-lg transition-colors disabled:opacity-40"
                            >
                              <Download className="w-2.5 h-2.5" />
                              {r.hostname}
                            </button>
                          )
                        })}
                      </div>
                    )}
                    {!m.fits && !m.downloaded && !isModelPulling(m.name) && (
                      <span className="text-[10px] text-gray-600">Too large</span>
                    )}
                    {(() => {
                      const opStatus = getModelOpStatus(m.name)
                      if (opStatus?.status === 'failed') return (
                        <span className="flex items-center gap-1 text-[10px] text-red-400">
                          <AlertCircle className="w-2.5 h-2.5" />
                          {opStatus.error || 'Pull failed'}
                        </span>
                      )
                      if (opStatus?.status === 'completed') return (
                        <span className="flex items-center gap-1 text-[10px] text-green-400">
                          <CheckCircle2 className="w-2.5 h-2.5" />
                          Done
                        </span>
                      )
                      return null
                    })()}
                  </div>
                </div>
              </div>

              {/* Expanded detail panel */}
              {isExpanded && (
                <div className="border-t border-gray-800 px-4 py-3 space-y-3">
                  {m.parameter_sizes.length > 0 && (
                    <div>
                      <p className="text-[10px] text-gray-500 uppercase tracking-wide mb-1.5">Available sizes</p>
                      <div className="flex gap-1.5 flex-wrap">
                        {m.parameter_sizes.map(s => {
                          const sizeModel = `${m.name}:${s}`
                          const pulling = isModelPulling(sizeModel)
                          const opStatus = getModelOpStatus(sizeModel)
                          return (
                          <button
                            key={s}
                            onClick={e => { e.stopPropagation(); handlePull(sizeModel, selectedRunner) }}
                            disabled={pulling || (pull.isPending && pullMsg?.model === sizeModel)}
                            className={`flex items-center gap-1 text-xs px-2 py-1 rounded transition-colors ${
                              pulling
                                ? 'bg-amber-900/30 text-amber-400'
                                : opStatus?.status === 'failed'
                                ? 'bg-red-900/30 text-red-400'
                                : opStatus?.status === 'completed'
                                ? 'bg-green-900/30 text-green-400'
                                : 'bg-blue-900/20 hover:bg-blue-900/40 text-blue-400'
                            }`}
                          >
                            {pulling
                              ? <Loader2 className="w-2.5 h-2.5 animate-spin" />
                              : opStatus?.status === 'completed'
                              ? <CheckCircle2 className="w-2.5 h-2.5" />
                              : opStatus?.status === 'failed'
                              ? <AlertCircle className="w-2.5 h-2.5" />
                              : <Download className="w-2.5 h-2.5" />}
                            {s}
                            {pulling
                              ? <span className="text-amber-500 ml-0.5">downloading...</span>
                              : opStatus?.status === 'completed'
                              ? <span className="text-green-500 ml-0.5">done</span>
                              : opStatus?.status === 'failed'
                              ? <span className="text-red-500 ml-0.5">failed</span>
                              : selectedRunnerHostname && <span className="text-gray-600 ml-0.5">→ {selectedRunnerHostname}</span>}
                          </button>
                          )
                        })}
                      </div>
                    </div>
                  )}

                  {m.categories.length > 0 && (
                    <div>
                      <p className="text-[10px] text-gray-500 uppercase tracking-wide mb-1.5">Categories</p>
                      <div className="flex gap-1.5 flex-wrap">
                        {m.categories.map(c => (
                          <span key={c} className="text-[10px] bg-indigo-900/30 text-indigo-400 px-2 py-0.5 rounded">{c}</span>
                        ))}
                      </div>
                    </div>
                  )}

                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                    <div>
                      <p className="text-[10px] text-gray-500 uppercase tracking-wide">VRAM estimate</p>
                      <p className="text-sm text-gray-200 mt-0.5">~{m.vram_estimate_gb} GB</p>
                    </div>
                    {m.pulls && (
                      <div>
                        <p className="text-[10px] text-gray-500 uppercase tracking-wide">Downloads</p>
                        <p className="text-sm text-gray-200 mt-0.5">{m.pulls}</p>
                      </div>
                    )}
                    <div>
                      <p className="text-[10px] text-gray-500 uppercase tracking-wide">Safety</p>
                      <p className={`text-sm mt-0.5 ${m.safety === 'safe' ? 'text-green-400' : 'text-red-400'}`}>{m.safety}</p>
                    </div>
                    {m.fits_on.length > 0 && (
                      <div>
                        <p className="text-[10px] text-gray-500 uppercase tracking-wide">Fits on</p>
                        <p className="text-sm text-gray-200 mt-0.5">{m.fits_on.map(r => r.runner).join(', ')}</p>
                      </div>
                    )}
                  </div>

                  {/* Per-runner download status */}
                  {m.downloaded && m.downloaded_on?.length > 0 && (
                    <div>
                      <p className="text-[10px] text-gray-500 uppercase tracking-wide mb-1.5">Downloaded on</p>
                      <div className="flex gap-1.5 flex-wrap">
                        {allRunners.map(r => (
                          <span
                            key={r.id}
                            className={`text-[10px] px-2 py-0.5 rounded ${
                              m.downloaded_on.includes(r.hostname)
                                ? 'bg-green-900/40 text-green-400'
                                : 'bg-gray-800 text-gray-600'
                            }`}
                          >
                            {r.hostname}: {m.downloaded_on.includes(r.hostname) ? 'yes' : 'no'}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
            )
          })
        )}
      </div>
    </section>
  )
}

// ── Image Models Section ────────────────────────────────────────────────────

function ImageModelsSection() {
  const status = useLlmStatus()
  const checkpoints = useCheckpoints()
  const switchCp = useSwitchCheckpoint()
  const startComfy = useStartComfyui()
  const stopComfy = useStopComfyui()
  const [switching, setSwitching] = useState<string | null>(null)

  const comfyRunning = status.data?.comfyui_running ?? false
  const activeCheckpoint = status.data?.comfyui_active_checkpoint ?? null
  const cpList = checkpoints.data ?? []

  async function handleSwitch(name: string) {
    setSwitching(name)
    try { await switchCp.mutateAsync(name) } finally { setSwitching(null) }
  }

  return (
    <section className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Image className="w-4 h-4 text-brand-400" />
          <h2 className="text-base font-semibold text-gray-200">Image Models (ComfyUI)</h2>
          <span className={`ml-1 text-xs px-2 py-0.5 rounded-full ${
            comfyRunning ? 'bg-green-900/40 text-green-400' : 'bg-gray-800 text-gray-500'
          }`}>
            {comfyRunning ? 'Running' : 'Stopped'}
          </span>
        </div>
        {comfyRunning ? (
          <button onClick={() => stopComfy.mutate()} disabled={stopComfy.isPending}
            className="flex items-center gap-1.5 text-xs bg-red-900/30 hover:bg-red-800/40 text-red-400 px-3 py-1.5 rounded-lg transition-colors disabled:opacity-40">
            {stopComfy.isPending ? <Loader2 className="w-3 h-3 animate-spin" /> : <Square className="w-3 h-3" />}
            Stop ComfyUI
          </button>
        ) : (
          <button onClick={() => startComfy.mutate()} disabled={startComfy.isPending}
            className="flex items-center gap-1.5 text-xs bg-green-900/30 hover:bg-green-800/40 text-green-400 px-3 py-1.5 rounded-lg transition-colors disabled:opacity-40">
            {startComfy.isPending ? <Loader2 className="w-3 h-3 animate-spin" /> : <Play className="w-3 h-3" />}
            Start ComfyUI
          </button>
        )}
      </div>

      {checkpoints.isLoading ? (
        <div className="py-8 text-center text-gray-600 text-sm">Loading checkpoints...</div>
      ) : cpList.length === 0 ? (
        <div className="bg-gray-900 border border-gray-800 rounded-xl py-8 text-center text-gray-600 text-sm">No checkpoints found</div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {cpList.map(cp => {
            const isActive = cp === activeCheckpoint
            const isLoading = switching === cp
            return (
              <button key={cp} onClick={() => !isActive && handleSwitch(cp)}
                disabled={isActive || isLoading || !comfyRunning}
                className={`text-left bg-gray-900 border rounded-xl p-4 transition-all ${
                  isActive ? 'border-brand-600 bg-brand-900/20 cursor-default'
                  : !comfyRunning ? 'border-gray-800 opacity-50 cursor-not-allowed'
                  : 'border-gray-800 hover:border-gray-700 hover:bg-gray-800/50 cursor-pointer'
                }`}>
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <p className={`text-sm font-medium truncate ${isActive ? 'text-brand-300' : 'text-gray-200'}`}>{stripExt(cp)}</p>
                    <p className="text-xs text-gray-600 truncate mt-0.5">{cp}</p>
                  </div>
                  {isLoading && <Loader2 className="w-4 h-4 text-brand-400 animate-spin flex-shrink-0 mt-0.5" />}
                  {isActive && !isLoading && (
                    <span className="text-xs bg-brand-900 text-brand-300 px-1.5 py-0.5 rounded flex-shrink-0">Active</span>
                  )}
                </div>
              </button>
            )
          })}
        </div>
      )}
    </section>
  )
}

// ── Cloud Models Section ────────────────────────────────────────────────────

function ApiKeyManager() {
  const keys = useCloudKeys()
  const store = useStoreCloudKey()
  const remove = useDeleteCloudKey()
  const [adding, setAdding] = useState<string | null>(null)
  const [keyInput, setKeyInput] = useState('')
  const [labelInput, setLabelInput] = useState('')

  const providers = [
    { id: 'anthropic', name: 'Anthropic (Claude)' },
    { id: 'openai', name: 'OpenAI' },
  ]

  const keysByProvider: Record<string, StoredApiKey[]> = {}
  for (const k of keys.data ?? []) {
    ;(keysByProvider[k.provider] ??= []).push(k)
  }

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 space-y-3">
      <h3 className="text-sm font-medium text-gray-300 flex items-center gap-2">
        <Shield className="w-4 h-4 text-brand-400" /> API Keys
      </h3>
      {providers.map(p => {
        const existing = keysByProvider[p.id] ?? []
        return (
          <div key={p.id} className="flex items-center justify-between gap-3 bg-gray-950 rounded-lg px-3 py-2">
            <div className="min-w-0">
              <p className="text-sm text-gray-200">{p.name}</p>
              {existing.length > 0 ? (
                <div className="flex items-center gap-2 mt-0.5">
                  <span className="text-xs text-gray-500 font-mono">{existing[0].key_preview}</span>
                  {existing[0].label && <span className="text-xs text-gray-600">{existing[0].label}</span>}
                </div>
              ) : (
                <p className="text-xs text-gray-600">No key configured</p>
              )}
            </div>
            <div className="flex items-center gap-1.5 flex-shrink-0">
              {existing.length > 0 && (
                <button onClick={() => remove.mutate(existing[0].id)} disabled={remove.isPending}
                  className="p-1.5 rounded-lg bg-gray-800 hover:bg-red-900/50 hover:text-red-400 text-gray-500 transition-colors" title="Remove key">
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              )}
              <button onClick={() => setAdding(adding === p.id ? null : p.id)}
                className="text-xs px-2.5 py-1.5 rounded-lg bg-brand-900/50 hover:bg-brand-800/50 text-brand-300 transition-colors">
                {existing.length > 0 ? 'Replace' : 'Add key'}
              </button>
            </div>
          </div>
        )
      })}
      {adding && (
        <div className="flex gap-2 mt-2">
          <input type="password" value={keyInput} onChange={e => setKeyInput(e.target.value)}
            placeholder={`${adding} API key`}
            className="flex-1 bg-gray-950 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-brand-600" />
          <input type="text" value={labelInput} onChange={e => setLabelInput(e.target.value)}
            placeholder="Label (optional)"
            className="w-32 bg-gray-950 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-brand-600" />
          <button onClick={() => {
            if (keyInput.trim()) {
              store.mutate({ provider: adding, key: keyInput.trim(), label: labelInput.trim() })
              setKeyInput(''); setLabelInput(''); setAdding(null)
            }
          }} disabled={store.isPending || !keyInput.trim()}
            className="px-3 py-1.5 rounded-lg bg-brand-600 hover:bg-brand-500 text-white text-sm transition-colors disabled:opacity-40">
            Save
          </button>
        </div>
      )}
    </div>
  )
}

function CloudModelsSection() {
  const models = useCloudModels()
  const status = useCloudStatus()
  const update = useUpdateCloudModel()
  const [editing, setEditing] = useState<string | null>(null)
  const [editValues, setEditValues] = useState<{ max_tokens?: number; temperature?: number }>({})

  const anthropicStatus = status.data?.anthropic

  return (
    <section className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Cloud className="w-4 h-4 text-brand-400" />
          <h2 className="text-base font-semibold text-gray-200">Anthropic (Claude)</h2>
          {anthropicStatus && (
            <span className={`inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded ${
              anthropicStatus.reachable ? 'bg-green-900/40 text-green-400'
              : anthropicStatus.configured ? 'bg-yellow-900/40 text-yellow-400'
              : 'bg-gray-800 text-gray-500'
            }`}>
              <span className={`w-1.5 h-1.5 rounded-full ${
                anthropicStatus.reachable ? 'bg-green-400' : anthropicStatus.configured ? 'bg-yellow-400' : 'bg-gray-600'
              }`} />
              {anthropicStatus.reachable ? 'Connected' : anthropicStatus.configured ? 'Unreachable' : 'Not configured'}
            </span>
          )}
        </div>
      </div>

      <ApiKeyManager />

      {!anthropicStatus?.configured ? (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6 text-center">
          <p className="text-sm text-gray-500">Add an Anthropic API key above to enable Claude models.</p>
        </div>
      ) : models.isLoading ? (
        <p className="text-xs text-gray-600 py-4 text-center">Loading cloud models...</p>
      ) : (
        <div className="space-y-2">
          {(models.data ?? []).map((m: CloudModel) => (
            <div key={m.id} className="bg-gray-900 border border-gray-800 rounded-xl p-4">
              <div className="flex items-center justify-between gap-3">
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-gray-200 font-medium">{m.display_name}</span>
                    <span className="text-[10px] bg-indigo-900/30 text-indigo-400 px-1.5 py-0.5 rounded">{m.provider}</span>
                    {!m.enabled && <span className="text-[10px] bg-gray-800 text-gray-500 px-1.5 py-0.5 rounded">disabled</span>}
                  </div>
                  <p className="text-xs text-gray-500 mt-0.5 font-mono">{m.id}</p>
                  <div className="flex items-center gap-3 mt-1.5 text-[10px] text-gray-500">
                    <span>Max tokens: {m.max_tokens}</span>
                    {m.temperature !== null && <span>Temperature: {m.temperature}</span>}
                  </div>
                </div>
                <div className="flex items-center gap-1.5 flex-shrink-0">
                  <button onClick={() => {
                    if (editing === m.id) { update.mutate({ modelId: m.id, ...editValues }); setEditing(null); setEditValues({}) }
                    else { setEditing(m.id); setEditValues({ max_tokens: m.max_tokens, temperature: m.temperature ?? undefined }) }
                  }} className="p-1.5 rounded-lg bg-gray-800 hover:bg-gray-700 text-gray-400 transition-colors"
                    title={editing === m.id ? 'Save' : 'Configure'}>
                    {editing === m.id ? <CheckCircle2 className="w-3.5 h-3.5 text-green-400" /> : <Settings2 className="w-3.5 h-3.5" />}
                  </button>
                  <button onClick={() => update.mutate({ modelId: m.id, enabled: !m.enabled })}
                    className={`p-1.5 rounded-lg transition-colors ${
                      m.enabled ? 'bg-green-900/30 hover:bg-green-900/50 text-green-400' : 'bg-gray-800 hover:bg-gray-700 text-gray-500'
                    }`} title={m.enabled ? 'Disable' : 'Enable'}>
                    <Power className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
              {editing === m.id && (
                <div className="mt-3 pt-3 border-t border-gray-800 flex gap-3">
                  <label className="flex items-center gap-2 text-xs text-gray-400">
                    Max tokens
                    <input type="number" value={editValues.max_tokens ?? ''} onChange={e => setEditValues(v => ({ ...v, max_tokens: parseInt(e.target.value) || undefined }))}
                      className="w-24 bg-gray-950 border border-gray-700 rounded px-2 py-1 text-xs text-gray-200 focus:outline-none focus:border-brand-600" />
                  </label>
                  <label className="flex items-center gap-2 text-xs text-gray-400">
                    Temperature
                    <input type="number" step="0.1" min="0" max="2" value={editValues.temperature ?? ''}
                      onChange={e => setEditValues(v => ({ ...v, temperature: parseFloat(e.target.value) || undefined }))}
                      className="w-20 bg-gray-950 border border-gray-700 rounded px-2 py-1 text-xs text-gray-200 focus:outline-none focus:border-brand-600" />
                  </label>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </section>
  )
}

// ── Main Models Page ────────────────────────────────────────────────────────

export function Models() {
  const [tab, setTab] = useState<'local' | 'cloud'>('local')
  const runners = useRunners()
  const runnerList = (runners.data ?? []).filter(r => r.enabled)
  const [selectedRunner, setSelectedRunner] = useState<number | undefined>(undefined)
  const sync = useSyncModels()
  const [syncMsg, setSyncMsg] = useState<string | null>(null)

  const selectedRunnerData = runnerList.find(r => r.id === selectedRunner)

  return (
    <div className="space-y-6">
      {/* Tab bar */}
      <div className="flex items-center justify-between">
        <div className="flex gap-1 bg-gray-900 rounded-lg p-1 w-fit">
          <button onClick={() => setTab('local')}
            className={`flex items-center gap-1.5 px-4 py-2 rounded-md text-sm transition-colors ${
              tab === 'local' ? 'bg-brand-900 text-brand-300' : 'text-gray-400 hover:text-gray-200'
            }`}>
            <Layers className="w-4 h-4" /> Local Models
          </button>
          <button onClick={() => setTab('cloud')}
            className={`flex items-center gap-1.5 px-4 py-2 rounded-md text-sm transition-colors ${
              tab === 'cloud' ? 'bg-brand-900 text-brand-300' : 'text-gray-400 hover:text-gray-200'
            }`}>
            <Cloud className="w-4 h-4" /> Cloud Models
          </button>
        </div>

        {tab === 'local' && runnerList.length >= 2 && (
          <button
            onClick={async () => {
              setSyncMsg(null)
              try {
                const result = await sync.mutateAsync()
                if (result.pulls.length === 0) {
                  setSyncMsg('All models already synced across runners')
                } else {
                  setSyncMsg(`Syncing ${result.pulls.length} model(s): ${result.pulls.map(p => `${p.model} → ${p.target}`).join(', ')}`)
                }
                setTimeout(() => setSyncMsg(null), 8000)
              } catch (e) {
                setSyncMsg(`Error: ${(e as Error).message}`)
              }
            }}
            disabled={sync.isPending}
            title="Sync models across all runners (pulls biggest weight that fits each runner)"
            className="flex items-center gap-1.5 text-xs bg-indigo-900/40 hover:bg-indigo-800/50 text-indigo-300 border border-indigo-800 px-3 py-2 rounded-lg transition-colors disabled:opacity-40"
          >
            {sync.isPending ? <Loader2 className="w-3 h-3 animate-spin" /> : <RefreshCcw className="w-3 h-3" />}
            Sync Models
          </button>
        )}
      </div>

      {syncMsg && (
        <div className="text-xs bg-indigo-900/20 border border-indigo-800/50 text-indigo-300 px-3 py-2 rounded-lg">
          {syncMsg}
        </div>
      )}

      {tab === 'local' ? (
        runnerList.length === 0 ? (
          <div className="bg-gray-900 border border-gray-800 rounded-xl py-12 text-center">
            <Server className="w-8 h-8 text-gray-700 mx-auto mb-3" />
            <p className="text-sm text-gray-500">No runners connected</p>
            <p className="text-xs text-gray-600 mt-1">Start an llm-agent on a GPU host to manage models.</p>
          </div>
        ) : (
          <div className="space-y-6">
            {/* Runner tabs */}
            <RunnerTabs runners={runnerList} selected={selectedRunner} onSelect={setSelectedRunner} />

            {/* Model content based on selection */}
            <div className="space-y-8">
              {selectedRunner === undefined ? (
                <>
                  <div className="flex items-center gap-2">
                    <Layers className="w-4 h-4 text-brand-400" />
                    <h2 className="text-base font-semibold text-gray-200">Models across all runners</h2>
                  </div>
                  <AllRunnersModelView runners={runnerList} />
                </>
              ) : (
                <RunnerModelView runnerId={selectedRunner} runnerHostname={selectedRunnerData?.hostname ?? ''} />
              )}

              <div className="border-t border-gray-800" />
              <LibraryBrowserSection
                selectedRunner={selectedRunner}
                selectedRunnerHostname={selectedRunnerData?.hostname}
                allRunners={runnerList}
              />
              <div className="border-t border-gray-800" />
              <ImageModelsSection />
            </div>
          </div>
        )
      ) : (
        <CloudModelsSection />
      )}
    </div>
  )
}
