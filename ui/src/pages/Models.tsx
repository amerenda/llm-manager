import { useState } from 'react'
import { Download, Trash2, Loader2, CheckCircle2, AlertCircle, Image, Layers, Cpu, Upload, Shield, ShieldOff, Play, Square, Search, RefreshCw, BookOpen } from 'lucide-react'
import { useLlmModels, usePullModel, useDeleteModel, useCheckpoints, useSwitchCheckpoint, useLlmStatus, useLoadModel, useUnloadModel, useStartComfyui, useStopComfyui, useLibrary, useRefreshLibrary } from '../hooks/useBackend'
import type { LlmModel, LibraryModel } from '../types'

function stripExt(filename: string): string {
  return filename.replace(/\.[^.]+$/, '')
}

function TextModelsSection() {
  const models = useLlmModels()
  const status = useLlmStatus()
  const pull = usePullModel()
  const del = useDeleteModel()
  const load = useLoadModel()
  const unload = useUnloadModel()
  const [pullInput, setPullInput] = useState('')
  const [pullMsg, setPullMsg] = useState<{ type: 'ok' | 'err'; text: string } | null>(null)
  const [showUnsafe, setShowUnsafe] = useState(false)

  const textModels = (models.data ?? []).filter((m: LlmModel) => m.type === 'text')
  const loadedModels = status.data?.loaded_ollama_models ?? []
  const loadedNames = new Set(loadedModels.map(m => m.name))

  async function handlePull() {
    const name = pullInput.trim()
    if (!name) return
    setPullMsg(null)
    try {
      const result = await pull.mutateAsync(name)
      if (result.ok) {
        setPullMsg({ type: 'ok', text: `Pulled ${name}` })
        setPullInput('')
      } else {
        setPullMsg({ type: 'err', text: 'Pull returned not ok' })
      }
    } catch (e) {
      setPullMsg({ type: 'err', text: (e as Error).message })
    }
  }

  return (
    <section className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-brand-400" />
          <h2 className="text-base font-semibold text-gray-200">Text Models (Ollama)</h2>
        </div>
        {/* Safe/unsafe filter toggle */}
        <button
          onClick={() => setShowUnsafe(!showUnsafe)}
          className={`flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-lg transition-colors ${
            showUnsafe
              ? 'bg-red-900/40 text-red-400 border border-red-800'
              : 'bg-gray-800 text-gray-400 border border-gray-700 hover:border-gray-600'
          }`}
        >
          {showUnsafe ? <ShieldOff className="w-3 h-3" /> : <Shield className="w-3 h-3" />}
          {showUnsafe ? 'Showing all' : 'Safe only'}
        </button>
      </div>

      {/* Pull new model */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
        <p className="text-xs text-gray-500 mb-2 font-medium uppercase tracking-wide">Pull new model</p>
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
            {pull.isPending
              ? <Loader2 className="w-4 h-4 animate-spin" />
              : <Download className="w-4 h-4" />}
            Pull
          </button>
        </div>
        {pullMsg && (
          <div className={`mt-2 flex items-center gap-1.5 text-xs ${pullMsg.type === 'ok' ? 'text-green-400' : 'text-red-400'}`}>
            {pullMsg.type === 'ok'
              ? <CheckCircle2 className="w-3.5 h-3.5" />
              : <AlertCircle className="w-3.5 h-3.5" />}
            {pullMsg.text}
          </div>
        )}
      </div>

      {/* Model table */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
        {models.isLoading ? (
          <div className="py-8 text-center text-gray-600 text-sm">Loading models…</div>
        ) : textModels.length === 0 ? (
          <div className="py-8 text-center text-gray-600 text-sm">No text models found</div>
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
                        <span className="flex items-center gap-1.5 text-xs text-green-400">
                          <Cpu className="w-3 h-3" />
                          In VRAM{loadedInfo ? ` (${loadedInfo.size_gb} GB)` : ''}
                        </span>
                      ) : (
                        <span className="text-xs text-gray-600">Not loaded</span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <div className="flex items-center justify-end gap-2">
                        {isLoaded ? (
                          <button
                            onClick={() => unload.mutate({ model: m.id })}
                            disabled={unload.isPending}
                            title="Unload from VRAM"
                            className="flex items-center gap-1 text-xs bg-yellow-900/30 hover:bg-yellow-800/40 text-yellow-400 px-2.5 py-1 rounded-lg transition-colors disabled:opacity-40"
                          >
                            {unload.isPending ? <Loader2 className="w-3 h-3 animate-spin" /> : <Upload className="w-3 h-3" />}
                            Unload
                          </button>
                        ) : (
                          <button
                            onClick={() => load.mutate({ model: m.id })}
                            disabled={load.isPending}
                            title="Load into VRAM"
                            className="flex items-center gap-1 text-xs bg-brand-900/50 hover:bg-brand-800/50 text-brand-300 px-2.5 py-1 rounded-lg transition-colors disabled:opacity-40"
                          >
                            {load.isPending ? <Loader2 className="w-3 h-3 animate-spin" /> : <Cpu className="w-3 h-3" />}
                            Load
                          </button>
                        )}
                        <button
                          onClick={() => del.mutate(m.id)}
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
    try {
      await switchCp.mutateAsync(name)
    } finally {
      setSwitching(null)
    }
  }

  return (
    <section className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Image className="w-4 h-4 text-brand-400" />
          <h2 className="text-base font-semibold text-gray-200">Image Models (ComfyUI)</h2>
          <span className={`ml-1 text-xs px-2 py-0.5 rounded-full ${
            comfyRunning
              ? 'bg-green-900/40 text-green-400'
              : 'bg-gray-800 text-gray-500'
          }`}>
            {comfyRunning ? 'Running' : 'Stopped'}
          </span>
        </div>
        {comfyRunning ? (
          <button
            onClick={() => stopComfy.mutate()}
            disabled={stopComfy.isPending}
            className="flex items-center gap-1.5 text-xs bg-red-900/30 hover:bg-red-800/40 text-red-400 px-3 py-1.5 rounded-lg transition-colors disabled:opacity-40"
          >
            {stopComfy.isPending ? <Loader2 className="w-3 h-3 animate-spin" /> : <Square className="w-3 h-3" />}
            Stop ComfyUI
          </button>
        ) : (
          <button
            onClick={() => startComfy.mutate()}
            disabled={startComfy.isPending}
            className="flex items-center gap-1.5 text-xs bg-green-900/30 hover:bg-green-800/40 text-green-400 px-3 py-1.5 rounded-lg transition-colors disabled:opacity-40"
          >
            {startComfy.isPending ? <Loader2 className="w-3 h-3 animate-spin" /> : <Play className="w-3 h-3" />}
            Start ComfyUI
          </button>
        )}
      </div>

      {checkpoints.isLoading ? (
        <div className="py-8 text-center text-gray-600 text-sm">Loading checkpoints…</div>
      ) : cpList.length === 0 ? (
        <div className="bg-gray-900 border border-gray-800 rounded-xl py-8 text-center text-gray-600 text-sm">
          No checkpoints found
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {cpList.map(cp => {
            const isActive = cp === activeCheckpoint
            const isLoading = switching === cp
            return (
              <button
                key={cp}
                onClick={() => !isActive && handleSwitch(cp)}
                disabled={isActive || isLoading || !comfyRunning}
                className={`text-left bg-gray-900 border rounded-xl p-4 transition-all ${
                  isActive
                    ? 'border-brand-600 bg-brand-900/20 cursor-default'
                    : !comfyRunning
                    ? 'border-gray-800 opacity-50 cursor-not-allowed'
                    : 'border-gray-800 hover:border-gray-700 hover:bg-gray-800/50 cursor-pointer'
                }`}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <p className={`text-sm font-medium truncate ${isActive ? 'text-brand-300' : 'text-gray-200'}`}>
                      {stripExt(cp)}
                    </p>
                    <p className="text-xs text-gray-600 truncate mt-0.5">{cp}</p>
                  </div>
                  {isLoading && <Loader2 className="w-4 h-4 text-brand-400 animate-spin flex-shrink-0 mt-0.5" />}
                  {isActive && !isLoading && (
                    <span className="text-xs bg-brand-900 text-brand-300 px-1.5 py-0.5 rounded flex-shrink-0">
                      Active
                    </span>
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

function LibraryBrowserSection() {
  const [search, setSearch] = useState('')
  const [safety, setSafety] = useState<string>('safe')
  const [fitsOnly, setFitsOnly] = useState(true)
  const [sort, setSort] = useState<string>('pulls')
  const pull = usePullModel()
  const refresh = useRefreshLibrary()
  const [pullMsg, setPullMsg] = useState<{ model: string; type: 'ok' | 'err' | 'pulling'; text: string } | null>(null)

  const library = useLibrary({
    search: search || undefined,
    safety,
    fits: fitsOnly || undefined,
  })

  // Client-side sort (API also sorts, but this is instant for filter changes)
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
  const runners = library.data?.runners ?? []
  const cacheAge = library.data?.cache_age_hours ?? 0

  async function handlePull(model: string) {
    setPullMsg({ model, type: 'pulling', text: 'Pulling...' })
    try {
      await pull.mutateAsync(model)
      setPullMsg({ model, type: 'ok', text: 'Pulled!' })
      setTimeout(() => setPullMsg(null), 3000)
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

      {/* Model list — card layout to avoid horizontal scroll */}
      <div className="space-y-2">
        {library.isLoading ? (
          <div className="py-8 text-center text-gray-600 text-sm">Loading library...</div>
        ) : models.length === 0 ? (
          <div className="py-8 text-center text-gray-600 text-sm">No models match your filters</div>
        ) : (
          models.map((m: LibraryModel) => (
            <div key={m.name} className="bg-gray-900 border border-gray-800 rounded-xl p-4 hover:bg-gray-800/40 transition-colors">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-sm text-gray-200 font-medium">{m.name}</span>
                    {m.safety === 'unsafe' && (
                      <span className="text-[10px] bg-red-900/40 text-red-400 px-1.5 py-0.5 rounded">unsafe</span>
                    )}
                    {m.downloaded && (
                      <span className="flex items-center gap-0.5 text-[10px] text-green-400">
                        <CheckCircle2 className="w-2.5 h-2.5" /> Downloaded
                      </span>
                    )}
                  </div>
                  {m.description && (
                    <p className="text-xs text-gray-500 mt-1 line-clamp-1">{m.description}</p>
                  )}
                  <div className="flex items-center gap-3 mt-2 flex-wrap">
                    {m.parameter_sizes.length > 0 && (
                      <div className="flex gap-1 flex-wrap">
                        {m.parameter_sizes.map(s => (
                          <span key={s} className="text-[10px] bg-blue-900/30 text-blue-400 px-1.5 py-0.5 rounded">{s}</span>
                        ))}
                      </div>
                    )}
                    {m.categories.length > 0 && (
                      <div className="flex gap-1">
                        {m.categories.map(c => (
                          <span key={c} className="text-[10px] bg-indigo-900/30 text-indigo-400 px-1.5 py-0.5 rounded">{c}</span>
                        ))}
                      </div>
                    )}
                    <span className="text-[10px] text-gray-600">~{m.vram_estimate_gb}GB VRAM</span>
                    {m.pulls && <span className="text-[10px] text-gray-600">{m.pulls} pulls</span>}
                  </div>
                </div>
                <div className="flex-shrink-0">
                  {!m.downloaded && m.fits && (
                    <button
                      onClick={() => handlePull(m.name)}
                      disabled={pullMsg?.model === m.name && pullMsg.type === 'pulling'}
                      className="flex items-center gap-1 text-xs bg-brand-900/50 hover:bg-brand-800/50 text-brand-300 px-3 py-1.5 rounded-lg transition-colors disabled:opacity-40"
                    >
                      {pullMsg?.model === m.name && pullMsg.type === 'pulling'
                        ? <Loader2 className="w-3 h-3 animate-spin" />
                        : <Download className="w-3 h-3" />}
                      Pull
                    </button>
                  )}
                  {!m.fits && !m.downloaded && (
                    <span className="text-[10px] text-gray-600">Too large</span>
                  )}
                  {pullMsg?.model === m.name && pullMsg.type !== 'pulling' && (
                    <span className={`text-[10px] ${pullMsg.type === 'ok' ? 'text-green-400' : 'text-red-400'}`}>
                      {pullMsg.text}
                    </span>
                  )}
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </section>
  )
}

export function Models() {
  return (
    <div className="space-y-8">
      <TextModelsSection />
      <div className="border-t border-gray-800" />
      <LibraryBrowserSection />
      <div className="border-t border-gray-800" />
      <ImageModelsSection />
    </div>
  )
}
