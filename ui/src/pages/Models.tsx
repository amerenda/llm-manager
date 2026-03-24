import { useState } from 'react'
import { Download, Trash2, Loader2, CheckCircle2, AlertCircle, Image, Layers, Cpu, Upload, Shield, ShieldOff, Play, Square, Search, RefreshCw, BookOpen, Cloud, Settings2, Power } from 'lucide-react'
import { useLlmModels, usePullModel, useDeleteModel, useCheckpoints, useSwitchCheckpoint, useLlmStatus, useLoadModel, useUnloadModel, useStartComfyui, useStopComfyui, useLibrary, useRefreshLibrary, useCloudModels, useCloudStatus, useUpdateCloudModel, useCloudKeys, useStoreCloudKey, useDeleteCloudKey, useRunners } from '../hooks/useBackend'
import type { LlmModel, LibraryModel, Runner } from '../types'
import type { CloudModel, StoredApiKey } from '../hooks/useBackend'

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
      const result = await pull.mutateAsync({ model: name })
      if (result.ok) {
        setPullMsg({ type: 'ok', text: `Pulling ${name}...` })
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
                          onClick={() => {
                            if (window.confirm(`Delete ${m.id}? This will remove the model from disk.`)) {
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

function LibraryBrowserSection({ selectedRunner }: { selectedRunner?: number }) {
  const [search, setSearch] = useState('')
  const [safety, setSafety] = useState<string>('safe')
  const [fitsOnly, setFitsOnly] = useState(true)
  const [hideDownloaded, setHideDownloaded] = useState(true)
  const [sort, setSort] = useState<string>('pulls')
  const [expanded, setExpanded] = useState<string | null>(null)
  const pull = usePullModel()
  const refresh = useRefreshLibrary()
  const [pullMsg, setPullMsg] = useState<{ model: string; type: 'ok' | 'err' | 'pulling'; text: string } | null>(null)

  const library = useLibrary({
    search: search || undefined,
    safety,
    fits: fitsOnly || undefined,
    downloaded: hideDownloaded ? false : undefined,
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
  const cacheAge = library.data?.cache_age_hours ?? 0

  async function handlePull(model: string) {
    setPullMsg({ model, type: 'pulling', text: 'Pulling...' })
    try {
      await pull.mutateAsync({ model, runner_id: selectedRunner })
      setPullMsg({ model, type: 'ok', text: 'Pulling in background' })
      setTimeout(() => setPullMsg(null), 5000)
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

      {/* Model list — card layout to avoid horizontal scroll */}
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
                      {m.downloaded && (
                        <span className="flex items-center gap-0.5 text-[10px] text-green-400">
                          <CheckCircle2 className="w-2.5 h-2.5" /> Downloaded
                        </span>
                      )}
                      {m.loaded && (
                        <span className="text-[10px] bg-green-900/40 text-green-400 px-1.5 py-0.5 rounded">In VRAM</span>
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
                  <div className="flex-shrink-0" onClick={e => e.stopPropagation()}>
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

              {/* Expanded detail panel */}
              {isExpanded && (
                <div className="border-t border-gray-800 px-4 py-3 space-y-3">
                  {/* Available sizes */}
                  {m.parameter_sizes.length > 0 && (
                    <div>
                      <p className="text-[10px] text-gray-500 uppercase tracking-wide mb-1.5">Available sizes</p>
                      <div className="flex gap-1.5 flex-wrap">
                        {m.parameter_sizes.map(s => (
                          <button
                            key={s}
                            onClick={e => { e.stopPropagation(); handlePull(`${m.name}:${s}`) }}
                            disabled={pullMsg?.model === `${m.name}:${s}` && pullMsg.type === 'pulling'}
                            className="flex items-center gap-1 text-xs bg-blue-900/20 hover:bg-blue-900/40 text-blue-400 px-2 py-1 rounded transition-colors"
                          >
                            <Download className="w-2.5 h-2.5" />
                            {s}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Categories */}
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

                  {/* Stats */}
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
        <Shield className="w-4 h-4 text-brand-400" />
        API Keys
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
                <button
                  onClick={() => remove.mutate(existing[0].id)}
                  disabled={remove.isPending}
                  className="p-1.5 rounded-lg bg-gray-800 hover:bg-red-900/50 hover:text-red-400 text-gray-500 transition-colors"
                  title="Remove key"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              )}
              <button
                onClick={() => setAdding(adding === p.id ? null : p.id)}
                className="text-xs px-2.5 py-1.5 rounded-lg bg-brand-900/50 hover:bg-brand-800/50 text-brand-300 transition-colors"
              >
                {existing.length > 0 ? 'Replace' : 'Add key'}
              </button>
            </div>
          </div>
        )
      })}
      {adding && (
        <div className="flex gap-2 mt-2">
          <input
            type="password"
            value={keyInput}
            onChange={e => setKeyInput(e.target.value)}
            placeholder={`${adding} API key`}
            className="flex-1 bg-gray-950 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-brand-600"
          />
          <input
            type="text"
            value={labelInput}
            onChange={e => setLabelInput(e.target.value)}
            placeholder="Label (optional)"
            className="w-32 bg-gray-950 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-brand-600"
          />
          <button
            onClick={() => {
              if (keyInput.trim()) {
                store.mutate({ provider: adding, key: keyInput.trim(), label: labelInput.trim() })
                setKeyInput('')
                setLabelInput('')
                setAdding(null)
              }
            }}
            disabled={store.isPending || !keyInput.trim()}
            className="px-3 py-1.5 rounded-lg bg-brand-600 hover:bg-brand-500 text-white text-sm transition-colors disabled:opacity-40"
          >
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
              anthropicStatus.reachable
                ? 'bg-green-900/40 text-green-400'
                : anthropicStatus.configured
                ? 'bg-yellow-900/40 text-yellow-400'
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
                    {!m.enabled && (
                      <span className="text-[10px] bg-gray-800 text-gray-500 px-1.5 py-0.5 rounded">disabled</span>
                    )}
                  </div>
                  <p className="text-xs text-gray-500 mt-0.5 font-mono">{m.id}</p>
                  <div className="flex items-center gap-3 mt-1.5 text-[10px] text-gray-500">
                    <span>Max tokens: {m.max_tokens}</span>
                    {m.temperature !== null && <span>Temperature: {m.temperature}</span>}
                  </div>
                </div>
                <div className="flex items-center gap-1.5 flex-shrink-0">
                  <button
                    onClick={() => {
                      if (editing === m.id) {
                        // Save
                        update.mutate({ modelId: m.id, ...editValues })
                        setEditing(null)
                        setEditValues({})
                      } else {
                        setEditing(m.id)
                        setEditValues({ max_tokens: m.max_tokens, temperature: m.temperature ?? undefined })
                      }
                    }}
                    className="p-1.5 rounded-lg bg-gray-800 hover:bg-gray-700 text-gray-400 transition-colors"
                    title={editing === m.id ? 'Save' : 'Configure'}
                  >
                    {editing === m.id ? <CheckCircle2 className="w-3.5 h-3.5 text-green-400" /> : <Settings2 className="w-3.5 h-3.5" />}
                  </button>
                  <button
                    onClick={() => update.mutate({ modelId: m.id, enabled: !m.enabled })}
                    className={`p-1.5 rounded-lg transition-colors ${
                      m.enabled
                        ? 'bg-green-900/30 hover:bg-green-900/50 text-green-400'
                        : 'bg-gray-800 hover:bg-gray-700 text-gray-500'
                    }`}
                    title={m.enabled ? 'Disable' : 'Enable'}
                  >
                    <Power className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
              {editing === m.id && (
                <div className="mt-3 pt-3 border-t border-gray-800 flex gap-3">
                  <label className="flex items-center gap-2 text-xs text-gray-400">
                    Max tokens
                    <input
                      type="number"
                      value={editValues.max_tokens ?? ''}
                      onChange={e => setEditValues(v => ({ ...v, max_tokens: parseInt(e.target.value) || undefined }))}
                      className="w-24 bg-gray-950 border border-gray-700 rounded px-2 py-1 text-xs text-gray-200 focus:outline-none focus:border-brand-600"
                    />
                  </label>
                  <label className="flex items-center gap-2 text-xs text-gray-400">
                    Temperature
                    <input
                      type="number"
                      step="0.1"
                      min="0"
                      max="2"
                      value={editValues.temperature ?? ''}
                      onChange={e => setEditValues(v => ({ ...v, temperature: parseFloat(e.target.value) || undefined }))}
                      className="w-20 bg-gray-950 border border-gray-700 rounded px-2 py-1 text-xs text-gray-200 focus:outline-none focus:border-brand-600"
                    />
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

export function Models() {
  const [tab, setTab] = useState<'local' | 'cloud'>('local')
  const runners = useRunners()
  const runnerList = runners.data ?? []
  const [selectedRunner, setSelectedRunner] = useState<number | undefined>(undefined)

  return (
    <div className="space-y-6">
      {/* Tab bar + runner selector */}
      <div className="flex items-center justify-between">
        <div className="flex gap-1 bg-gray-900 rounded-lg p-1 w-fit">
          <button
            onClick={() => setTab('local')}
            className={`flex items-center gap-1.5 px-4 py-2 rounded-md text-sm transition-colors ${
              tab === 'local' ? 'bg-brand-900 text-brand-300' : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            <Layers className="w-4 h-4" />
            Local Models
          </button>
          <button
            onClick={() => setTab('cloud')}
            className={`flex items-center gap-1.5 px-4 py-2 rounded-md text-sm transition-colors ${
              tab === 'cloud' ? 'bg-brand-900 text-brand-300' : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            <Cloud className="w-4 h-4" />
            Cloud Models
          </button>
        </div>

        {tab === 'local' && runnerList.length > 0 && (
          <select
            value={selectedRunner ?? ''}
            onChange={e => setSelectedRunner(e.target.value ? parseInt(e.target.value) : undefined)}
            className="text-xs bg-gray-900 border border-gray-800 rounded-lg px-2.5 py-2 text-gray-400 focus:outline-none focus:border-brand-600"
          >
            <option value="">All runners</option>
            {runnerList.map((r: Runner) => (
              <option key={r.id} value={r.id}>{r.hostname}</option>
            ))}
          </select>
        )}
      </div>

      {tab === 'local' ? (
        <div className="space-y-8">
          <TextModelsSection />
          <div className="border-t border-gray-800" />
          <LibraryBrowserSection selectedRunner={selectedRunner} />
          <div className="border-t border-gray-800" />
          <ImageModelsSection />
        </div>
      ) : (
        <CloudModelsSection />
      )}
    </div>
  )
}
