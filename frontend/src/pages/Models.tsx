import { useState, type ReactNode } from 'react'
import { Download, Trash2, Loader2, CheckCircle2, AlertCircle, Image, Layers, Cpu, Upload, Shield, ShieldOff, Play, Square, Search, RefreshCw, BookOpen, Cloud, Settings2, Power, RefreshCcw, Server, X, Pin } from 'lucide-react'
import { usePullModel, useDeleteModel, useCheckpoints, useSwitchCheckpoint, useLlmStatus, useLoadModel, useUnloadFromVram, useStartComfyui, useStopComfyui, useLibrary, useRefreshLibrary, useRefreshRemoteDigests, useUpdateOutdatedModels, useForceUpdateModel, useCommunityModels, useCloudModels, useCloudStatus, useUpdateCloudModel, useCloudKeys, useStoreCloudKey, useDeleteCloudKey, useRunners, useSyncModels, useOps, useDismissOp, useModelList, useUpdateModelSettings, useAliasesForModel, useUpsertAlias, useDeleteAlias, useRunnerParamsForModel, useUpsertRunnerParams, useDeleteRunnerParams, usePinModelOnRunner } from '../hooks/useBackend'
import type { ModelAlias, ModelRunnerParams } from '../hooks/useBackend'
import type { LibraryModel, Runner } from '../types'
import type { CloudModel, ModelInfo, StoredApiKey } from '../hooks/useBackend'

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

// ── Model Settings Modal ─────────────────────────────────────────────────────

const STANDARD_CATEGORIES = ['tools', 'vision', 'thinking', 'embedding']
const PARAM_FIELDS: { key: string; label: string; min: number; max: number; step: number }[] = [
  { key: 'temperature', label: 'Temperature', min: 0, max: 2, step: 0.05 },
  { key: 'top_p', label: 'Top P', min: 0, max: 1, step: 0.05 },
  { key: 'top_k', label: 'Top K', min: 1, max: 200, step: 1 },
  { key: 'num_ctx', label: 'Context (tokens)', min: 512, max: 131072, step: 512 },
  { key: 'repeat_penalty', label: 'Repeat Penalty', min: 0, max: 2, step: 0.05 },
  { key: 'num_predict', label: 'Max Predict', min: -1, max: 8192, step: 64 },
]

function ParamEditor({
  params, onChange,
}: { params: Record<string, unknown>; onChange: (p: Record<string, unknown>) => void }) {
  function set(key: string, val: string) {
    const n = parseFloat(val)
    onChange({ ...params, [key]: isNaN(n) ? undefined : n })
  }
  function clear(key: string) {
    const next = { ...params }
    delete next[key]
    onChange(next)
  }
  return (
    <div className="space-y-2">
      {PARAM_FIELDS.map(f => (
        <div key={f.key} className="flex items-center gap-2">
          <span className="text-xs text-gray-400 w-32 shrink-0">{f.label}</span>
          <input
            type="number" min={f.min} max={f.max} step={f.step}
            value={params[f.key] !== undefined ? String(params[f.key]) : ''}
            placeholder="default"
            onChange={e => set(f.key, e.target.value)}
            className="w-24 bg-gray-950 border border-gray-700 rounded px-2 py-1 text-xs text-gray-200 placeholder-gray-600 focus:outline-none focus:border-brand-600"
          />
          {params[f.key] !== undefined && (
            <button onClick={() => clear(f.key)} className="text-gray-600 hover:text-gray-400">
              <X className="w-3 h-3" />
            </button>
          )}
        </div>
      ))}
    </div>
  )
}

function RunnerParamsTab({ modelName, runners }: { modelName: string; runners: import('../types').Runner[] }) {
  const { data: allParams = [] } = useRunnerParamsForModel(modelName)
  const upsert = useUpsertRunnerParams()
  const remove = useDeleteRunnerParams()
  const [editing, setEditing] = useState<number | null>(null)
  const [draft, setDraft] = useState<{ system_prompt: string; parameters: Record<string, unknown> }>({
    system_prompt: '', parameters: {},
  })

  function startEdit(runner_id: number) {
    const existing = allParams.find(p => p.runner_id === runner_id)
    setDraft({
      system_prompt: existing?.system_prompt ?? '',
      parameters: existing?.parameters ?? {},
    })
    setEditing(runner_id)
  }

  function save(runner_id: number) {
    upsert.mutate({
      model_name: modelName,
      runner_id,
      hostname: null,
      system_prompt: draft.system_prompt || null,
      parameters: draft.parameters,
    }, { onSuccess: () => setEditing(null) })
  }

  return (
    <div className="space-y-3">
      {runners.map(r => {
        const existing = allParams.find(p => p.runner_id === r.id)
        const isEditing = editing === r.id
        return (
          <div key={r.id} className="bg-gray-800/60 rounded-lg p-3">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-medium text-gray-300">{r.hostname}</span>
              <div className="flex gap-1.5">
                {existing && !isEditing && (
                  <button onClick={() => remove.mutate({ model_name: modelName, runner_id: r.id })}
                    className="text-[10px] text-red-500 hover:text-red-400 px-1.5 py-0.5 rounded">
                    Reset
                  </button>
                )}
                <button onClick={() => isEditing ? setEditing(null) : startEdit(r.id)}
                  className="text-[10px] text-gray-400 hover:text-gray-200 px-1.5 py-0.5 rounded bg-gray-700">
                  {isEditing ? 'Cancel' : existing ? 'Edit' : 'Add override'}
                </button>
              </div>
            </div>
            {!isEditing && existing && (
              <div className="text-[10px] text-gray-500 space-y-0.5">
                {existing.system_prompt && <p className="truncate">System: {existing.system_prompt}</p>}
                {Object.entries(existing.parameters).map(([k, v]) => (
                  <span key={k} className="inline-block mr-2">{k}={String(v)}</span>
                ))}
              </div>
            )}
            {!isEditing && !existing && (
              <p className="text-[10px] text-gray-600">Using defaults</p>
            )}
            {isEditing && (
              <div className="space-y-3 mt-2">
                <div>
                  <p className="text-[10px] text-gray-500 uppercase tracking-wide mb-1">System Prompt</p>
                  <textarea
                    rows={2}
                    value={draft.system_prompt}
                    onChange={e => setDraft(d => ({ ...d, system_prompt: e.target.value }))}
                    placeholder="Override system prompt (leave blank to inherit)"
                    className="w-full bg-gray-950 border border-gray-700 rounded px-2 py-1.5 text-xs text-gray-200 placeholder-gray-600 resize-none focus:outline-none focus:border-brand-600"
                  />
                </div>
                <div>
                  <p className="text-[10px] text-gray-500 uppercase tracking-wide mb-2">Parameters</p>
                  <ParamEditor params={draft.parameters} onChange={p => setDraft(d => ({ ...d, parameters: p }))} />
                </div>
                <button onClick={() => save(r.id)} disabled={upsert.isPending}
                  className="w-full text-xs py-1.5 rounded bg-brand-600 text-white hover:bg-brand-500 disabled:opacity-40 transition-colors">
                  Save
                </button>
              </div>
            )}
          </div>
        )
      })}
      {runners.length === 0 && <p className="text-xs text-gray-600">No runners available</p>}
    </div>
  )
}

function AliasesTab({ baseModel }: { baseModel: string }) {
  const { data: aliases = [], isLoading } = useAliasesForModel(baseModel)
  const upsert = useUpsertAlias()
  const remove = useDeleteAlias()
  const [creating, setCreating] = useState(false)
  const [editingId, setEditingId] = useState<number | null>(null)
  const [saveError, setSaveError] = useState<string | null>(null)
  const blankDraft = { alias_name: '', base_model: baseModel, system_prompt: '', parameters: {} as Record<string, unknown>, description: '' }
  const [draft, setDraft] = useState(blankDraft)

  function startCreate() {
    setDraft({ ...blankDraft })
    setCreating(true)
    setEditingId(null)
  }

  function startEdit(a: ModelAlias) {
    setDraft({ alias_name: a.alias_name, base_model: baseModel, system_prompt: a.system_prompt ?? '', parameters: a.parameters, description: a.description })
    setEditingId(a.id)
    setCreating(false)
  }

  function save() {
    setSaveError(null)
    upsert.mutate(
      { ...draft, system_prompt: draft.system_prompt || null },
      {
        onSuccess: () => { setCreating(false); setEditingId(null); setSaveError(null) },
        onError: (e) => setSaveError((e as Error).message),
      }
    )
  }

  const editForm = (
    <div className="bg-gray-800/60 rounded-lg p-3 space-y-2 mt-2">
      <input value={draft.alias_name} onChange={e => setDraft(d => ({ ...d, alias_name: e.target.value }))}
        placeholder="Alias name (e.g. qwen3-thinking)"
        disabled={editingId !== null}
        className="w-full bg-gray-950 border border-gray-700 rounded px-2 py-1.5 text-xs text-gray-200 placeholder-gray-600 focus:outline-none focus:border-brand-600 disabled:opacity-50"
      />
      <input value={draft.description} onChange={e => setDraft(d => ({ ...d, description: e.target.value }))}
        placeholder="Description (optional)"
        className="w-full bg-gray-950 border border-gray-700 rounded px-2 py-1.5 text-xs text-gray-200 placeholder-gray-600 focus:outline-none focus:border-brand-600"
      />
      <textarea rows={2} value={draft.system_prompt ?? ''} onChange={e => setDraft(d => ({ ...d, system_prompt: e.target.value }))}
        placeholder="System prompt (optional)"
        className="w-full bg-gray-950 border border-gray-700 rounded px-2 py-1.5 text-xs text-gray-200 placeholder-gray-600 resize-none focus:outline-none focus:border-brand-600"
      />
      <ParamEditor params={draft.parameters} onChange={p => setDraft(d => ({ ...d, parameters: p }))} />
      {saveError && <p className="text-xs text-red-400">{saveError}</p>}
      <div className="flex gap-2 pt-1">
        <button onClick={() => { setCreating(false); setEditingId(null); setSaveError(null) }}
          className="flex-1 text-xs py-1.5 rounded bg-gray-700 text-gray-400 hover:bg-gray-600 transition-colors">
          Cancel
        </button>
        <button onClick={save} disabled={!draft.alias_name.trim() || upsert.isPending}
          className="flex-1 text-xs py-1.5 rounded bg-brand-600 text-white hover:bg-brand-500 disabled:opacity-40 transition-colors">
          {upsert.isPending ? 'Saving…' : 'Save'}
        </button>
      </div>
    </div>
  )

  return (
    <div className="space-y-2">
      {isLoading && <p className="text-xs text-gray-600">Loading…</p>}
      {aliases.map(a => (
        <div key={a.id}>
          {editingId === a.id ? editForm : (
            <div className="bg-gray-800/60 rounded-lg p-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono text-brand-300">{a.alias_name}</span>
                <div className="flex gap-1.5">
                  <button onClick={() => startEdit(a)} className="text-[10px] text-gray-400 hover:text-gray-200 px-1.5 py-0.5 rounded bg-gray-700">Edit</button>
                  <button onClick={() => remove.mutate(a.alias_name)} className="text-[10px] text-red-500 hover:text-red-400 px-1.5 py-0.5 rounded">Delete</button>
                </div>
              </div>
              {a.description && <p className="text-[10px] text-gray-500 mt-0.5">{a.description}</p>}
              {a.system_prompt && <p className="text-[10px] text-gray-500 mt-0.5 truncate">System: {a.system_prompt}</p>}
              {Object.keys(a.parameters).length > 0 && (
                <p className="text-[10px] text-gray-600 mt-0.5">
                  {Object.entries(a.parameters).map(([k, v]) => `${k}=${v}`).join(' · ')}
                </p>
              )}
            </div>
          )}
        </div>
      ))}
      {creating ? editForm : (
        <button onClick={startCreate}
          className="w-full text-xs py-1.5 rounded border border-dashed border-gray-700 text-gray-500 hover:border-gray-500 hover:text-gray-300 transition-colors">
          + New alias
        </button>
      )}
    </div>
  )
}

type SettingsTab = 'general' | 'runner-params' | 'aliases'

function AliasSettingsModal({ model, onClose }: { model: ModelInfo; onClose: () => void }) {
  const mAny = model as unknown as { base_model: string; alias_description?: string; alias_parameters?: Record<string, unknown>; alias_system_prompt?: string | null }
  const upsert = useUpsertAlias()
  const [description, setDescription] = useState(mAny.alias_description ?? '')
  const [systemPrompt, setSystemPrompt] = useState(mAny.alias_system_prompt ?? '')
  const [parameters, setParameters] = useState<Record<string, unknown>>(mAny.alias_parameters ?? {})
  const [saveError, setSaveError] = useState<string | null>(null)

  function save() {
    setSaveError(null)
    upsert.mutate(
      { alias_name: model.name, base_model: mAny.base_model, system_prompt: systemPrompt || null, parameters, description },
      { onSuccess: onClose, onError: (e) => setSaveError((e as Error).message) }
    )
  }

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50" onClick={onClose}>
      <div className="bg-gray-900 border border-gray-700 rounded-xl p-5 w-[28rem] max-h-[90vh] overflow-y-auto" onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-1">
          <h3 className="text-sm font-semibold text-gray-200">Alias Settings</h3>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-300"><X className="w-4 h-4" /></button>
        </div>
        <p className="text-xs text-gray-500 font-mono mb-1 truncate">{model.name}</p>
        <p className="text-[10px] text-gray-600 mb-4">alias of <span className="text-indigo-400">{mAny.base_model}</span></p>

        <div className="space-y-3">
          <div>
            <p className="text-[10px] text-gray-500 uppercase tracking-wide mb-1.5">Description</p>
            <input value={description} onChange={e => setDescription(e.target.value)}
              placeholder="Short description (optional)"
              className="w-full bg-gray-950 border border-gray-700 rounded-lg px-2.5 py-1.5 text-xs text-gray-200 placeholder-gray-600 focus:outline-none focus:border-brand-600"
            />
          </div>
          <div>
            <p className="text-[10px] text-gray-500 uppercase tracking-wide mb-1.5">System Prompt</p>
            <textarea rows={4} value={systemPrompt} onChange={e => setSystemPrompt(e.target.value)}
              placeholder="System prompt injected on every request (optional)"
              className="w-full bg-gray-950 border border-gray-700 rounded-lg px-2.5 py-1.5 text-xs text-gray-200 placeholder-gray-600 resize-none focus:outline-none focus:border-brand-600"
            />
          </div>
          <div>
            <p className="text-[10px] text-gray-500 uppercase tracking-wide mb-1.5">Parameters</p>
            <ParamEditor params={parameters} onChange={setParameters} />
          </div>
          {saveError && <p className="text-xs text-red-400">{saveError}</p>}
          <div className="flex gap-2 pt-1">
            <button onClick={onClose} className="flex-1 text-xs px-3 py-2 rounded-lg bg-gray-800 text-gray-400 hover:bg-gray-700 transition-colors">
              Cancel
            </button>
            <button onClick={save} disabled={upsert.isPending}
              className="flex-1 text-xs px-3 py-2 rounded-lg bg-brand-600 text-white hover:bg-brand-500 disabled:opacity-40 transition-colors">
              {upsert.isPending ? 'Saving…' : 'Save'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

function ModelSettingsModal({ model, runners, onClose }: { model: ModelInfo; runners: import('../types').Runner[]; onClose: () => void }) {
  const update = useUpdateModelSettings()
  const [tab, setTab] = useState<SettingsTab>('general')
  const [categories, setCategories] = useState<string[]>(model.categories ?? [])
  const [safety, setSafety] = useState(model.safety ?? 'safe')
  const [customCat, setCustomCat] = useState('')

  function toggleCategory(cat: string) {
    setCategories(prev => prev.includes(cat) ? prev.filter(c => c !== cat) : [...prev, cat])
  }

  function addCustom() {
    const c = customCat.trim().toLowerCase()
    if (c && !categories.includes(c)) {
      setCategories(prev => [...prev, c])
      setCustomCat('')
    }
  }

  function save() {
    update.mutate({ model: model.name, categories, safety }, { onSuccess: onClose })
  }

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50" onClick={onClose}>
      <div className="bg-gray-900 border border-gray-700 rounded-xl p-5 w-[28rem] max-h-[90vh] overflow-y-auto" onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-1">
          <h3 className="text-sm font-semibold text-gray-200">Model Settings</h3>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-300"><X className="w-4 h-4" /></button>
        </div>
        <p className="text-xs text-gray-500 font-mono mb-3 truncate">{model.name}</p>

        <div className="flex gap-1 bg-gray-800 rounded-lg p-0.5 mb-4 text-[11px]">
          {(['general', 'runner-params', 'aliases'] as SettingsTab[]).map(t => (
            <button key={t} onClick={() => setTab(t)}
              className={`flex-1 py-1.5 rounded-md transition-colors font-medium ${
                tab === t ? 'bg-gray-700 text-gray-200' : 'text-gray-500 hover:text-gray-300'
              }`}>
              {t === 'general' ? 'General' : t === 'runner-params' ? 'Runner Params' : 'Aliases'}
            </button>
          ))}
        </div>

        {tab === 'general' && (
          <div className="space-y-4">
            <div>
              <p className="text-[10px] text-gray-500 uppercase tracking-wide mb-2">Safety</p>
              <div className="flex gap-2">
                {(['safe', 'unsafe'] as const).map(s => (
                  <button key={s} onClick={() => setSafety(s)}
                    className={`text-xs px-3 py-1.5 rounded-lg transition-colors border ${
                      safety === s
                        ? s === 'safe' ? 'bg-green-900/50 text-green-400 border-green-700' : 'bg-red-900/50 text-red-400 border-red-700'
                        : 'bg-gray-800 text-gray-400 hover:bg-gray-700 border-transparent'
                    }`}>
                    {s === 'safe' ? 'Safe' : 'Unsafe'}
                  </button>
                ))}
              </div>
            </div>
            <div>
              <p className="text-[10px] text-gray-500 uppercase tracking-wide mb-2">Categories</p>
              <div className="flex gap-1.5 flex-wrap mb-2">
                {STANDARD_CATEGORIES.map(cat => (
                  <button key={cat} onClick={() => toggleCategory(cat)}
                    className={`text-xs px-2.5 py-1 rounded-full transition-colors border ${
                      categories.includes(cat)
                        ? 'bg-brand-900 text-brand-300 border-brand-700'
                        : 'bg-gray-800 text-gray-500 hover:bg-gray-700 hover:text-gray-300 border-transparent'
                    }`}>
                    {cat}
                  </button>
                ))}
              </div>
              {categories.filter(c => !STANDARD_CATEGORIES.includes(c)).length > 0 && (
                <div className="flex flex-wrap gap-1.5 mb-2">
                  {categories.filter(c => !STANDARD_CATEGORIES.includes(c)).map(c => (
                    <span key={c} className="inline-flex items-center gap-1 text-xs bg-indigo-900/40 text-indigo-300 px-2 py-0.5 rounded-full">
                      {c}
                      <button onClick={() => toggleCategory(c)} className="hover:text-red-400"><X className="w-2.5 h-2.5" /></button>
                    </span>
                  ))}
                </div>
              )}
              <div className="flex gap-2 mt-1">
                <input type="text" value={customCat} onChange={e => setCustomCat(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && addCustom()}
                  placeholder="Add custom category"
                  className="flex-1 bg-gray-950 border border-gray-700 rounded-lg px-2.5 py-1.5 text-xs text-gray-200 placeholder-gray-600 focus:outline-none focus:border-brand-600"
                />
                <button onClick={addCustom} disabled={!customCat.trim()}
                  className="text-xs px-2.5 py-1.5 rounded-lg bg-gray-800 text-gray-400 hover:bg-gray-700 disabled:opacity-30 transition-colors">
                  Add
                </button>
              </div>
            </div>
            <div className="flex gap-2 pt-1">
              <button onClick={onClose} className="flex-1 text-xs px-3 py-2 rounded-lg bg-gray-800 text-gray-400 hover:bg-gray-700 transition-colors">
                Cancel
              </button>
              <button onClick={save} disabled={update.isPending}
                className="flex-1 text-xs px-3 py-2 rounded-lg bg-brand-600 text-white hover:bg-brand-500 disabled:opacity-40 transition-colors">
                {update.isPending ? 'Saving...' : 'Save'}
              </button>
            </div>
          </div>
        )}

        {tab === 'runner-params' && (
          <RunnerParamsTab modelName={model.name} runners={runners} />
        )}

        {tab === 'aliases' && (
          <AliasesTab baseModel={model.name} />
        )}
      </div>
    </div>
  )
}

// ── Installed Models View ────────────────────────────────────────────────────

function InstalledModelsView({ runners, selectedRunner, selectedRunnerHostname }: {
  runners: Runner[]
  selectedRunner?: number
  selectedRunnerHostname?: string
}) {
  const modelList = useModelList()
  const status = useLlmStatus()
  const pull = usePullModel()
  const del = useDeleteModel()
  const load = useLoadModel()
  const unloadVram = useUnloadFromVram()
  const updateSettings = useUpdateModelSettings()
  const pinRunner = usePinModelOnRunner()
  const ops = useOps()
  const [pullInput, setPullInput] = useState('')
  const [pullMsg, setPullMsg] = useState<{ type: 'ok' | 'err'; text: string } | null>(null)
  const [editingModel, setEditingModel] = useState<ModelInfo | null>(null)
  const [expandedModel, setExpandedModel] = useState<string | null>(null)

  const activeOps = (ops.data ?? []).filter(op => op.status === 'running')
  const failedOps = (ops.data ?? []).filter(op => op.status === 'failed')
  const dismissOp = useDismissOp()
  const allLoadedModels = status.data?.loaded_ollama_models ?? []
  const runnerStatuses = status.data?.runners ?? []

  const [searchFilter, setSearchFilter] = useState('')
  const [safetyFilter, setSafetyFilter] = useState('')
  const [capFilter, setCapFilter] = useState('')

  const allModels = (modelList.data ?? []).sort((a, b) => a.name.localeCompare(b.name))


  const models = (selectedRunner !== undefined
    ? allModels.filter(m => (m.runners ?? []).some(r => r.runner_id === selectedRunner))
    : allModels
  )
    .filter(m => !searchFilter || m.name.toLowerCase().includes(searchFilter.toLowerCase()))
    .filter(m => !safetyFilter || m.safety === safetyFilter)
    .filter(m => !capFilter || (m.categories ?? []).includes(capFilter))

  async function handlePull() {
    const name = pullInput.trim()
    if (!name) return
    setPullMsg(null)
    try {
      const result = await pull.mutateAsync({ model: name, runner_id: selectedRunner })
      if (result.ok) {
        setPullMsg({ type: 'ok', text: `Pulling ${name}...` })
        setPullInput('')
      }
    } catch (e) {
      setPullMsg({ type: 'err', text: (e as Error).message })
    }
  }

  const displayRunners = selectedRunner !== undefined
    ? runners.filter(r => r.id === selectedRunner)
    : runners

  return (
    <section className="space-y-4">
      {/* Resource bars */}
      <div className={`grid grid-cols-1 ${runners.length >= 2 && selectedRunner === undefined ? 'sm:grid-cols-2' : ''} gap-2`}>
        {displayRunners.map(r => {
          const rs = runnerStatuses.find(s => s.runner_id === r.id)
          const vramUsed = rs?.gpu_vram_used_gb ?? 0
          const vramTotal = rs?.gpu_vram_total_gb ?? 0
          const vramPct = vramTotal > 0 ? Math.round(vramUsed / vramTotal * 100) : 0
          const diskUsed = rs?.disk_used_gb ?? 0
          const diskTotal = rs?.disk_total_gb ?? 0
          const diskFree = rs?.disk_free_gb ?? 0
          const diskPct = diskTotal > 0 ? Math.round(diskUsed / diskTotal * 100) : 0
          if (!vramTotal && !diskTotal) return null
          return (
            <div key={r.id} className="bg-gray-900 border border-gray-800 rounded-xl p-3 space-y-2">
              <p className="text-[10px] text-gray-500 font-medium uppercase tracking-wide">{r.hostname}</p>
              {vramTotal > 0 && (
                <div>
                  <div className="flex items-center justify-between text-xs text-gray-400 mb-1">
                    <span>VRAM</span>
                    <span>{vramUsed.toFixed(1)} / {vramTotal.toFixed(1)} GB ({vramPct}%)</span>
                  </div>
                  <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                    <div className={`h-full rounded-full transition-all ${vramPct > 90 ? 'bg-red-500' : vramPct > 70 ? 'bg-yellow-500' : 'bg-brand-500'}`}
                      style={{ width: `${vramPct}%` }} />
                  </div>
                </div>
              )}
              {diskTotal > 0 && (
                <div>
                  <div className="flex items-center justify-between text-xs text-gray-400 mb-1">
                    <span>Disk</span>
                    <span>{diskFree.toFixed(1)} GB free of {diskTotal.toFixed(1)} GB</span>
                  </div>
                  <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                    <div className={`h-full rounded-full transition-all ${diskPct > 95 ? 'bg-red-500' : diskPct > 85 ? 'bg-yellow-500' : 'bg-emerald-500'}`}
                      style={{ width: `${diskPct}%` }} />
                  </div>
                  {diskFree < 5 && <p className="text-[10px] text-red-400 mt-1">Low disk space</p>}
                </div>
              )}
              {(rs?.loaded_ollama_models ?? []).length > 0 && (
                <div>
                  <p className="text-[10px] text-gray-500 font-medium uppercase tracking-wide mb-1.5">In VRAM</p>
                  <div className="space-y-1">
                    {(rs?.loaded_ollama_models ?? []).map(lm => {
                      const loadedInfo = allLoadedModels.find(x => x.name === lm.name && x.runner === r.hostname)
                      const isPinned = loadedInfo?.do_not_evict ?? false
                      return (
                        <div key={lm.name} className="flex items-center justify-between gap-2">
                          <div className="min-w-0 flex items-center gap-1.5">
                            {isPinned && <Pin className="w-2.5 h-2.5 text-indigo-400 shrink-0" />}
                            <div className="min-w-0">
                              <span className="text-xs text-gray-300 truncate block">{lm.name.replace(/:latest$/, '')}</span>
                            </div>
                          </div>
                          <div className="flex items-center gap-1 shrink-0">
                            <button
                              onClick={() => pinRunner.mutate({ model_name: lm.name, runner_id: r.id, do_not_evict: !isPinned })}
                              disabled={pinRunner.isPending}
                              title={isPinned ? 'Unpin from this runner' : 'Pin to this runner'}
                              className={`p-0.5 rounded transition-colors disabled:opacity-40 ${isPinned ? 'text-indigo-400 hover:text-indigo-300' : 'text-gray-600 hover:text-gray-400'}`}
                            >
                              <Pin className="w-3 h-3" />
                            </button>
                            <button
                              onClick={() => unloadVram.mutate({ model: lm.name, runner_id: r.id })}
                              disabled={unloadVram.isPending}
                              title="Evict from VRAM"
                              className="flex items-center gap-1 text-[10px] bg-yellow-900/30 hover:bg-yellow-800/40 text-yellow-400 px-1.5 py-0.5 rounded transition-colors disabled:opacity-40"
                            >
                              <Upload className="w-2.5 h-2.5" /> Evict
                            </button>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>
              )}
            </div>
          )
        })}
      </div>

      {/* Active downloads */}
      {activeOps.length > 0 && selectedRunner !== undefined && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 space-y-3">
          <p className="text-xs text-gray-500 font-medium uppercase tracking-wide">Downloads in progress ({activeOps.length})</p>
          {activeOps.map((op, i) => {
            let pct = 0
            let statusText = op.progress || 'starting...'
            try {
              const p = JSON.parse(op.progress || '{}')
              if (p.completed && p.total) {
                pct = Math.round((p.completed / p.total) * 100)
                statusText = `${p.status || 'downloading'} ${(p.completed / 1e9).toFixed(1)}/${(p.total / 1e9).toFixed(1)} GB`
              } else if (p.status) statusText = p.status
            } catch { statusText = op.progress || 'starting...' }
            return (
              <div key={`${op.model}-${i}`}>
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className="text-gray-300 font-medium">{op.model}</span>
                  <span className="text-gray-500">{pct > 0 ? `${pct}%` : statusText.slice(0, 40)}</span>
                </div>
                <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                  <div className="h-full rounded-full bg-brand-500 transition-all" style={{ width: `${pct}%` }} />
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* Failed operations */}
      {failedOps.length > 0 && (
        <div className="bg-red-950/40 border border-red-900/60 rounded-xl p-4 space-y-2">
          <p className="text-xs text-red-300 font-medium uppercase tracking-wide flex items-center gap-1.5">
            <AlertCircle className="w-3.5 h-3.5" />
            Failed ({failedOps.length})
          </p>
          {failedOps.map(op => (
            <div key={op.op_id ?? op.model} className="flex items-start justify-between gap-3 text-xs">
              <div className="min-w-0 flex-1">
                <div className="text-gray-200 font-medium truncate">
                  {op.type} · {op.model}
                </div>
                <div className="text-red-300/90 break-words mt-0.5">
                  {op.error || op.progress || 'unknown error'}
                </div>
              </div>
              {op.op_id && (
                <button
                  onClick={() => dismissOp.mutate(op.op_id!)}
                  disabled={dismissOp.isPending}
                  title="Dismiss"
                  className="shrink-0 text-gray-500 hover:text-gray-300 disabled:opacity-40 p-1 rounded hover:bg-red-900/40"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Pull input (runner view only) */}
      {selectedRunner !== undefined && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
          <p className="text-xs text-gray-500 mb-2 font-medium uppercase tracking-wide">
            Pull model to {selectedRunnerHostname}
          </p>
          <div className="flex gap-2">
            <input type="text" value={pullInput} onChange={e => setPullInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handlePull()}
              placeholder="e.g. qwen2.5:7b"
              className="flex-1 bg-gray-950 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-brand-600"
            />
            <button onClick={handlePull} disabled={pull.isPending || !pullInput.trim()}
              className="flex items-center gap-2 bg-brand-600 hover:bg-brand-500 disabled:opacity-40 text-white text-sm font-medium px-4 py-2 rounded-lg transition-colors">
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
      )}

      {/* Filters */}
      <div className="flex flex-wrap gap-2 items-center">
        <div className="relative flex-1 min-w-[160px]">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-gray-600 pointer-events-none" />
          <input
            type="text"
            value={searchFilter}
            onChange={e => setSearchFilter(e.target.value)}
            placeholder="Filter models…"
            className="w-full bg-gray-900 border border-gray-800 rounded-lg pl-8 pr-3 py-1.5 text-xs text-gray-200 placeholder-gray-600 focus:outline-none focus:border-brand-600"
          />
        </div>
        <div className="flex gap-1">
          {(['', 'safe', 'unsafe'] as const).map(s => (
            <button key={s} onClick={() => setSafetyFilter(s)}
              className={`text-xs px-2.5 py-1.5 rounded-lg transition-colors border ${
                safetyFilter === s
                  ? s === 'unsafe' ? 'bg-red-900/50 text-red-400 border-red-700'
                    : s === 'safe' ? 'bg-green-900/50 text-green-400 border-green-700'
                    : 'bg-gray-700 text-gray-200 border-gray-600'
                  : 'bg-gray-900 text-gray-500 hover:bg-gray-800 border-transparent'
              }`}>
              {s === '' ? 'All' : s === 'safe' ? 'Safe' : 'Unsafe'}
            </button>
          ))}
        </div>
        <div className="flex gap-1">
          {STANDARD_CATEGORIES.map(cat => (
            <button key={cat} onClick={() => setCapFilter(capFilter === cat ? '' : cat)}
              className={`text-xs px-2.5 py-1.5 rounded-lg transition-colors border ${
                capFilter === cat
                  ? 'bg-indigo-900/50 text-indigo-300 border-indigo-700'
                  : 'bg-gray-900 text-gray-500 hover:bg-gray-800 border-transparent'
              }`}>
              {cat}
            </button>
          ))}
        </div>
      </div>

      {/* Model list */}
      <div className="space-y-2">
        {modelList.isLoading ? (
          <div className="py-8 text-center text-gray-600 text-sm">Loading models...</div>
        ) : models.length === 0 ? (
          <div className="bg-gray-900 border border-gray-800 rounded-xl py-8 text-center text-gray-600 text-sm">
            {selectedRunner !== undefined ? `No models on ${selectedRunnerHostname}` : 'No models downloaded on any runner'}
          </div>
        ) : (
          models.map(m => {
            const mAny = m as unknown as { is_alias?: boolean; base_model?: string }
            const realName = mAny.is_alias ? (mAny.base_model ?? m.name) : m.name
            const isLoaded = selectedRunner !== undefined
              ? allLoadedModels.some(lm => lm.name === realName && lm.runner === selectedRunnerHostname)
              : allLoadedModels.some(lm => lm.name === realName)
            // Per-runner pin state: read from the loaded model entry for this runner
            const loadedEntry = selectedRunner !== undefined
              ? allLoadedModels.find(lm => lm.name === realName && lm.runner === selectedRunnerHostname)
              : undefined
            const isRunnerPinned = loadedEntry?.do_not_evict ?? false
            const displayName = m.name.replace(/:latest$/, '')
            const isExpanded = expandedModel === m.name

            return (
              <div key={m.name} className={`bg-gray-900 border rounded-xl overflow-hidden transition-colors ${isExpanded ? 'border-brand-700/60' : 'border-gray-800'}`}>
                {/* Header row — click to expand */}
                <div
                  className="px-4 py-3 flex items-center gap-3 cursor-pointer hover:bg-gray-800/40 transition-colors"
                  onClick={() => setExpandedModel(isExpanded ? null : m.name)}
                >
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-1.5 flex-wrap">
                      <span className="text-sm text-gray-200 font-medium">{displayName}</span>
                      {(m as unknown as { is_alias?: boolean }).is_alias && (
                        <span className="text-[10px] bg-indigo-900/50 text-indigo-400 px-1.5 py-0.5 rounded-full">
                          alias → {(m as unknown as { base_model?: string }).base_model}
                        </span>
                      )}
                      {m.parameter_count && !(m as unknown as { is_alias?: boolean }).is_alias && (
                        <span className="text-[10px] bg-gray-800 text-gray-400 px-1.5 py-0.5 rounded">{m.parameter_count}</span>
                      )}
                      {m.quantization && !(m as unknown as { is_alias?: boolean }).is_alias && (
                        <span className="text-[10px] bg-gray-800 text-gray-500 px-1.5 py-0.5 rounded">{m.quantization}</span>
                      )}
                      {(m.categories ?? []).map(cat => (
                        <span key={cat} className="text-[10px] bg-indigo-900/30 text-indigo-400 px-1.5 py-0.5 rounded-full">{cat}</span>
                      ))}
                      <span className={`text-[10px] px-1.5 py-0.5 rounded-full ${
                        m.safety === 'unsafe' ? 'bg-red-900/30 text-red-400' : 'bg-green-900/20 text-green-600'
                      }`}>{m.safety}</span>
                      {isLoaded && (
                        <span className="text-[10px] bg-blue-900/40 text-blue-400 px-1.5 py-0.5 rounded-full flex items-center gap-0.5">
                          <Cpu className="w-2.5 h-2.5" /> VRAM
                        </span>
                      )}
                      {isRunnerPinned && (
                        <span className="text-[10px] bg-indigo-900/50 text-indigo-400 px-1.5 py-0.5 rounded-full flex items-center gap-0.5">
                          <Pin className="w-2.5 h-2.5" /> pinned
                        </span>
                      )}
                    </div>
                    <p className="text-[10px] text-gray-600 mt-0.5">
                      ~{m.vram_estimate_gb} GB VRAM{m.size_gb ? ` · ${m.size_gb} GB disk` : ''}
                      {selectedRunner === undefined && (m.runners ?? []).length > 0 && (
                        <span className="ml-2">
                          {(m.runners ?? []).map(r => (
                            <span key={r.runner_id} className="text-[10px] bg-gray-800 text-gray-500 px-1.5 py-0.5 rounded mr-1">{r.hostname}</span>
                          ))}
                        </span>
                      )}
                    </p>
                  </div>
                  <div className="flex items-center gap-1.5 flex-shrink-0" onClick={e => e.stopPropagation()}>
                    {selectedRunner !== undefined && (
                      <button
                        onClick={() => pinRunner.mutate({ model_name: realName, runner_id: selectedRunner, do_not_evict: !isRunnerPinned })}
                        disabled={pinRunner.isPending}
                        title={isRunnerPinned ? 'Unpin from this runner' : 'Pin to this runner'}
                        className={`p-1.5 rounded-lg transition-colors disabled:opacity-40 ${isRunnerPinned ? 'bg-indigo-900/50 text-indigo-400 hover:bg-indigo-900/70' : 'bg-gray-800 hover:bg-gray-700 text-gray-500 hover:text-gray-300'}`}
                      >
                        <Pin className="w-3.5 h-3.5" />
                      </button>
                    )}
                    <button onClick={() => setEditingModel(m)} title="Edit categories / safety"
                      className="p-1.5 rounded-lg bg-gray-800 hover:bg-gray-700 text-gray-500 hover:text-gray-300 transition-colors">
                      <Settings2 className="w-3.5 h-3.5" />
                    </button>
                    {selectedRunner !== undefined && (
                      isLoaded ? (
                        <button onClick={() => unloadVram.mutate({ model: realName, runner_id: selectedRunner })}
                          disabled={unloadVram.isPending} title="Free VRAM"
                          className="flex items-center gap-1 text-xs bg-yellow-900/30 hover:bg-yellow-800/40 text-yellow-400 px-2.5 py-1 rounded-lg transition-colors disabled:opacity-40">
                          {unloadVram.isPending ? <Loader2 className="w-3 h-3 animate-spin" /> : <Upload className="w-3 h-3" />}
                          Unload
                        </button>
                      ) : (
                        <button onClick={() => load.mutate({ model: m.name, runner_id: selectedRunner })}
                          disabled={load.isPending} title="Load into VRAM"
                          className="flex items-center gap-1 text-xs bg-brand-900/50 hover:bg-brand-800/50 text-brand-300 px-2.5 py-1 rounded-lg transition-colors disabled:opacity-40">
                          {load.isPending ? <Loader2 className="w-3 h-3 animate-spin" /> : <Cpu className="w-3 h-3" />}
                          Load
                        </button>
                      )
                    )}
                    {selectedRunner !== undefined && (
                      <button
                        onClick={() => {
                          if (window.confirm(`Delete ${m.name} from ${selectedRunnerHostname}?`))
                            del.mutate({ model: m.name, runner_id: selectedRunner })
                        }}
                        disabled={del.isPending} title="Delete from disk"
                        className="p-1.5 rounded-lg bg-gray-800 hover:bg-red-900/50 hover:text-red-400 text-gray-500 transition-colors">
                        {del.isPending ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Trash2 className="w-3.5 h-3.5" />}
                      </button>
                    )}
                  </div>
                </div>

                {/* Expanded details */}
                {isExpanded && (
                  <div className="border-t border-gray-800 px-4 py-3 space-y-3">
                    {m.description && (
                      <p className="text-xs text-gray-400 leading-relaxed">{m.description}</p>
                    )}
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                      <div>
                        <p className="text-[10px] text-gray-500 uppercase tracking-wide">VRAM estimate</p>
                        <p className="text-xs text-gray-300 mt-0.5">~{m.vram_estimate_gb} GB</p>
                      </div>
                      {m.size_gb > 0 && (
                        <div>
                          <p className="text-[10px] text-gray-500 uppercase tracking-wide">Disk size</p>
                          <p className="text-xs text-gray-300 mt-0.5">{m.size_gb} GB</p>
                        </div>
                      )}
                      <div>
                        <p className="text-[10px] text-gray-500 uppercase tracking-wide">Parameters</p>
                        <p className="text-xs text-gray-300 mt-0.5">{m.parameter_count || '—'}</p>
                      </div>
                      <div>
                        <p className="text-[10px] text-gray-500 uppercase tracking-wide">Quantization</p>
                        <p className="text-xs text-gray-300 mt-0.5">{m.quantization || 'Default'}</p>
                      </div>
                      <div>
                        <p className="text-[10px] text-gray-500 uppercase tracking-wide">Safety</p>
                        <p className={`text-xs mt-0.5 ${m.safety === 'unsafe' ? 'text-red-400' : 'text-green-500'}`}>{m.safety}</p>
                      </div>
                      {(m.categories ?? []).length > 0 && (
                        <div className="col-span-2">
                          <p className="text-[10px] text-gray-500 uppercase tracking-wide">Categories</p>
                          <div className="flex gap-1 flex-wrap mt-0.5">
                            {(m.categories ?? []).map(cat => (
                              <span key={cat} className="text-[10px] bg-indigo-900/30 text-indigo-400 px-1.5 py-0.5 rounded-full">{cat}</span>
                            ))}
                          </div>
                        </div>
                      )}
                      <div className="col-span-2">
                        <p className="text-[10px] text-gray-500 uppercase tracking-wide">Full name</p>
                        <p className="text-xs text-gray-500 mt-0.5 font-mono truncate">{m.name}</p>
                      </div>
                      {(m.runners ?? []).length > 0 && (
                        <div className="col-span-2 sm:col-span-4">
                          <p className="text-[10px] text-gray-500 uppercase tracking-wide">On runners</p>
                          <div className="flex gap-1.5 flex-wrap mt-0.5">
                            {(m.runners ?? []).map(r => (
                              <span key={r.runner_id} className="text-[10px] bg-gray-800 text-gray-400 px-1.5 py-0.5 rounded">{r.hostname}</span>
                            ))}
                          </div>
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

      {editingModel && (
        (editingModel as unknown as { is_alias?: boolean }).is_alias
          ? <AliasSettingsModal model={editingModel} onClose={() => setEditingModel(null)} />
          : <ModelSettingsModal model={editingModel} runners={runners} onClose={() => setEditingModel(null)} />
      )}
    </section>
  )
}

// ── Library Browser ─────────────────────────────────────────────────────────

function LibraryBrowserSection({ selectedRunner, selectedRunnerHostname, allRunners }: {
  selectedRunner?: number
  selectedRunnerHostname?: string
  allRunners: Runner[]
}) {
  const modelList = useModelList()
  // Scope downloaded-ness to the selected runner. /api/models is fleet-wide
  // and returns every runner's models; without this filter a model downloaded
  // only on archlinux would light up green on the murderbot tab too.
  const scopedModelEntries = (modelList.data ?? []).filter(
    m => selectedRunner === undefined || (m.runners ?? []).some(r => r.runner_id === selectedRunner)
  )
  const downloadedNames = new Set(scopedModelEntries.map((m: { name: string }) => m.name))
  const fitsMap = new Map(scopedModelEntries.map((m: { name: string; fits: boolean }) => [m.name, m.fits]))
  const [search, setSearch] = useState('')
  const [safety, setSafety] = useState<string>('safe')
  const [fitsOnly, setFitsOnly] = useState(true)
  const [hideDownloaded, setHideDownloaded] = useState(true)
  const [sort, setSort] = useState<string>('pulls')
  const [capFilter, setCapFilter] = useState<string>('')
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
    // Scope the entire library view to the selected runner — downloaded_on,
    // outdated_on, fits_on all reflect ONLY that runner. When unset,
    // fleet-wide behavior.
    runner_id: selectedRunner,
  })

  const rawModels = (library.data?.models ?? []).filter(
    m => !capFilter || m.categories.includes(capFilter)
  )
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
    // Only show "done" if the model is actually still downloaded on a runner
    if (completed && downloadedNames.has(model)) return { status: 'completed' as const }
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
          <h2 className="text-base font-semibold text-gray-200">
            {selectedRunner !== undefined
              ? `Ollama Library — ${selectedRunnerHostname ?? 'runner'}`
              : 'Ollama Library'}
          </h2>
          <span className="text-xs text-gray-600">{library.data?.total ?? 0} models</span>
        </div>
        <LibraryToolbar
          cacheAge={cacheAge}
          onRefreshCatalog={() => refresh.mutate()}
          refreshCatalogPending={refresh.isPending}
          selectedRunner={selectedRunner}
          selectedRunnerHostname={selectedRunnerHostname}
        />
      </div>

      {/* Legend */}
      <div className="flex gap-3 flex-wrap text-[10px] text-gray-600">
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-blue-500" /> Available</span>
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-green-500" /> Downloaded</span>
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-yellow-500" /> Downloaded, won't fit</span>
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-red-500" /> Won't fit</span>
      </div>

      {/* Community / custom models — downloaded tags not in the library catalog */}
      <CommunityModelsSection selectedRunner={selectedRunner} selectedRunnerHostname={selectedRunnerHostname} />


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
          value={capFilter}
          onChange={e => setCapFilter(e.target.value)}
          className="text-xs bg-gray-900 border border-gray-800 rounded-lg px-2.5 py-2 text-gray-400 focus:outline-none focus:border-brand-600"
        >
          <option value="">Any capability</option>
          <option value="tools">Tools</option>
          <option value="vision">Vision</option>
          <option value="thinking">Thinking</option>
          <option value="embedding">Embedding</option>
        </select>
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
                      {(m.outdated_on ?? []).length > 0 && (
                        <span
                          className="flex items-center gap-0.5 text-[10px] text-amber-400"
                          title={`Outdated on: ${(m.outdated_on ?? []).join(', ')} — remote manifest digest differs from local`}
                        >
                          <AlertCircle className="w-2.5 h-2.5" />
                          update available
                        </span>
                      )}
                    </div>
                    {m.description && (
                      <p className={`text-xs text-gray-500 mt-1 ${isExpanded ? '' : 'line-clamp-1'}`}>{m.description}</p>
                    )}
                    {!isExpanded && (
                      <div className="space-y-2 mt-2">
                        {m.parameter_sizes.length > 0 && (
                          <LibraryTagGrid
                            model={m}
                            downloadedNames={downloadedNames}
                            fitsMap={fitsMap}
                            allRunners={allRunners}
                            selectedRunner={selectedRunner}
                            selectedRunnerHostname={selectedRunnerHostname}
                            onPull={handlePull}
                            pullPending={pull.isPending}
                            isModelPulling={isModelPulling}
                            getModelOpStatus={getModelOpStatus}
                          />
                        )}
                        <div className="flex items-center gap-3 flex-wrap">
                          {m.pulls && <span className="text-[10px] text-gray-600">{m.pulls} pulls</span>}
                          {m.categories.includes('tools') && (
                            <span className="text-[10px] bg-emerald-900/30 text-emerald-400 px-1.5 py-0.5 rounded">tools</span>
                          )}
                          {m.categories.includes('vision') && (
                            <span className="text-[10px] bg-violet-900/30 text-violet-400 px-1.5 py-0.5 rounded">vision</span>
                          )}
                          {m.categories.includes('thinking') && (
                            <span className="text-[10px] bg-cyan-900/30 text-cyan-400 px-1.5 py-0.5 rounded">thinking</span>
                          )}
                          {m.categories.includes('embedding') && (
                            <span className="text-[10px] bg-gray-800 text-gray-400 px-1.5 py-0.5 rounded">embedding</span>
                          )}
                        </div>
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
                    {!isModelPulling(m.name) && !m.downloaded && m.fits && !isExpanded && (
                      <button
                        onClick={e => { e.stopPropagation(); setExpanded(m.name) }}
                        className="flex items-center gap-1 text-xs bg-brand-900/50 hover:bg-brand-800/50 text-brand-300 px-3 py-1.5 rounded-lg transition-colors"
                      >
                        <Download className="w-3 h-3" />
                        Select version
                      </button>
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
                          const isDownloaded = downloadedNames.has(sizeModel) || downloadedNames.has(`${m.name}:latest`)
                          const sizeFits = m.size_info?.[s]?.fits ?? fitsMap.get(sizeModel) ?? m.fits
                          const baseClass = isDownloaded && sizeFits
                            ? 'bg-green-900/30 text-green-400'
                            : isDownloaded && !sizeFits
                            ? 'bg-yellow-900/30 text-yellow-400'
                            : !isDownloaded && sizeFits
                            ? 'bg-blue-900/20 hover:bg-blue-900/40 text-blue-400'
                            : 'bg-red-900/30 text-red-400'
                          return (
                          <div key={s} className="flex items-center gap-1">
                            <span className={`text-xs px-2 py-1 rounded ${baseClass}`}>
                              {s}
                            </span>
                            {pulling ? (
                              <span className="flex items-center gap-1 text-[10px] text-amber-400">
                                <Loader2 className="w-2.5 h-2.5 animate-spin" /> downloading...
                              </span>
                            ) : opStatus?.status === 'completed' ? (
                              <span className="flex items-center gap-1 text-[10px] text-green-400">
                                <CheckCircle2 className="w-2.5 h-2.5" /> done
                              </span>
                            ) : opStatus?.status === 'failed' ? (
                              <span className="flex items-center gap-1 text-[10px] text-red-400">
                                <AlertCircle className="w-2.5 h-2.5" /> failed
                              </span>
                            ) : !isDownloaded && sizeFits ? (
                              selectedRunner !== undefined ? (
                                <button
                                  onClick={e => { e.stopPropagation(); handlePull(sizeModel, selectedRunner) }}
                                  disabled={pull.isPending && pullMsg?.model === sizeModel}
                                  className="flex items-center gap-1 text-[10px] bg-brand-900/50 hover:bg-brand-800/50 text-brand-300 px-2 py-0.5 rounded transition-colors disabled:opacity-40"
                                >
                                  <Download className="w-2.5 h-2.5" /> {selectedRunnerHostname}
                                </button>
                              ) : (
                                <div className="flex gap-1">
                                  {allRunners.map(r => {
                                    const fitsOnThis = m.fits_on.some(f => f.runner === r.hostname)
                                    if (!fitsOnThis) return null
                                    return (
                                      <button
                                        key={r.id}
                                        onClick={e => { e.stopPropagation(); handlePull(sizeModel, r.id) }}
                                        disabled={pull.isPending && pullMsg?.model === sizeModel}
                                        className="flex items-center gap-1 text-[10px] bg-brand-900/50 hover:bg-brand-800/50 text-brand-300 px-2 py-0.5 rounded transition-colors disabled:opacity-40"
                                      >
                                        <Download className="w-2.5 h-2.5" /> {r.hostname}
                                      </button>
                                    )
                                  })}
                                </div>
                              )
                            ) : isDownloaded ? (
                              <ForceUpdateButton model={sizeModel} runnerId={selectedRunner} />
                            ) : null}
                          </div>
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

        {/* Sync Models is a fleet-level action — hide it when a specific
            runner is selected. Per-runner work happens via the library
            section's Update outdated / Force update buttons. */}
        {tab === 'local' && runnerList.length >= 2 && selectedRunner === undefined && (
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
              {selectedRunner === undefined && (
                <div className="flex items-center gap-2">
                  <Layers className="w-4 h-4 text-brand-400" />
                  <h2 className="text-base font-semibold text-gray-200">Installed Models</h2>
                </div>
              )}
              <InstalledModelsView
                runners={runnerList}
                selectedRunner={selectedRunner}
                selectedRunnerHostname={selectedRunnerData?.hostname}
              />

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


// ── LibraryTagGrid ──────────────────────────────────────────────────────
// Bigger, tappable per-tag buttons on the collapsed library card. Replaces
// the old inline pill badges that were read-only info. Each tag shows state
// (3-bucket: downloaded / outdated / available), size, and hover/click-to-
// pull when appropriate.
//
// Click behavior:
//   - Not downloaded + fits → pulls to selected runner (or first-fit runner)
//   - Downloaded → no-op (force-update + delete live in expanded detail to
//     avoid accidental clicks during browsing)
//   - Not downloaded + won't fit → disabled with tooltip

function LibraryTagGrid({
  model,
  downloadedNames,
  fitsMap,
  allRunners,
  selectedRunner,
  selectedRunnerHostname,
  onPull,
  pullPending,
  isModelPulling,
  getModelOpStatus,
}: {
  model: LibraryModel
  downloadedNames: Set<string>
  fitsMap: Map<string, boolean>
  allRunners: Runner[]
  selectedRunner?: number
  selectedRunnerHostname?: string
  onPull: (model: string, runnerId?: number) => void
  pullPending: boolean
  isModelPulling: (m: string) => boolean
  getModelOpStatus: (m: string) => { status: 'running' | 'completed' | 'failed'; error?: string } | null
}) {
  return (
    <div className="flex gap-1.5 flex-wrap">
      {model.parameter_sizes.map(s => {
        const tag = `${model.name}:${s}`
        const isDownloaded = downloadedNames.has(tag) || downloadedNames.has(`${model.name}:latest`)
        const fits = model.size_info?.[s]?.fits ?? fitsMap.get(tag) ?? model.fits
        const isOutdated = (model.outdated_on ?? []).length > 0 && isDownloaded
        const vramGb = model.size_info?.[s]?.vram_gb
        const pulling = isModelPulling(tag)
        const op = getModelOpStatus(tag)

        // State-driven styling (3 buckets + fit modifier)
        let tone: string
        let icon: ReactNode = null
        let stateLabel = ''
        if (pulling) {
          tone = 'bg-amber-900/30 text-amber-400 border-amber-800/60'
          icon = <Loader2 className="w-3 h-3 animate-spin" />
          stateLabel = 'pulling'
        } else if (op?.status === 'failed') {
          tone = 'bg-red-900/30 text-red-400 border-red-800/60'
          icon = <AlertCircle className="w-3 h-3" />
          stateLabel = 'failed'
        } else if (isDownloaded && isOutdated) {
          tone = 'bg-amber-900/30 text-amber-400 border-amber-800/60'
          icon = <AlertCircle className="w-3 h-3" />
          stateLabel = 'update'
        } else if (isDownloaded && fits) {
          tone = 'bg-green-900/30 text-green-400 border-green-800/60'
          icon = <CheckCircle2 className="w-3 h-3" />
          stateLabel = 'installed'
        } else if (isDownloaded && !fits) {
          tone = 'bg-yellow-900/30 text-yellow-400 border-yellow-800/60'
          icon = <AlertCircle className="w-3 h-3" />
          stateLabel = "won't fit"
        } else if (!isDownloaded && fits) {
          tone = 'bg-blue-900/30 text-blue-400 hover:bg-blue-800/50 border-blue-800/60 cursor-pointer'
          icon = <Download className="w-3 h-3" />
          stateLabel = 'pull'
        } else {
          tone = 'bg-gray-800/60 text-gray-500 border-gray-700 cursor-not-allowed'
          icon = <X className="w-3 h-3" />
          stateLabel = "won't fit"
        }

        const canPull = !isDownloaded && fits && !pulling
        const title = [
          tag,
          vramGb ? `~${vramGb}GB VRAM` : null,
          isDownloaded ? `downloaded on ${(model.downloaded_on ?? []).join(', ') || '?'}` : null,
          isOutdated ? `update available on ${(model.outdated_on ?? []).join(', ')}` : null,
          canPull && selectedRunnerHostname ? `click to pull to ${selectedRunnerHostname}` :
          canPull ? 'click to pull' : null,
        ].filter(Boolean).join(' · ')

        const onClick = (e: React.MouseEvent) => {
          e.stopPropagation()
          if (!canPull) return
          if (selectedRunner !== undefined) {
            onPull(tag, selectedRunner)
            return
          }
          // No scope — pick first runner that fits
          const runner = allRunners.find(r => (model.fits_on ?? []).some(f => f.runner === r.hostname))
          onPull(tag, runner?.id)
        }

        return (
          <button
            key={s}
            type="button"
            onClick={onClick}
            disabled={!canPull || pullPending}
            title={title}
            className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-md border text-xs font-medium transition-colors ${tone} ${!canPull ? 'disabled:opacity-100' : ''}`}
          >
            {icon}
            <span className="font-mono">{s}</span>
            {vramGb !== undefined && (
              <span className="text-[10px] opacity-70">{vramGb}GB</span>
            )}
            <span className="text-[10px] opacity-70">· {stateLabel}</span>
          </button>
        )
      })}
    </div>
  )
}


// ── Community / custom models ───────────────────────────────────────────
// Tags downloaded on a runner that aren't in the Ollama library catalog
// (e.g. MFDoom/*, hf.co/*, user imports). Catalog scrape only covers
// ollama.com/library so these never show in the main grid.
//
// Renders as a collapsible section above the main library cards. Shows
// per-tag size + runners + "update available" + delete + force-update
// (same capabilities as a library card, just with minimal metadata).

function CommunityModelsSection({
  selectedRunner,
  selectedRunnerHostname,
}: {
  selectedRunner?: number
  selectedRunnerHostname?: string
}) {
  const community = useCommunityModels(selectedRunner)
  const forceUpdate = useForceUpdateModel()
  const deleteModel = useDeleteModel()
  const models = community.data?.models ?? []
  if (models.length === 0) return null
  return (
    <div className="bg-gray-900/60 border border-gray-800 rounded-xl p-4 space-y-2">
      <div className="flex items-center justify-between">
        <p className="text-xs text-gray-400 font-medium uppercase tracking-wide">
          Community / custom models
          {selectedRunnerHostname && <span className="ml-2 text-gray-600 normal-case"> on {selectedRunnerHostname}</span>}
          <span className="ml-2 text-gray-600 normal-case">· {models.length}</span>
        </p>
        <span className="text-[10px] text-gray-600">Downloaded but not in the Ollama library catalog</span>
      </div>
      <div className="space-y-1">
        {models.map(m => {
          const sizeGb = m.size_bytes ? (m.size_bytes / 1e9).toFixed(1) : null
          const outdated = m.outdated_on.length > 0
          return (
            <div key={m.name} className="flex items-center justify-between gap-2 text-xs bg-gray-950/40 border border-gray-800 rounded-lg px-3 py-2">
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-gray-200 font-mono truncate">{m.name}</span>
                  {sizeGb && <span className="text-[10px] text-gray-500">{sizeGb} GB</span>}
                  <span className="flex items-center gap-0.5 text-[10px] text-green-400">
                    <CheckCircle2 className="w-2.5 h-2.5" />
                    {m.downloaded_on.join(', ')}
                  </span>
                  {outdated && (
                    <span className="flex items-center gap-0.5 text-[10px] text-amber-400" title={`Outdated on: ${m.outdated_on.join(', ')}`}>
                      <AlertCircle className="w-2.5 h-2.5" /> update available
                    </span>
                  )}
                </div>
              </div>
              <div className="flex items-center gap-1 shrink-0">
                <button
                  onClick={() => {
                    if (!window.confirm(`Force re-pull ${m.name}?`)) return
                    forceUpdate.mutate({ model: m.name, runner_id: selectedRunner })
                  }}
                  disabled={forceUpdate.isPending}
                  title={`Force re-pull ${m.name}`}
                  className="flex items-center gap-1 text-[10px] bg-gray-800 hover:bg-gray-700 text-gray-300 px-2 py-1 rounded transition-colors disabled:opacity-40"
                >
                  <RefreshCcw className="w-2.5 h-2.5" />
                  {outdated ? 'Update' : 'Re-pull'}
                </button>
                <button
                  onClick={() => {
                    const where = selectedRunnerHostname ?? m.downloaded_on.join(', ')
                    if (!window.confirm(`Delete ${m.name} from ${where}? This removes the blobs from disk.`)) return
                    deleteModel.mutate({ model: m.name, runner_id: selectedRunner })
                  }}
                  disabled={deleteModel.isPending}
                  title="Delete from runner"
                  className="flex items-center gap-1 text-[10px] bg-gray-800 hover:bg-red-900/50 hover:text-red-300 text-gray-400 px-2 py-1 rounded transition-colors disabled:opacity-40"
                >
                  <Trash2 className="w-2.5 h-2.5" />
                </button>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}


// ── Force-update single model ───────────────────────────────────────────
// Tiny inline button for the expanded library detail — re-pulls a specific
// `model:tag` regardless of local state. Used on downloaded variants.

function ForceUpdateButton({ model, runnerId }: { model: string; runnerId: number | undefined }) {
  const forceUpdate = useForceUpdateModel()
  const [status, setStatus] = useState<'idle' | 'ok' | 'err'>('idle')
  return (
    <button
      onClick={e => {
        e.stopPropagation()
        if (!window.confirm(`Force re-pull ${model}? Runs the pull even if the model is already up to date.`)) return
        setStatus('idle')
        forceUpdate.mutate({ model, runner_id: runnerId }, {
          onSuccess: () => setStatus('ok'),
          onError: () => setStatus('err'),
        })
      }}
      disabled={forceUpdate.isPending}
      className="flex items-center gap-1 text-[10px] bg-gray-800 hover:bg-gray-700 text-gray-300 px-2 py-0.5 rounded transition-colors disabled:opacity-40"
      title="Force re-pull this model (ignores local/remote digest)"
    >
      {forceUpdate.isPending ? <Loader2 className="w-2.5 h-2.5 animate-spin" />
        : status === 'ok' ? <CheckCircle2 className="w-2.5 h-2.5 text-green-400" />
        : status === 'err' ? <AlertCircle className="w-2.5 h-2.5 text-red-400" />
        : <RefreshCcw className="w-2.5 h-2.5" />}
      Force update
    </button>
  )
}


// ── Library update toolbar ───────────────────────────────────────────────
// Top-right of the Ollama Library section. Three actions:
//   - Refresh catalog: scrape ollama.com/library for the curated list (keeps
//     the model cards current).
//   - Check for updates: ping registry.ollama.ai for the current manifest
//     digest of each *downloaded* tag and cache it. Drives the "update
//     available" indicator.
//   - Update all outdated: pull every tag whose local digest differs from
//     the remote. Each pull becomes a background op visible in the
//     downloads-in-progress card.

function LibraryToolbar({
  cacheAge,
  onRefreshCatalog,
  refreshCatalogPending,
  selectedRunner,
  selectedRunnerHostname,
}: {
  cacheAge: number
  onRefreshCatalog: () => void
  refreshCatalogPending: boolean
  selectedRunner?: number
  selectedRunnerHostname?: string
}) {
  const refreshRemote = useRefreshRemoteDigests()
  const updateOutdated = useUpdateOutdatedModels()
  const [msg, setMsg] = useState<string | null>(null)
  const scopeSuffix = selectedRunnerHostname ? ` on ${selectedRunnerHostname}` : ''
  return (
    <div className="flex items-center gap-3">
      {msg && <span className="text-[10px] text-gray-500">{msg}</span>}
      <button
        onClick={() => {
          setMsg(null)
          refreshRemote.mutate({ runner_id: selectedRunner }, {
            onSuccess: d => setMsg(
              d.status === 'idle'
                ? 'No downloaded models to check'
                : `Checked ${d.tags_checked ?? 0} tags (${d.errors ?? 0} errors)`
            ),
            onError: e => setMsg(`Error: ${(e as Error).message}`),
          })
        }}
        disabled={refreshRemote.isPending}
        className="flex items-center gap-1 text-xs text-gray-500 hover:text-gray-300 disabled:opacity-40 transition-colors"
        title={`Ask registry.ollama.ai for the current manifest digest of each downloaded tag${scopeSuffix}`}
      >
        {refreshRemote.isPending ? <Loader2 className="w-3 h-3 animate-spin" /> : <RefreshCw className="w-3 h-3" />}
        Check for updates{scopeSuffix}
      </button>
      <button
        onClick={() => {
          const where = selectedRunnerHostname ?? 'every runner'
          if (!window.confirm(`Pull every outdated tag on ${where}? Backend auto-refreshes digests first. This could take a while for large models.`)) return
          setMsg(null)
          updateOutdated.mutate({ runner_id: selectedRunner }, {
            onSuccess: d => setMsg(
              d.count === 0
                ? `No outdated tags (${d.up_to_date ?? 0} up-to-date${d.skipped_no_remote ? `, ${d.skipped_no_remote} unknown-remote` : ''})`
                : `Started ${d.count} pull${d.count === 1 ? '' : 's'}`
            ),
            onError: e => setMsg(`Error: ${(e as Error).message}`),
          })
        }}
        disabled={updateOutdated.isPending}
        className="flex items-center gap-1 text-xs text-amber-400 hover:text-amber-300 disabled:opacity-40 transition-colors"
        title={`Pull every model whose local digest differs from the cached remote digest${scopeSuffix}`}
      >
        {updateOutdated.isPending ? <Loader2 className="w-3 h-3 animate-spin" /> : <Upload className="w-3 h-3" />}
        Update outdated{scopeSuffix}
      </button>
      <button
        onClick={onRefreshCatalog}
        disabled={refreshCatalogPending}
        className="flex items-center gap-1 text-xs text-gray-500 hover:text-gray-300 disabled:opacity-40 transition-colors"
        title={`Catalog cache age: ${cacheAge.toFixed(1)}h`}
      >
        {refreshCatalogPending ? <Loader2 className="w-3 h-3 animate-spin" /> : <RefreshCw className="w-3 h-3" />}
        Refresh catalog
      </button>
    </div>
  )
}
