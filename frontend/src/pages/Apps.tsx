import { useState } from 'react'
import { AppWindow, Plus, Loader2, CheckCircle2, AlertCircle, Copy, Check, Shield, RefreshCw, Cpu, X, Cloud, Server, Trash2 } from 'lucide-react'
import { useApps, useRegisterApp, useApproveApp, useUpdateAppPermissions, useUpdateAppAllowedModels, useModelList, useCloudModels, useRunners, useUpdateAppAllowedRunners, useUpdateAppCategories, useDeleteApp } from '../hooks/useBackend'
import { StatusDot } from '../components/StatusDot'
import type { RegisteredApp, Runner } from '../types'

function relativeTime(iso: string | null): string {
  if (!iso) return 'Never'
  const diffMs = Date.now() - new Date(iso).getTime()
  const diffSec = Math.floor(diffMs / 1000)
  if (diffSec < 60) return `${diffSec}s ago`
  const diffMin = Math.floor(diffSec / 60)
  if (diffMin < 60) return `${diffMin}m ago`
  const diffHr = Math.floor(diffMin / 60)
  if (diffHr < 24) return `${diffHr}h ago`
  return `${Math.floor(diffHr / 24)}d ago`
}

function isOnline(lastSeen: string | null): boolean {
  if (!lastSeen) return false
  return Date.now() - new Date(lastSeen).getTime() < 2 * 60 * 1000
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)

  function copy() {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  return (
    <button
      onClick={copy}
      title="Copy to clipboard"
      className="p-1 rounded text-gray-500 hover:text-gray-300 transition-colors"
    >
      {copied ? <Check className="w-3.5 h-3.5 text-green-400" /> : <Copy className="w-3.5 h-3.5" />}
    </button>
  )
}

function ModelRestrictionEditor({ appId, currentModels }: { appId: number; currentModels: string[] }) {
  const [open, setOpen] = useState(false)
  const [selected, setSelected] = useState<string[]>(currentModels)
  const [customPattern, setCustomPattern] = useState('')
  const updateModels = useUpdateAppAllowedModels()
  const cloudModels = useCloudModels()

  const cloudModelList = cloudModels.data?.filter((m: { enabled: boolean }) => m.enabled).map((m: { id: string }) => m.id) ?? []

  // Dynamic pattern helpers
  const DYNAMIC_PATTERNS = ['@all', '@safe', '@unsafe'] as const
  const hasDynamic = (p: string) => selected.includes(p)

  // Custom patterns = anything that's not a dynamic pattern and not a known cloud model
  const knownSet = new Set([...DYNAMIC_PATTERNS, ...cloudModelList, 'claude-*'])
  const customPatterns = selected.filter(s => !knownSet.has(s))

  function toggle(model: string) {
    setSelected(prev =>
      prev.includes(model) ? prev.filter(m => m !== model) : [...prev, model]
    )
  }

  function setLocalPolicy(policy: '@all' | '@safe' | '@unsafe' | null) {
    // Remove all dynamic patterns, then add the new one
    setSelected(prev => {
      const withoutDynamic = prev.filter(p => !DYNAMIC_PATTERNS.includes(p as typeof DYNAMIC_PATTERNS[number]))
      return policy ? [...withoutDynamic, policy] : withoutDynamic
    })
  }

  function addPattern() {
    const p = customPattern.trim()
    if (p && !selected.includes(p)) {
      setSelected(prev => [...prev, p])
      setCustomPattern('')
    }
  }

  function save() {
    updateModels.mutate({ appId, allowed_models: selected }, {
      onSuccess: () => setOpen(false),
    })
  }

  // Determine current local policy for display
  const activeLocal = selected.find(p => DYNAMIC_PATTERNS.includes(p as typeof DYNAMIC_PATTERNS[number])) as string | undefined
  const hasCloud = selected.some(m => m.startsWith('claude-') || cloudModelList.includes(m))
  const isDefault = currentModels.length === 0

  // Summary for closed state
  function summaryText(): string {
    if (isDefault) return 'Safe local'
    if (!activeLocal && customPatterns.length === 0 && !hasCloud) return 'No local models'
    const parts: string[] = []
    if (activeLocal === '@all') parts.push('All local')
    else if (activeLocal === '@safe') parts.push('Safe local')
    else if (activeLocal === '@unsafe') parts.push('Unsafe local')
    if (customPatterns.length > 0) parts.push(`${customPatterns.length} pattern${customPatterns.length !== 1 ? 's' : ''}`)
    if (hasCloud) parts.push('+ cloud')
    return parts.join(' ') || 'Custom'
  }

  if (!open) {
    return (
      <button
        onClick={() => { setSelected(currentModels); setOpen(true) }}
        title={isDefault ? 'Default — all safe local models, no cloud' : `${currentModels.length} pattern(s)`}
        className={`flex items-center gap-1 text-xs px-2 py-1 rounded-lg transition-colors ${
          isDefault
            ? 'bg-gray-800 text-gray-500 border border-gray-700 hover:border-gray-600'
            : 'bg-blue-900/30 text-blue-400 border border-blue-800'
        }`}
      >
        <Cpu className="w-3 h-3" />
        {summaryText()}
        {hasCloud && <Cloud className="w-3 h-3 text-indigo-400" />}
      </button>
    )
  }

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50" onClick={() => setOpen(false)}>
      <div className="bg-gray-900 border border-gray-700 rounded-xl p-5 w-[28rem] max-h-[80vh] flex flex-col" onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-semibold text-gray-200">Model Access</h3>
          <button onClick={() => setOpen(false)} className="text-gray-500 hover:text-gray-300">
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto space-y-4 mb-4">
          {/* Local model policy */}
          <div>
            <p className="text-[10px] text-gray-500 uppercase tracking-wide mb-2 flex items-center gap-1">
              <Server className="w-3 h-3" /> Local models
            </p>
            <p className="text-[10px] text-gray-600 mb-2">Includes models added in the future.</p>
            <div className="flex gap-1.5 flex-wrap">
              {([
                { id: null, label: 'None', desc: 'No local models (use custom patterns or cloud only)' },
                { id: '@safe' as const, label: 'Safe only', desc: 'All safe local models, current + future' },
                { id: '@all' as const, label: 'All models', desc: 'All local models including unsafe, current + future' },
                { id: '@unsafe' as const, label: 'Unsafe only', desc: 'Only unsafe local models, current + future' },
              ] as const).map(p => {
                const isActive = p.id === null
                  ? !selected.some(s => DYNAMIC_PATTERNS.includes(s as typeof DYNAMIC_PATTERNS[number]))
                  : hasDynamic(p.id)
                return (
                  <button
                    key={p.id ?? 'none'}
                    onClick={() => setLocalPolicy(p.id)}
                    title={p.desc}
                    className={`text-xs px-2.5 py-1.5 rounded-lg transition-colors ${
                      isActive
                        ? 'bg-brand-900 text-brand-300 border border-brand-700'
                        : 'bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-gray-200 border border-transparent'
                    }`}
                  >
                    {p.label}
                  </button>
                )
              })}
            </div>
          </div>

          {/* Cloud models — individual opt-in */}
          <div>
            <p className="text-[10px] text-gray-500 uppercase tracking-wide mb-1.5 flex items-center gap-1">
              <Cloud className="w-3 h-3" /> Cloud models
              <span className="text-gray-600">(opt-in individually)</span>
            </p>
            <div className="space-y-1">
              {cloudModelList.length === 0 ? (
                <p className="text-xs text-gray-600 py-2 text-center">No cloud models configured</p>
              ) : (
                cloudModelList.sort().map((id: string) => (
                  <label key={id} className="flex items-center gap-2 px-3 py-1.5 rounded-lg hover:bg-gray-800 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={selected.includes(id)}
                      onChange={() => toggle(id)}
                      className="rounded border-gray-600 bg-gray-800 text-brand-500 focus:ring-brand-600"
                    />
                    <span className="text-sm text-gray-300 font-mono">{id}</span>
                  </label>
                ))
              )}
              {/* Wildcard shortcut */}
              <label className="flex items-center gap-2 px-3 py-1.5 rounded-lg hover:bg-gray-800 cursor-pointer border border-dashed border-gray-700">
                <input
                  type="checkbox"
                  checked={selected.includes('claude-*')}
                  onChange={() => toggle('claude-*')}
                  className="rounded border-gray-600 bg-gray-800 text-brand-500 focus:ring-brand-600"
                />
                <span className="text-sm text-gray-400 font-mono">claude-*</span>
                <span className="text-[10px] text-gray-600">All current + future Claude models</span>
              </label>
            </div>
          </div>

          {/* Custom patterns */}
          <div>
            <p className="text-[10px] text-gray-500 uppercase tracking-wide mb-1.5">Custom patterns</p>
            {customPatterns.length > 0 && (
              <div className="space-y-1 mb-2">
                {customPatterns.map(p => (
                  <div key={p} className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-gray-800">
                    <span className="text-sm text-gray-300 font-mono flex-1">{p}</span>
                    <button onClick={() => toggle(p)} className="text-gray-500 hover:text-red-400">
                      <X className="w-3 h-3" />
                    </button>
                  </div>
                ))}
              </div>
            )}
            <div className="flex gap-2">
              <input
                type="text"
                value={customPattern}
                onChange={e => setCustomPattern(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && addPattern()}
                placeholder="e.g. llama3.2:* or *:7b"
                className="flex-1 bg-gray-950 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-brand-600 font-mono"
              />
              <button
                onClick={addPattern}
                disabled={!customPattern.trim()}
                className="text-xs px-3 py-1.5 rounded-lg bg-gray-800 text-gray-400 hover:bg-gray-700 disabled:opacity-30 transition-colors"
              >
                Add
              </button>
            </div>
            <p className="text-[10px] text-gray-600 mt-1">Glob patterns: * matches any, ? matches one character</p>
          </div>
        </div>

        {/* Footer info + save */}
        <div className="border-t border-gray-800 pt-3">
          <p className="text-[10px] text-gray-600 mb-3">
            {selected.length === 0
              ? 'Default — app gets all safe local models, no cloud access.'
              : `${selected.length} rule${selected.length !== 1 ? 's' : ''} — only matched models are accessible.`}
          </p>
          <div className="flex gap-2">
            <button
              onClick={() => { setSelected([]); save() }}
              className="flex-1 text-xs px-3 py-2 rounded-lg bg-gray-800 text-gray-400 hover:bg-gray-700 transition-colors"
            >
              Reset to default
            </button>
            <button
              onClick={save}
              disabled={updateModels.isPending}
              className="flex-1 text-xs px-3 py-2 rounded-lg bg-brand-600 text-white hover:bg-brand-500 disabled:opacity-40 transition-colors"
            >
              {updateModels.isPending ? 'Saving...' : 'Save'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

const STANDARD_CATEGORIES = ['tools', 'vision', 'thinking', 'embedding']

function CategoryFilterEditor({ appId, currentAllowed, currentExcluded, allowProfileSwitch }: {
  appId: number
  currentAllowed: string[]
  currentExcluded: string[]
  allowProfileSwitch: boolean
}) {
  const [open, setOpen] = useState(false)
  const [allowed, setAllowed] = useState<string[]>(currentAllowed)
  const [excluded, setExcluded] = useState<string[]>(currentExcluded)
  const [customCat, setCustomCat] = useState('')
  const update = useUpdateAppCategories()

  function toggleAllowed(cat: string) {
    setAllowed(prev => prev.includes(cat) ? prev.filter(c => c !== cat) : [...prev, cat])
    setExcluded(prev => prev.filter(c => c !== cat))  // can't be in both
  }

  function toggleExcluded(cat: string) {
    setExcluded(prev => prev.includes(cat) ? prev.filter(c => c !== cat) : [...prev, cat])
    setAllowed(prev => prev.filter(c => c !== cat))  // can't be in both
  }

  function addCustom() {
    const c = customCat.trim().toLowerCase()
    if (c && !allowed.includes(c) && !excluded.includes(c)) {
      setAllowed(prev => [...prev, c])
      setCustomCat('')
    }
  }

  function save() {
    update.mutate(
      { appId, allow_profile_switch: allowProfileSwitch, allowed_categories: allowed, excluded_categories: excluded },
      { onSuccess: () => setOpen(false) }
    )
  }

  const hasFilter = currentAllowed.length > 0 || currentExcluded.length > 0

  if (!open) {
    return (
      <button
        onClick={() => { setAllowed(currentAllowed); setExcluded(currentExcluded); setOpen(true) }}
        title={hasFilter ? `Category filter active` : 'No category filter'}
        className={`flex items-center gap-1 text-xs px-2 py-1 rounded-lg transition-colors ${
          hasFilter
            ? 'bg-indigo-900/30 text-indigo-400 border border-indigo-800'
            : 'bg-gray-800 text-gray-500 border border-gray-700 hover:border-gray-600'
        }`}
      >
        <Shield className="w-3 h-3" />
        {hasFilter ? `${currentAllowed.length + currentExcluded.length} cat filter` : 'No cat filter'}
      </button>
    )
  }

  const allCats = [...new Set([...STANDARD_CATEGORIES, ...allowed, ...excluded])]

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50" onClick={() => setOpen(false)}>
      <div className="bg-gray-900 border border-gray-700 rounded-xl p-5 w-[28rem] max-h-[80vh] flex flex-col" onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-semibold text-gray-200">Category Filters</h3>
          <button onClick={() => setOpen(false)} className="text-gray-500 hover:text-gray-300"><X className="w-4 h-4" /></button>
        </div>
        <p className="text-[10px] text-gray-500 mb-4">
          Require: app sees only models with at least one of these categories (uncategorized models always pass).<br />
          Exclude: app cannot use models whose <em>only</em> categories are all in this set.
        </p>

        <div className="flex-1 overflow-y-auto space-y-3 mb-4">
          <div className="grid grid-cols-3 gap-1 text-[10px] text-center mb-1">
            <span className="text-gray-600">Category</span>
            <span className="text-green-600">Require</span>
            <span className="text-red-600">Exclude</span>
          </div>
          {allCats.map(cat => (
            <div key={cat} className="grid grid-cols-3 gap-1 items-center">
              <span className="text-xs text-gray-300">{cat}</span>
              <div className="flex justify-center">
                <button onClick={() => toggleAllowed(cat)}
                  className={`w-6 h-6 rounded transition-colors ${
                    allowed.includes(cat) ? 'bg-green-700 text-white' : 'bg-gray-800 text-gray-600 hover:bg-gray-700'
                  }`}>
                  {allowed.includes(cat) ? '✓' : ''}
                </button>
              </div>
              <div className="flex justify-center">
                <button onClick={() => toggleExcluded(cat)}
                  className={`w-6 h-6 rounded transition-colors ${
                    excluded.includes(cat) ? 'bg-red-800 text-white' : 'bg-gray-800 text-gray-600 hover:bg-gray-700'
                  }`}>
                  {excluded.includes(cat) ? '✓' : ''}
                </button>
              </div>
            </div>
          ))}
          <div className="flex gap-2 mt-2">
            <input type="text" value={customCat} onChange={e => setCustomCat(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && addCustom()}
              placeholder="Add category"
              className="flex-1 bg-gray-950 border border-gray-700 rounded-lg px-2.5 py-1.5 text-xs text-gray-200 placeholder-gray-600 focus:outline-none focus:border-brand-600"
            />
            <button onClick={addCustom} disabled={!customCat.trim()}
              className="text-xs px-2.5 py-1.5 rounded-lg bg-gray-800 text-gray-400 hover:bg-gray-700 disabled:opacity-30 transition-colors">
              Add
            </button>
          </div>
        </div>

        <div className="border-t border-gray-800 pt-3">
          <div className="flex gap-2">
            <button onClick={() => { setAllowed([]); setExcluded([]) }}
              className="flex-1 text-xs px-3 py-2 rounded-lg bg-gray-800 text-gray-400 hover:bg-gray-700 transition-colors">
              Clear all
            </button>
            <button onClick={save} disabled={update.isPending}
              className="flex-1 text-xs px-3 py-2 rounded-lg bg-brand-600 text-white hover:bg-brand-500 disabled:opacity-40 transition-colors">
              {update.isPending ? 'Saving...' : 'Save'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

function RunnerAffinityEditor({ appId, currentRunnerIds }: { appId: number; currentRunnerIds: number[] }) {
  const [open, setOpen] = useState(false)
  const [selected, setSelected] = useState<number[]>(currentRunnerIds)
  const runners = useRunners()
  const update = useUpdateAppAllowedRunners()
  const runnerList = runners.data ?? []

  const isUnrestricted = currentRunnerIds.length === 0

  function toggle(id: number) {
    setSelected(prev => prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id])
  }

  function save() {
    update.mutate({ appId, allowed_runner_ids: selected }, {
      onSuccess: () => setOpen(false),
    })
  }

  if (!open) {
    return (
      <button
        onClick={() => { setSelected(currentRunnerIds); setOpen(true) }}
        title={isUnrestricted ? 'Any runner' : `Restricted to ${currentRunnerIds.length} runner(s)`}
        className={`flex items-center gap-1 text-xs px-2 py-1 rounded-lg transition-colors ${
          isUnrestricted
            ? 'bg-gray-800 text-gray-500 border border-gray-700 hover:border-gray-600'
            : 'bg-purple-900/30 text-purple-400 border border-purple-800'
        }`}
      >
        <Server className="w-3 h-3" />
        {isUnrestricted ? 'Any runner' : `${currentRunnerIds.length} runner${currentRunnerIds.length !== 1 ? 's' : ''}`}
      </button>
    )
  }

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50" onClick={() => setOpen(false)}>
      <div className="bg-gray-900 border border-gray-700 rounded-xl p-5 w-80" onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-semibold text-gray-200">Runner Affinity</h3>
          <button onClick={() => setOpen(false)} className="text-gray-500 hover:text-gray-300">
            <X className="w-4 h-4" />
          </button>
        </div>
        <p className="text-[10px] text-gray-500 mb-3">
          {selected.length === 0
            ? 'No restriction — app can use any enabled runner.'
            : `Restricted to ${selected.length} runner(s).`}
        </p>
        <div className="space-y-1 mb-4">
          {runnerList.map((r: Runner) => (
            <label key={r.id} className="flex items-center gap-2 px-3 py-1.5 rounded-lg hover:bg-gray-800 cursor-pointer">
              <input
                type="checkbox"
                checked={selected.includes(r.id)}
                onChange={() => toggle(r.id)}
                className="rounded border-gray-600 bg-gray-800 text-brand-500 focus:ring-brand-600"
              />
              <span className="text-sm text-gray-300">{r.hostname}</span>
              {!r.enabled && <span className="text-[10px] text-gray-600">disabled</span>}
            </label>
          ))}
          {runnerList.length === 0 && (
            <p className="text-xs text-gray-600 py-2 text-center">No runners available</p>
          )}
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => { setSelected([]); update.mutate({ appId, allowed_runner_ids: [] }, { onSuccess: () => setOpen(false) }) }}
            className="flex-1 text-xs px-3 py-2 rounded-lg bg-gray-800 text-gray-400 hover:bg-gray-700 transition-colors"
          >
            Any runner
          </button>
          <button
            onClick={save}
            disabled={update.isPending}
            className="flex-1 text-xs px-3 py-2 rounded-lg bg-brand-600 text-white hover:bg-brand-500 disabled:opacity-40 transition-colors"
          >
            {update.isPending ? 'Saving...' : 'Save'}
          </button>
        </div>
      </div>
    </div>
  )
}

export function Apps() {
  const apps = useApps()
  const register = useRegisterApp()
  const approve = useApproveApp()
  const updatePerms = useUpdateAppPermissions()
  const deleteApp = useDeleteApp()
  const [name, setName] = useState('')
  const [url, setUrl] = useState('')
  const [generatedKey, setGeneratedKey] = useState<string | null>(null)
  const [regMsg, setRegMsg] = useState<{ type: 'ok' | 'err'; text: string } | null>(null)

  const appList = apps.data ?? []
  const pendingApps = appList.filter((a: RegisteredApp) => a.status === 'pending')
  const activeApps = appList.filter((a: RegisteredApp) => a.status !== 'pending')

  async function handleRegister() {
    const trimName = name.trim()
    const trimUrl = url.trim()
    if (!trimName || !trimUrl) return
    setRegMsg(null)
    setGeneratedKey(null)
    try {
      const result = await register.mutateAsync({ name: trimName, base_url: trimUrl })
      if (result.ok) {
        setRegMsg({ type: 'ok', text: `Registered "${trimName}"` })
        setGeneratedKey(result.api_key)
        setName('')
        setUrl('')
      } else {
        setRegMsg({ type: 'err', text: 'Registration returned not ok' })
      }
    } catch (e) {
      setRegMsg({ type: 'err', text: (e as Error).message })
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2">
        <AppWindow className="w-4 h-4 text-brand-400" />
        <h1 className="text-base font-semibold text-gray-200">Applications</h1>
      </div>

      {/* Pending apps */}
      {pendingApps.length > 0 && (
        <div className="bg-yellow-900/20 border border-yellow-800/50 rounded-xl p-4 space-y-3">
          <div className="flex items-center gap-2">
            <RefreshCw className="w-4 h-4 text-yellow-400" />
            <h2 className="text-sm font-semibold text-yellow-300">Pending Approval</h2>
          </div>
          {pendingApps.map((a: RegisteredApp) => (
            <div key={a.id} className="flex items-center justify-between bg-gray-900/60 rounded-lg px-4 py-3">
              <div>
                <p className="text-sm text-gray-200 font-medium">{a.name}</p>
                <p className="text-xs text-gray-500">{a.base_url}</p>
              </div>
              <button
                onClick={() => approve.mutate(a.id)}
                disabled={approve.isPending}
                className="flex items-center gap-1.5 bg-green-700 hover:bg-green-600 disabled:opacity-40 text-white text-xs px-3 py-1.5 rounded-lg transition-colors"
              >
                {approve.isPending ? <Loader2 className="w-3 h-3 animate-spin" /> : <CheckCircle2 className="w-3 h-3" />}
                Approve
              </button>
            </div>
          ))}
        </div>
      )}

      {/* App table */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
        {apps.isLoading ? (
          <div className="py-8 text-center text-gray-600 text-sm">Loading apps…</div>
        ) : activeApps.length === 0 ? (
          <div className="py-8 text-center text-gray-600 text-sm">
            No apps registered yet
          </div>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800 text-xs text-gray-500 uppercase tracking-wide">
                <th className="text-left px-4 py-3 font-medium">Status</th>
                <th className="text-left px-4 py-3 font-medium">Name</th>
                <th className="text-left px-4 py-3 font-medium hidden md:table-cell">URL</th>
                <th className="text-left px-4 py-3 font-medium">Last Seen</th>
                <th className="text-left px-4 py-3 font-medium">Permissions</th>
              </tr>
            </thead>
            <tbody>
              {activeApps.map((a: RegisteredApp) => (
                <tr key={a.id} className="border-b border-gray-800 last:border-0 hover:bg-gray-800/40 transition-colors">
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <StatusDot online={isOnline(a.last_seen)} />
                      <span className={`text-xs ${isOnline(a.last_seen) ? 'text-green-400' : 'text-gray-500'}`}>
                        {isOnline(a.last_seen) ? 'Online' : 'Offline'}
                      </span>
                    </div>
                  </td>
                  <td className="px-4 py-3 text-gray-200 font-medium">{a.name}</td>
                  <td className="px-4 py-3 text-gray-500 text-xs truncate max-w-xs hidden md:table-cell">
                    {a.base_url}
                  </td>
                  <td className="px-4 py-3 text-gray-500 text-xs tabular-nums">
                    {relativeTime(a.last_seen)}
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2 flex-wrap">
                      <button
                        onClick={() => updatePerms.mutate({ appId: a.id, allow_profile_switch: !a.allow_profile_switch })}
                        disabled={updatePerms.isPending}
                        title={a.allow_profile_switch ? 'Profile switching enabled' : 'Profile switching disabled'}
                        className={`flex items-center gap-1 text-xs px-2 py-1 rounded-lg transition-colors ${
                          a.allow_profile_switch
                            ? 'bg-green-900/30 text-green-400 border border-green-800'
                            : 'bg-gray-800 text-gray-500 border border-gray-700 hover:border-gray-600'
                        }`}
                      >
                        <Shield className="w-3 h-3" />
                        {a.allow_profile_switch ? 'Profiles' : 'No profiles'}
                      </button>
                      <ModelRestrictionEditor appId={a.id} currentModels={a.allowed_models ?? []} />
                      <CategoryFilterEditor
                        appId={a.id}
                        currentAllowed={a.allowed_categories ?? []}
                        currentExcluded={a.excluded_categories ?? []}
                        allowProfileSwitch={a.allow_profile_switch}
                      />
                      <RunnerAffinityEditor appId={a.id} currentRunnerIds={a.allowed_runner_ids ?? []} />
                      <button
                        onClick={() => {
                          if (confirm(`Delete registered app "${a.name}"? Historical queue jobs are preserved (app_id becomes null), but its permissions and rate limits are removed. The app will need to re-register to come back.`)) {
                            deleteApp.mutate(a.id)
                          }
                        }}
                        disabled={deleteApp.isPending}
                        title="Delete application"
                        className="flex items-center gap-1 text-xs px-2 py-1 rounded-lg bg-gray-800 text-gray-500 border border-gray-700 hover:border-red-700 hover:text-red-400 hover:bg-red-950/30 disabled:opacity-40 transition-colors"
                      >
                        {deleteApp.isPending && deleteApp.variables === a.id
                          ? <Loader2 className="w-3 h-3 animate-spin" />
                          : <Trash2 className="w-3 h-3" />}
                        Delete
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Register new app */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <div className="flex items-center gap-2 mb-4">
          <Plus className="w-4 h-4 text-brand-400" />
          <h2 className="text-sm font-semibold text-gray-300">Register Application</h2>
        </div>
        <div className="space-y-3">
          <div>
            <label className="block text-xs text-gray-500 mb-1">App Name</label>
            <input
              type="text"
              value={name}
              onChange={e => setName(e.target.value)}
              placeholder="e.g. murderbot-telegram"
              className="w-full bg-gray-950 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-brand-600"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1">Base URL</label>
            <input
              type="text"
              value={url}
              onChange={e => setUrl(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleRegister()}
              placeholder="e.g. http://murderbot-telegram:8080"
              className="w-full bg-gray-950 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-brand-600"
            />
          </div>
          <button
            onClick={handleRegister}
            disabled={register.isPending || !name.trim() || !url.trim()}
            className="flex items-center gap-2 bg-brand-600 hover:bg-brand-500 disabled:opacity-40 text-white text-sm font-medium px-4 py-2 rounded-lg transition-colors"
          >
            {register.isPending
              ? <Loader2 className="w-4 h-4 animate-spin" />
              : <Plus className="w-4 h-4" />}
            Register
          </button>
        </div>

        {/* Result feedback */}
        {regMsg && (
          <div className={`mt-3 flex items-center gap-1.5 text-xs ${regMsg.type === 'ok' ? 'text-green-400' : 'text-red-400'}`}>
            {regMsg.type === 'ok'
              ? <CheckCircle2 className="w-3.5 h-3.5" />
              : <AlertCircle className="w-3.5 h-3.5" />}
            {regMsg.text}
          </div>
        )}

        {/* Generated API key display */}
        {generatedKey && (
          <div className="mt-4 bg-gray-950 border border-gray-700 rounded-lg p-3">
            <p className="text-xs text-gray-500 mb-1.5 font-medium uppercase tracking-wide">Generated API Key</p>
            <p className="text-xs text-amber-400 mb-2">Save this key — it will not be shown again.</p>
            <div className="flex items-center gap-2 bg-gray-900 rounded-lg px-3 py-2">
              <code className="flex-1 text-xs text-brand-300 font-mono break-all">{generatedKey}</code>
              <CopyButton text={generatedKey} />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
