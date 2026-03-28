import { useState } from 'react'
import { AppWindow, Plus, Loader2, CheckCircle2, AlertCircle, Copy, Check, Shield, RefreshCw, Cpu, X, Cloud, Server } from 'lucide-react'
import { useApps, useRegisterApp, useApproveApp, useUpdateAppPermissions, useUpdateAppAllowedModels, useModelList, useCloudModels, useRunners, useUpdateAppAllowedRunners } from '../hooks/useBackend'
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
  const modelList = useModelList()
  const cloudModels = useCloudModels()

  const localModels = modelList.data?.map(m => ({ id: m.name, safety: m.safety })) ?? []
  const cloudModelList = cloudModels.data?.filter((m: { enabled: boolean }) => m.enabled).map((m: { id: string }) => m.id) ?? []

  // Separate custom patterns from specific model selections
  const knownModels = new Set([...localModels.map(m => m.id), ...cloudModelList])
  const customPatterns = selected.filter(s => !knownModels.has(s) && (s.includes('*') || s.includes('?')))

  function toggle(model: string) {
    setSelected(prev =>
      prev.includes(model) ? prev.filter(m => m !== model) : [...prev, model]
    )
  }

  function addPattern() {
    const p = customPattern.trim()
    if (p && !selected.includes(p)) {
      setSelected(prev => [...prev, p])
      setCustomPattern('')
    }
  }

  function applyPreset(preset: 'unrestricted' | 'safe-local' | 'all-local' | 'safe-all') {
    const safeLocal = localModels.filter(m => m.safety !== 'unsafe').map(m => m.id)
    const allLocal = localModels.map(m => m.id)

    switch (preset) {
      case 'unrestricted':
        setSelected([])
        break
      case 'safe-local':
        setSelected(safeLocal)
        break
      case 'all-local':
        setSelected(allLocal)
        break
      case 'safe-all':
        setSelected([...safeLocal, ...cloudModelList])
        break
    }
  }

  function save() {
    updateModels.mutate({ appId, allowed_models: selected }, {
      onSuccess: () => setOpen(false),
    })
  }

  const isUnrestricted = currentModels.length === 0
  const hasCloud = currentModels.some(m => m.startsWith('claude-'))
  const modelCount = currentModels.length

  if (!open) {
    return (
      <button
        onClick={() => { setSelected(currentModels); setOpen(true) }}
        title={isUnrestricted ? 'All safe local models (no cloud)' : `${modelCount} model pattern(s)`}
        className={`flex items-center gap-1 text-xs px-2 py-1 rounded-lg transition-colors ${
          isUnrestricted
            ? 'bg-gray-800 text-gray-500 border border-gray-700 hover:border-gray-600'
            : 'bg-blue-900/30 text-blue-400 border border-blue-800'
        }`}
      >
        <Cpu className="w-3 h-3" />
        {isUnrestricted ? 'Safe local' : `${modelCount} model${modelCount !== 1 ? 's' : ''}`}
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

        {/* Presets */}
        <div className="mb-4">
          <p className="text-[10px] text-gray-500 uppercase tracking-wide mb-2">Quick presets</p>
          <div className="flex gap-1.5 flex-wrap">
            {[
              { id: 'unrestricted' as const, label: 'Safe local only', desc: 'Default — all safe local models, no cloud' },
              { id: 'safe-local' as const, label: 'Safe local (explicit)', desc: 'Same but listed explicitly' },
              { id: 'all-local' as const, label: 'All local', desc: 'Including unsafe models' },
              { id: 'safe-all' as const, label: 'Safe + Cloud', desc: 'Safe local + all cloud models' },
            ].map(p => (
              <button
                key={p.id}
                onClick={() => applyPreset(p.id)}
                title={p.desc}
                className="text-xs px-2.5 py-1.5 rounded-lg bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-gray-200 transition-colors"
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>

        <div className="flex-1 overflow-y-auto space-y-4 mb-4">
          {/* Local models */}
          <div>
            <p className="text-[10px] text-gray-500 uppercase tracking-wide mb-1.5 flex items-center gap-1">
              <Cpu className="w-3 h-3" /> Local models
            </p>
            <div className="space-y-1">
              {localModels.length === 0 ? (
                <p className="text-xs text-gray-600 py-2 text-center">No local models downloaded</p>
              ) : (
                localModels.sort((a, b) => a.id.localeCompare(b.id)).map(m => (
                  <label key={m.id} className="flex items-center gap-2 px-3 py-1.5 rounded-lg hover:bg-gray-800 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={selected.includes(m.id)}
                      onChange={() => toggle(m.id)}
                      className="rounded border-gray-600 bg-gray-800 text-brand-500 focus:ring-brand-600"
                    />
                    <span className="text-sm text-gray-300 font-mono flex-1">{m.id}</span>
                    {m.safety === 'unsafe' && (
                      <span className="text-[10px] bg-red-900/40 text-red-400 px-1.5 py-0.5 rounded">unsafe</span>
                    )}
                  </label>
                ))
              )}
            </div>
          </div>

          {/* Cloud models */}
          <div>
            <p className="text-[10px] text-gray-500 uppercase tracking-wide mb-1.5 flex items-center gap-1">
              <Cloud className="w-3 h-3" /> Cloud models
              <span className="text-gray-600">(denied by default)</span>
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
              ? 'No restrictions — app gets all safe local models, no cloud access.'
              : `${selected.length} pattern${selected.length !== 1 ? 's' : ''} — only matched models are accessible.`}
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
                      <RunnerAffinityEditor appId={a.id} currentRunnerIds={a.allowed_runner_ids ?? []} />
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
