import { useState } from 'react'
import { AppWindow, Plus, Loader2, CheckCircle2, AlertCircle, Copy, Check } from 'lucide-react'
import { useApps, useRegisterApp } from '../hooks/useBackend'
import { StatusDot } from '../components/StatusDot'

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

export function Apps() {
  const apps = useApps()
  const register = useRegisterApp()
  const [name, setName] = useState('')
  const [url, setUrl] = useState('')
  const [generatedKey, setGeneratedKey] = useState<string | null>(null)
  const [regMsg, setRegMsg] = useState<{ type: 'ok' | 'err'; text: string } | null>(null)

  const appList = apps.data ?? []

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

      {/* App table */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
        {apps.isLoading ? (
          <div className="py-8 text-center text-gray-600 text-sm">Loading apps…</div>
        ) : appList.length === 0 ? (
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
              </tr>
            </thead>
            <tbody>
              {appList.map(app => (
                <tr key={app.id} className="border-b border-gray-800 last:border-0 hover:bg-gray-800/40 transition-colors">
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <StatusDot online={isOnline(app.last_seen)} />
                      <span className={`text-xs ${isOnline(app.last_seen) ? 'text-green-400' : 'text-gray-500'}`}>
                        {isOnline(app.last_seen) ? 'Online' : 'Offline'}
                      </span>
                    </div>
                  </td>
                  <td className="px-4 py-3 text-gray-200 font-medium">{app.name}</td>
                  <td className="px-4 py-3 text-gray-500 text-xs truncate max-w-xs hidden md:table-cell">
                    {app.base_url}
                  </td>
                  <td className="px-4 py-3 text-gray-500 text-xs tabular-nums">
                    {relativeTime(app.last_seen)}
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
