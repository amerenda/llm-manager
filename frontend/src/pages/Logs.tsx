import { useState, useEffect, useRef } from 'react'
import { RefreshCw, Pause, Play } from 'lucide-react'
import { useRunners, useRunnerLogs } from '../hooks/useBackend'

const TAIL_OPTIONS = [50, 100, 200, 500]
const SEVERITY_OPTIONS = ['all', 'error', 'warning', 'info', 'debug'] as const
type Severity = typeof SEVERITY_OPTIONS[number]

function lineSeverity(line: string): Severity {
  if (/\bERROR\b|\berror\b/.test(line)) return 'error'
  if (/\bWARN(ING)?\b/.test(line)) return 'warning'
  if (/\bINFO\b/.test(line)) return 'info'
  if (/\bDEBUG\b/.test(line)) return 'debug'
  return 'info'
}

function filterLines(lines: string[], severity: Severity): string[] {
  if (severity === 'all') return lines
  return lines.filter(l => lineSeverity(l) === severity)
}

function LogPane({ lines, label, empty }: { lines: string[]; label: string; empty: string }) {
  const bottomRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [lines])

  return (
    <div className="flex flex-col flex-1 min-h-0">
      <div className="text-xs font-medium text-gray-400 px-1 pb-1">{label}</div>
      <div className="flex-1 overflow-y-auto bg-gray-950 border border-gray-800 rounded-lg p-3 font-mono text-xs text-gray-300 leading-relaxed min-h-0">
        {lines.length === 0 ? (
          <span className="text-gray-600">{empty}</span>
        ) : (
          lines.map((line, i) => (
            <div key={i} className="whitespace-pre-wrap break-all">
              {colorize(line)}
            </div>
          ))
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}

function colorize(line: string) {
  if (/\bERROR\b|\berror\b/.test(line)) return <span className="text-red-400">{line}</span>
  if (/\bWARN(ING)?\b/.test(line)) return <span className="text-yellow-400">{line}</span>
  if (/\bINFO\b/.test(line)) return <span className="text-gray-300">{line}</span>
  if (/\bDEBUG\b/.test(line)) return <span className="text-gray-500">{line}</span>
  return <span>{line}</span>
}

export function Logs() {
  const runners = useRunners()
  const [runnerId, setRunnerId] = useState<number | null>(null)
  const [service, setService] = useState<'all' | 'agent' | 'ollama'>('all')
  const [tail, setTail] = useState(200)
  const [live, setLive] = useState(true)
  const [severity, setSeverity] = useState<Severity>('all')

  const activeRunners = runners.data ?? []

  useEffect(() => {
    if (runnerId === null && activeRunners.length > 0) {
      setRunnerId(activeRunners[0].id)
    }
  }, [activeRunners, runnerId])

  const logs = useRunnerLogs(runnerId, tail, service, live)

  const agentLines = filterLines(logs.data?.agent_logs ?? [], severity)
  const ollamaLines = filterLines(logs.data?.ollama_logs ?? [], severity)

  return (
    <div className="flex flex-col gap-4 h-[calc(100vh-8rem)]">
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-3">
        <select
          value={runnerId ?? ''}
          onChange={e => setRunnerId(Number(e.target.value))}
          className="bg-gray-900 border border-gray-700 text-gray-200 text-sm rounded-lg px-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-brand-500"
        >
          {activeRunners.map(r => (
            <option key={r.id} value={r.id}>{r.hostname}</option>
          ))}
        </select>

        <select
          value={service}
          onChange={e => setService(e.target.value as 'all' | 'agent' | 'ollama')}
          className="bg-gray-900 border border-gray-700 text-gray-200 text-sm rounded-lg px-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-brand-500"
        >
          <option value="all">Agent + Ollama</option>
          <option value="agent">Agent only</option>
          <option value="ollama">Ollama only</option>
        </select>

        <select
          value={tail}
          onChange={e => setTail(Number(e.target.value))}
          className="bg-gray-900 border border-gray-700 text-gray-200 text-sm rounded-lg px-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-brand-500"
        >
          {TAIL_OPTIONS.map(n => (
            <option key={n} value={n}>Last {n} lines</option>
          ))}
        </select>

        <select
          value={severity}
          onChange={e => setSeverity(e.target.value as Severity)}
          className="bg-gray-900 border border-gray-700 text-gray-200 text-sm rounded-lg px-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-brand-500"
        >
          <option value="all">All levels</option>
          <option value="error">Error</option>
          <option value="warning">Warning</option>
          <option value="info">Info</option>
          <option value="debug">Debug</option>
        </select>

        <button
          onClick={() => setLive(v => !v)}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm transition-colors ${
            live
              ? 'bg-brand-900 text-brand-300 hover:bg-brand-800'
              : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
          }`}
        >
          {live ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
          {live ? 'Pause' : 'Resume'}
        </button>

        <button
          onClick={() => logs.refetch()}
          disabled={logs.isFetching}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm bg-gray-800 text-gray-400 hover:bg-gray-700 disabled:opacity-50 transition-colors"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${logs.isFetching ? 'animate-spin' : ''}`} />
          Refresh
        </button>

        {logs.data && (
          <span className="text-xs text-gray-600 ml-auto">
            {agentLines.length}/{logs.data.agent_logs.length} agent · {ollamaLines.length}/{logs.data.ollama_logs.length} ollama
          </span>
        )}
      </div>

      {/* Error */}
      {logs.isError && (
        <div className="text-sm text-red-400 bg-red-950/30 border border-red-900/50 rounded-lg px-3 py-2">
          Failed to fetch logs: {String(logs.error)}
        </div>
      )}

      {/* Log panes */}
      {service === 'all' ? (
        <div className="flex gap-4 flex-1 min-h-0">
          <div className="flex flex-col flex-1 min-h-0">
            <LogPane
              lines={agentLines}
              label="Agent"
              empty="No agent logs captured yet."
            />
          </div>
          {logs.data?.ollama_available !== false && (
            <div className="flex flex-col flex-1 min-h-0">
              <LogPane lines={ollamaLines} label="Ollama" empty="No Ollama logs." />
            </div>
          )}
        </div>
      ) : service === 'agent' ? (
        <LogPane lines={agentLines} label="Agent" empty="No agent logs captured yet." />
      ) : (
        <LogPane
          lines={ollamaLines}
          label="Ollama"
          empty={
            logs.data && !logs.data.ollama_available
              ? 'Ollama container logs not available (Docker not configured).'
              : 'No Ollama logs.'
          }
        />
      )}
    </div>
  )
}
