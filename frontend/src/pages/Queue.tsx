import { useState } from 'react'
import { ListOrdered, X, ChevronUp, ChevronDown, Clock, Zap, CheckCircle2, XCircle, Loader2, RefreshCw, ChevronRight } from 'lucide-react'
import { useQueueJobs, useQueueHistory, useQueueMetrics, useCancelQueueJob, useSetJobPriority } from '../hooks/useBackend'
import { StatCard } from '../components/StatCard'
import type { QueueJob } from '../types'

function formatDuration(secs: number): string {
  if (secs < 1) return '<1s'
  if (secs < 60) return `${Math.round(secs)}s`
  if (secs < 3600) return `${Math.floor(secs / 60)}m ${Math.round(secs % 60)}s`
  return `${Math.floor(secs / 3600)}h ${Math.floor((secs % 3600) / 60)}m`
}

function timeAgo(iso: string | null): string {
  if (!iso) return '—'
  const diffSec = Math.floor((Date.now() - new Date(iso).getTime()) / 1000)
  if (diffSec < 0) return 'just now'
  if (diffSec < 60) return `${diffSec}s ago`
  if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`
  return `${Math.floor(diffSec / 3600)}h ago`
}

function waitTime(iso: string | null): string {
  if (!iso) return '—'
  const diffSec = Math.max(0, Math.floor((Date.now() - new Date(iso).getTime()) / 1000))
  return formatDuration(diffSec)
}

function statusBadge(status: string) {
  const styles: Record<string, string> = {
    running: 'bg-blue-900/40 text-blue-400',
    loading_model: 'bg-yellow-900/40 text-yellow-400',
    queued: 'bg-gray-800 text-gray-400',
    waiting_for_eviction: 'bg-orange-900/40 text-orange-400',
    completed: 'bg-green-900/40 text-green-400',
    failed: 'bg-red-900/40 text-red-400',
    cancelled: 'bg-gray-800 text-gray-500',
  }
  return (
    <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${styles[status] || 'bg-gray-800 text-gray-400'}`}>
      {status.replace('_', ' ')}
    </span>
  )
}

function JobRow({ job, onCancel, onPriorityUp, onPriorityDown, isPending }: {
  job: QueueJob
  onCancel: () => void
  onPriorityUp: () => void
  onPriorityDown: () => void
  isPending: boolean
}) {
  const isQueued = job.status === 'queued' || job.status === 'waiting_for_eviction'
  const isRunning = job.status === 'running' || job.status === 'loading_model'
  const canCancel = isQueued || isRunning
  const showRunnerHost =
    !!job.runner_hostname &&
    (job.status === 'loading_model' ||
      job.status === 'running' ||
      job.status === 'waiting_for_eviction')

  return (
    <div className="flex items-center gap-3 bg-gray-950 rounded-lg px-3 py-2.5">
      <div className="flex items-center gap-2 w-28 flex-shrink-0">
        {isRunning && <Loader2 className="w-3 h-3 text-blue-400 animate-spin" />}
        {statusBadge(job.status)}
      </div>
      <div className="min-w-0 flex-1">
        <p className="text-sm text-gray-200 truncate">
          {job.model}
          {showRunnerHost && (
            <span className="ml-2 text-xs font-normal text-gray-500">
              on <span className="text-gray-300">{job.runner_hostname}</span>
            </span>
          )}
        </p>
        <p className="text-xs text-gray-500 truncate">
          {job.app_name || 'unknown'} · {job.id}
          {job.metadata?.slot !== undefined && <span> · slot {String(job.metadata.slot)}</span>}
        </p>
      </div>
      <div className="text-xs text-gray-500 tabular-nums w-16 text-right flex-shrink-0">
        {job.priority !== 0 && <span className="text-brand-400">p{job.priority} </span>}
        {waitTime(job.created_at)}
      </div>
      <div className="flex items-center gap-0.5 flex-shrink-0">
        {isQueued && (
          <>
            <button
              onClick={onPriorityUp}
              disabled={isPending}
              className="p-1 rounded hover:bg-gray-800 text-gray-500 hover:text-gray-300 transition-colors"
              title="Increase priority"
            >
              <ChevronUp className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={onPriorityDown}
              disabled={isPending}
              className="p-1 rounded hover:bg-gray-800 text-gray-500 hover:text-gray-300 transition-colors"
              title="Decrease priority"
            >
              <ChevronDown className="w-3.5 h-3.5" />
            </button>
          </>
        )}
        {canCancel && (
          <button
            onClick={onCancel}
            disabled={isPending}
            className="p-1 rounded hover:bg-red-900/50 text-gray-500 hover:text-red-400 transition-colors"
            title="Cancel job"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        )}
      </div>
    </div>
  )
}

export function Queue() {
  const jobs = useQueueJobs()
  const history = useQueueHistory()
  const metrics = useQueueMetrics()
  const cancelJob = useCancelQueueJob()
  const setPriority = useSetJobPriority()
  const [showHistory, setShowHistory] = useState(false)

  const m = metrics.data
  const jobList = jobs.data ?? []
  const historyList = (history.data ?? []).slice(0, 20)

  const totalActive = m ? m.active.queued + m.active.running + m.active.loading_model + m.active.waiting_for_eviction : 0

  return (
    <div className="space-y-6">
      {/* Stats */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <StatCard
          label="Queue Depth"
          value={m ? String(m.active.queued + m.active.waiting_for_eviction) : '—'}
          sub={totalActive > 0 ? `${totalActive} total active` : undefined}
          icon={<ListOrdered className="w-4 h-4" />}
        />
        <StatCard
          label="Running"
          value={m ? String(m.active.running) : '—'}
          sub={m?.active.loading_model ? `${m.active.loading_model} loading` : undefined}
          icon={<Zap className="w-4 h-4" />}
        />
        <StatCard
          label="Completed / hr"
          value={m ? String(m.last_hour.completed) : '—'}
          sub={m?.last_hour.failed ? `${m.last_hour.failed} failed` : undefined}
          icon={<CheckCircle2 className="w-4 h-4" />}
        />
        <StatCard
          label="Avg Time"
          value={m ? formatDuration(m.timing.avg_processing_secs) : '—'}
          sub={m ? `${formatDuration(m.timing.avg_wait_secs)} wait` : undefined}
          icon={<Clock className="w-4 h-4" />}
        />
      </div>

      {/* Active Queue */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-medium text-gray-300 flex items-center gap-2">
            <ListOrdered className="w-4 h-4 text-brand-400" />
            Active Queue
          </h2>
          {jobs.isFetching && <RefreshCw className="w-3 h-3 text-gray-600 animate-spin" />}
        </div>
        {jobs.isLoading ? (
          <p className="text-xs text-gray-600 py-4 text-center">Loading...</p>
        ) : jobList.length === 0 ? (
          <p className="text-xs text-gray-600 py-4 text-center">Queue is empty</p>
        ) : (
          <div className="space-y-1.5">
            {jobList.map((job: QueueJob) => (
              <JobRow
                key={job.id}
                job={job}
                onCancel={() => cancelJob.mutate(job.id)}
                onPriorityUp={() => setPriority.mutate({ jobId: job.id, priority: job.priority + 1 })}
                onPriorityDown={() => setPriority.mutate({ jobId: job.id, priority: Math.max(0, job.priority - 1) })}
                isPending={cancelJob.isPending || setPriority.isPending}
              />
            ))}
          </div>
        )}
      </div>

      {/* Breakdown: By Model + By App */}
      {m && (m.by_model.length > 0 || m.by_app.length > 0) && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {m.by_model.length > 0 && (
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
              <h2 className="text-sm font-medium text-gray-300 mb-3">By Model (1h)</h2>
              <div className="space-y-1.5">
                {m.by_model.map(row => (
                  <div key={row.model} className="flex items-center justify-between bg-gray-950 rounded-lg px-3 py-2">
                    <p className="text-sm text-gray-200 truncate flex-1 min-w-0">{row.model}</p>
                    <div className="flex items-center gap-3 text-xs tabular-nums flex-shrink-0 ml-2">
                      <span className="text-gray-400">{row.total} total</span>
                      <span className="text-green-400">{row.completed}</span>
                      {row.failed > 0 && <span className="text-red-400">{row.failed}</span>}
                      <span className="text-gray-500">{formatDuration(row.avg_secs)}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
          {m.by_app.length > 0 && (
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
              <h2 className="text-sm font-medium text-gray-300 mb-3">By App (1h)</h2>
              <div className="space-y-1.5">
                {m.by_app.map(row => (
                  <div key={row.app_name} className="flex items-center justify-between bg-gray-950 rounded-lg px-3 py-2">
                    <p className="text-sm text-gray-200 truncate flex-1 min-w-0">{row.app_name}</p>
                    <div className="flex items-center gap-3 text-xs tabular-nums flex-shrink-0 ml-2">
                      <span className="text-gray-400">{row.total} total</span>
                      <span className="text-green-400">{row.completed}</span>
                      {row.failed > 0 && <span className="text-red-400">{row.failed}</span>}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* All-time totals */}
      {m && (
        <div className="text-xs text-gray-600 text-center">
          All time: {m.totals.all_time.toLocaleString()} jobs · {m.totals.completed.toLocaleString()} completed · {m.totals.failed.toLocaleString()} failed
        </div>
      )}

      {/* History (collapsible) */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
        <button
          onClick={() => setShowHistory(!showHistory)}
          className="flex items-center gap-2 w-full text-left"
        >
          <ChevronRight className={`w-4 h-4 text-gray-500 transition-transform ${showHistory ? 'rotate-90' : ''}`} />
          <h2 className="text-sm font-medium text-gray-300">Recent History</h2>
          <span className="text-xs text-gray-600 ml-auto">{historyList.length} jobs</span>
        </button>
        {showHistory && (
          <div className="mt-3 space-y-1.5">
            {historyList.length === 0 ? (
              <p className="text-xs text-gray-600 py-2 text-center">No recent jobs</p>
            ) : (
              historyList.map((job: QueueJob) => {
                const duration = job.started_at && job.completed_at
                  ? (new Date(job.completed_at).getTime() - new Date(job.started_at).getTime()) / 1000
                  : null
                return (
                  <div key={job.id} className="flex items-center gap-3 bg-gray-950 rounded-lg px-3 py-2">
                    <div className="flex-shrink-0">
                      {job.status === 'completed' ? (
                        <CheckCircle2 className="w-3.5 h-3.5 text-green-500" />
                      ) : job.status === 'cancelled' ? (
                        <X className="w-3.5 h-3.5 text-gray-500" />
                      ) : (
                        <XCircle className="w-3.5 h-3.5 text-red-500" />
                      )}
                    </div>
                    <div className="min-w-0 flex-1">
                      <p className="text-sm text-gray-300 truncate">{job.model}</p>
                      <p className="text-xs text-gray-500 truncate">{job.app_name || 'unknown'} · {job.id}</p>
                    </div>
                    <div className="text-xs text-gray-500 tabular-nums flex-shrink-0">
                      {duration !== null && <span>{formatDuration(duration)} · </span>}
                      {timeAgo(job.completed_at ?? job.started_at ?? job.created_at)}
                    </div>
                  </div>
                )
              })
            )}
          </div>
        )}
      </div>
    </div>
  )
}
