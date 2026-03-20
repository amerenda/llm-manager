import type { ReactNode } from 'react'

interface StatCardProps {
  label: string
  value: ReactNode
  sub?: ReactNode
  /** 0–100, enables progress bar when provided */
  progress?: number
  icon?: ReactNode
}

function progressColor(pct: number): string {
  if (pct >= 90) return 'bg-red-500'
  if (pct >= 70) return 'bg-yellow-500'
  return 'bg-brand-500'
}

export function StatCard({ label, value, sub, progress, icon }: StatCardProps) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="text-xs text-gray-500 uppercase tracking-wide font-medium">{label}</span>
        {icon && <span className="text-gray-600">{icon}</span>}
      </div>
      <div className="text-xl font-semibold text-gray-100">{value}</div>
      {sub && <div className="text-xs text-gray-500">{sub}</div>}
      {progress !== undefined && (
        <div className="mt-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all ${progressColor(progress)}`}
            style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
          />
        </div>
      )}
    </div>
  )
}
