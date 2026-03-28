interface StatusDotProps {
  online: boolean
  className?: string
}

export function StatusDot({ online, className = '' }: StatusDotProps) {
  return (
    <span
      className={`inline-block w-2 h-2 rounded-full flex-shrink-0 ${online ? 'bg-green-400' : 'bg-gray-600'} ${className}`}
      title={online ? 'Online' : 'Offline'}
    />
  )
}
