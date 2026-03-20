import { NavLink } from 'react-router-dom'
import { LayoutDashboard, Layers, AppWindow, Server } from 'lucide-react'

const navItems = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/runners', label: 'Runners', icon: Server },
  { to: '/models', label: 'Models', icon: Layers },
  { to: '/apps', label: 'Apps', icon: AppWindow },
]

export function NavBar() {
  return (
    <nav className="bg-gray-900 border-b border-gray-800">
      <div className="max-w-5xl mx-auto px-4 flex items-center justify-between h-14">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-brand-600 flex items-center justify-center">
            <span className="text-white text-xs font-bold">L</span>
          </div>
          <span className="font-semibold text-gray-100 text-sm">LLM Manager</span>
        </div>
        <div className="flex gap-1">
          {navItems.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm transition-colors ${
                  isActive
                    ? 'bg-brand-900 text-brand-300'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'
                }`
              }
            >
              <Icon className="w-4 h-4" />
              <span className="hidden sm:inline">{label}</span>
            </NavLink>
          ))}
        </div>
      </div>
    </nav>
  )
}
