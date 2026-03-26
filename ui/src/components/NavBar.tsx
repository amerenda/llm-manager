import { NavLink } from 'react-router-dom'
import { LayoutDashboard, Layers, AppWindow, Server, Sliders, LogIn, LogOut } from 'lucide-react'

const navItems = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/runners', label: 'Runners', icon: Server },
  { to: '/models', label: 'Models', icon: Layers },
  { to: '/profiles', label: 'Profiles', icon: Sliders },
  { to: '/apps', label: 'Apps', icon: AppWindow },
]

interface NavBarProps {
  isAdmin: boolean
  user?: string
  environment?: string
}

export function NavBar({ isAdmin, user, environment }: NavBarProps) {
  const isUat = environment === 'uat'
  return (
    <nav className={`border-b ${isUat ? 'bg-yellow-900/30 border-yellow-800/50' : 'bg-gray-900 border-gray-800'}`}>
      <div className="max-w-5xl mx-auto px-4 flex items-center justify-between h-14">
        <div className="flex items-center gap-2">
          <div className={`w-7 h-7 rounded-lg flex items-center justify-center ${isUat ? 'bg-yellow-600' : 'bg-brand-600'}`}>
            <span className="text-white text-xs font-bold">L</span>
          </div>
          <span className="font-semibold text-gray-100 text-sm">LLM Manager</span>
          {isUat && (
            <span className="text-[10px] bg-yellow-800 text-yellow-300 px-2 py-0.5 rounded-full font-bold uppercase tracking-wider">
              UAT
            </span>
          )}
        </div>

        <div className="flex items-center gap-1">
          {isAdmin && navItems.map(({ to, label, icon: Icon }) => (
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

          {isAdmin ? (
            <div className="flex items-center gap-2 ml-2 pl-2 border-l border-gray-700">
              <span className="text-xs text-gray-500 hidden sm:inline">{user}</span>
              <a
                href="/auth/logout"
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm text-gray-400 hover:text-gray-200 hover:bg-gray-800 transition-colors"
              >
                <LogOut className="w-4 h-4" />
                <span className="hidden sm:inline">Logout</span>
              </a>
            </div>
          ) : (
            <a
              href="/auth/login"
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm text-brand-400 hover:text-brand-300 hover:bg-brand-900/50 transition-colors"
            >
              <LogIn className="w-4 h-4" />
              <span>Admin Login</span>
            </a>
          )}
        </div>
      </div>
    </nav>
  )
}
