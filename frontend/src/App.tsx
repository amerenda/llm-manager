import { Routes, Route } from 'react-router-dom'
import { NavBar } from './components/NavBar'
import { Dashboard } from './pages/Dashboard'
import { PublicDashboard } from './pages/PublicDashboard'
import { Models } from './pages/Models'
import { Apps } from './pages/Apps'
import { Runners } from './pages/Runners'
import { Profiles } from './pages/Profiles'
import { Queue } from './pages/Queue'
import { useAuth } from './hooks/useBackend'

export default function App() {
  const auth = useAuth()
  const isAdmin = auth.data?.admin === true
  const isLoading = auth.isLoading

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p className="text-gray-500 text-sm">Loading…</p>
      </div>
    )
  }

  return (
    <div className="min-h-screen">
      <NavBar isAdmin={isAdmin} user={auth.data?.user} environment={auth.data?.environment} />
      <main className="max-w-5xl mx-auto px-4 py-6">
        {isAdmin ? (
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/runners" element={<Runners />} />
            <Route path="/models" element={<Models />} />
            <Route path="/profiles" element={<Profiles />} />
            <Route path="/apps" element={<Apps />} />
            <Route path="/queue" element={<Queue />} />
          </Routes>
        ) : (
          <PublicDashboard />
        )}
      </main>
    </div>
  )
}
