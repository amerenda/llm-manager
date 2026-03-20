import { Routes, Route } from 'react-router-dom'
import { NavBar } from './components/NavBar'
import { Dashboard } from './pages/Dashboard'
import { Models } from './pages/Models'
import { Apps } from './pages/Apps'
import { Runners } from './pages/Runners'
import { Profiles } from './pages/Profiles'

export default function App() {
  return (
    <div className="min-h-screen">
      <NavBar />
      <main className="max-w-5xl mx-auto px-4 py-6">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/runners" element={<Runners />} />
          <Route path="/models" element={<Models />} />
          <Route path="/profiles" element={<Profiles />} />
          <Route path="/apps" element={<Apps />} />
        </Routes>
      </main>
    </div>
  )
}
