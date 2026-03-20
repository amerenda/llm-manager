import { useState } from 'react'
import {
  Sliders, Plus, Trash2, Loader2, ChevronDown, ChevronRight,
  Play, Square, AlertCircle, Shield, ShieldOff, Server,
  Layers, Image, RefreshCw,
} from 'lucide-react'
import {
  useProfiles, useProfile, useCreateProfile, useUpdateProfile, useDeleteProfile,
  useAddProfileModel, useDeleteProfileModel,
  useAddProfileImage, useDeleteProfileImage,
  useActivateProfile, useDeactivateProfile,
  useProfileActivations, useRunners, useLlmModels, useLlmStatus,
} from '../hooks/useBackend'
import type { Profile, ProfileModelEntry, ProfileImageEntry } from '../types'

function ProfileSelector({
  profiles,
  selectedId,
  onSelect,
}: {
  profiles: Profile[]
  selectedId: number | null
  onSelect: (id: number) => void
}) {
  const create = useCreateProfile()
  const [creating, setCreating] = useState(false)
  const [newName, setNewName] = useState('')

  async function handleCreate() {
    const name = newName.trim()
    if (!name) return
    try {
      const result = await create.mutateAsync({ name })
      setNewName('')
      setCreating(false)
      onSelect(result.id)
    } catch { /* error shown in UI via mutation state */ }
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <h2 className="text-xs text-gray-500 font-medium uppercase tracking-wide">Profiles</h2>
        <button
          onClick={() => setCreating(!creating)}
          className="p-1 rounded-lg text-gray-500 hover:text-brand-300 hover:bg-gray-800 transition-colors"
        >
          <Plus className="w-4 h-4" />
        </button>
      </div>

      {creating && (
        <div className="flex gap-2">
          <input
            type="text"
            value={newName}
            onChange={e => setNewName(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleCreate()}
            placeholder="Profile name"
            className="flex-1 bg-gray-950 border border-gray-700 rounded-lg px-2 py-1.5 text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-brand-600"
            autoFocus
          />
          <button
            onClick={handleCreate}
            disabled={create.isPending || !newName.trim()}
            className="bg-brand-600 hover:bg-brand-500 disabled:opacity-40 text-white text-xs px-3 py-1.5 rounded-lg"
          >
            {create.isPending ? <Loader2 className="w-3 h-3 animate-spin" /> : 'Add'}
          </button>
        </div>
      )}

      <div className="space-y-1">
        {profiles.map(p => (
          <button
            key={p.id}
            onClick={() => onSelect(p.id)}
            className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors flex items-center justify-between ${
              selectedId === p.id
                ? 'bg-brand-900/40 text-brand-300 border border-brand-800'
                : 'text-gray-300 hover:bg-gray-800 border border-transparent'
            }`}
          >
            <span className="truncate">{p.name}</span>
            <div className="flex items-center gap-1.5">
              {p.unsafe_enabled && <ShieldOff className="w-3 h-3 text-red-400" />}
              {p.is_default && (
                <span className="text-[10px] bg-gray-800 text-gray-500 px-1.5 py-0.5 rounded">default</span>
              )}
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}

function ModelEntryRow({
  entry,
  profileId,
  unsafeEnabled,
}: {
  entry: ProfileModelEntry
  profileId: number
  unsafeEnabled: boolean
}) {
  const deleteEntry = useDeleteProfileModel()

  return (
    <div className="flex items-center gap-3 bg-gray-950 border border-gray-800 rounded-lg px-3 py-2">
      <Layers className="w-3.5 h-3.5 text-gray-600 flex-shrink-0" />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-200 font-medium truncate">{entry.model_safe}</span>
          {entry.count > 1 && (
            <span className="text-[10px] bg-gray-800 text-gray-400 px-1.5 py-0.5 rounded">x{entry.count}</span>
          )}
          {entry.label && (
            <span className="text-[10px] bg-brand-900/40 text-brand-400 px-1.5 py-0.5 rounded flex items-center gap-0.5">
              <Layers className="w-2.5 h-2.5" />{entry.label}
            </span>
          )}
        </div>
        {unsafeEnabled && entry.model_unsafe && (
          <p className="text-xs text-red-400/70 mt-0.5">unsafe: {entry.model_unsafe}</p>
        )}
      </div>
      <button
        onClick={() => deleteEntry.mutate({ profileId, entryId: entry.id })}
        disabled={deleteEntry.isPending}
        className="p-1 rounded text-gray-600 hover:text-red-400 hover:bg-red-900/20 transition-colors"
      >
        <Trash2 className="w-3.5 h-3.5" />
      </button>
    </div>
  )
}

function ImageEntryRow({
  entry,
  profileId,
  unsafeEnabled,
}: {
  entry: ProfileImageEntry
  profileId: number
  unsafeEnabled: boolean
}) {
  const deleteEntry = useDeleteProfileImage()

  return (
    <div className="flex items-center gap-3 bg-gray-950 border border-gray-800 rounded-lg px-3 py-2">
      <Image className="w-3.5 h-3.5 text-gray-600 flex-shrink-0" />
      <div className="flex-1 min-w-0">
        <span className="text-sm text-gray-200 font-medium truncate block">{entry.checkpoint_safe}</span>
        {unsafeEnabled && entry.checkpoint_unsafe && (
          <p className="text-xs text-red-400/70 mt-0.5">unsafe: {entry.checkpoint_unsafe}</p>
        )}
      </div>
      <button
        onClick={() => deleteEntry.mutate({ profileId, entryId: entry.id })}
        disabled={deleteEntry.isPending}
        className="p-1 rounded text-gray-600 hover:text-red-400 hover:bg-red-900/20 transition-colors"
      >
        <Trash2 className="w-3.5 h-3.5" />
      </button>
    </div>
  )
}

function AddModelForm({ profileId, unsafeEnabled }: { profileId: number; unsafeEnabled: boolean }) {
  const addModel = useAddProfileModel()
  const models = useLlmModels()
  const [modelSafe, setModelSafe] = useState('')
  const [modelUnsafe, setModelUnsafe] = useState('')
  const [count, setCount] = useState(1)
  const [label, setLabel] = useState('')
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [parameters, setParameters] = useState<Record<string, unknown>>({})

  const textModels = (models.data ?? []).filter(m => m.type === 'text')

  async function handleAdd() {
    if (!modelSafe.trim()) return
    try {
      await addModel.mutateAsync({
        profileId,
        model_safe: modelSafe.trim(),
        model_unsafe: unsafeEnabled && modelUnsafe.trim() ? modelUnsafe.trim() : undefined,
        count,
        label: label.trim() || undefined,
        parameters: Object.keys(parameters).length > 0 ? parameters : undefined,
      })
      setModelSafe('')
      setModelUnsafe('')
      setCount(1)
      setLabel('')
      setParameters({})
    } catch { /* handled by mutation */ }
  }

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-3 space-y-3">
      <p className="text-xs text-gray-500 font-medium uppercase tracking-wide">Add text model</p>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        <div>
          <label className="block text-xs text-gray-600 mb-1">Model (safe)</label>
          <input
            type="text"
            value={modelSafe}
            onChange={e => setModelSafe(e.target.value)}
            placeholder="e.g. qwen2.5:7b"
            list="available-models"
            className="w-full bg-gray-950 border border-gray-700 rounded-lg px-2 py-1.5 text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-brand-600"
          />
          <datalist id="available-models">
            {textModels.map(m => <option key={m.id} value={m.id} />)}
          </datalist>
        </div>
        {unsafeEnabled && (
          <div>
            <label className="block text-xs text-gray-600 mb-1">Model (unsafe)</label>
            <input
              type="text"
              value={modelUnsafe}
              onChange={e => setModelUnsafe(e.target.value)}
              placeholder="e.g. dolphin-mistral:7b"
              className="w-full bg-gray-950 border border-gray-700 rounded-lg px-2 py-1.5 text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-brand-600"
            />
          </div>
        )}
        <div>
          <label className="block text-xs text-gray-600 mb-1">Instances</label>
          <input
            type="number"
            min={1}
            max={10}
            value={count}
            onChange={e => setCount(parseInt(e.target.value) || 1)}
            className="w-full bg-gray-950 border border-gray-700 rounded-lg px-2 py-1.5 text-sm text-gray-200 focus:outline-none focus:border-brand-600"
          />
        </div>
        <div>
          <label className="block text-xs text-gray-600 mb-1">Label (optional)</label>
          <input
            type="text"
            value={label}
            onChange={e => setLabel(e.target.value)}
            placeholder="e.g. voice-assistant"
            className="w-full bg-gray-950 border border-gray-700 rounded-lg px-2 py-1.5 text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-brand-600"
          />
        </div>
      </div>

      <button
        onClick={() => setShowAdvanced(!showAdvanced)}
        className="flex items-center gap-1 text-xs text-gray-500 hover:text-gray-300 transition-colors"
      >
        <RefreshCw className="w-3 h-3" />
        Advanced parameters
        {showAdvanced ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
      </button>

      {showAdvanced && (
        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="block text-xs text-gray-600 mb-1">Context window</label>
            <input
              type="number"
              value={(parameters.num_ctx as number) || ''}
              onChange={e => setParameters(p => ({ ...p, num_ctx: parseInt(e.target.value) || undefined }))}
              placeholder="4096"
              className="w-full bg-gray-950 border border-gray-700 rounded-lg px-2 py-1.5 text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-brand-600"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-600 mb-1">Temperature</label>
            <input
              type="number"
              step="0.1"
              min="0"
              max="2"
              value={(parameters.temperature as number) || ''}
              onChange={e => setParameters(p => ({ ...p, temperature: parseFloat(e.target.value) || undefined }))}
              placeholder="0.7"
              className="w-full bg-gray-950 border border-gray-700 rounded-lg px-2 py-1.5 text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-brand-600"
            />
          </div>
        </div>
      )}

      <button
        onClick={handleAdd}
        disabled={addModel.isPending || !modelSafe.trim()}
        className="flex items-center gap-1.5 bg-brand-600 hover:bg-brand-500 disabled:opacity-40 text-white text-xs px-3 py-1.5 rounded-lg transition-colors"
      >
        {addModel.isPending ? <Loader2 className="w-3 h-3 animate-spin" /> : <Plus className="w-3 h-3" />}
        Add model
      </button>
    </div>
  )
}

function AddImageForm({ profileId, unsafeEnabled }: { profileId: number; unsafeEnabled: boolean }) {
  const addImage = useAddProfileImage()
  const status = useLlmStatus()
  const [cpSafe, setCpSafe] = useState('')
  const [cpUnsafe, setCpUnsafe] = useState('')

  const checkpoints = status.data?.comfyui_checkpoints ?? []

  async function handleAdd() {
    if (!cpSafe.trim()) return
    try {
      await addImage.mutateAsync({
        profileId,
        checkpoint_safe: cpSafe.trim(),
        checkpoint_unsafe: unsafeEnabled && cpUnsafe.trim() ? cpUnsafe.trim() : undefined,
      })
      setCpSafe('')
      setCpUnsafe('')
    } catch { /* handled by mutation */ }
  }

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-3 space-y-3">
      <p className="text-xs text-gray-500 font-medium uppercase tracking-wide">Add image checkpoint</p>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        <div>
          <label className="block text-xs text-gray-600 mb-1">Checkpoint (safe)</label>
          <select
            value={cpSafe}
            onChange={e => setCpSafe(e.target.value)}
            className="w-full bg-gray-950 border border-gray-700 rounded-lg px-2 py-1.5 text-sm text-gray-200 focus:outline-none focus:border-brand-600"
          >
            <option value="">Select checkpoint…</option>
            {checkpoints.map(cp => <option key={cp} value={cp}>{cp}</option>)}
          </select>
        </div>
        {unsafeEnabled && (
          <div>
            <label className="block text-xs text-gray-600 mb-1">Checkpoint (unsafe)</label>
            <select
              value={cpUnsafe}
              onChange={e => setCpUnsafe(e.target.value)}
              className="w-full bg-gray-950 border border-gray-700 rounded-lg px-2 py-1.5 text-sm text-gray-200 focus:outline-none focus:border-brand-600"
            >
              <option value="">Select checkpoint…</option>
              {checkpoints.map(cp => <option key={cp} value={cp}>{cp}</option>)}
            </select>
          </div>
        )}
      </div>
      <button
        onClick={handleAdd}
        disabled={addImage.isPending || !cpSafe.trim()}
        className="flex items-center gap-1.5 bg-brand-600 hover:bg-brand-500 disabled:opacity-40 text-white text-xs px-3 py-1.5 rounded-lg transition-colors"
      >
        {addImage.isPending ? <Loader2 className="w-3 h-3 animate-spin" /> : <Plus className="w-3 h-3" />}
        Add checkpoint
      </button>
    </div>
  )
}

function ActivationPanel({ profileId, profileName }: { profileId: number; profileName: string }) {
  const runners = useRunners()
  const activations = useProfileActivations()
  const activate = useActivateProfile()
  const deactivate = useDeactivateProfile()
  const [selectedRunner, setSelectedRunner] = useState<number | null>(null)
  const [force, setForce] = useState(false)
  const [result, setResult] = useState<{ ok: boolean; warnings?: string[] } | null>(null)

  const runnerList = runners.data ?? []
  const activeList = activations.data ?? []

  // Find activations for this profile
  const profileActivations = activeList.filter(a => a.profile_id === profileId)

  async function handleActivate() {
    if (!selectedRunner) return
    setResult(null)
    try {
      const res = await activate.mutateAsync({ profileId, runner_id: selectedRunner, force })
      setResult(res)
    } catch (e) {
      setResult({ ok: false, warnings: [(e as Error).message] })
    }
  }

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 space-y-3">
      <div className="flex items-center gap-2">
        <Server className="w-4 h-4 text-brand-400" />
        <h3 className="text-sm font-semibold text-gray-300">Activate on Runner</h3>
      </div>

      {/* Current activations */}
      {profileActivations.length > 0 && (
        <div className="space-y-1">
          {profileActivations.map(a => (
            <div key={a.runner_id} className="flex items-center justify-between bg-gray-950 rounded-lg px-3 py-2">
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${
                  a.activation_status === 'active' ? 'bg-green-400' :
                  a.activation_status === 'activating' ? 'bg-yellow-400 animate-pulse' :
                  'bg-red-400'
                }`} />
                <span className="text-xs text-gray-300">Runner {a.runner_id}</span>
                <span className="text-[10px] text-gray-500">{a.activation_status}</span>
              </div>
              <button
                onClick={() => deactivate.mutate({ profileId, runner_id: a.runner_id })}
                disabled={deactivate.isPending}
                className="text-xs text-gray-500 hover:text-red-400 transition-colors"
              >
                <Square className="w-3 h-3" />
              </button>
            </div>
          ))}
        </div>
      )}

      <div className="flex items-center gap-2">
        <select
          value={selectedRunner ?? ''}
          onChange={e => setSelectedRunner(e.target.value ? parseInt(e.target.value) : null)}
          className="flex-1 bg-gray-950 border border-gray-700 rounded-lg px-2 py-1.5 text-sm text-gray-200 focus:outline-none focus:border-brand-600"
        >
          <option value="">Select runner…</option>
          {runnerList.map(r => (
            <option key={r.id} value={r.id}>
              {r.hostname} ({r.capabilities.gpu_vram_total_bytes
                ? `${(r.capabilities.gpu_vram_free_bytes! / 1e9).toFixed(1)}/${(r.capabilities.gpu_vram_total_bytes / 1e9).toFixed(1)} GB free`
                : 'no GPU info'})
            </option>
          ))}
        </select>
        <label className="flex items-center gap-1 text-xs text-gray-500">
          <input
            type="checkbox"
            checked={force}
            onChange={e => setForce(e.target.checked)}
            className="rounded border-gray-700"
          />
          Force
        </label>
        <button
          onClick={handleActivate}
          disabled={activate.isPending || !selectedRunner}
          className="flex items-center gap-1.5 bg-green-700 hover:bg-green-600 disabled:opacity-40 text-white text-xs px-3 py-1.5 rounded-lg transition-colors"
        >
          {activate.isPending ? <Loader2 className="w-3 h-3 animate-spin" /> : <Play className="w-3 h-3" />}
          Activate
        </button>
      </div>

      {result && (
        <div className={`text-xs p-2 rounded-lg ${result.ok ? 'bg-green-900/30 text-green-400' : 'bg-red-900/30 text-red-400'}`}>
          {result.ok ? 'Profile activated' : 'Activation failed'}
          {result.warnings && result.warnings.length > 0 && (
            <ul className="mt-1 space-y-0.5">
              {result.warnings.map((w, i) => (
                <li key={i} className="flex items-start gap-1">
                  <AlertCircle className="w-3 h-3 mt-0.5 flex-shrink-0" />
                  {w}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}

function ProfileEditor({ profileId }: { profileId: number }) {
  const { data: profile, isLoading } = useProfile(profileId)
  const updateProfile = useUpdateProfile()
  const deleteProfile = useDeleteProfile()

  if (isLoading || !profile) {
    return <div className="text-center text-gray-600 text-sm py-8">Loading profile…</div>
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-gray-200">{profile.name}</h2>
        <div className="flex items-center gap-3">
          {/* Unsafe toggle */}
          <button
            onClick={() => updateProfile.mutate({ id: profile.id, unsafe_enabled: !profile.unsafe_enabled })}
            disabled={updateProfile.isPending}
            className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg transition-colors ${
              profile.unsafe_enabled
                ? 'bg-red-900/40 text-red-400 border border-red-800'
                : 'bg-gray-800 text-gray-400 border border-gray-700 hover:border-gray-600'
            }`}
          >
            {profile.unsafe_enabled ? <ShieldOff className="w-3 h-3" /> : <Shield className="w-3 h-3" />}
            {profile.unsafe_enabled ? 'Unsafe' : 'Safe'}
          </button>

          {/* Delete */}
          {!profile.is_default && (
            <button
              onClick={() => {
                if (confirm(`Delete profile "${profile.name}"?`)) {
                  deleteProfile.mutate(profile.id)
                }
              }}
              disabled={deleteProfile.isPending}
              className="p-1.5 rounded-lg text-gray-600 hover:text-red-400 hover:bg-red-900/20 transition-colors"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Model entries */}
      <section className="space-y-3">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-brand-400" />
          <h3 className="text-sm font-semibold text-gray-300">Text Models</h3>
          <span className="text-[10px] bg-gray-800 text-gray-500 px-1.5 py-0.5 rounded">
            {profile.model_entries.length}
          </span>
        </div>
        {profile.model_entries.map(entry => (
          <ModelEntryRow
            key={entry.id}
            entry={entry}
            profileId={profile.id}
            unsafeEnabled={profile.unsafe_enabled}
          />
        ))}
        <AddModelForm profileId={profile.id} unsafeEnabled={profile.unsafe_enabled} />
      </section>

      <div className="border-t border-gray-800" />

      {/* Image entries */}
      <section className="space-y-3">
        <div className="flex items-center gap-2">
          <Image className="w-4 h-4 text-brand-400" />
          <h3 className="text-sm font-semibold text-gray-300">Image Checkpoints</h3>
          <span className="text-[10px] bg-gray-800 text-gray-500 px-1.5 py-0.5 rounded">
            {profile.image_entries.length}
          </span>
        </div>
        {profile.image_entries.map(entry => (
          <ImageEntryRow
            key={entry.id}
            entry={entry}
            profileId={profile.id}
            unsafeEnabled={profile.unsafe_enabled}
          />
        ))}
        <AddImageForm profileId={profile.id} unsafeEnabled={profile.unsafe_enabled} />
      </section>

      <div className="border-t border-gray-800" />

      {/* Activation */}
      <ActivationPanel profileId={profile.id} profileName={profile.name} />
    </div>
  )
}

export function Profiles() {
  const profiles = useProfiles()
  const [selectedId, setSelectedId] = useState<number | null>(null)

  const profileList = profiles.data ?? []

  // Auto-select default profile on first load
  if (selectedId === null && profileList.length > 0) {
    const defaultProfile = profileList.find(p => p.is_default)
    if (defaultProfile) {
      setSelectedId(defaultProfile.id)
    } else {
      setSelectedId(profileList[0].id)
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2">
        <Sliders className="w-4 h-4 text-brand-400" />
        <h1 className="text-base font-semibold text-gray-200">Profiles</h1>
      </div>

      {profiles.isLoading ? (
        <div className="py-8 text-center text-gray-600 text-sm">Loading profiles…</div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-[220px_1fr] gap-6">
          <ProfileSelector
            profiles={profileList}
            selectedId={selectedId}
            onSelect={setSelectedId}
          />
          <div>
            {selectedId ? (
              <ProfileEditor profileId={selectedId} />
            ) : (
              <div className="text-center text-gray-600 text-sm py-8">
                Select a profile to edit
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
