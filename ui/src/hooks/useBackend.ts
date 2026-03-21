import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import type { LlmStatus, LlmModel, RegisteredApp, Agent, Runner, Profile, ProfileActivation, LibraryModel, SafetyTag } from '../types'

// nginx proxies /api to the backend service
async function get<T>(path: string): Promise<T> {
  const r = await fetch(path)
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`)
  return r.json()
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  const r = await fetch(path, {
    method: 'POST',
    headers: body ? { 'Content-Type': 'application/json' } : {},
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`)
  return r.json()
}

async function patch<T>(path: string, body?: unknown): Promise<T> {
  const r = await fetch(path, {
    method: 'PATCH',
    headers: body ? { 'Content-Type': 'application/json' } : {},
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`)
  return r.json()
}

async function del<T>(path: string): Promise<T> {
  const r = await fetch(path, { method: 'DELETE' })
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`)
  return r.json()
}

// ── LLM Status ────────────────────────────────────────────────────────────────

export function useLlmStatus() {
  return useQuery<LlmStatus>({
    queryKey: ['llm-status'],
    queryFn: () => get('/api/llm/status'),
    refetchInterval: 10_000,
    retry: 0,
  })
}

// ── Models ────────────────────────────────────────────────────────────────────

export function useLlmModels() {
  return useQuery<LlmModel[]>({
    queryKey: ['llm-models'],
    queryFn: () => get<{ data: LlmModel[] }>('/api/llm/models').then(r => r.data),
    refetchInterval: 15_000,
  })
}

export function usePullModel() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (model: string) => post<{ ok: boolean }>('/api/llm/models/pull', { model }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['llm-models'] })
      qc.invalidateQueries({ queryKey: ['llm-status'] })
    },
  })
}

export function useDeleteModel() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (model: string) => del<{ ok: boolean }>(`/api/llm/models/${encodeURIComponent(model)}`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['llm-models'] })
      qc.invalidateQueries({ queryKey: ['llm-status'] })
    },
  })
}

// ── ComfyUI ───────────────────────────────────────────────────────────────────

export function useCheckpoints() {
  return useQuery<string[]>({
    queryKey: ['checkpoints'],
    queryFn: () => get<{ checkpoints: string[] }>('/api/llm/checkpoints').then(r => r.checkpoints),
    refetchInterval: 30_000,
  })
}

export function useStartComfyui() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: () => post<{ ok: boolean }>('/api/llm/comfyui/start'),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['llm-status'] })
      qc.invalidateQueries({ queryKey: ['checkpoints'] })
    },
  })
}

export function useStopComfyui() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: () => post<{ ok: boolean }>('/api/llm/comfyui/stop'),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['llm-status'] })
    },
  })
}

export function useSwitchCheckpoint() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (name: string) =>
      post<{ ok: boolean }>('/api/llm/comfyui/checkpoint', { name }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['llm-status'] })
      qc.invalidateQueries({ queryKey: ['checkpoints'] })
    },
  })
}

// ── Apps ──────────────────────────────────────────────────────────────────────

export function useApps() {
  return useQuery<RegisteredApp[]>({
    queryKey: ['apps'],
    queryFn: () => get('/api/apps'),
    refetchInterval: 15_000,
  })
}

export function useRegisterApp() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ name, base_url }: { name: string; base_url: string }) =>
      post<{ ok: boolean; api_key: string }>('/api/apps', { name, base_url }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['apps'] }),
  })
}

// ── Runners ───────────────────────────────────────────────────────────────────

export function useRunners() {
  return useQuery<Runner[]>({
    queryKey: ['runners'],
    queryFn: () => get('/api/runners'),
    refetchInterval: 15_000,
  })
}

// ── App management ───────────────────────────────────────────────────────────

export function useApproveApp() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (appId: number) => post<{ ok: boolean; api_key: string }>(`/api/apps/${appId}/approve`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['apps'] }),
  })
}

export function useUpdateAppPermissions() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ appId, allow_profile_switch }: { appId: number; allow_profile_switch: boolean }) =>
      patch<{ ok: boolean }>(`/api/apps/${appId}/permissions`, { allow_profile_switch }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['apps'] }),
  })
}

export function useUpdateAppAllowedModels() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ appId, allowed_models }: { appId: number; allowed_models: string[] }) =>
      fetch(`/api/apps/${appId}/allowed-models`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ allowed_models }),
      }).then(r => { if (!r.ok) throw new Error('Failed'); return r.json() }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['apps'] }),
  })
}

// ── Model load/unload ────────────────────────────────────────────────────────

export function useLoadModel() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ model, runner_id }: { model: string; runner_id?: number }) =>
      post<{ ok: boolean }>(`/api/llm/models/load${runner_id ? `?runner_id=${runner_id}` : ''}`, { model }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['llm-status'] })
      qc.invalidateQueries({ queryKey: ['llm-models'] })
    },
  })
}

export function useUnloadModel() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ model, runner_id }: { model: string; runner_id?: number }) =>
      post<{ ok: boolean }>(`/api/llm/models/unload${runner_id ? `?runner_id=${runner_id}` : ''}`, { model }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['llm-status'] })
      qc.invalidateQueries({ queryKey: ['llm-models'] })
    },
  })
}

// ── Profiles ─────────────────────────────────────────────────────────────────

export function useProfiles() {
  return useQuery<Profile[]>({
    queryKey: ['profiles'],
    queryFn: () => get('/api/profiles'),
    refetchInterval: 30_000,
  })
}

export function useProfile(id: number) {
  return useQuery<Profile>({
    queryKey: ['profiles', id],
    queryFn: () => get(`/api/profiles/${id}`),
    enabled: id > 0,
  })
}

export function useCreateProfile() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (data: { name: string; unsafe_enabled?: boolean }) =>
      post<{ ok: boolean; id: number }>('/api/profiles', data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['profiles'] }),
  })
}

export function useUpdateProfile() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ id, ...data }: { id: number; name?: string; unsafe_enabled?: boolean }) =>
      patch<{ ok: boolean }>(`/api/profiles/${id}`, data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['profiles'] }),
  })
}

export function useDeleteProfile() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: number) => del<{ ok: boolean }>(`/api/profiles/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['profiles'] }),
  })
}

export function useAddProfileModel() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ profileId, ...data }: {
      profileId: number; model_safe: string; model_unsafe?: string;
      count?: number; label?: string; parameters?: Record<string, unknown>
    }) => post<{ ok: boolean; id: number }>(`/api/profiles/${profileId}/models`, data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['profiles'] }),
  })
}

export function useUpdateProfileModel() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ profileId, entryId, ...data }: {
      profileId: number; entryId: number; model_safe: string; model_unsafe?: string;
      count?: number; label?: string; parameters?: Record<string, unknown>
    }) => patch<{ ok: boolean }>(`/api/profiles/${profileId}/models/${entryId}`, data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['profiles'] }),
  })
}

export function useDeleteProfileModel() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ profileId, entryId }: { profileId: number; entryId: number }) =>
      del<{ ok: boolean }>(`/api/profiles/${profileId}/models/${entryId}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['profiles'] }),
  })
}

export function useAddProfileImage() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ profileId, ...data }: {
      profileId: number; checkpoint_safe: string; checkpoint_unsafe?: string;
      label?: string; parameters?: Record<string, unknown>
    }) => post<{ ok: boolean; id: number }>(`/api/profiles/${profileId}/images`, data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['profiles'] }),
  })
}

export function useDeleteProfileImage() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ profileId, entryId }: { profileId: number; entryId: number }) =>
      del<{ ok: boolean }>(`/api/profiles/${profileId}/images/${entryId}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['profiles'] }),
  })
}

export function useActivateProfile() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ profileId, runner_id, force }: { profileId: number; runner_id: number; force?: boolean }) =>
      post<{ ok: boolean; warnings: string[] }>(`/api/profiles/${profileId}/activate`, { runner_id, force }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['profiles'] })
      qc.invalidateQueries({ queryKey: ['llm-status'] })
      qc.invalidateQueries({ queryKey: ['profile-activations'] })
    },
  })
}

export function useDeactivateProfile() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ profileId, runner_id }: { profileId: number; runner_id: number }) =>
      post<{ ok: boolean }>(`/api/profiles/${profileId}/deactivate?runner_id=${runner_id}`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['profiles'] })
      qc.invalidateQueries({ queryKey: ['profile-activations'] })
    },
  })
}

export function useProfileActivations() {
  return useQuery<ProfileActivation[]>({
    queryKey: ['profile-activations'],
    queryFn: () => get('/api/profiles/activations'),
    refetchInterval: 15_000,
  })
}

// ── Library ──────────────────────────────────────────────────────────────────

export function useLibrary(params: { search?: string; safety?: string; fits?: boolean; downloaded?: boolean } = {}) {
  const qs = new URLSearchParams()
  if (params.search) qs.set('search', params.search)
  if (params.safety) qs.set('safety', params.safety)
  if (params.fits !== undefined) qs.set('fits', String(params.fits))
  if (params.downloaded !== undefined) qs.set('downloaded', String(params.downloaded))
  const query = qs.toString()
  return useQuery<{ models: LibraryModel[]; total: number; cache_age_hours: number; runners: string[] }>({
    queryKey: ['library', query],
    queryFn: () => get(`/api/library${query ? `?${query}` : ''}`),
  })
}

export function useRefreshLibrary() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: () => post<{ status: string; model_count?: number }>('/api/library/refresh'),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['library'] }),
  })
}

export function useSafetyTags() {
  return useQuery<SafetyTag[]>({
    queryKey: ['safety-tags'],
    queryFn: () => get('/api/safety-tags'),
  })
}

// ── Agents (moltbook) ─────────────────────────────────────────────────────────

export function useAgents() {
  return useQuery<Agent[]>({
    queryKey: ['agents'],
    queryFn: () => get('/api/agents'),
    refetchInterval: 30_000,
  })
}
