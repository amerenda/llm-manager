import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import type { LlmStatus, LlmModel, RegisteredApp, Agent, Runner, RunnerStatus, Profile, ProfileActivation, LibraryModel, SafetyTag, QueueJob, QueueMetrics } from '../types'

// nginx proxies /api to the backend service
async function get<T>(path: string): Promise<T> {
  const r = await fetch(path, { credentials: 'include' })
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`)
  return r.json()
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  const r = await fetch(path, {
    method: 'POST',
    credentials: 'include',
    headers: body ? { 'Content-Type': 'application/json' } : {},
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`)
  return r.json()
}

async function patch<T>(path: string, body?: unknown): Promise<T> {
  const r = await fetch(path, {
    method: 'PATCH',
    credentials: 'include',
    headers: body ? { 'Content-Type': 'application/json' } : {},
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`)
  return r.json()
}

async function put<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(path, {
    method: 'PUT',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`)
  return r.json()
}

async function del<T>(path: string): Promise<T> {
  const r = await fetch(path, { method: 'DELETE', credentials: 'include' })
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`)
  return r.json()
}

// ── Auth ──────────────────────────────────────────────────────────────────────

export interface AuthUser {
  user: string
  admin: boolean
  environment?: string
}

export interface PublicStats {
  gpu: { vram_total_gb: number; vram_used_gb: number; vram_free_gb: number }
  active_models: number
  connected_apps: number
  active_runners: number
}

export function useAuth() {
  return useQuery<AuthUser>({
    queryKey: ['auth'],
    queryFn: () => get('/auth/me'),
    retry: false,
    refetchOnWindowFocus: true,
  })
}

export function usePublicStats() {
  return useQuery<PublicStats>({
    queryKey: ['public-stats'],
    queryFn: () => get('/api/stats'),
    refetchInterval: 10_000,
  })
}

// ── Agent target version ─────────────────────────────────────────────────────

export function useAgentTargetVersion() {
  return useQuery<{ target_version: string }>({
    queryKey: ['agent-target-version'],
    queryFn: () => get('/api/runners/target-version'),
    refetchInterval: 30_000,
  })
}

export function useSetAgentTargetVersion() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (target_version: string) =>
      put<{ ok: boolean }>('/api/runners/target-version', { target_version }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['agent-target-version'] }),
  })
}

// ── Models with safety info (from /api/models) ──────────────────────────────

export interface ModelInfo {
  name: string
  size_gb: number
  vram_estimate_gb: number
  parameter_count: string | null
  quantization: string | null
  safety: string
  categories: string[]
  description: string
  runners: { runner_id: number; hostname: string }[]
  downloaded: boolean
  loaded: boolean
  fits: boolean
  fits_on: { runner: string; vram_total_gb: number }[]
  do_not_evict?: boolean
}

export function useModelList() {
  return useQuery<ModelInfo[]>({
    queryKey: ['model-list'],
    queryFn: () => get('/api/models'),
    refetchInterval: 10_000,
  })
}

export function useUpdateModelSettings() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ model, ...body }: { model: string; categories?: string[]; safety?: string; vram_estimate_gb?: number | null; do_not_evict?: boolean }) =>
      patch<{ model_name: string }>(`/api/models/${encodeURIComponent(model)}/settings`, body),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['model-list'] }),
  })
}

// ── LLM Status ────────────────────────────────────────────────────────────────

export function useLlmStatus() {
  return useQuery<LlmStatus>({
    queryKey: ['llm-status'],
    queryFn: () => get('/api/llm/status'),
    refetchInterval: 5_000,
    retry: 0,
  })
}

// ── Models ────────────────────────────────────────────────────────────────────

export function useLlmModels() {
  return useQuery<LlmModel[]>({
    queryKey: ['llm-models'],
    queryFn: () => get<{ data: LlmModel[] }>('/api/llm/models').then(r => r.data),
    refetchInterval: 5_000,
  })
}

export function usePullModel() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ model, runner_id }: { model: string; runner_id?: number }) =>
      post<{ ok: boolean; op_id: string; message: string }>(`/api/llm/models/pull${runner_id ? `?runner_id=${runner_id}` : ''}`, { model }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['ops'] })
      qc.invalidateQueries({ queryKey: ['llm-models'] })
      qc.invalidateQueries({ queryKey: ['llm-status'] })
      setTimeout(() => {
        qc.invalidateQueries({ queryKey: ['llm-models'] })
        qc.invalidateQueries({ queryKey: ['llm-status'] })
        qc.invalidateQueries({ queryKey: ['library'] })
      }, 5000)
    },
  })
}

export function useSyncModels() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: () =>
      post<{ ok: boolean; pulls: { model: string; target: string }[]; message: string }>('/api/llm/models/sync'),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['llm-models'] })
      qc.invalidateQueries({ queryKey: ['llm-status'] })
      qc.invalidateQueries({ queryKey: ['library'] })
      setTimeout(() => {
        qc.invalidateQueries({ queryKey: ['llm-models'] })
        qc.invalidateQueries({ queryKey: ['llm-status'] })
      }, 10000)
    },
  })
}

export function useDeleteModel() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ model, runner_id }: { model: string; runner_id?: number }) => {
      const params = runner_id ? `&runner_id=${runner_id}` : ''
      return post<{ ok: boolean }>(`/api/llm/models/delete?model=${encodeURIComponent(model)}${params}`)
    },
    onMutate: async ({ model }) => {
      // Optimistic: remove from model-list cache immediately
      await qc.cancelQueries({ queryKey: ['model-list'] })
      const prev = qc.getQueryData<ModelInfo[]>(['model-list'])
      if (prev) {
        qc.setQueryData(['model-list'], prev.filter(m => m.name !== model))
      }
      return { prev }
    },
    onError: (_err, _vars, ctx) => {
      // Restore on failure
      if (ctx?.prev) qc.setQueryData(['model-list'], ctx.prev)
    },
    onSettled: () => {
      qc.invalidateQueries({ queryKey: ['model-list'] })
      qc.invalidateQueries({ queryKey: ['llm-models'] })
      qc.invalidateQueries({ queryKey: ['llm-status'] })
    },
  })
}

export function useUnloadFromVram() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ model, runner_id }: { model: string; runner_id?: number }) => {
      const params = runner_id ? `?runner_id=${runner_id}` : ''
      return post<{ ok: boolean }>(`/api/llm/models/unload${params}`, { model })
    },
    onSettled: () => {
      qc.invalidateQueries({ queryKey: ['llm-status'] })
      qc.invalidateQueries({ queryKey: ['model-list'] })
    },
  })
}

export function useFlushRunnerVram() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (runner_id: number) =>
      post<{ ok: boolean; unloaded: string[]; errors: { model: string; error: string }[]; message: string }>(`/api/llm/runners/${runner_id}/flush`),
    onSettled: () => {
      qc.invalidateQueries({ queryKey: ['llm-status'] })
      qc.invalidateQueries({ queryKey: ['runners'] })
    },
  })
}

export function useRestartOllama() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (runner_id: number) =>
      post<{ ok: boolean; message: string }>(`/api/llm/runners/${runner_id}/restart-ollama`),
    onSettled: () => {
      qc.invalidateQueries({ queryKey: ['llm-status'] })
      qc.invalidateQueries({ queryKey: ['runners'] })
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
    refetchInterval: 10_000,
  })
}

export function useDeleteApp() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (app_id: number) =>
      del<{ ok: boolean; app_id: number }>(`/api/apps/by-id/${app_id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['apps'] }),
  })
}

export function useRegisterApp() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ name, base_url }: { name: string; base_url: string }) =>
      post<{ ok: boolean; api_key: string }>('/api/apps/register', { name, base_url }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['apps'] }),
  })
}

// ── Runners ───────────────────────────────────────────────────────────────────

export function useRunners() {
  return useQuery<Runner[]>({
    queryKey: ['runners'],
    queryFn: () => get('/api/runners'),
    refetchInterval: 5_000,
  })
}

export function useUpdateAppAllowedRunners() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ appId, allowed_runner_ids }: { appId: number; allowed_runner_ids: number[] }) =>
      fetch(`/api/apps/${appId}/allowed-runners`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ allowed_runner_ids }),
      }).then(r => { if (!r.ok) throw new Error('Failed'); return r.json() }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['apps'] }),
  })
}

export function useUpdateRunner() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ runnerId, ...body }: { runnerId: number; enabled?: boolean; auto_update?: boolean; draining?: boolean; pinned_model?: string | null }) =>
      patch<{ ok: boolean }>(`/api/runners/${runnerId}`, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['runners'] })
      qc.invalidateQueries({ queryKey: ['llm-status'] })
    },
  })
}

// ── Ollama runtime settings (per-runner) ──────────────────────────────────

export interface OllamaSettingsResponse {
  settings: Record<string, string>
  allowlist: Record<string, 'int' | 'bool' | 'enum' | 'duration' | 'str'>
  env_file: string
}

export interface OllamaVersionResponse {
  version: string | null
  image_tag: string
  commit: string | null
}

export function useOllamaVersion(runnerId: number | null, enabled = true) {
  return useQuery<OllamaVersionResponse>({
    queryKey: ['ollama-version', runnerId],
    queryFn: () => get(`/api/llm/runners/${runnerId}/ollama-version`),
    enabled: enabled && runnerId != null,
    refetchInterval: false,
  })
}

export function useUpgradeOllama() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ runnerId, tag }: { runnerId: number; tag: string }) =>
      post<{ ok: boolean; image: string; tag: string; commit: string | null; message: string }>(
        `/api/llm/runners/${runnerId}/ollama-upgrade`, { tag }),
    onSuccess: (_d, v) => {
      qc.invalidateQueries({ queryKey: ['ollama-version', v.runnerId] })
      qc.invalidateQueries({ queryKey: ['runners'] })
      qc.invalidateQueries({ queryKey: ['llm-status'] })
    },
  })
}

export function useOllamaSettings(runnerId: number | null, enabled = true) {
  return useQuery<OllamaSettingsResponse>({
    queryKey: ['ollama-settings', runnerId],
    queryFn: () => get(`/api/llm/runners/${runnerId}/ollama-settings`),
    enabled: enabled && runnerId != null,
    refetchInterval: false,
  })
}

export function useUpdateOllamaSettings() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ runnerId, settings }: { runnerId: number; settings: Record<string, string> }) =>
      put<{ ok: boolean; applied: Record<string, string>; message: string }>(
        `/api/llm/runners/${runnerId}/ollama-settings`, { settings }),
    onSuccess: (_d, v) => {
      qc.invalidateQueries({ queryKey: ['ollama-settings', v.runnerId] })
      qc.invalidateQueries({ queryKey: ['runners'] })
      qc.invalidateQueries({ queryKey: ['llm-status'] })
    },
  })
}

export function useRunnerStatus(runnerId: number | null) {
  return useQuery<RunnerStatus>({
    queryKey: ['runner-status', runnerId],
    queryFn: () => get(`/api/llm/status?runner_id=${runnerId}`),
    enabled: runnerId !== null,
    refetchInterval: 5_000,
  })
}

export function useTriggerRunnerUpdate() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ runnerId, target_version }: { runnerId: number; target_version?: string }) =>
      post<{ ok: boolean; message: string }>(`/api/runners/${runnerId}/update`, target_version ? { target_version } : {}),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['runners'] })
    },
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

export function useUpdateAppCategories() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ appId, allowed_categories, excluded_categories, allow_profile_switch }: {
      appId: number
      allow_profile_switch: boolean
      allowed_categories: string[]
      excluded_categories: string[]
    }) =>
      patch<{ ok: boolean }>(`/api/apps/${appId}/permissions`, { allow_profile_switch, allowed_categories, excluded_categories }),
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

export function useUpdateAppExcludedModels() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ appId, excluded_models }: { appId: number; excluded_models: string[] }) =>
      fetch(`/api/apps/${appId}/excluded-models`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ excluded_models }),
      }).then(r => { if (!r.ok) throw new Error('Failed'); return r.json() }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['apps'] }),
  })
}

// ── Model load/unload ────────────────────────────────────────────────────────

export function useLoadModel() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ model, runner_id }: { model: string; runner_id?: number }) =>
      post<{ ok: boolean; op_id: string }>(`/api/llm/models/load${runner_id ? `?runner_id=${runner_id}` : ''}`, { model }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['llm-status'] })
      qc.invalidateQueries({ queryKey: ['llm-models'] })
      setTimeout(() => {
        qc.invalidateQueries({ queryKey: ['llm-status'] })
        qc.invalidateQueries({ queryKey: ['llm-models'] })
      }, 3000)
    },
  })
}

export function useUnloadModel() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ model, runner_id }: { model: string; runner_id?: number }) =>
      post<{ ok: boolean; op_id: string }>(`/api/llm/models/unload${runner_id ? `?runner_id=${runner_id}` : ''}`, { model }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['llm-status'] })
      qc.invalidateQueries({ queryKey: ['llm-models'] })
      setTimeout(() => {
        qc.invalidateQueries({ queryKey: ['llm-status'] })
        qc.invalidateQueries({ queryKey: ['llm-models'] })
      }, 2000)
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

// ── Operations (background pull/sync tracking) ─────────────────────────────

export interface Op {
  op_id?: string
  status: 'running' | 'completed' | 'failed'
  model: string
  type: string
  progress?: string
  error?: string
  target?: string
}

export function useOps(enabled = true) {
  return useQuery<Op[]>({
    queryKey: ['ops'],
    queryFn: () => get<Op[]>('/api/ops'),
    refetchInterval: enabled ? 2_000 : false,
  })
}

export function useDismissOp() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (op_id: string) =>
      del<{ ok: boolean; op_id: string }>(`/api/ops/${encodeURIComponent(op_id)}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['ops'] }),
  })
}

// ── Library updates (remote-digest checks and pulls) ─────────────────────

export function useRefreshRemoteDigests() {
  const qc = useQueryClient()
  return useMutation({
    // User-initiated → force=true bypasses the 1h cache so the click feels live.
    // runner_id optional: when present the refresh only touches tags downloaded
    // on that runner.
    mutationFn: ({ runner_id }: { runner_id?: number } = {}) => {
      const qs = new URLSearchParams({ force: 'true' })
      if (runner_id != null) qs.set('runner_id', String(runner_id))
      return post<{ status: string; checked?: number; errors?: number; skipped?: number; tags_checked?: number; message?: string }>(
        `/api/library/refresh-remote-digests?${qs.toString()}`, {})
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ['library'] }),
  })
}

export function useUpdateOutdatedModels() {
  const qc = useQueryClient()
  return useMutation({
    // Backend auto-refreshes digests first (refresh=true is the default) so a
    // lone click always produces a correct result. runner_id scopes both the
    // refresh and the pulls.
    mutationFn: ({ runner_id }: { runner_id?: number } = {}) => {
      const qs = new URLSearchParams()
      if (runner_id != null) qs.set('runner_id', String(runner_id))
      const q = qs.toString() ? `?${qs.toString()}` : ''
      return post<{ status: string; pulls: Array<{ runner: string; model: string; op_id?: string; error?: string }>; count: number; up_to_date?: number; skipped_no_remote?: number; scope?: string; message?: string }>(
        `/api/library/update-outdated${q}`, {})
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['library'] })
      qc.invalidateQueries({ queryKey: ['ops'] })
    },
  })
}

export function useForceUpdateModel() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ model, runner_id }: { model: string; runner_id?: number }) => {
      const qs = runner_id != null ? `?runner_id=${runner_id}` : ''
      return post<{ ok: boolean; op_id: string; message: string }>(
        `/api/library/models/${encodeURIComponent(model)}/force-update${qs}`, {})
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['library'] })
      qc.invalidateQueries({ queryKey: ['ops'] })
    },
  })
}

// ── Library ──────────────────────────────────────────────────────────────────

export function useLibrary(params: { search?: string; safety?: string; fits?: boolean; downloaded?: boolean; hasPulling?: boolean; runner_id?: number } = {}) {
  const qs = new URLSearchParams()
  if (params.search) qs.set('search', params.search)
  if (params.safety) qs.set('safety', params.safety)
  if (params.fits !== undefined) qs.set('fits', String(params.fits))
  if (params.downloaded !== undefined) qs.set('downloaded', String(params.downloaded))
  if (params.runner_id != null) qs.set('runner_id', String(params.runner_id))
  const query = qs.toString()
  return useQuery<{ models: LibraryModel[]; total: number; cache_age_hours: number; runners: string[] }>({
    queryKey: ['library', query],
    queryFn: () => get(`/api/library${query ? `?${query}` : ''}`),
    refetchInterval: params.hasPulling ? 5_000 : false,
  })
}

// Models downloaded on a runner (or fleet-wide) that aren't in the Ollama
// library catalog — community / user-namespaced tags like MFDoom/..., hf.co/...
export interface CommunityModel {
  name: string
  downloaded_on: string[]
  outdated_on: string[]
  size_bytes: number
  digest: string
}

export function useCommunityModels(runner_id?: number) {
  const qs = new URLSearchParams()
  if (runner_id != null) qs.set('runner_id', String(runner_id))
  const query = qs.toString()
  return useQuery<{ models: CommunityModel[] }>({
    queryKey: ['community-models', query],
    queryFn: () => get(`/api/library/community${query ? `?${query}` : ''}`),
    refetchInterval: 15_000,
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

// ── Cloud models ─────────────────────────────────────────────────────────────

export interface CloudModel {
  id: string
  display_name: string
  provider: string
  enabled: boolean
  max_tokens: number
  temperature: number | null
  config: Record<string, unknown>
}

export interface CloudProviderStatus {
  configured: boolean
  reachable: boolean
  model_count: number
}

export function useCloudModels() {
  return useQuery<CloudModel[]>({
    queryKey: ['cloud-models'],
    queryFn: () => get('/api/cloud/models'),
    refetchInterval: 30_000,
  })
}

export function useCloudStatus() {
  return useQuery<Record<string, CloudProviderStatus>>({
    queryKey: ['cloud-status'],
    queryFn: () => get('/api/cloud/status'),
    refetchInterval: 30_000,
  })
}

export function useUpdateCloudModel() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ modelId, ...data }: { modelId: string; enabled?: boolean; max_tokens?: number; temperature?: number; display_name?: string }) =>
      patch(`/api/cloud/models/${encodeURIComponent(modelId)}`, data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['cloud-models'] }),
  })
}

// ── API keys ─────────────────────────────────────────────────────────────────

export interface StoredApiKey {
  id: number
  provider: string
  user_id: number | null
  key_preview: string
  label: string
  created_at: string
}

export function useCloudKeys() {
  return useQuery<StoredApiKey[]>({
    queryKey: ['cloud-keys'],
    queryFn: () => get('/api/cloud/keys'),
  })
}

export function useStoreCloudKey() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (data: { provider: string; key: string; label?: string }) =>
      post('/api/cloud/keys', data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['cloud-keys'] })
      qc.invalidateQueries({ queryKey: ['cloud-status'] })
      qc.invalidateQueries({ queryKey: ['cloud-models'] })
    },
  })
}

export function useDeleteCloudKey() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (keyId: number) => del(`/api/cloud/keys/${keyId}`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['cloud-keys'] })
      qc.invalidateQueries({ queryKey: ['cloud-status'] })
      qc.invalidateQueries({ queryKey: ['cloud-models'] })
    },
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

// ── Queue ────────────────────────────────────────────────────────────────────

export function useQueueJobs() {
  return useQuery<QueueJob[]>({
    queryKey: ['queue-jobs'],
    queryFn: () => get('/api/queue/jobs'),
    refetchInterval: 3_000,
  })
}

export function useQueueHistory() {
  return useQuery<QueueJob[]>({
    queryKey: ['queue-history'],
    queryFn: () => get('/api/queue/history'),
    refetchInterval: 10_000,
  })
}

export function useQueueMetrics() {
  return useQuery<QueueMetrics>({
    queryKey: ['queue-metrics'],
    queryFn: () => get('/api/queue/metrics'),
    refetchInterval: 5_000,
  })
}

export function useCancelQueueJob() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (jobId: string) => del<{ ok: boolean }>(`/api/queue/jobs/${jobId}`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['queue-jobs'] })
      qc.invalidateQueries({ queryKey: ['queue-metrics'] })
    },
  })
}

export function useSetJobPriority() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ jobId, priority }: { jobId: string; priority: number }) =>
      patch<{ ok: boolean }>(`/api/queue/jobs/${jobId}/priority`, { priority }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['queue-jobs'] }),
  })
}

// ── Model Aliases ─────────────────────────────────────────────────────────────

export interface ModelAlias {
  id: number
  alias_name: string
  base_model: string
  system_prompt: string | null
  parameters: Record<string, unknown>
  description: string
}

export interface ModelRunnerParams {
  model_name: string
  runner_id: number
  hostname: string | null
  system_prompt: string | null
  parameters: Record<string, unknown>
}

export function useModelAliases() {
  return useQuery<ModelAlias[]>({
    queryKey: ['model-aliases'],
    queryFn: () => get('/api/model-aliases'),
    refetchInterval: 30_000,
  })
}

export function useAliasesForModel(baseModel: string) {
  return useQuery<ModelAlias[]>({
    queryKey: ['model-aliases', baseModel],
    queryFn: () => get(`/api/models/${encodeURIComponent(baseModel)}/aliases`),
    enabled: !!baseModel,
  })
}

export function useUpsertAlias() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ alias_name, ...body }: Omit<ModelAlias, 'id'>) =>
      put<ModelAlias>(`/api/model-aliases/${encodeURIComponent(alias_name)}`, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['model-aliases'] })
      qc.invalidateQueries({ queryKey: ['model-list'] })
    },
  })
}

export function useDeleteAlias() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (alias_name: string) =>
      del<{ ok: boolean }>(`/api/model-aliases/${encodeURIComponent(alias_name)}`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['model-aliases'] })
      qc.invalidateQueries({ queryKey: ['model-list'] })
    },
  })
}

export function useRunnerParamsForModel(modelName: string) {
  return useQuery<ModelRunnerParams[]>({
    queryKey: ['runner-params', modelName],
    queryFn: () => get(`/api/models/${encodeURIComponent(modelName)}/runner-params`),
    enabled: !!modelName,
  })
}

export function useUpsertRunnerParams() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ model_name, runner_id, ...body }: ModelRunnerParams) =>
      put<ModelRunnerParams>(
        `/api/models/${encodeURIComponent(model_name)}/runner-params/${runner_id}`,
        body,
      ),
    onSuccess: (_data, vars) => qc.invalidateQueries({ queryKey: ['runner-params', vars.model_name] }),
  })
}

export function usePinModelOnRunner() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ model_name, runner_id, do_not_evict }: { model_name: string; runner_id: number; do_not_evict: boolean }) =>
      patch<{ ok: boolean }>(`/api/models/${encodeURIComponent(model_name)}/runner-params/${runner_id}/pin`, { do_not_evict }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['llm-status'] })
      qc.invalidateQueries({ queryKey: ['runner-params'] })
    },
  })
}

export function useDeleteRunnerParams() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ model_name, runner_id }: { model_name: string; runner_id: number }) =>
      del<{ ok: boolean }>(`/api/models/${encodeURIComponent(model_name)}/runner-params/${runner_id}`),
    onSuccess: (_data, vars) => qc.invalidateQueries({ queryKey: ['runner-params', vars.model_name] }),
  })
}
