import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import type { LlmStatus, LlmModel, RegisteredApp, Agent } from '../types'

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
    queryFn: () => get('/api/llm/models'),
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
    queryFn: () => get('/api/llm/checkpoints'),
    refetchInterval: 30_000,
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

// ── Agents (moltbook) ─────────────────────────────────────────────────────────

export function useAgents() {
  return useQuery<Agent[]>({
    queryKey: ['agents'],
    queryFn: () => get('/api/agents'),
    refetchInterval: 30_000,
  })
}
