import { useState } from 'react'
import { Download, Trash2, Loader2, CheckCircle2, AlertCircle, Image, Layers } from 'lucide-react'
import { useLlmModels, usePullModel, useDeleteModel, useCheckpoints, useSwitchCheckpoint, useLlmStatus } from '../hooks/useBackend'
import type { LlmModel } from '../types'

function stripExt(filename: string): string {
  return filename.replace(/\.[^.]+$/, '')
}

function TextModelsSection() {
  const models = useLlmModels()
  const pull = usePullModel()
  const del = useDeleteModel()
  const [pullInput, setPullInput] = useState('')
  const [pullMsg, setPullMsg] = useState<{ type: 'ok' | 'err'; text: string } | null>(null)

  const textModels = (models.data ?? []).filter((m: LlmModel) => m.type === 'text')

  async function handlePull() {
    const name = pullInput.trim()
    if (!name) return
    setPullMsg(null)
    try {
      const result = await pull.mutateAsync(name)
      if (result.ok) {
        setPullMsg({ type: 'ok', text: `Pulled ${name}` })
        setPullInput('')
      } else {
        setPullMsg({ type: 'err', text: 'Pull returned not ok' })
      }
    } catch (e) {
      setPullMsg({ type: 'err', text: (e as Error).message })
    }
  }

  return (
    <section className="space-y-4">
      <div className="flex items-center gap-2">
        <Layers className="w-4 h-4 text-brand-400" />
        <h2 className="text-base font-semibold text-gray-200">Text Models (Ollama)</h2>
      </div>

      {/* Pull new model */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
        <p className="text-xs text-gray-500 mb-2 font-medium uppercase tracking-wide">Pull new model</p>
        <div className="flex gap-2">
          <input
            type="text"
            value={pullInput}
            onChange={e => setPullInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handlePull()}
            placeholder="e.g. qwen2.5:7b"
            className="flex-1 bg-gray-950 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-brand-600"
          />
          <button
            onClick={handlePull}
            disabled={pull.isPending || !pullInput.trim()}
            className="flex items-center gap-2 bg-brand-600 hover:bg-brand-500 disabled:opacity-40 text-white text-sm font-medium px-4 py-2 rounded-lg transition-colors"
          >
            {pull.isPending
              ? <Loader2 className="w-4 h-4 animate-spin" />
              : <Download className="w-4 h-4" />}
            Pull
          </button>
        </div>
        {pullMsg && (
          <div className={`mt-2 flex items-center gap-1.5 text-xs ${pullMsg.type === 'ok' ? 'text-green-400' : 'text-red-400'}`}>
            {pullMsg.type === 'ok'
              ? <CheckCircle2 className="w-3.5 h-3.5" />
              : <AlertCircle className="w-3.5 h-3.5" />}
            {pullMsg.text}
          </div>
        )}
      </div>

      {/* Model table */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
        {models.isLoading ? (
          <div className="py-8 text-center text-gray-600 text-sm">Loading models…</div>
        ) : textModels.length === 0 ? (
          <div className="py-8 text-center text-gray-600 text-sm">No text models found</div>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800 text-xs text-gray-500 uppercase tracking-wide">
                <th className="text-left px-4 py-3 font-medium">Model</th>
                <th className="text-left px-4 py-3 font-medium">Size</th>
                <th className="text-left px-4 py-3 font-medium">Status</th>
                <th className="px-4 py-3" />
              </tr>
            </thead>
            <tbody>
              {textModels.map((m: LlmModel) => (
                <tr key={m.id} className="border-b border-gray-800 last:border-0 hover:bg-gray-800/40 transition-colors">
                  <td className="px-4 py-3 text-gray-200 font-medium truncate max-w-xs">{m.name}</td>
                  <td className="px-4 py-3 text-gray-400 tabular-nums">{m.size_gb.toFixed(1)} GB</td>
                  <td className="px-4 py-3">
                    {m.is_loaded ? (
                      <span className="inline-flex items-center gap-1 bg-green-900/40 text-green-400 text-xs px-2 py-0.5 rounded-full">
                        <span className="w-1.5 h-1.5 rounded-full bg-green-400" />
                        Loaded
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 bg-gray-800 text-gray-500 text-xs px-2 py-0.5 rounded-full">
                        <span className="w-1.5 h-1.5 rounded-full bg-gray-600" />
                        Available
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-right">
                    <div className="flex items-center justify-end gap-2">
                      {!m.is_loaded && (
                        <button
                          onClick={() => pull.mutate(m.name)}
                          disabled={pull.isPending}
                          title="Load model"
                          className="flex items-center gap-1 text-xs bg-brand-900/50 hover:bg-brand-800/50 text-brand-300 px-2.5 py-1 rounded-lg transition-colors disabled:opacity-40"
                        >
                          {pull.isPending ? <Loader2 className="w-3 h-3 animate-spin" /> : <Download className="w-3 h-3" />}
                          Load
                        </button>
                      )}
                      <button
                        onClick={() => del.mutate(m.name)}
                        disabled={del.isPending}
                        title="Delete model"
                        className="p-1.5 rounded-lg bg-gray-800 hover:bg-red-900/50 hover:text-red-400 text-gray-500 transition-colors"
                      >
                        {del.isPending ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Trash2 className="w-3.5 h-3.5" />}
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </section>
  )
}

function ImageModelsSection() {
  const status = useLlmStatus()
  const checkpoints = useCheckpoints()
  const switchCp = useSwitchCheckpoint()
  const [switching, setSwitching] = useState<string | null>(null)

  const comfyRunning = status.data?.comfyui_running ?? false
  const activeCheckpoint = status.data?.comfyui_active_checkpoint ?? null
  const cpList = checkpoints.data ?? []

  async function handleSwitch(name: string) {
    setSwitching(name)
    try {
      await switchCp.mutateAsync(name)
    } finally {
      setSwitching(null)
    }
  }

  return (
    <section className="space-y-4">
      <div className="flex items-center gap-2">
        <Image className="w-4 h-4 text-brand-400" />
        <h2 className="text-base font-semibold text-gray-200">Image Models (ComfyUI)</h2>
        <span className={`ml-1 text-xs px-2 py-0.5 rounded-full ${
          comfyRunning
            ? 'bg-green-900/40 text-green-400'
            : 'bg-gray-800 text-gray-500'
        }`}>
          {comfyRunning ? 'Running' : 'Stopped'}
        </span>
      </div>

      {!comfyRunning && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6 text-center">
          <Image className="w-8 h-8 text-gray-700 mx-auto mb-2" />
          <p className="text-sm text-gray-500">ComfyUI is not running</p>
          <p className="text-xs text-gray-600 mt-1">Start ComfyUI to manage image checkpoints</p>
        </div>
      )}

      {comfyRunning && (
        <>
          {checkpoints.isLoading ? (
            <div className="py-8 text-center text-gray-600 text-sm">Loading checkpoints…</div>
          ) : cpList.length === 0 ? (
            <div className="bg-gray-900 border border-gray-800 rounded-xl py-8 text-center text-gray-600 text-sm">
              No checkpoints found
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {cpList.map(cp => {
                const isActive = cp === activeCheckpoint
                const isLoading = switching === cp
                return (
                  <button
                    key={cp}
                    onClick={() => !isActive && handleSwitch(cp)}
                    disabled={isActive || isLoading}
                    className={`text-left bg-gray-900 border rounded-xl p-4 transition-all ${
                      isActive
                        ? 'border-brand-600 bg-brand-900/20 cursor-default'
                        : 'border-gray-800 hover:border-gray-700 hover:bg-gray-800/50 cursor-pointer'
                    }`}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0">
                        <p className={`text-sm font-medium truncate ${isActive ? 'text-brand-300' : 'text-gray-200'}`}>
                          {stripExt(cp)}
                        </p>
                        <p className="text-xs text-gray-600 truncate mt-0.5">{cp}</p>
                      </div>
                      {isLoading && <Loader2 className="w-4 h-4 text-brand-400 animate-spin flex-shrink-0 mt-0.5" />}
                      {isActive && !isLoading && (
                        <span className="text-xs bg-brand-900 text-brand-300 px-1.5 py-0.5 rounded flex-shrink-0">
                          Active
                        </span>
                      )}
                    </div>
                  </button>
                )
              })}
            </div>
          )}
        </>
      )}
    </section>
  )
}

export function Models() {
  return (
    <div className="space-y-8">
      <TextModelsSection />
      <div className="border-t border-gray-800" />
      <ImageModelsSection />
    </div>
  )
}
