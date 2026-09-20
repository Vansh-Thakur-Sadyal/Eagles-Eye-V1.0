/**
 * The on-screen GPU switch.
 *
 * Two independent controls, because they govern different machines:
 *
 *  - Server inference: moves every loaded torch model between CUDA and CPU on
 *    the API host, live. Turning it off pushes the NVIDIA models to CPU only.
 *  - Client rendering: whether this browser uses its high-performance WebGPU
 *    adapter for the digital twin and heatmaps. WebGPU needs no permission
 *    prompt, so a dashboard served from a CPU-only VPS still uses the viewer's
 *    discrete GPU for the visual layer.
 *
 * Remote GPU workers are listed here too: that is how server-side inference
 * runs on a machine that is not the VPS.
 */
import { useState } from 'react'
import { useAuth } from '../lib/auth'
import { useGpu, useWebGpu } from '../lib/hooks'
import { Icon, Modal, Toggle, toast } from './ui'

export function GpuToggle({ compact = false }: { compact?: boolean }) {
  const { can } = useAuth()
  const { status, busy, error, setServerGpu, clientGpu, setClientGpu } = useGpu()
  const webgpu = useWebGpu(clientGpu)
  const [open, setOpen] = useState(false)
  const allowed = can('system:gpu')

  const active = status?.active_device ?? 'cpu'
  const onGpu = active.startsWith('cuda')
  const gpuName = status?.gpus?.[0]?.name

  async function toggleServer(next: boolean) {
    try {
      const result = await setServerGpu(next)
      if (result.warning) toast(result.warning, 'info')
      else toast(next ? `Server inference moved to ${result.active_device}` : 'Server inference moved to CPU')
    } catch {
      toast('Could not change the compute device', 'error')
    }
  }

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        title={
          onGpu
            ? `Server inference on ${gpuName ?? active}`
            : 'Server inference on CPU — click to configure acceleration'
        }
        className={`pill transition-colors ${
          onGpu
            ? 'bg-emerald-50 border-emerald-200 text-emerald-800 hover:bg-emerald-100'
            : 'bg-surface-container border-outline-variant text-on-surface-variant hover:bg-surface-container-high'
        }`}
      >
        <Icon name={onGpu ? 'memory' : 'developer_board_off'} className="text-[14px]" />
        {compact ? (onGpu ? 'GPU' : 'CPU') : onGpu ? `GPU · ${gpuName ?? active}` : 'CPU only'}
      </button>

      <Modal open={open} onClose={() => setOpen(false)} title="Compute acceleration" width="max-w-3xl">
        <div className="flex flex-col gap-space-lg">
          {/* ------------------------------------------------ server tier */}
          <section className="card p-space-md flex flex-col gap-space-md">
            <div className="flex items-start justify-between gap-space-lg">
              <div className="min-w-0">
                <h3 className="font-headline-md text-headline-md text-on-surface">Server inference</h3>
                <p className="font-body-sm text-body-sm text-on-surface-variant mt-0.5">
                  Where detection, tracking, re-identification and face matching run. Switching this
                  relocates every loaded model immediately — no restart, and the next frame uses the
                  new device.
                </p>
              </div>
              <Toggle
                checked={Boolean(status?.gpu_enabled)}
                onChange={toggleServer}
                disabled={!allowed}
                busy={busy}
              />
            </div>

            {!allowed ? (
              <p className="font-body-sm text-body-sm text-amber-800 bg-amber-50 border border-amber-200 rounded-lg p-space-sm">
                Your role cannot change the compute device. Ask a commander or administrator.
              </p>
            ) : null}

            {error ? (
              <p className="font-body-sm text-body-sm text-amber-900 bg-amber-50 border border-amber-200 rounded-lg p-space-sm">
                {error}
              </p>
            ) : null}

            <dl className="grid grid-cols-2 sm:grid-cols-4 gap-space-md">
              <Stat label="Active device" value={active} />
              <Stat label="CUDA available" value={status?.cuda_available ? 'Yes' : 'No'} />
              <Stat label="Torch" value={status?.torch_version ?? 'not installed'} />
              <Stat label="Precision" value={status?.half_precision ? 'FP16' : 'FP32'} />
            </dl>

            {status?.gpus?.length ? (
              <div className="flex flex-col gap-space-sm">
                {status.gpus.map((gpu) => (
                  <div
                    key={gpu.index}
                    className="flex items-center justify-between gap-space-md p-space-sm rounded-lg bg-surface border border-outline-variant/40"
                  >
                    <div className="min-w-0">
                      <p className="font-headline-sm text-headline-sm truncate">{gpu.name}</p>
                      <p className="mono text-on-surface-variant">
                        {(gpu.total_memory_mb / 1024).toFixed(1)} GB
                        {gpu.compute_capability ? ` · SM ${gpu.compute_capability}` : ''}
                        {gpu.driver ? ` · driver ${gpu.driver}` : ''}
                      </p>
                    </div>
                    {status.memory?.total_mb ? (
                      <span className="mono text-on-surface-variant shrink-0">
                        {status.memory.used_mb} / {status.memory.total_mb} MB used
                      </span>
                    ) : null}
                  </div>
                ))}
              </div>
            ) : (
              <p className="font-body-sm text-body-sm text-on-surface-variant">
                No NVIDIA GPU was detected on the API host.
              </p>
            )}

            {status && status.gpus.length > 0 && !status.cuda_available ? (
              <p className="font-body-sm text-body-sm text-amber-900 bg-amber-50 border border-amber-200 rounded-lg p-space-sm">
                A GPU is present but CUDA is not available to the API process. Install a CUDA-enabled
                torch build (see docs/INSTALL.md) to enable GPU inference.
              </p>
            ) : null}

            {status?.loaded_models?.length ? (
              <p className="mono text-on-surface-variant">
                Relocatable models: {status.loaded_models.join(', ')}
              </p>
            ) : null}
          </section>

          {/* ------------------------------------------------ client tier */}
          <section className="card p-space-md flex flex-col gap-space-md">
            <div className="flex items-start justify-between gap-space-lg">
              <div className="min-w-0">
                <h3 className="font-headline-md text-headline-md text-on-surface">
                  This browser&rsquo;s GPU
                </h3>
                <p className="font-body-sm text-body-sm text-on-surface-variant mt-0.5">
                  Hardware acceleration for the digital twin, heatmaps and overlays. The
                  high-performance adapter is requested, which selects a discrete GPU without asking
                  permission. Switching this off falls back to software rendering.
                </p>
              </div>
              <Toggle checked={clientGpu} onChange={setClientGpu} />
            </div>
            <dl className="grid grid-cols-2 gap-space-md">
              <Stat label="WebGPU" value={webgpu.available ? 'Active' : 'Unavailable'} />
              <Stat label="Adapter" value={webgpu.adapter ?? webgpu.reason ?? '—'} />
            </dl>
          </section>

          {/* ---------------------------------------------- remote workers */}
          <section className="card p-space-md flex flex-col gap-space-sm">
            <h3 className="font-headline-md text-headline-md text-on-surface">Remote GPU workers</h3>
            <p className="font-body-sm text-body-sm text-on-surface-variant">
              A browser cannot run CUDA models, so a dashboard hosted on a CPU-only VPS reaches a
              discrete GPU by registering a worker: run the agent in <code>tools/gpu_worker.py</code>{' '}
              on the machine with the GPU and it appears here.
            </p>
            {status?.workers?.length ? (
              <div className="flex flex-col gap-1">
                {status.workers.map((worker) => (
                  <div
                    key={worker.node_id}
                    className="flex items-center justify-between gap-space-md p-space-sm rounded-lg bg-surface border border-outline-variant/40"
                  >
                    <div className="min-w-0">
                      <p className="font-headline-sm text-headline-sm truncate">{worker.name}</p>
                      <p className="mono text-on-surface-variant truncate">
                        {worker.gpu_name} · {(worker.total_memory_mb / 1024).toFixed(1)} GB ·{' '}
                        {worker.jobs_completed} jobs
                      </p>
                    </div>
                    <span
                      className={`pill ${
                        worker.online
                          ? 'bg-emerald-50 border-emerald-200 text-emerald-800'
                          : 'bg-surface-container border-outline-variant text-on-surface-variant'
                      }`}
                    >
                      <span className={`pip ${worker.online ? 'bg-emerald-500' : 'bg-outline'}`} />
                      {worker.online ? 'online' : 'stale'}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="mono text-on-surface-variant">No workers registered.</p>
            )}
          </section>
        </div>
      </Modal>
    </>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0">
      <dt className="label">{label}</dt>
      <dd className="mono text-on-surface truncate" title={value}>
        {value}
      </dd>
    </div>
  )
}
