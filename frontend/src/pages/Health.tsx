import { useEffect } from 'react'
import { Card, Empty, ErrorNote, Icon, Kpi, Loading, Meter, PageHeader, StatusPill } from '../components/ui'
import { number, relativeTime, titleCase } from '../lib/format'
import { useApi } from '../lib/hooks'
import type { EdgeStatus } from '../lib/api'

interface Info {
  environment: string
  host: { platform: string; python: string; processor: string }
  compute: {
    active_device: string
    cuda_available: boolean
    torch_available: boolean
    gpus: { index: number; name: string; total_memory_mb: number }[]
    memory: { used_mb?: number; total_mb?: number; utilisation_percent?: number }
  }
  pipeline: {
    running: number
    total: number
    capacity: number
    cameras: {
      camera_id: string
      name: string
      status: string
      status_detail: string | null
      measured_fps: number
      target_fps: number
      frame_index: number
      inference_ms: number
      track_count: number
      last_frame_at: string | null
      source: Record<string, any> | null
    }[]
  }
  orchestrator: { frames_processed: number; findings_emitted: number; incidents_opened: number }
  persistence: { incidents_written: number; events_written: number; tracks_written: number; errors: number }
  automation: { enabled: boolean; dispatched: number; failed: number; queue_depth: number; running: boolean }
  llm: { provider: string; enabled: boolean }
  vector_store: { backend: string; documents: number }
  websocket_clients: number
  counts: Record<string, number>
}

export default function HealthPage() {
  const { data, loading, error, refresh } = useApi<Info>('/api/system/info', undefined, { pollMs: 5000 })
  const edge = useApi<EdgeStatus>('/api/edge/nodes', undefined, { pollMs: 5000 })

  useEffect(() => {
    document.title = 'System Health · Eagles Eye'
  }, [])

  if (error) return <ErrorNote message={error} onRetry={refresh} />
  if (loading && !data) return <Loading label="Reading system state" />
  if (!data) return <Empty icon="monitor_heart" title="No telemetry" />

  const healthy = data.pipeline.cameras.filter((c) => c.status === 'online').length
  const uptimePct = data.pipeline.total ? (healthy / data.pipeline.total) * 100 : 100

  return (
    <>
      <PageHeader
        title="System Health"
        subtitle="Pipeline, compute, persistence and automation — measured, not estimated"
        actions={
          <button className="btn-ghost btn-xs" onClick={refresh}>
            <Icon name="refresh" className="text-[16px]" />
          </button>
        }
      />

      <section className="grid grid-cols-2 lg:grid-cols-5 gap-space-md">
        <Kpi
          label="Pipeline health"
          value={uptimePct.toFixed(0)}
          unit="%"
          tone={uptimePct === 100 ? 'good' : uptimePct >= 80 ? 'warning' : 'critical'}
          footer={`${healthy}/${data.pipeline.total} cameras online`}
        />
        <Kpi
          label="Capacity"
          value={`${data.pipeline.running}/${data.pipeline.capacity}`}
          footer="concurrent camera workers"
        />
        <Kpi label="Frames analysed" value={number(data.orchestrator.frames_processed)} />
        <Kpi
          label="Persistence errors"
          value={data.persistence.errors}
          tone={data.persistence.errors ? 'critical' : 'good'}
          footer={`${number(data.persistence.events_written)} events written`}
        />
        <Kpi label="Dashboard clients" value={data.websocket_clients} footer="live websocket sessions" />
      </section>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-space-lg items-start">
        <Card className="xl:col-span-2" title="Camera workers" bodyClassName="p-0">
          {data.pipeline.cameras.length === 0 ? (
            <Empty icon="videocam_off" title="No workers running" />
          ) : (
            <div className="table-wrap">
              <table className="tbl">
                <thead>
                  <tr>
                    <th>Camera</th>
                    <th>Status</th>
                    <th>FPS</th>
                    <th>Inference</th>
                    <th>Tracks</th>
                    <th>Frames</th>
                    <th>Last frame</th>
                    <th>Reconnects</th>
                  </tr>
                </thead>
                <tbody>
                  {data.pipeline.cameras.map((camera) => (
                    <tr key={camera.camera_id}>
                      <td className="truncate max-w-[12rem]" title={camera.status_detail ?? ''}>
                        {camera.name}
                      </td>
                      <td><StatusPill status={camera.status} pulse={camera.status === 'online'} /></td>
                      <td>
                        <div className="flex items-center gap-2">
                          <span className="mono tabular-nums w-10">{camera.measured_fps.toFixed(1)}</span>
                          <Meter
                            value={camera.measured_fps}
                            max={Math.max(1, camera.target_fps)}
                            tone={camera.measured_fps >= camera.target_fps * 0.8 ? 'bg-emerald-500' : 'bg-amber-500'}
                          />
                        </div>
                      </td>
                      <td className="mono tabular-nums">{camera.inference_ms.toFixed(0)} ms</td>
                      <td className="mono tabular-nums">{camera.track_count}</td>
                      <td className="mono tabular-nums">{number(camera.frame_index)}</td>
                      <td className="mono text-on-surface-variant">
                        {camera.last_frame_at ? relativeTime(camera.last_frame_at) : '—'}
                      </td>
                      <td className="mono tabular-nums">{camera.source?.reconnects ?? 0}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        <div className="flex flex-col gap-space-lg">
          <Card title="Edge GPU nodes">
            {!edge.data || edge.data.nodes.length === 0 ? (
              <p className="font-body-sm text-body-sm text-on-surface-variant">
                No edge nodes connected. Run <span className="mono">tools/gpu_worker.py</span> on a GPU machine to
                process cameras there. Placement: <span className="mono">{edge.data?.placement ?? '—'}</span>
              </p>
            ) : (
              <ul className="flex flex-col gap-space-sm">
                {edge.data.nodes.map((n) => (
                  <li key={n.node_id} className="flex flex-col gap-0.5 min-w-0">
                    <span className="flex items-center gap-1.5 min-w-0">
                      <span className={`pip ${n.online ? 'bg-emerald-500' : 'bg-amber-500'}`} />
                      <span className="font-headline-sm text-headline-sm truncate">{n.name}</span>
                    </span>
                    <span className="mono text-on-surface-variant truncate">
                      {n.gpu_name} · {n.cameras.length}/{n.max_cameras} cameras ·{' '}
                      {n.online ? `seen ${n.seconds_since_seen.toFixed(0)}s ago` : 'offline'}
                    </span>
                    <span className="mono text-outline">
                      {number(n.incidents_received)} incidents · {number(n.events_received)} findings ·{' '}
                      {number(n.frames_received)} previews
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </Card>

          <Card title="Compute">
            <dl className="grid grid-cols-2 gap-space-md">
              <Stat label="Device" value={data.compute.active_device} />
              <Stat label="CUDA" value={data.compute.cuda_available ? 'available' : 'unavailable'} />
              <Stat label="Torch" value={data.compute.torch_available ? 'installed' : 'not installed'} />
              <Stat label="Environment" value={data.environment} />
            </dl>
            {data.compute.gpus.map((gpu) => (
              <div key={gpu.index} className="mt-space-md">
                <p className="font-headline-sm text-headline-sm">{gpu.name}</p>
                {data.compute.memory.total_mb ? (
                  <>
                    <Meter
                      value={data.compute.memory.used_mb ?? 0}
                      max={data.compute.memory.total_mb}
                      tone={(data.compute.memory.utilisation_percent ?? 0) > 85 ? 'bg-red-600' : 'bg-primary-container'}
                    />
                    <p className="mono text-on-surface-variant mt-1">
                      {data.compute.memory.used_mb} / {data.compute.memory.total_mb} MB
                    </p>
                  </>
                ) : (
                  <p className="mono text-on-surface-variant">
                    {(gpu.total_memory_mb / 1024).toFixed(1)} GB · not in use by this process
                  </p>
                )}
              </div>
            ))}
          </Card>

          <Card title="Subsystems">
            <dl className="flex flex-col gap-space-sm">
              <Row label="Vector store" value={`${data.vector_store.backend} · ${data.vector_store.documents} docs`} ok />
              <Row label="Language model" value={data.llm.enabled ? data.llm.provider : 'not configured'} ok={data.llm.enabled} />
              <Row
                label="Automation"
                value={
                  data.automation.running
                    ? `${data.automation.dispatched} dispatched, ${data.automation.failed} failed`
                    : 'stopped'
                }
                ok={data.automation.running && data.automation.failed === 0}
              />
              <Row
                label="Persistence"
                value={data.persistence.errors ? `${data.persistence.errors} errors` : 'healthy'}
                ok={data.persistence.errors === 0}
              />
            </dl>
          </Card>

          <Card title="Host">
            <dl className="flex flex-col gap-space-sm">
              <Stat label="Platform" value={data.host.platform} />
              <Stat label="Python" value={data.host.python} />
              <Stat label="Processor" value={data.host.processor || '—'} />
            </dl>
          </Card>
        </div>
      </div>
    </>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0">
      <dt className="label">{label}</dt>
      <dd className="mono text-on-surface break-words">{value}</dd>
    </div>
  )
}

function Row({ label, value, ok }: { label: string; value: string; ok: boolean }) {
  return (
    <div className="flex items-center justify-between gap-space-md">
      <span className="font-body-sm text-body-sm">{label}</span>
      <span className="flex items-center gap-1.5 min-w-0">
        <span className={`pip ${ok ? 'bg-emerald-500' : 'bg-amber-500'}`} />
        <span className="mono text-on-surface-variant truncate">{value}</span>
      </span>
    </div>
  )
}
