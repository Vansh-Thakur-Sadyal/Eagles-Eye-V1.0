/** Command Center — the live operational picture. */
import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { Card, Empty, ErrorNote, Icon, Kpi, Loading, Meter, PageHeader, SeverityPill, StatusPill } from '../components/ui'
import type { Camera, CrowdRow, Incident, Paged } from '../lib/api'
import { streamUrl } from '../lib/api'
import { behaviorIcon, behaviorLabel, densityColor, number, relativeTime, riskBand, titleCase } from '../lib/format'
import { useApi, useLiveEvents } from '../lib/hooks'

interface Summary {
  window_hours: number
  open: number
  opened_in_window: number
  resolved_in_window: number
  by_severity: Record<string, number>
  by_event_type: Record<string, number>
  average_open_risk: number
  mean_time_to_acknowledge_seconds: number | null
}

interface SystemInfo {
  counts: { cameras_total: number; cameras_online: number; cameras_running: number; incidents_open: number; tracks_active: number }
  compute: { active_device: string; gpu_enabled: boolean }
  orchestrator: { frames_processed: number; findings_emitted: number }
  pipeline: { running: number; total: number }
}

export default function CommandCenterPage() {
  const { data: summary, refresh: refreshSummary, error: summaryError } = useApi<Summary>(
    '/api/incidents/summary', { since_hours: 24 }, { pollMs: 20000 },
  )
  const { data: info, refresh: refreshInfo } = useApi<SystemInfo>('/api/system/info', undefined, { pollMs: 10000 })
  const { data: incidents, refresh: refreshIncidents } = useApi<Paged<Incident>>(
    '/api/incidents', { open_only: true, limit: 12 }, { pollMs: 15000 },
  )
  const { data: cameras } = useApi<Paged<Camera>>('/api/cameras', { limit: 200 }, { pollMs: 15000 })
  const { data: crowd } = useApi<{ items: CrowdRow[]; total_people: number }>(
    '/api/intel/crowd/live', undefined, { pollMs: 8000 },
  )
  const { data: tracks } = useApi<{ total_tracks: number }>('/api/intel/tracks/live', undefined, { pollMs: 8000 })

  const [feed, setFeed] = useState<{ id: string; behavior: string; camera: string; confidence: number; at: string; explanation: string }[]>([])

  useLiveEvents(['finding', 'incident'], (event) => {
    if (event.topic === 'finding') {
      setFeed((prev) =>
        [
          {
            id: `${event.at}-${event.payload.behavior}-${event.payload.track_id ?? ''}`,
            behavior: String(event.payload.behavior),
            camera: String(event.payload.camera_name ?? event.payload.camera_id ?? ''),
            confidence: Number(event.payload.confidence ?? 0),
            at: event.at,
            explanation: String(event.payload.explanation ?? ''),
          },
          ...prev,
        ].slice(0, 40),
      )
    } else {
      refreshIncidents()
      refreshSummary()
      refreshInfo()
    }
  })

  useEffect(() => {
    document.title = 'Command Center · Eagles Eye'
  }, [])

  const runningCameras = useMemo(
    () => (cameras?.items ?? []).filter((c) => c.running).slice(0, 4),
    [cameras],
  )

  const bySeverity = summary?.by_severity ?? {}
  const critical = (bySeverity.critical ?? 0) + (bySeverity.high ?? 0)

  if (summaryError) return <ErrorNote message={summaryError} onRetry={refreshSummary} />

  return (
    <>
      <PageHeader
        title="Command Center"
        subtitle="Live operational picture across every connected camera"
        actions={
          <Link to="/cameras" className="btn-secondary">
            <Icon name="videocam" className="text-[16px]" />
            Cameras
          </Link>
        }
      />

      {/* -------------------------------------------------------------- KPIs */}
      <section className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-space-md">
        <Kpi
          label="Open incidents"
          value={summary ? String(summary.open).padStart(2, '0') : '—'}
          tone={critical > 0 ? 'critical' : 'default'}
          footer={
            summary
              ? `${bySeverity.critical ?? 0} critical · ${bySeverity.high ?? 0} high · ${bySeverity.medium ?? 0} medium`
              : undefined
          }
        />
        <Kpi
          label="Cameras online"
          value={info?.counts.cameras_online ?? '—'}
          unit={info ? `/ ${info.counts.cameras_total}` : undefined}
          tone={info && info.counts.cameras_online < info.counts.cameras_total ? 'warning' : 'good'}
          footer={info ? `${info.counts.cameras_running} pipelines running` : undefined}
        />
        <Kpi
          label="People tracked"
          value={number(tracks?.total_tracks ?? 0)}
          footer={info ? `${number(info.orchestrator.frames_processed)} frames analysed` : undefined}
        />
        <Kpi
          label="In view now"
          value={number(crowd?.total_people ?? 0)}
          footer={
            crowd?.items?.length
              ? `${crowd.items.filter((c) => c.risk === 'high' || c.risk === 'critical').length} zones elevated`
              : 'no crowd telemetry yet'
          }
        />
        <Kpi
          label="Average open risk"
          value={summary ? summary.average_open_risk.toFixed(0) : '—'}
          unit="%"
          tone={summary && summary.average_open_risk >= 60 ? 'warning' : 'default'}
          footer={summary ? `${summary.opened_in_window} opened in 24h` : undefined}
        />
        <Kpi
          label="Compute"
          value={info?.compute.active_device.startsWith('cuda') ? 'GPU' : 'CPU'}
          tone={info?.compute.active_device.startsWith('cuda') ? 'good' : 'default'}
          footer={info?.compute.active_device}
        />
      </section>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-space-lg items-start">
        {/* -------------------------------------------------------- incidents */}
        <Card
          className="xl:col-span-2"
          title="Active incidents"
          bodyClassName="p-0"
          actions={
            <Link to="/incidents" className="btn-ghost btn-xs">
              View all
              <Icon name="chevron_right" className="text-[14px]" />
            </Link>
          }
        >
          {!incidents ? (
            <Loading />
          ) : incidents.items.length === 0 ? (
            <Empty
              icon="check_circle"
              title="No open incidents"
              hint="Agents are running. Anything they raise will appear here immediately."
            />
          ) : (
            <div className="table-wrap">
              <table className="tbl">
                <thead>
                  <tr>
                    <th>Incident</th>
                    <th>Camera</th>
                    <th>Risk</th>
                    <th>Severity</th>
                    <th>Status</th>
                    <th>Updated</th>
                  </tr>
                </thead>
                <tbody>
                  {incidents.items.map((incident) => (
                    <tr key={incident.id}>
                      <td className="max-w-xs">
                        <Link to={`/incidents/${incident.id}`} className="flex items-center gap-2 min-w-0 group">
                          <Icon
                            name={behaviorIcon(incident.event_type)}
                            className="text-[16px] text-on-surface-variant shrink-0"
                          />
                          <span className="truncate group-hover:text-primary-container">{incident.title}</span>
                        </Link>
                      </td>
                      <td className="mono text-on-surface-variant truncate max-w-[10rem]">
                        {String((incident.meta as any)?.camera_name ?? incident.camera_id ?? '—')}
                      </td>
                      <td className="w-28">
                        <div className="flex items-center gap-2">
                          <span className="mono font-semibold tabular-nums w-8">
                            {incident.risk_score.toFixed(0)}
                          </span>
                          <Meter
                            value={incident.risk_score}
                            tone={
                              riskBand(incident.risk_score) === 'critical'
                                ? 'bg-red-600'
                                : riskBand(incident.risk_score) === 'high'
                                  ? 'bg-amber-500'
                                  : 'bg-primary-container'
                            }
                          />
                        </div>
                      </td>
                      <td><SeverityPill severity={incident.severity} /></td>
                      <td className="mono text-on-surface-variant">{titleCase(incident.status)}</td>
                      <td className="mono text-on-surface-variant whitespace-nowrap">
                        {relativeTime(incident.last_update_at)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        {/* ------------------------------------------------------- live feed */}
        <Card title="Agent activity" bodyClassName="p-0" className="xl:row-span-2">
          <div className="max-h-[520px] overflow-y-auto divide-y divide-outline-variant/25">
            {feed.length === 0 ? (
              <Empty
                icon="sensors"
                title="Waiting for observations"
                hint="Start a camera and the agents' findings stream here in real time."
              />
            ) : (
              feed.map((item) => (
                <div key={item.id} className="p-space-md flex gap-space-sm">
                  <Icon
                    name={behaviorIcon(item.behavior)}
                    className="text-[18px] text-primary-container shrink-0 mt-0.5"
                  />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-headline-sm text-headline-sm truncate">
                        {behaviorLabel(item.behavior)}
                      </span>
                      <span className="mono text-on-surface-variant shrink-0">
                        {Math.round(item.confidence * 100)}%
                      </span>
                    </div>
                    <p className="font-body-sm text-body-sm text-on-surface-variant line-clamp-2">
                      {item.explanation}
                    </p>
                    <p className="mono text-outline mt-0.5">
                      {item.camera} · {relativeTime(item.at)}
                    </p>
                  </div>
                </div>
              ))
            )}
          </div>
        </Card>

        {/* ---------------------------------------------------------- feeds */}
        <Card
          className="xl:col-span-2"
          title="Live feeds"
          actions={
            <Link to="/cameras" className="btn-ghost btn-xs">
              Manage
            </Link>
          }
        >
          {runningCameras.length === 0 ? (
            <Empty
              icon="videocam_off"
              title="No camera is running"
              hint="Attach a webcam, an ESP32-CAM or an RTSP stream, then press Start."
              action={
                <Link to="/cameras" className="btn-primary">
                  <Icon name="add" className="text-[16px]" />
                  Attach a camera
                </Link>
              }
            />
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-space-md">
              {runningCameras.map((camera) => (
                <figure key={camera.id} className="min-w-0">
                  <div className="relative rounded-lg overflow-hidden border border-outline-variant/60 bg-slate-900 aspect-video">
                    <img
                      src={streamUrl(camera.id)}
                      alt={`${camera.name} live view`}
                      className="w-full h-full object-cover"
                    />
                    <span className="absolute top-2 left-2 pill bg-black/55 border-white/20 text-white">
                      <span className="pip bg-red-500 animate-pulse" />
                      live
                    </span>
                  </div>
                  <figcaption className="flex items-center justify-between gap-2 mt-1.5">
                    <span className="font-headline-sm text-headline-sm truncate">{camera.name}</span>
                    <span className="mono text-on-surface-variant shrink-0">
                      {camera.runtime?.measured_fps?.toFixed(1) ?? '—'} fps ·{' '}
                      {camera.runtime?.track_count ?? 0} tracks
                    </span>
                  </figcaption>
                </figure>
              ))}
            </div>
          )}
        </Card>
      </div>

      {/* ------------------------------------------------------ crowd + types */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-space-lg items-start">
        <Card title="Crowd by camera" bodyClassName="p-0">
          {!crowd || crowd.items.length === 0 ? (
            <Empty icon="groups" title="No crowd telemetry" hint="Telemetry appears once a camera is running." />
          ) : (
            <div className="table-wrap">
              <table className="tbl">
                <thead>
                  <tr>
                    <th>Camera</th>
                    <th>People</th>
                    <th>Density</th>
                    <th>Counter-flow</th>
                    <th>Risk</th>
                  </tr>
                </thead>
                <tbody>
                  {crowd.items.map((row) => (
                    <tr key={row.camera_id}>
                      <td className="truncate max-w-[12rem]">{row.camera_name ?? row.camera_id}</td>
                      <td className="mono tabular-nums">{row.count}</td>
                      <td>
                        <div className="flex items-center gap-2">
                          <span className={`pip ${densityColor(row.density_band)}`} />
                          <span className="mono">{row.density_band}</span>
                        </div>
                      </td>
                      <td className="mono tabular-nums">{Math.round(row.counter_flow_ratio * 100)}%</td>
                      <td><SeverityPill severity={row.risk === 'low' ? 'low' : row.risk} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        <Card title="Incident types · last 24 hours">
          {!summary || Object.keys(summary.by_event_type).length === 0 ? (
            <Empty icon="analytics" title="No incidents in the window" />
          ) : (
            <ul className="flex flex-col gap-space-sm">
              {Object.entries(summary.by_event_type)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 8)
                .map(([type, count]) => {
                  const max = Math.max(...Object.values(summary.by_event_type))
                  return (
                    <li key={type} className="flex items-center gap-space-md">
                      <Icon name={behaviorIcon(type)} className="text-[16px] text-on-surface-variant shrink-0" />
                      <span className="font-body-sm text-body-sm w-44 truncate">{behaviorLabel(type)}</span>
                      <div className="flex-1 min-w-0">
                        <Meter value={count} max={max} />
                      </div>
                      <span className="mono tabular-nums w-8 text-right">{count}</span>
                    </li>
                  )
                })}
            </ul>
          )}
        </Card>
      </div>
    </>
  )
}
