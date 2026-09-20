import { useEffect, useMemo, useState } from 'react'
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Card, Empty, ErrorNote, Icon, Kpi, Loading, PageHeader, SeverityPill } from '../components/ui'
import type { CrowdRow } from '../lib/api'
import { densityColor, number, timeOf, titleCase } from '../lib/format'
import { useApi } from '../lib/hooks'

interface Heatmap {
  camera_id: string
  rows: number
  cols: number
  heatmap: number[][]
  density_band: string
  count: number
}

interface HistoryRow {
  id: string
  camera_id: string
  sampled_at: string
  count: number
  density: number
  density_band: string
  mean_speed: number
  counter_flow_ratio: number
  compression: number
  risk: string
}

export default function CrowdPage() {
  const [camera, setCamera] = useState<string>('')

  const { data: live, loading, error, refresh } = useApi<{ items: CrowdRow[]; total_people: number }>(
    '/api/intel/crowd/live', undefined, { pollMs: 5000 },
  )
  const { data: heatmap } = useApi<Heatmap>(
    camera ? `/api/intel/crowd/${camera}/heatmap` : null, undefined, { pollMs: 4000 },
  )
  const { data: history } = useApi<{ items: HistoryRow[] }>(
    '/api/intel/crowd/history',
    { camera_id: camera || undefined, since_minutes: 60, limit: 400 },
    { pollMs: 15000 },
  )

  useEffect(() => {
    document.title = 'Crowd Intelligence · Eagles Eye'
  }, [])

  useEffect(() => {
    if (!camera && live?.items.length) setCamera(live.items[0].camera_id)
  }, [live, camera])

  const elevated = (live?.items ?? []).filter((c) => c.risk === 'high' || c.risk === 'critical')
  const peak = useMemo(
    () => (live?.items ?? []).reduce<CrowdRow | null>((best, row) => (!best || row.count > best.count ? row : best), null),
    [live],
  )

  const chartData = useMemo(
    () =>
      (history?.items ?? []).map((row) => ({
        t: timeOf(row.sampled_at),
        people: row.count,
        density: Number((row.density * 100).toFixed(1)),
        counterflow: Number((row.counter_flow_ratio * 100).toFixed(1)),
      })),
    [history],
  )

  const maxCell = useMemo(
    () => Math.max(1, ...(heatmap?.heatmap ?? []).flat()),
    [heatmap],
  )

  return (
    <>
      <PageHeader
        title="Crowd Intelligence"
        subtitle="Density, flow, compression and surge — measured continuously per camera"
      />

      <section className="grid grid-cols-2 lg:grid-cols-4 gap-space-md">
        <Kpi label="People in view" value={number(live?.total_people ?? 0)} footer={`${live?.items.length ?? 0} cameras reporting`} />
        <Kpi
          label="Elevated zones"
          value={elevated.length}
          tone={elevated.length ? 'warning' : 'good'}
          footer={elevated.map((e) => e.camera_name ?? e.camera_id).join(', ') || 'all within limits'}
        />
        <Kpi label="Busiest camera" value={peak?.count ?? 0} footer={peak?.camera_name ?? peak?.camera_id ?? '—'} />
        <Kpi
          label="Peak density band"
          value={titleCase(peak?.density_band ?? 'low')}
          tone={peak?.density_band === 'critical' ? 'critical' : peak?.density_band === 'high' ? 'warning' : 'good'}
          footer={peak ? `compression ${peak.compression.toFixed(2)}` : undefined}
        />
      </section>

      {error ? <ErrorNote message={error} onRetry={refresh} /> : null}

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-space-lg items-start">
        <Card className="xl:col-span-2" title="Per-camera telemetry" bodyClassName="p-0">
          {loading && !live ? (
            <Loading />
          ) : !live || live.items.length === 0 ? (
            <Empty icon="groups" title="No crowd telemetry" hint="Start a camera and the crowd agent samples it every second." />
          ) : (
            <div className="table-wrap">
              <table className="tbl">
                <thead>
                  <tr>
                    <th>Camera</th>
                    <th>People</th>
                    <th>Density</th>
                    <th>Mean speed</th>
                    <th>Flow</th>
                    <th>Counter-flow</th>
                    <th>Compression</th>
                    <th>Risk</th>
                  </tr>
                </thead>
                <tbody>
                  {live.items.map((row) => (
                    <tr
                      key={row.camera_id}
                      className={`cursor-pointer ${camera === row.camera_id ? 'bg-surface-container-low' : ''}`}
                      onClick={() => setCamera(row.camera_id)}
                    >
                      <td className="truncate max-w-[12rem]">{row.camera_name ?? row.camera_id}</td>
                      <td className="mono tabular-nums font-semibold">{row.count}</td>
                      <td>
                        <span className="flex items-center gap-2">
                          <span className={`pip ${densityColor(row.density_band)}`} />
                          <span className="mono">{row.density_band}</span>
                        </span>
                      </td>
                      <td className="mono tabular-nums">{row.mean_speed.toFixed(0)} px/s</td>
                      <td className="mono tabular-nums">
                        {row.flow_direction_deg === null ? '—' : `${row.flow_direction_deg.toFixed(0)}°`}
                      </td>
                      <td className="mono tabular-nums">{Math.round(row.counter_flow_ratio * 100)}%</td>
                      <td className="mono tabular-nums">{row.compression.toFixed(2)}</td>
                      <td><SeverityPill severity={row.risk === 'low' ? 'low' : row.risk} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        <Card title={`Occupancy grid${heatmap ? ` · ${heatmap.count} people` : ''}`}>
          {!heatmap ? (
            <Empty icon="grid_on" title="Select a camera" hint="Pick a row on the left to see its occupancy grid." />
          ) : (
            <div className="flex flex-col gap-space-sm">
              <div
                className="grid gap-[2px] w-full aspect-video"
                style={{ gridTemplateColumns: `repeat(${heatmap.cols}, minmax(0, 1fr))` }}
              >
                {heatmap.heatmap.flatMap((row, r) =>
                  row.map((cell, c) => {
                    const intensity = cell / maxCell
                    return (
                      <div
                        key={`${r}-${c}`}
                        title={cell ? `${cell} in cell` : 'empty'}
                        className="rounded-[1px]"
                        style={{
                          backgroundColor:
                            cell === 0
                              ? 'rgba(148,163,184,0.16)'
                              : `rgba(220,38,38,${0.18 + intensity * 0.75})`,
                        }}
                      />
                    )
                  }),
                )}
              </div>
              <p className="mono text-on-surface-variant">
                {heatmap.rows} × {heatmap.cols} cells · band {heatmap.density_band}
              </p>
            </div>
          )}
        </Card>
      </div>

      <Card title="Last hour">
        {chartData.length < 2 ? (
          <Empty icon="show_chart" title="Not enough samples yet" hint="The chart fills in as telemetry accumulates." />
        ) : (
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
                <defs>
                  <linearGradient id="people" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#1d4ed8" stopOpacity={0.35} />
                    <stop offset="100%" stopColor="#1d4ed8" stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="#e2e8f0" vertical={false} />
                <XAxis dataKey="t" tick={{ fontSize: 11, fill: '#434655' }} minTickGap={40} />
                <YAxis tick={{ fontSize: 11, fill: '#434655' }} width={40} />
                <Tooltip
                  contentStyle={{
                    borderRadius: 8, border: '1px solid #c4c5d7', fontSize: 12, fontFamily: 'Inter',
                  }}
                />
                <Area type="monotone" dataKey="people" stroke="#1d4ed8" strokeWidth={2} fill="url(#people)" />
                <Area type="monotone" dataKey="counterflow" stroke="#d97706" strokeWidth={1.5} fillOpacity={0} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        )}
      </Card>
    </>
  )
}
