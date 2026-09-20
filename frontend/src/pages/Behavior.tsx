import { useEffect, useState } from 'react'
import { Card, Empty, ErrorNote, Icon, Loading, Meter, PageHeader, SeverityPill } from '../components/ui'
import type { BehaviorEventRow, Paged } from '../lib/api'
import { behaviorIcon, behaviorLabel, confidence, dateTimeOf, duration, relativeTime } from '../lib/format'
import { useApi, useLiveEvents } from '../lib/hooks'

interface Stats {
  window_hours: number
  total: number
  items: { behavior: string; count: number; mean_confidence: number }[]
}

export default function BehaviorPage() {
  const [behavior, setBehavior] = useState('')
  const [minConfidence, setMinConfidence] = useState(0)
  const [selected, setSelected] = useState<BehaviorEventRow | null>(null)

  const { data: stats, refresh: refreshStats } = useApi<Stats>(
    '/api/intel/behavior/stats', { since_hours: 24 }, { pollMs: 30000 },
  )
  const { data, loading, error, refresh } = useApi<Paged<BehaviorEventRow>>(
    '/api/intel/behavior',
    { limit: 200, behavior: behavior || undefined, min_confidence: minConfidence || undefined, since_minutes: 1440 },
    { pollMs: 15000 },
  )

  useLiveEvents(['finding'], () => {
    refresh()
    refreshStats()
  })

  useEffect(() => {
    document.title = 'Behavior Intelligence · Eagles Eye'
  }, [])

  const maxCount = Math.max(1, ...(stats?.items ?? []).map((i) => i.count))

  return (
    <>
      <PageHeader
        title="Behavior Intelligence"
        subtitle="Loitering, counter-flow, running, erratic movement, restricted entry and following patterns"
      />

      <div className="grid grid-cols-1 xl:grid-cols-4 gap-space-lg items-start">
        <Card title={`Last 24 hours · ${stats?.total ?? 0} observations`}>
          {!stats || stats.items.length === 0 ? (
            <Empty icon="analytics" title="No observations yet" />
          ) : (
            <ul className="flex flex-col gap-space-sm">
              {stats.items.map((item) => (
                <li key={item.behavior}>
                  <button
                    onClick={() => setBehavior(behavior === item.behavior ? '' : item.behavior)}
                    className={`w-full text-left p-space-sm rounded-lg transition-colors ${
                      behavior === item.behavior ? 'bg-surface-container-low ring-1 ring-primary-container' : 'hover:bg-surface'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="flex items-center gap-2 min-w-0">
                        <Icon name={behaviorIcon(item.behavior)} className="text-[16px] text-on-surface-variant" />
                        <span className="font-body-sm text-body-sm truncate">{behaviorLabel(item.behavior)}</span>
                      </span>
                      <span className="mono tabular-nums shrink-0">{item.count}</span>
                    </div>
                    <div className="mt-1 flex items-center gap-2">
                      <Meter value={item.count} max={maxCount} />
                      <span className="mono text-outline shrink-0">{confidence(item.mean_confidence)}</span>
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </Card>

        <Card
          className="xl:col-span-3"
          title={behavior ? `${behaviorLabel(behavior)} events` : 'All behaviour events'}
          bodyClassName="p-0"
          actions={
            <div className="flex items-center gap-space-sm">
              <select
                className="input w-40 h-6 text-label-sm"
                value={minConfidence}
                onChange={(e) => setMinConfidence(Number(e.target.value))}
              >
                <option value={0}>Any confidence</option>
                <option value={0.5}>≥ 50%</option>
                <option value={0.7}>≥ 70%</option>
                <option value={0.85}>≥ 85%</option>
              </select>
              {behavior ? (
                <button className="btn-ghost btn-xs" onClick={() => setBehavior('')}>
                  <Icon name="filter_alt_off" className="text-[14px]" />
                  Clear
                </button>
              ) : null}
            </div>
          }
        >
          {error ? (
            <div className="p-space-md"><ErrorNote message={error} onRetry={refresh} /></div>
          ) : loading && !data ? (
            <Loading />
          ) : !data || data.items.length === 0 ? (
            <Empty
              icon="psychology"
              title="No behaviour events"
              hint="The behaviour agent writes an event whenever it observes a pattern that clears the configured threshold."
            />
          ) : (
            <div className="table-wrap max-h-[70vh] overflow-y-auto">
              <table className="tbl">
                <thead className="sticky top-0 z-10">
                  <tr>
                    <th>When</th>
                    <th>Behaviour</th>
                    <th>Camera</th>
                    <th>Track</th>
                    <th>Confidence</th>
                    <th>Duration</th>
                    <th>Severity</th>
                  </tr>
                </thead>
                <tbody>
                  {data.items.map((event) => (
                    <tr key={event.id} className="cursor-pointer" onClick={() => setSelected(event)}>
                      <td className="mono whitespace-nowrap" title={dateTimeOf(event.started_at)}>
                        {relativeTime(event.started_at)}
                      </td>
                      <td>
                        <span className="flex items-center gap-2">
                          <Icon name={behaviorIcon(event.behavior)} className="text-[16px] text-on-surface-variant" />
                          {behaviorLabel(event.behavior)}
                        </span>
                      </td>
                      <td className="mono text-on-surface-variant truncate max-w-[10rem]">{event.camera_id}</td>
                      <td className="mono">
                        {event.track_id ?? '—'}
                        {event.secondary_track_id ? ` → ${event.secondary_track_id}` : ''}
                      </td>
                      <td className="mono tabular-nums">{confidence(event.confidence)}</td>
                      <td className="mono">{duration(event.duration_seconds)}</td>
                      <td><SeverityPill severity={event.severity} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      </div>

      {selected ? (
        <Card
          title={`${behaviorLabel(selected.behavior)} · evidence`}
          actions={
            <button className="btn-ghost btn-xs" onClick={() => setSelected(null)}>
              <Icon name="close" className="text-[16px]" />
            </button>
          }
        >
          <div className="flex flex-col gap-space-md">
            <p className="font-body-lg text-body-lg">{selected.explanation}</p>
            <div className="table-wrap">
              <table className="tbl">
                <tbody>
                  {Object.entries(selected.evidence).map(([key, value]) => (
                    <tr key={key}>
                      <td className="label w-56 align-top pt-2">{key.replace(/_/g, ' ')}</td>
                      <td className="mono break-all">
                        {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </Card>
      ) : null}
    </>
  )
}
