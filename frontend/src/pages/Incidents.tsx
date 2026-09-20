import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Card, Empty, ErrorNote, Icon, Loading, Meter, PageHeader, SeverityPill, Tabs } from '../components/ui'
import type { Incident, Paged } from '../lib/api'
import { behaviorIcon, dateTimeOf, relativeTime, riskBand, titleCase } from '../lib/format'
import { useApi, useLiveEvents } from '../lib/hooks'

const TABS = [
  { key: 'open', label: 'Open' },
  { key: 'all', label: 'All' },
  { key: 'resolved', label: 'Resolved' },
] as const

type TabKey = (typeof TABS)[number]['key']

export default function IncidentsPage() {
  const [tab, setTab] = useState<TabKey>('open')
  const [severity, setSeverity] = useState('')
  const [eventType, setEventType] = useState('')

  const params: Record<string, unknown> = { limit: 100 }
  if (tab === 'open') params.open_only = true
  if (tab === 'resolved') params.status_filter = 'resolved'
  if (severity) params.severity = severity
  if (eventType) params.event_type = eventType

  const { data, loading, error, refresh } = useApi<Paged<Incident>>('/api/incidents', params, { pollMs: 15000 })
  const { data: summary } = useApi<{ by_event_type: Record<string, number>; by_severity: Record<string, number> }>(
    '/api/incidents/summary', { since_hours: 168 }, { pollMs: 60000 },
  )

  useLiveEvents(['incident'], () => refresh())

  useEffect(() => {
    document.title = 'Incidents · Eagles Eye'
  }, [])

  return (
    <>
      <PageHeader
        title="Incidents"
        subtitle="Correlated multi-signal events, each with an explainable risk score"
      />

      <Card bodyClassName="p-0">
        <div className="flex items-center justify-between gap-space-md px-space-md flex-wrap">
          <Tabs tabs={TABS.map((t) => ({ ...t }))} active={tab} onChange={setTab} />
          <div className="flex items-center gap-space-sm py-space-sm">
            <select className="input w-36" value={severity} onChange={(e) => setSeverity(e.target.value)}>
              <option value="">All severities</option>
              {['critical', 'high', 'medium', 'low', 'info'].map((s) => (
                <option key={s} value={s}>
                  {titleCase(s)}
                  {summary?.by_severity?.[s] ? ` (${summary.by_severity[s]})` : ''}
                </option>
              ))}
            </select>
            <select className="input w-48" value={eventType} onChange={(e) => setEventType(e.target.value)}>
              <option value="">All event types</option>
              {Object.entries(summary?.by_event_type ?? {}).map(([type, count]) => (
                <option key={type} value={type}>
                  {titleCase(type)} ({count})
                </option>
              ))}
            </select>
            <button className="btn-ghost btn-xs" onClick={refresh} title="Refresh">
              <Icon name="refresh" className="text-[16px]" />
            </button>
          </div>
        </div>

        {error ? (
          <div className="p-space-md">
            <ErrorNote message={error} onRetry={refresh} />
          </div>
        ) : loading && !data ? (
          <Loading label="Loading incidents" />
        ) : !data || data.items.length === 0 ? (
          <Empty
            icon="check_circle"
            title="Nothing matches these filters"
            hint="Incidents appear here as soon as the agents correlate signals from a running camera."
          />
        ) : (
          <div className="table-wrap">
            <table className="tbl">
              <thead>
                <tr>
                  <th>Incident</th>
                  <th>Camera</th>
                  <th className="w-32">Risk</th>
                  <th>Severity</th>
                  <th>Status</th>
                  <th>Started</th>
                  <th>Updated</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {data.items.map((incident) => (
                  <tr key={incident.id}>
                    <td className="max-w-sm">
                      <Link to={`/incidents/${incident.id}`} className="flex items-center gap-2 min-w-0 group">
                        <Icon name={behaviorIcon(incident.event_type)} className="text-[16px] text-on-surface-variant shrink-0" />
                        <span className="min-w-0">
                          <span className="block truncate group-hover:text-primary-container">{incident.title}</span>
                          <span className="mono text-outline">{incident.id}</span>
                        </span>
                      </Link>
                    </td>
                    <td className="mono text-on-surface-variant truncate max-w-[10rem]">
                      {String((incident.meta as any)?.camera_name ?? incident.camera_id ?? '—')}
                    </td>
                    <td>
                      <div className="flex items-center gap-2">
                        <span className="mono font-semibold tabular-nums w-8">{incident.risk_score.toFixed(0)}</span>
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
                    <td className="mono text-on-surface-variant whitespace-nowrap" title={dateTimeOf(incident.started_at)}>
                      {relativeTime(incident.started_at)}
                    </td>
                    <td className="mono text-on-surface-variant whitespace-nowrap">
                      {relativeTime(incident.last_update_at)}
                    </td>
                    <td className="w-10">
                      <Link to={`/incidents/${incident.id}`} className="btn-ghost btn-xs">
                        <Icon name="chevron_right" className="text-[16px]" />
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </>
  )
}
