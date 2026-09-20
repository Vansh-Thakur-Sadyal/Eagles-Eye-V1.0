import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Card, Empty, ErrorNote, Icon, Loading, PageHeader, SeverityPill, Tabs, toast } from '../components/ui'
import { ApiError, api, type Paged } from '../lib/api'
import { useAuth } from '../lib/auth'
import { dateTimeOf, relativeTime } from '../lib/format'
import { useApi, useLiveEvents } from '../lib/hooks'

interface Alert {
  id: string
  incident_id: string | null
  channel: string
  severity: string
  title: string
  body: string | null
  acknowledged: boolean
  acknowledged_by: string | null
  created_at: string
  payload: Record<string, unknown>
}

const TABS = [
  { key: 'unack', label: 'Unacknowledged' },
  { key: 'all', label: 'All' },
] as const
type TabKey = (typeof TABS)[number]['key']

export default function AlertsPage() {
  const { can } = useAuth()
  const [tab, setTab] = useState<TabKey>('unack')
  const { data, loading, error, refresh, reload } = useApi<Paged<Alert>>(
    '/api/alerts',
    { unacknowledged_only: tab === 'unack', limit: 200 },
    { pollMs: 15000 },
  )

  useLiveEvents(['incident'], () => refresh())

  useEffect(() => {
    document.title = 'Alerts · Eagles Eye'
  }, [])

  async function acknowledge(alertId: string) {
    try {
      await api.post(`/api/alerts/${alertId}/acknowledge`)
      reload()
    } catch (err) {
      toast(err instanceof ApiError ? err.message : String(err), 'error')
    }
  }

  async function acknowledgeAll() {
    try {
      const out = await api.post<{ acknowledged: number }>('/api/alerts/acknowledge-all')
      toast(`${out.acknowledged} alerts acknowledged`)
      reload()
    } catch (err) {
      toast(err instanceof ApiError ? err.message : String(err), 'error')
    }
  }

  return (
    <>
      <PageHeader
        title="Alerts"
        subtitle="Operational notifications raised by the incident pipeline"
        actions={
          can('incident:write') && data && data.total > 0 && tab === 'unack' ? (
            <button className="btn-secondary" onClick={acknowledgeAll}>
              <Icon name="done_all" className="text-[16px]" />
              Acknowledge all
            </button>
          ) : null
        }
      />

      <Card bodyClassName="p-0">
        <div className="px-space-md">
          <Tabs tabs={TABS.map((t) => ({ ...t }))} active={tab} onChange={setTab} />
        </div>

        {error ? (
          <div className="p-space-md"><ErrorNote message={error} onRetry={reload} /></div>
        ) : loading && !data ? (
          <Loading />
        ) : !data || data.items.length === 0 ? (
          <Empty
            icon="notifications_off"
            title={tab === 'unack' ? 'Nothing needs attention' : 'No alerts recorded'}
            hint="An alert is raised the first time an incident is opened."
          />
        ) : (
          <ul className="divide-y divide-outline-variant/25">
            {data.items.map((alert) => (
              <li key={alert.id} className={`p-space-md flex items-start gap-space-md ${alert.acknowledged ? 'opacity-60' : ''}`}>
                <SeverityPill severity={alert.severity} />
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2 flex-wrap">
                    {alert.incident_id ? (
                      <Link to={`/incidents/${alert.incident_id}`} className="font-headline-sm text-headline-sm hover:text-primary-container">
                        {alert.title}
                      </Link>
                    ) : (
                      <span className="font-headline-sm text-headline-sm">{alert.title}</span>
                    )}
                    <span className="mono text-outline">{alert.channel}</span>
                  </div>
                  {alert.body ? (
                    <p className="font-body-sm text-body-sm text-on-surface-variant line-clamp-2">{alert.body}</p>
                  ) : null}
                  <p className="mono text-outline mt-0.5" title={dateTimeOf(alert.created_at)}>
                    {relativeTime(alert.created_at)}
                    {alert.acknowledged_by ? ` · acknowledged by ${alert.acknowledged_by}` : ''}
                    {typeof alert.payload?.camera_name === 'string' ? ` · ${alert.payload.camera_name}` : ''}
                  </p>
                </div>
                {!alert.acknowledged && can('incident:write') ? (
                  <button className="btn-secondary btn-xs shrink-0" onClick={() => acknowledge(alert.id)}>
                    Acknowledge
                  </button>
                ) : null}
              </li>
            ))}
          </ul>
        )}
      </Card>
    </>
  )
}
