/** Privacy & Audit — the controls, and proof they are actually enforced. */
import { useEffect, useState } from 'react'
import { Card, Empty, ErrorNote, Icon, InfoNote, Kpi, Loading, PageHeader, Tabs, toast } from '../components/ui'
import { ApiError, api, type Paged } from '../lib/api'
import { useAuth } from '../lib/auth'
import { dateTimeOf, number, relativeTime, titleCase } from '../lib/format'
import { useApi } from '../lib/hooks'

interface PrivacyStatus {
  settings: {
    bystander_redaction: boolean
    anonymous_by_default: boolean
    retention_days: number
    audit_all_reads: boolean
  }
  runtime: { redactions_applied: number; redaction_enabled: boolean; cameras_with_exemptions: number }
  active_watchlist_subjects: number
  pending_approvals: number
  records_past_retention: number
  controls: { control: string; enforced: boolean }[]
}

interface AuditRow {
  id: string
  at: string
  actor: string
  actor_role: string
  action: string
  resource_type: string
  resource_id: string | null
  justification: string | null
  ip_address: string | null
  outcome: string
  detail: Record<string, unknown>
}

const TABS = [
  { key: 'controls', label: 'Controls' },
  { key: 'audit', label: 'Audit log' },
] as const
type TabKey = (typeof TABS)[number]['key']

export default function PrivacyPage() {
  const { can } = useAuth()
  const [tab, setTab] = useState<TabKey>('controls')
  const [actionFilter, setActionFilter] = useState('')

  const { data: status, loading, error, refresh } = useApi<PrivacyStatus>(
    '/api/admin/privacy', undefined, { pollMs: 20000 },
  )
  const { data: audit, refresh: refreshAudit } = useApi<Paged<AuditRow>>(
    '/api/admin/audit',
    { limit: 200, action: actionFilter || undefined, since_hours: 168 },
    { pollMs: 30000, enabled: can('audit:read') },
  )
  const { data: summary } = useApi<{
    by_action: { action: string; count: number }[]
    identity_sensitive_reads: number
    failed_actions: number
  }>('/api/admin/audit/summary', { since_hours: 24 }, { enabled: can('audit:read') })

  useEffect(() => {
    document.title = 'Privacy & Audit · Eagles Eye'
  }, [])

  async function purge(dryRun: boolean) {
    if (!dryRun && !confirm('Apply the retention policy and permanently delete records past the retention window?\n\nSealed evidence and the incidents it references are kept.')) {
      return
    }
    try {
      const result = await api.post<Record<string, unknown>>('/api/admin/privacy/purge-expired', undefined, { dry_run: dryRun })
      toast(
        dryRun
          ? `Dry run: ${result.incidents_past_retention} incidents and ${result.behavior_events_past_retention} events are past retention`
          : `Deleted ${result.deleted_incidents} incidents and ${result.deleted_events} events`,
      )
      refresh()
    } catch (err) {
      toast(err instanceof ApiError ? err.message : String(err), 'error')
    }
  }

  if (error) return <ErrorNote message={error} onRetry={refresh} />
  if (loading && !status) return <Loading />
  if (!status) return null

  return (
    <>
      <PageHeader
        title="Privacy & Audit"
        subtitle="Privacy-by-design controls and the immutable record of who looked at what"
      />

      <section className="grid grid-cols-2 lg:grid-cols-4 gap-space-md">
        <Kpi
          label="Redactions applied"
          value={number(status.runtime.redactions_applied)}
          footer="bystander faces blurred before transmission"
        />
        <Kpi
          label="Active watchlist subjects"
          value={status.active_watchlist_subjects}
          tone={status.active_watchlist_subjects ? 'warning' : 'good'}
          footer="each required an approval"
        />
        <Kpi
          label="Identity-sensitive reads"
          value={summary?.identity_sensitive_reads ?? 0}
          footer="last 24 hours, all logged"
        />
        <Kpi
          label="Past retention"
          value={status.records_past_retention}
          tone={status.records_past_retention ? 'warning' : 'good'}
          footer={`policy: ${status.settings.retention_days} days`}
        />
      </section>

      <Tabs tabs={TABS.map((t) => ({ ...t }))} active={tab} onChange={setTab} />

      {tab === 'controls' ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-space-lg items-start">
          <Card title="Enforced controls">
            <ul className="flex flex-col gap-space-sm">
              {status.controls.map((control) => (
                <li key={control.control} className="flex items-start gap-space-sm">
                  <Icon
                    name={control.enforced ? 'check_circle' : 'cancel'}
                    className={`text-[18px] shrink-0 mt-0.5 ${control.enforced ? 'text-emerald-600' : 'text-amber-600'}`}
                  />
                  <span className="font-body-sm text-body-sm">{control.control}</span>
                </li>
              ))}
            </ul>
          </Card>

          <div className="flex flex-col gap-space-lg">
            <Card title="Data retention">
              <p className="font-body-sm text-body-sm text-on-surface-variant mb-space-md">
                Records older than {status.settings.retention_days} days are eligible for deletion.
                Sealed evidence — and the incidents it references — is exempt.
              </p>
              {can('*') ? (
                <div className="flex gap-space-sm">
                  <button className="btn-secondary" onClick={() => purge(true)}>
                    <Icon name="preview" className="text-[16px]" />
                    Dry run
                  </button>
                  <button className="btn-danger" onClick={() => purge(false)}>
                    <Icon name="delete_sweep" className="text-[16px]" />
                    Apply retention policy
                  </button>
                </div>
              ) : (
                <InfoNote icon="lock">Only an administrator can apply the retention policy.</InfoNote>
              )}
            </Card>

            {summary ? (
              <Card title="Activity · last 24 hours">
                <ul className="flex flex-col gap-1 max-h-60 overflow-y-auto">
                  {summary.by_action.map((row) => (
                    <li key={row.action} className="flex items-center justify-between gap-space-md">
                      <button
                        className="mono text-on-surface-variant hover:text-primary-container truncate"
                        onClick={() => {
                          setActionFilter(row.action)
                          setTab('audit')
                        }}
                      >
                        {row.action}
                      </button>
                      <span className="mono tabular-nums">{row.count}</span>
                    </li>
                  ))}
                </ul>
                {summary.failed_actions > 0 ? (
                  <p className="mono text-amber-800 mt-space-md pt-space-md border-t border-outline-variant/40">
                    {summary.failed_actions} failed action(s) recorded
                  </p>
                ) : null}
              </Card>
            ) : null}
          </div>
        </div>
      ) : !can('audit:read') ? (
        <InfoNote icon="lock">Your role cannot read the audit log.</InfoNote>
      ) : (
        <Card
          title={`Audit log${actionFilter ? ` · ${actionFilter}` : ''}`}
          bodyClassName="p-0"
          actions={
            <div className="flex items-center gap-space-sm">
              <input
                className="input w-48 h-6 text-label-sm"
                placeholder="Filter by action prefix"
                value={actionFilter}
                onChange={(e) => setActionFilter(e.target.value)}
              />
              <button className="btn-ghost btn-xs" onClick={refreshAudit}>
                <Icon name="refresh" className="text-[14px]" />
              </button>
            </div>
          }
        >
          {!audit || audit.items.length === 0 ? (
            <Empty icon="history" title="No matching entries" />
          ) : (
            <div className="table-wrap max-h-[65vh] overflow-y-auto">
              <table className="tbl">
                <thead className="sticky top-0 z-10">
                  <tr>
                    <th>When</th>
                    <th>Actor</th>
                    <th>Action</th>
                    <th>Resource</th>
                    <th>Outcome</th>
                    <th>Justification</th>
                  </tr>
                </thead>
                <tbody>
                  {audit.items.map((row) => (
                    <tr key={row.id}>
                      <td className="mono whitespace-nowrap" title={dateTimeOf(row.at)}>
                        {relativeTime(row.at)}
                      </td>
                      <td className="mono">
                        {row.actor}
                        {row.actor_role ? <span className="text-outline"> · {row.actor_role}</span> : null}
                      </td>
                      <td className="mono">{row.action}</td>
                      <td className="mono text-on-surface-variant truncate max-w-[14rem]">
                        {row.resource_type}
                        {row.resource_id ? ` · ${row.resource_id}` : ''}
                      </td>
                      <td>
                        <span className={`mono ${row.outcome === 'success' ? 'text-emerald-700' : 'text-red-700'}`}>
                          {row.outcome}
                        </span>
                      </td>
                      <td className="text-on-surface-variant max-w-sm truncate" title={row.justification ?? ''}>
                        {row.justification ?? '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      )}
    </>
  )
}
