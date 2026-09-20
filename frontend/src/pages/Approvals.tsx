/** Human approval queue — the gate in front of every high-consequence action. */
import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Card, Empty, ErrorNote, Icon, InfoNote, Loading, PageHeader, Tabs, toast } from '../components/ui'
import { ApiError, api, type Paged } from '../lib/api'
import { useAuth } from '../lib/auth'
import { dateTimeOf, relativeTime, titleCase } from '../lib/format'
import { useApi, useLiveEvents } from '../lib/hooks'

interface Approval {
  id: string
  kind: string
  title: string
  detail: string | null
  justification: string | null
  requested_by: string
  required_role: string
  resource_type: string | null
  resource_id: string | null
  status: string
  decided_by: string | null
  decided_at: string | null
  decision_note: string | null
  created_at: string
  payload: Record<string, unknown>
}

const TABS = [
  { key: 'pending', label: 'Pending' },
  { key: 'approved', label: 'Approved' },
  { key: 'rejected', label: 'Rejected' },
  { key: 'all', label: 'All' },
] as const
type TabKey = (typeof TABS)[number]['key']

const KIND_ICON: Record<string, string> = {
  identity_escalation: 'fingerprint',
  watchlist_enrol: 'person_add',
  evidence_export: 'download',
  dispatch: 'local_police',
  retention_override: 'schedule',
  model_promotion: 'model_training',
}

export default function ApprovalsPage() {
  const { user, can } = useAuth()
  const [tab, setTab] = useState<TabKey>('pending')
  const { data, loading, error, refresh, reload } = useApi<Paged<Approval>>(
    '/api/admin/approvals', { status_filter: tab, limit: 100 }, { pollMs: 20000 },
  )

  useLiveEvents(['approval'], () => refresh())

  useEffect(() => {
    document.title = 'Approvals · Eagles Eye'
  }, [])

  async function decide(approval: Approval, decision: 'approve' | 'reject') {
    const note = prompt(
      `${decision === 'approve' ? 'Approve' : 'Reject'}: ${approval.title}\n\nRequested by ${approval.requested_by}\nJustification: ${approval.justification ?? 'none given'}\n\nOptional note:`,
    )
    if (note === null) return
    try {
      await api.post(`/api/admin/approvals/${approval.id}/decide`, { decision, note: note || undefined })
      toast(`Request ${decision === 'approve' ? 'approved' : 'rejected'}`)
      reload()
    } catch (err) {
      toast(err instanceof ApiError ? err.message : String(err), 'error')
    }
  }

  const canDecide = can('approval:decide')

  return (
    <>
      <PageHeader
        title="Approvals"
        subtitle="Identity escalation, watchlist enrolment and every action with a real-world consequence"
      />

      <InfoNote icon="gavel">
        Eagles Eye recommends; a person decides. Nothing in this queue has taken effect, and you cannot
        approve a request you raised yourself.
      </InfoNote>

      <Card bodyClassName="p-0">
        <div className="px-space-md">
          <Tabs tabs={TABS.map((t) => ({ ...t }))} active={tab} onChange={setTab} />
        </div>

        {error ? (
          <div className="p-space-md"><ErrorNote message={error} onRetry={reload} /></div>
        ) : loading && !data ? (
          <Loading />
        ) : !data || data.items.length === 0 ? (
          <Empty icon="task_alt" title={tab === 'pending' ? 'Nothing awaiting a decision' : 'No requests here'} />
        ) : (
          <ul className="divide-y divide-outline-variant/25">
            {data.items.map((approval) => {
              const ownRequest = approval.requested_by === user?.username
              return (
                <li key={approval.id} className="p-space-md flex items-start gap-space-md">
                  <span className="w-8 h-8 rounded-lg bg-surface-container flex items-center justify-center shrink-0">
                    <Icon name={KIND_ICON[approval.kind] ?? 'help'} className="text-[18px] text-on-surface-variant" />
                  </span>

                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="font-headline-sm text-headline-sm">{approval.title}</span>
                      <span className="pill bg-surface-container border-outline-variant text-on-surface-variant">
                        {titleCase(approval.kind)}
                      </span>
                      {approval.status !== 'pending' ? (
                        <span
                          className={`pill ${
                            approval.status === 'approved'
                              ? 'bg-emerald-50 border-emerald-200 text-emerald-800'
                              : 'bg-red-50 border-red-200 text-red-800'
                          }`}
                        >
                          {approval.status}
                        </span>
                      ) : null}
                    </div>

                    {approval.detail ? (
                      <p className="font-body-sm text-body-sm text-on-surface-variant">{approval.detail}</p>
                    ) : null}
                    {approval.justification ? (
                      <p className="font-body-sm text-body-sm text-on-surface mt-1 border-l-[3px] border-outline-variant pl-space-sm">
                        {approval.justification}
                      </p>
                    ) : null}

                    <p className="mono text-outline mt-1">
                      requested by {approval.requested_by} · {relativeTime(approval.created_at)} · needs{' '}
                      {approval.required_role}
                      {approval.decided_by ? ` · decided by ${approval.decided_by} ${relativeTime(approval.decided_at)}` : ''}
                    </p>

                    {approval.resource_type === 'incident' && approval.resource_id ? (
                      <Link to={`/incidents/${approval.resource_id}`} className="mono text-primary-container">
                        View incident →
                      </Link>
                    ) : approval.resource_type === 'watchlist_subject' && approval.resource_id ? (
                      <Link to={`/watchlist/${approval.resource_id}`} className="mono text-primary-container">
                        View subject →
                      </Link>
                    ) : null}
                  </div>

                  {approval.status === 'pending' && canDecide ? (
                    <div className="flex gap-space-sm shrink-0">
                      <button
                        className="btn-secondary btn-xs"
                        onClick={() => decide(approval, 'reject')}
                        disabled={ownRequest}
                        title={ownRequest ? 'You cannot decide your own request' : undefined}
                      >
                        Reject
                      </button>
                      <button
                        className="btn-primary btn-xs"
                        onClick={() => decide(approval, 'approve')}
                        disabled={ownRequest}
                        title={ownRequest ? 'You cannot decide your own request' : undefined}
                      >
                        Approve
                      </button>
                    </div>
                  ) : null}
                </li>
              )
            })}
          </ul>
        )}
      </Card>
    </>
  )
}
