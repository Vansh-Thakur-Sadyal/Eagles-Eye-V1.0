import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Card, Empty, ErrorNote, Field, Icon, Loading, Modal, PageHeader, SeverityPill, Tabs, toast } from '../components/ui'
import { ApiError, api, type Incident, type Paged } from '../lib/api'
import { useAuth } from '../lib/auth'
import { dateTimeOf, relativeTime, titleCase } from '../lib/format'
import { useApi } from '../lib/hooks'

interface Case {
  id: string
  title: string
  reference: string | null
  description: string | null
  status: string
  priority: string
  lead: string | null
  incident_ids: string[]
  tags: string[]
  created_at: string
  updated_at: string
  closed_at: string | null
}

interface CaseDetail extends Case {
  incidents: Incident[]
  evidence: { id: string; kind: string; sha256: string | null; sealed: boolean; created_at: string; note: string | null }[]
}

const TABS = [
  { key: 'open', label: 'Open' },
  { key: 'all', label: 'All' },
  { key: 'closed', label: 'Closed' },
] as const
type TabKey = (typeof TABS)[number]['key']

export default function CasesPage() {
  const { can } = useAuth()
  const [tab, setTab] = useState<TabKey>('open')
  const [selected, setSelected] = useState<string | null>(null)
  const [open, setOpen] = useState(false)
  const [form, setForm] = useState({ title: '', reference: '', description: '', priority: 'medium' })

  const { data, loading, error, reload } = useApi<Paged<Case>>(
    '/api/admin/cases',
    { status_filter: tab === 'all' ? undefined : tab, limit: 100 },
    { pollMs: 30000 },
  )
  const { data: detail } = useApi<CaseDetail>(selected ? `/api/admin/cases/${selected}` : null)

  useEffect(() => {
    document.title = 'Cases · Eagles Eye'
  }, [])

  async function create() {
    try {
      await api.post('/api/admin/cases', {
        title: form.title,
        reference: form.reference || null,
        description: form.description || null,
        priority: form.priority,
      })
      toast('Case opened')
      setOpen(false)
      setForm({ title: '', reference: '', description: '', priority: 'medium' })
      reload()
    } catch (err) {
      toast(err instanceof ApiError ? err.message : String(err), 'error')
    }
  }

  async function close(caseId: string) {
    try {
      await api.patch(`/api/admin/cases/${caseId}`, { status: 'closed' })
      toast('Case closed')
      reload()
    } catch (err) {
      toast(err instanceof ApiError ? err.message : String(err), 'error')
    }
  }

  return (
    <>
      <PageHeader
        title="Cases"
        subtitle="Group related incidents and evidence into an investigation"
        actions={
          can('case:write') ? (
            <button className="btn-primary" onClick={() => setOpen(true)}>
              <Icon name="create_new_folder" className="text-[16px]" />
              Open case
            </button>
          ) : null
        }
      />

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-space-lg items-start">
        <Card className="xl:col-span-2" bodyClassName="p-0">
          <div className="px-space-md">
            <Tabs tabs={TABS.map((t) => ({ ...t }))} active={tab} onChange={setTab} />
          </div>

          {error ? (
            <div className="p-space-md"><ErrorNote message={error} onRetry={reload} /></div>
          ) : loading && !data ? (
            <Loading />
          ) : !data || data.items.length === 0 ? (
            <Empty icon="folder_off" title="No cases" hint="Open a case to collect incidents, evidence and notes in one place." />
          ) : (
            <div className="table-wrap">
              <table className="tbl">
                <thead>
                  <tr>
                    <th>Case</th>
                    <th>Priority</th>
                    <th>Lead</th>
                    <th>Incidents</th>
                    <th>Status</th>
                    <th>Updated</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {data.items.map((row) => (
                    <tr
                      key={row.id}
                      className={`cursor-pointer ${selected === row.id ? 'bg-surface-container-low' : ''}`}
                      onClick={() => setSelected(row.id)}
                    >
                      <td className="max-w-xs">
                        <span className="block truncate">{row.title}</span>
                        <span className="mono text-outline">{row.reference ?? row.id}</span>
                      </td>
                      <td><SeverityPill severity={row.priority === 'critical' ? 'critical' : row.priority} /></td>
                      <td className="mono text-on-surface-variant">{row.lead ?? '—'}</td>
                      <td className="mono tabular-nums">{row.incident_ids.length}</td>
                      <td className="mono">{titleCase(row.status)}</td>
                      <td className="mono text-on-surface-variant">{relativeTime(row.updated_at)}</td>
                      <td className="w-24">
                        {can('case:write') && row.status !== 'closed' ? (
                          <button
                            className="btn-ghost btn-xs"
                            onClick={(e) => {
                              e.stopPropagation()
                              close(row.id)
                            }}
                          >
                            Close
                          </button>
                        ) : null}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        <Card title={detail ? detail.title : 'Case detail'}>
          {!detail ? (
            <Empty icon="folder_open" title="Select a case" hint="Pick a row to see its incidents and evidence." />
          ) : (
            <div className="flex flex-col gap-space-md">
              {detail.description ? (
                <p className="font-body-sm text-body-sm text-on-surface-variant">{detail.description}</p>
              ) : null}
              <dl className="grid grid-cols-2 gap-space-md">
                <Stat label="Reference" value={detail.reference ?? '—'} />
                <Stat label="Lead" value={detail.lead ?? '—'} />
                <Stat label="Opened" value={dateTimeOf(detail.created_at)} />
                <Stat label="Status" value={titleCase(detail.status)} />
              </dl>

              <div>
                <p className="label">Incidents ({detail.incidents.length})</p>
                {detail.incidents.length === 0 ? (
                  <p className="mono text-on-surface-variant">none linked</p>
                ) : (
                  <ul className="flex flex-col gap-1">
                    {detail.incidents.map((incident) => (
                      <li key={incident.id}>
                        <Link
                          to={`/incidents/${incident.id}`}
                          className="flex items-center justify-between gap-2 hover:text-primary-container"
                        >
                          <span className="font-body-sm text-body-sm truncate">{incident.title}</span>
                          <SeverityPill severity={incident.severity} />
                        </Link>
                      </li>
                    ))}
                  </ul>
                )}
              </div>

              <div>
                <p className="label">Evidence ({detail.evidence.length})</p>
                {detail.evidence.length === 0 ? (
                  <p className="mono text-on-surface-variant">none collected</p>
                ) : (
                  <ul className="flex flex-col gap-1">
                    {detail.evidence.map((item) => (
                      <li key={item.id} className="flex items-center gap-2 mono text-on-surface-variant">
                        <Icon name={item.sealed ? 'lock' : 'lock_open'} className="text-[14px]" />
                        {item.kind} · {item.sha256?.slice(0, 12) ?? 'no hash'}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          )}
        </Card>
      </div>

      <Modal
        open={open}
        onClose={() => setOpen(false)}
        title="Open a case"
        footer={
          <>
            <button className="btn-secondary" onClick={() => setOpen(false)}>Cancel</button>
            <button className="btn-primary" onClick={create} disabled={!form.title.trim()}>Open case</button>
          </>
        }
      >
        <div className="flex flex-col gap-space-md">
          <Field label="Title" required>
            <input className="input" value={form.title} onChange={(e) => setForm({ ...form, title: e.target.value })} />
          </Field>
          <Field label="Reference">
            <input className="input" value={form.reference} onChange={(e) => setForm({ ...form, reference: e.target.value })} />
          </Field>
          <Field label="Priority">
            <select className="input" value={form.priority} onChange={(e) => setForm({ ...form, priority: e.target.value })}>
              {['low', 'medium', 'high', 'critical'].map((p) => (
                <option key={p} value={p}>{titleCase(p)}</option>
              ))}
            </select>
          </Field>
          <Field label="Description">
            <textarea
              className="input h-24 py-2 resize-none"
              value={form.description}
              onChange={(e) => setForm({ ...form, description: e.target.value })}
            />
          </Field>
        </div>
      </Modal>
    </>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0">
      <dt className="label">{label}</dt>
      <dd className="mono text-on-surface truncate" title={value}>{value}</dd>
    </div>
  )
}
