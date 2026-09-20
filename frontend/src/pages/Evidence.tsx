/** Evidence Locker — hashed captures with a chain of custody you can verify. */
import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Card, Empty, ErrorNote, Icon, InfoNote, Kpi, Loading, PageHeader, toast } from '../components/ui'
import { ApiError, api, type Paged } from '../lib/api'
import { useAuth } from '../lib/auth'
import { bytes, dateTimeOf, relativeTime, titleCase } from '../lib/format'
import { useApi } from '../lib/hooks'

interface EvidenceRow {
  id: string
  case_id: string | null
  incident_id: string | null
  kind: string
  file_path: string | null
  sha256: string | null
  size_bytes: number
  collected_by: string
  chain_of_custody: { at: string; actor: string; role: string; action: string; sha256?: string }[]
  sealed: boolean
  retention_until: string | null
  note: string | null
  created_at: string
}

export default function EvidencePage() {
  const { can } = useAuth()
  const [verified, setVerified] = useState<Record<string, { verified: boolean; reason?: string }>>({})
  const [expanded, setExpanded] = useState<string | null>(null)

  const { data, loading, error, reload } = useApi<Paged<EvidenceRow>>(
    '/api/admin/evidence', { limit: 200 }, { pollMs: 30000 },
  )

  useEffect(() => {
    document.title = 'Evidence Locker · Eagles Eye'
  }, [])

  async function verify(row: EvidenceRow) {
    try {
      const result = await api.get<{ verified: boolean; reason?: string }>(`/api/admin/evidence/${row.id}/verify`)
      setVerified((prev) => ({ ...prev, [row.id]: result }))
      toast(result.verified ? 'Hash matches — file is intact' : `Verification failed: ${result.reason ?? 'hash mismatch'}`,
            result.verified ? 'ok' : 'error')
    } catch (err) {
      toast(err instanceof ApiError ? err.message : String(err), 'error')
    }
  }

  async function seal(row: EvidenceRow) {
    if (!confirm('Seal this item? It becomes exempt from the retention purge.')) return
    try {
      await api.post(`/api/admin/evidence/${row.id}/seal`)
      toast('Evidence sealed')
      reload()
    } catch (err) {
      toast(err instanceof ApiError ? err.message : String(err), 'error')
    }
  }

  const sealed = (data?.items ?? []).filter((e) => e.sealed).length
  const totalBytes = (data?.items ?? []).reduce((sum, e) => sum + e.size_bytes, 0)

  return (
    <>
      <PageHeader title="Evidence Locker" subtitle="Hashed captures with a verifiable chain of custody" />

      <InfoNote icon="fingerprint">
        Every item is hashed with SHA-256 on capture and carries a chain-of-custody entry. Verify
        re-reads the file and recomputes the hash, so tampering after the fact is detectable.
      </InfoNote>

      <section className="grid grid-cols-2 lg:grid-cols-4 gap-space-md">
        <Kpi label="Items held" value={data?.total ?? 0} />
        <Kpi label="Sealed" value={sealed} footer="exempt from retention purge" />
        <Kpi label="Storage" value={bytes(totalBytes)} />
        <Kpi
          label="Verified this session"
          value={Object.values(verified).filter((v) => v.verified).length}
          tone={Object.values(verified).some((v) => !v.verified) ? 'critical' : 'good'}
        />
      </section>

      <Card bodyClassName="p-0">
        {error ? (
          <div className="p-space-md"><ErrorNote message={error} onRetry={reload} /></div>
        ) : loading && !data ? (
          <Loading />
        ) : !data || data.items.length === 0 ? (
          <Empty
            icon="lock_open"
            title="No evidence collected"
            hint="Capture evidence from an incident — the current frame is hashed and stored with a custody record."
          />
        ) : (
          <div className="table-wrap">
            <table className="tbl">
              <thead>
                <tr>
                  <th>Item</th>
                  <th>Incident</th>
                  <th>Kind</th>
                  <th>SHA-256</th>
                  <th>Size</th>
                  <th>Collected by</th>
                  <th>When</th>
                  <th>State</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {data.items.map((row) => (
                  <>
                    <tr key={row.id} className="cursor-pointer" onClick={() => setExpanded(expanded === row.id ? null : row.id)}>
                      <td className="mono">{row.id}</td>
                      <td className="mono">
                        {row.incident_id ? (
                          <Link to={`/incidents/${row.incident_id}`} className="hover:text-primary-container" onClick={(e) => e.stopPropagation()}>
                            {row.incident_id}
                          </Link>
                        ) : (
                          '—'
                        )}
                      </td>
                      <td>{titleCase(row.kind)}</td>
                      <td className="mono text-on-surface-variant" title={row.sha256 ?? ''}>
                        {row.sha256 ? `${row.sha256.slice(0, 16)}…` : '—'}
                      </td>
                      <td className="mono tabular-nums">{bytes(row.size_bytes)}</td>
                      <td className="mono">{row.collected_by}</td>
                      <td className="mono text-on-surface-variant" title={dateTimeOf(row.created_at)}>
                        {relativeTime(row.created_at)}
                      </td>
                      <td>
                        <span className="flex items-center gap-1.5">
                          <Icon
                            name={row.sealed ? 'lock' : 'lock_open'}
                            className={`text-[15px] ${row.sealed ? 'text-emerald-600' : 'text-outline'}`}
                          />
                          {verified[row.id] ? (
                            <span className={`mono ${verified[row.id].verified ? 'text-emerald-700' : 'text-red-700'}`}>
                              {verified[row.id].verified ? 'verified' : 'mismatch'}
                            </span>
                          ) : null}
                        </span>
                      </td>
                      <td className="w-32">
                        <div className="flex gap-1" onClick={(e) => e.stopPropagation()}>
                          <button className="btn-secondary btn-xs" onClick={() => verify(row)}>Verify</button>
                          {!row.sealed && can('evidence:write') ? (
                            <button className="btn-ghost btn-xs" onClick={() => seal(row)}>Seal</button>
                          ) : null}
                        </div>
                      </td>
                    </tr>
                    {expanded === row.id ? (
                      <tr key={`${row.id}-detail`}>
                        <td colSpan={9} className="!h-auto bg-surface p-space-md">
                          <p className="label">Chain of custody</p>
                          <ol className="flex flex-col gap-1">
                            {row.chain_of_custody.map((entry, index) => (
                              <li key={index} className="mono text-on-surface-variant">
                                {dateTimeOf(entry.at)} · {entry.actor} ({entry.role}) · {entry.action}
                                {entry.sha256 ? ` · ${entry.sha256.slice(0, 16)}…` : ''}
                              </li>
                            ))}
                          </ol>
                          {row.note ? (
                            <p className="font-body-sm text-body-sm text-on-surface-variant mt-space-sm">{row.note}</p>
                          ) : null}
                          {row.retention_until ? (
                            <p className="mono text-outline mt-1">
                              retention until {dateTimeOf(row.retention_until)}
                              {row.sealed ? ' (sealed — exempt)' : ''}
                            </p>
                          ) : null}
                        </td>
                      </tr>
                    ) : null}
                  </>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </>
  )
}
