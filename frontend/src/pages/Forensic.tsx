/** Forensic Search — natural language over the incident record, with citations. */
import { FormEvent, useEffect, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { Card, Empty, ErrorNote, Icon, InfoNote, Loading, PageHeader, SeverityPill, toast } from '../components/ui'
import { ApiError, api, type Incident } from '../lib/api'
import { behaviorIcon, behaviorLabel, dateTimeOf, relativeTime, titleCase } from '../lib/format'
import { useApi } from '../lib/hooks'

interface SearchResult {
  question: string
  answer: string
  answer_source: string
  parsed_query: {
    behaviors: string[]
    severities: string[]
    zone_hints: string[]
    start: string | null
    end: string | null
    intent: string
    limit: number
  }
  citations: {
    incident_id: string
    camera_name: string | null
    severity: string | null
    risk_score: number | null
    started_at: string | null
    relevance: number
  }[]
  incidents: Incident[]
  result_count: number
  store_backend: string
}

export default function ForensicPage() {
  const [params, setParams] = useSearchParams()
  const [query, setQuery] = useState(params.get('q') ?? '')
  const [result, setResult] = useState<SearchResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const { data: suggestions } = useApi<{ suggestions: string[] }>('/api/forensic/suggestions')

  useEffect(() => {
    document.title = 'Forensic Search · Eagles Eye'
  }, [])

  useEffect(() => {
    const q = params.get('q')
    if (q && q !== result?.question) run(q)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [params])

  async function run(text: string) {
    setBusy(true)
    setError(null)
    try {
      setResult(await api.post<SearchResult>('/api/forensic/search', { query: text }))
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  function submit(e: FormEvent) {
    e.preventDefault()
    if (!query.trim()) return
    setParams({ q: query.trim() })
    run(query.trim())
  }

  async function reindex() {
    try {
      const out = await api.post<{ indexed: number }>('/api/forensic/reindex')
      toast(`${out.indexed} incidents indexed`)
      if (result) run(result.question)
    } catch (err) {
      toast(err instanceof ApiError ? err.message : String(err), 'error')
    }
  }

  const parsed = result?.parsed_query

  return (
    <>
      <PageHeader
        title="Forensic Search"
        subtitle="Ask in plain language — answers are built only from retrieved evidence"
        actions={
          <button className="btn-secondary" onClick={reindex} title="Rebuild the vector index from the incident table">
            <Icon name="sync" className="text-[16px]" />
            Reindex
          </button>
        }
      />

      <Card>
        <form onSubmit={submit} className="flex flex-col gap-space-md">
          <div className="relative flex items-center">
            <Icon name="manage_search" className="absolute left-2.5 text-[18px] text-on-surface-variant" />
            <input
              ref={inputRef}
              className="input pl-9 h-10"
              placeholder="Show me all incidents near Gate 4 between 7 PM and 9 PM"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            <button className="btn-primary absolute right-1.5 h-7" disabled={busy || !query.trim()}>
              {busy ? <Icon name="progress_activity" className="animate-spin text-[16px]" /> : 'Search'}
            </button>
          </div>

          <div className="flex flex-wrap gap-space-sm">
            {(suggestions?.suggestions ?? []).map((text) => (
              <button
                key={text}
                type="button"
                className="pill bg-surface-container border-outline-variant text-on-surface-variant hover:bg-surface-container-high normal-case tracking-normal"
                onClick={() => {
                  setQuery(text)
                  setParams({ q: text })
                  run(text)
                }}
              >
                {text}
              </button>
            ))}
          </div>
        </form>
      </Card>

      {error ? <ErrorNote message={error} onRetry={() => run(query)} /> : null}
      {busy && !result ? <Loading label="Searching" /> : null}

      {result ? (
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-space-lg items-start">
          <div className="xl:col-span-2 flex flex-col gap-space-lg">
            <Card title="Answer">
              <p className="font-body-lg text-body-lg whitespace-pre-wrap">{result.answer}</p>
              <p className="mono text-outline mt-space-md pt-space-md border-t border-outline-variant/40">
                source: {result.answer_source} · retrieval: {result.store_backend} · {result.result_count} match(es)
              </p>
            </Card>

            {result.incidents.length > 0 ? (
              <Card title="Matching incidents" bodyClassName="p-0">
                <div className="table-wrap">
                  <table className="tbl">
                    <thead>
                      <tr>
                        <th>Incident</th>
                        <th>Camera</th>
                        <th>Risk</th>
                        <th>Severity</th>
                        <th>Started</th>
                        <th>Relevance</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.incidents.map((incident) => {
                        const citation = result.citations.find((c) => c.incident_id === incident.id)
                        return (
                          <tr key={incident.id}>
                            <td className="max-w-sm">
                              <Link to={`/incidents/${incident.id}`} className="flex items-center gap-2 min-w-0 group">
                                <Icon name={behaviorIcon(incident.event_type)} className="text-[16px] text-on-surface-variant shrink-0" />
                                <span className="truncate group-hover:text-primary-container">{incident.title}</span>
                              </Link>
                            </td>
                            <td className="mono text-on-surface-variant truncate max-w-[10rem]">
                              {String((incident.meta as any)?.camera_name ?? incident.camera_id ?? '—')}
                            </td>
                            <td className="mono tabular-nums">{incident.risk_score.toFixed(0)}</td>
                            <td><SeverityPill severity={incident.severity} /></td>
                            <td className="mono" title={dateTimeOf(incident.started_at)}>
                              {relativeTime(incident.started_at)}
                            </td>
                            <td className="mono tabular-nums">
                              {citation ? citation.relevance.toFixed(2) : '—'}
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </Card>
            ) : (
              <Card>
                <Empty
                  icon="search_off"
                  title="No incidents matched"
                  hint="Widen the time window, relax the filters, or reindex if incidents were recorded before the index was built."
                />
              </Card>
            )}
          </div>

          <Card title="How the query was read">
            {!parsed ? null : (
              <dl className="flex flex-col gap-space-md">
                <Row label="Intent" value={titleCase(parsed.intent)} />
                <Row
                  label="Event types"
                  value={parsed.behaviors.length ? parsed.behaviors.map(behaviorLabel).join(', ') : 'any'}
                />
                <Row label="Severities" value={parsed.severities.join(', ') || 'any'} />
                <Row label="Location hints" value={parsed.zone_hints.join(', ') || 'none'} />
                <Row
                  label="Time window"
                  value={parsed.start && parsed.end ? `${dateTimeOf(parsed.start)} → ${dateTimeOf(parsed.end)}` : 'unbounded'}
                />
              </dl>
            )}
            <InfoNote icon="fact_check">
              Filters are extracted deterministically before retrieval, so an answer can never invent
              an incident that is not in the store.
            </InfoNote>
          </Card>
        </div>
      ) : null}
    </>
  )
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0">
      <dt className="label">{label}</dt>
      <dd className="font-body-sm text-body-sm text-on-surface break-words">{value}</dd>
    </div>
  )
}
