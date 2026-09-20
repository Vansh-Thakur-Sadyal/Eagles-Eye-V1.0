/** Conversational investigation — multi-turn, evidence-grounded, with citations. */
import { FormEvent, useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { Card, Empty, Icon, InfoNote, PageHeader } from '../components/ui'
import { ApiError, api } from '../lib/api'
import { relativeTime } from '../lib/format'
import { useApi } from '../lib/hooks'

interface Turn {
  role: 'user' | 'assistant'
  text: string
  at: string
  source?: string
  latency?: number
  citations?: { incident_id: string; camera_name: string | null; severity: string | null; started_at: string | null }[]
}

export default function AssistantPage() {
  const [turns, setTurns] = useState<Turn[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const endRef = useRef<HTMLDivElement>(null)

  const { data: suggestions } = useApi<{ suggestions: string[] }>('/api/forensic/suggestions')
  const { data: llm } = useApi<{ provider: string; model: string | null; enabled: boolean }>('/api/system/llm/health')

  useEffect(() => {
    document.title = 'AI Assistant · Eagles Eye'
  }, [])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [turns])

  async function send(text: string) {
    if (!text.trim() || busy) return
    const now = new Date().toISOString()
    setTurns((prev) => [...prev, { role: 'user', text, at: now }])
    setInput('')
    setBusy(true)
    try {
      const reply = await api.post<{
        session_id: string
        answer: string
        answer_source: string
        latency_ms: number
        citations: Turn['citations']
      }>('/api/forensic/chat', { message: text, session_id: sessionId })
      setSessionId(reply.session_id)
      setTurns((prev) => [
        ...prev,
        {
          role: 'assistant',
          text: reply.answer,
          at: new Date().toISOString(),
          source: reply.answer_source,
          latency: reply.latency_ms,
          citations: reply.citations,
        },
      ])
    } catch (err) {
      setTurns((prev) => [
        ...prev,
        {
          role: 'assistant',
          text: err instanceof ApiError ? err.message : String(err),
          at: new Date().toISOString(),
          source: 'error',
        },
      ])
    } finally {
      setBusy(false)
    }
  }

  function submit(e: FormEvent) {
    e.preventDefault()
    send(input.trim())
  }

  return (
    <>
      <PageHeader
        title="AI Assistant"
        subtitle="Ask about what the cameras have recorded — answers cite the incidents they came from"
        actions={
          turns.length > 0 ? (
            <button
              className="btn-secondary"
              onClick={() => {
                setTurns([])
                setSessionId(null)
              }}
            >
              <Icon name="restart_alt" className="text-[16px]" />
              New session
            </button>
          ) : null
        }
      />

      {llm && !llm.enabled ? (
        <InfoNote icon="offline_bolt">
          No language model is configured, so answers are composed deterministically from the
          retrieved evidence. That keeps the assistant fully offline and unable to invent anything.
          Configure a provider in Settings for more natural phrasing.
        </InfoNote>
      ) : null}

      <Card bodyClassName="p-0" className="min-h-[60vh] flex flex-col">
        <div className="flex-1 overflow-y-auto p-space-lg flex flex-col gap-space-lg">
          {turns.length === 0 ? (
            <Empty
              icon="auto_awesome"
              title="Start an investigation"
              hint="Ask about incidents, locations, time windows, crowd risk or a subject's movement."
              action={
                <div className="flex flex-wrap gap-space-sm justify-center max-w-2xl">
                  {(suggestions?.suggestions ?? []).map((text) => (
                    <button
                      key={text}
                      className="pill bg-surface-container border-outline-variant text-on-surface-variant hover:bg-surface-container-high normal-case tracking-normal"
                      onClick={() => send(text)}
                    >
                      {text}
                    </button>
                  ))}
                </div>
              }
            />
          ) : (
            turns.map((turn, index) => (
              <div
                key={`${turn.at}-${index}`}
                className={`flex gap-space-md ${turn.role === 'user' ? 'justify-end' : ''}`}
              >
                {turn.role === 'assistant' ? (
                  <div className="w-7 h-7 rounded-full bg-primary-container flex items-center justify-center shrink-0">
                    <Icon name="smart_toy" className="text-on-primary text-[16px]" />
                  </div>
                ) : null}
                <div className={`max-w-2xl min-w-0 ${turn.role === 'user' ? 'order-1' : ''}`}>
                  <div
                    className={`rounded-xl px-space-md py-space-sm ${
                      turn.role === 'user'
                        ? 'bg-primary-container text-on-primary'
                        : turn.source === 'error'
                          ? 'bg-red-50 border border-red-200 text-red-900'
                          : 'bg-surface-container-low border border-outline-variant/40'
                    }`}
                  >
                    <p className="font-body-lg text-body-lg whitespace-pre-wrap">{turn.text}</p>
                  </div>

                  {turn.citations && turn.citations.length > 0 ? (
                    <div className="flex flex-wrap gap-1.5 mt-1.5">
                      {turn.citations.map((citation) => (
                        <Link
                          key={citation.incident_id}
                          to={`/incidents/${citation.incident_id}`}
                          className="pill bg-surface-container border-outline-variant text-on-surface-variant hover:bg-surface-container-high"
                        >
                          <Icon name="link" className="text-[12px]" />
                          {citation.incident_id}
                        </Link>
                      ))}
                    </div>
                  ) : null}

                  <p className={`mono text-outline mt-1 ${turn.role === 'user' ? 'text-right' : ''}`}>
                    {relativeTime(turn.at)}
                    {turn.source && turn.source !== 'error' ? ` · ${turn.source}` : ''}
                    {turn.latency ? ` · ${turn.latency.toFixed(0)} ms` : ''}
                  </p>
                </div>
              </div>
            ))
          )}
          {busy ? (
            <div className="flex items-center gap-2 text-on-surface-variant">
              <Icon name="progress_activity" className="animate-spin text-[18px]" />
              <span className="font-body-sm text-body-sm">Retrieving evidence…</span>
            </div>
          ) : null}
          <div ref={endRef} />
        </div>

        <form onSubmit={submit} className="border-t border-outline-variant/40 p-space-md flex gap-space-sm">
          <input
            className="input flex-1 h-10"
            placeholder="Ask about an incident, a location, a time window…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={busy}
          />
          <button className="btn-primary h-10 px-4" disabled={busy || !input.trim()}>
            <Icon name="send" className="text-[16px]" />
            Send
          </button>
        </form>
      </Card>
    </>
  )
}
