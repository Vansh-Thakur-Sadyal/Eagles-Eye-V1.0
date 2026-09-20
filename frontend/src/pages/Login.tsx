import { FormEvent, useEffect, useState } from 'react'
import { Navigate, useNavigate } from 'react-router-dom'
import { Icon } from '../components/ui'
import { useAuth } from '../lib/auth'

export default function LoginPage() {
  const { login, user, loading, error } = useAuth()
  const navigate = useNavigate()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    document.title = 'Sign in · Eagles Eye'
  }, [])

  if (!loading && user) return <Navigate to="/" replace />

  async function submit(e: FormEvent) {
    e.preventDefault()
    setBusy(true)
    try {
      await login(username.trim(), password)
      navigate('/')
    } catch {
      /* the error is surfaced from context */
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="min-h-screen grid lg:grid-cols-2">
      <div className="hidden lg:flex flex-col justify-between p-space-xl bg-inverse-surface text-inverse-on-surface">
        <div className="flex items-center gap-space-md">
          <div className="w-9 h-9 rounded-lg bg-primary-container flex items-center justify-center">
            <Icon name="shield_person" className="text-on-primary text-[22px]" />
          </div>
          <div className="leading-tight">
            <p className="font-headline-md text-headline-md tracking-wider">EAGLES EYE</p>
            <p className="font-label-sm text-label-sm uppercase tracking-widest opacity-70">
              Public Safety Intelligence Platform
            </p>
          </div>
        </div>

        <div className="max-w-md">
          <p className="font-display-lg text-display-lg leading-tight">
            See. Track. Understand. Correlate. Assess. Explain.
          </p>
          <p className="font-body-lg text-body-lg opacity-80 mt-space-md">
            Eagles Eye turns distributed CCTV into one spatio-temporal intelligence network. Subjects
            stay anonymous by default, every association carries a confidence, and the system
            recommends — a human decides.
          </p>
        </div>

        <ul className="flex flex-col gap-2 font-body-sm text-body-sm opacity-75">
          {[
            'Anonymous-by-default tracking with bystander redaction',
            'Explainable risk scoring, never a bare number',
            'Identity escalation gated behind human approval',
            'Every identity-sensitive read written to the audit log',
          ].map((line) => (
            <li key={line} className="flex items-center gap-2">
              <Icon name="check_circle" className="text-[16px] text-emerald-400" />
              {line}
            </li>
          ))}
        </ul>
      </div>

      <div className="flex items-center justify-center p-space-xl bg-background">
        <form onSubmit={submit} className="card p-space-xl w-full max-w-sm flex flex-col gap-space-lg">
          <div className="lg:hidden flex items-center gap-space-sm">
            <div className="w-8 h-8 rounded-lg bg-primary-container flex items-center justify-center">
              <Icon name="shield_person" className="text-on-primary text-[20px]" />
            </div>
            <span className="font-headline-md text-headline-md tracking-wider">EAGLES EYE</span>
          </div>

          <div>
            <h1 className="font-headline-lg text-headline-lg text-on-surface">Operator sign-in</h1>
            <p className="font-body-sm text-body-sm text-on-surface-variant mt-0.5">
              Authorised personnel only. Sessions are logged.
            </p>
          </div>

          <label className="block">
            <span className="label">Username</span>
            <input
              className="input"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              autoComplete="username"
              autoFocus
              required
            />
          </label>

          <label className="block">
            <span className="label">Password</span>
            <input
              className="input"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="current-password"
              required
            />
          </label>

          {error ? (
            <div className="flex items-start gap-2 p-space-sm rounded-lg bg-red-50 border border-red-200">
              <Icon name="error" className="text-[16px] text-red-600 shrink-0 mt-0.5" />
              <span className="font-body-sm text-body-sm text-red-800">{error}</span>
            </div>
          ) : null}

          <button className="btn-primary w-full" disabled={busy}>
            {busy ? <Icon name="progress_activity" className="animate-spin text-[16px]" /> : null}
            {busy ? 'Signing in' : 'Sign in'}
          </button>

          <p className="font-body-sm text-body-sm text-on-surface-variant border-t border-outline-variant/40 pt-space-md">
            First run? The generated administrator password is printed in the API log and saved to{' '}
            <code className="mono">storage/FIRST_RUN_CREDENTIALS.txt</code>. Change it, then delete
            that file.
          </p>
        </form>
      </div>
    </div>
  )
}
