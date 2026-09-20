/** Shared primitives built on the Stitch design system. */
import { ReactNode, useEffect, useRef, useState } from 'react'
import { severityStyle, statusStyle } from '../lib/format'

export function Icon({ name, className = '' }: { name: string; className?: string }) {
  return <span className={`material-symbols-outlined ${className}`}>{name}</span>
}

export function PageHeader({
  title,
  subtitle,
  actions,
}: {
  title: string
  subtitle?: ReactNode
  actions?: ReactNode
}) {
  return (
    <div className="flex items-start justify-between gap-space-lg flex-wrap">
      <div className="min-w-0">
        <h1 className="font-headline-lg text-headline-lg text-on-surface">{title}</h1>
        {subtitle ? (
          <p className="font-body-sm text-body-sm text-on-surface-variant mt-0.5">{subtitle}</p>
        ) : null}
      </div>
      {actions ? <div className="flex items-center gap-space-sm shrink-0">{actions}</div> : null}
    </div>
  )
}

export function Card({
  title,
  actions,
  children,
  className = '',
  bodyClassName = 'p-space-md',
}: {
  title?: ReactNode
  actions?: ReactNode
  children: ReactNode
  className?: string
  bodyClassName?: string
}) {
  return (
    <section className={`card flex flex-col min-w-0 ${className}`}>
      {title ? (
        <header className="card-header shrink-0">
          <span className="card-title truncate">{title}</span>
          {actions ? <div className="flex items-center gap-space-sm shrink-0">{actions}</div> : null}
        </header>
      ) : null}
      <div className={`${bodyClassName} min-w-0 flex-1`}>{children}</div>
    </section>
  )
}

export function Kpi({
  label,
  value,
  unit,
  trend,
  footer,
  tone = 'default',
}: {
  label: string
  value: ReactNode
  unit?: string
  trend?: { text: string; tone: 'up' | 'down' | 'flat' }
  footer?: ReactNode
  tone?: 'default' | 'critical' | 'warning' | 'good'
}) {
  const accent = {
    default: 'text-on-surface',
    critical: 'text-red-700',
    warning: 'text-amber-700',
    good: 'text-emerald-700',
  }[tone]
  const trendTone = {
    up: 'text-red-700 bg-red-50',
    down: 'text-emerald-700 bg-emerald-50',
    flat: 'text-on-surface-variant bg-surface-container',
  }

  return (
    <div className="card p-space-md flex flex-col justify-between gap-1 min-w-0">
      <div className="flex items-center justify-between gap-2">
        <span className="font-label-sm text-label-sm uppercase tracking-wider text-on-surface-variant truncate">
          {label}
        </span>
        {trend ? (
          <span className={`mono px-1.5 py-0.5 rounded-full font-semibold ${trendTone[trend.tone]}`}>
            {trend.text}
          </span>
        ) : null}
      </div>
      <div className="flex items-baseline gap-1.5">
        <span className={`font-display-lg text-display-lg font-bold tracking-tight ${accent}`}>{value}</span>
        {unit ? <span className="mono text-on-surface-variant">{unit}</span> : null}
      </div>
      {footer ? <div className="mono text-on-surface-variant truncate">{footer}</div> : null}
    </div>
  )
}

export function SeverityPill({ severity, label }: { severity?: string | null; label?: string }) {
  const style = severityStyle(severity)
  return (
    <span className={`pill ${style.pill}`}>
      <span className={`pip ${style.pip}`} />
      {label ?? severity ?? 'info'}
    </span>
  )
}

export function StatusPill({ status, label, pulse }: { status?: string | null; label?: string; pulse?: boolean }) {
  const style = statusStyle(status)
  return (
    <span className={`pill ${style.pill}`}>
      <span className={`pip ${style.pip} ${pulse ? 'animate-pulse' : ''}`} />
      {label ?? status ?? 'offline'}
    </span>
  )
}

export function Toggle({
  checked,
  onChange,
  label,
  description,
  disabled,
  busy,
}: {
  checked: boolean
  onChange: (next: boolean) => void
  label?: ReactNode
  description?: ReactNode
  disabled?: boolean
  busy?: boolean
}) {
  return (
    <label className={`flex items-start gap-space-sm ${disabled ? 'opacity-60' : 'cursor-pointer'}`}>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        disabled={disabled || busy}
        onClick={() => onChange(!checked)}
        className={`relative w-9 h-5 rounded-full shrink-0 transition-colors mt-0.5 ${
          checked ? 'bg-primary-container' : 'bg-outline-variant'
        } ${disabled || busy ? 'cursor-not-allowed' : 'cursor-pointer'}`}
      >
        <span
          className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow-sm transition-all ${
            checked ? 'left-[18px]' : 'left-0.5'
          } ${busy ? 'animate-pulse' : ''}`}
        />
      </button>
      {label || description ? (
        <span className="min-w-0">
          {label ? <span className="block font-headline-sm text-headline-sm text-on-surface">{label}</span> : null}
          {description ? (
            <span className="block font-body-sm text-body-sm text-on-surface-variant">{description}</span>
          ) : null}
        </span>
      ) : null}
    </label>
  )
}

export function Loading({ label = 'Loading' }: { label?: string }) {
  return (
    <div className="flex items-center justify-center gap-2 py-10 text-on-surface-variant">
      <Icon name="progress_activity" className="animate-spin text-[18px]" />
      <span className="font-body-sm text-body-sm">{label}…</span>
    </div>
  )
}

export function Empty({
  icon = 'inbox',
  title,
  hint,
  action,
}: {
  icon?: string
  title: string
  hint?: ReactNode
  action?: ReactNode
}) {
  return (
    <div className="flex flex-col items-center justify-center text-center gap-2 py-12 px-space-lg">
      <Icon name={icon} className="text-[32px] text-outline-variant" />
      <p className="font-headline-sm text-headline-sm text-on-surface">{title}</p>
      {hint ? <p className="font-body-sm text-body-sm text-on-surface-variant max-w-md">{hint}</p> : null}
      {action ? <div className="mt-2">{action}</div> : null}
    </div>
  )
}

export function ErrorNote({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="flex items-start gap-space-sm p-space-md rounded-lg bg-red-50 border border-red-200">
      <Icon name="error" className="text-[18px] text-red-600 shrink-0 mt-0.5" />
      <div className="min-w-0 flex-1">
        <p className="font-headline-sm text-headline-sm text-red-900">Something went wrong</p>
        <p className="font-body-sm text-body-sm text-red-800 break-words">{message}</p>
      </div>
      {onRetry ? (
        <button className="btn-secondary btn-xs" onClick={onRetry}>
          Retry
        </button>
      ) : null}
    </div>
  )
}

export function InfoNote({ children, icon = 'info' }: { children: ReactNode; icon?: string }) {
  return (
    <div className="flex items-start gap-space-sm p-space-md rounded-lg bg-surface-container-low border-l-[3px] border-primary-container">
      <Icon name={icon} className="text-[18px] text-primary-container shrink-0 mt-0.5" />
      <div className="font-body-sm text-body-sm text-on-surface-variant min-w-0">{children}</div>
    </div>
  )
}

export function Modal({
  open,
  onClose,
  title,
  children,
  footer,
  width = 'max-w-2xl',
}: {
  open: boolean
  onClose: () => void
  title: ReactNode
  children: ReactNode
  footer?: ReactNode
  width?: string
}) {
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  if (!open) return null
  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-[1px]">
      <div className={`card shadow-lg w-full ${width} max-h-[88vh] flex flex-col`}>
        <header className="flex items-center justify-between h-11 px-space-lg border-b border-outline-variant/40 shrink-0">
          <h2 className="font-headline-md text-headline-md text-on-surface truncate">{title}</h2>
          <button className="btn-ghost btn-xs" onClick={onClose} aria-label="Close">
            <Icon name="close" className="text-[18px]" />
          </button>
        </header>
        <div className="p-space-lg overflow-y-auto flex-1 min-h-0">{children}</div>
        {footer ? (
          <footer className="flex items-center justify-end gap-space-sm h-14 px-space-lg border-t border-outline-variant/40 bg-surface shrink-0">
            {footer}
          </footer>
        ) : null}
      </div>
    </div>
  )
}

export function Field({
  label,
  hint,
  children,
  required,
}: {
  label: string
  hint?: ReactNode
  children: ReactNode
  required?: boolean
}) {
  return (
    <label className="block min-w-0">
      <span className="label">
        {label}
        {required ? <span className="text-error ml-0.5">*</span> : null}
      </span>
      {children}
      {hint ? <span className="block font-body-sm text-body-sm text-on-surface-variant mt-1">{hint}</span> : null}
    </label>
  )
}

export function Tabs<T extends string>({
  tabs,
  active,
  onChange,
}: {
  tabs: { key: T; label: string; count?: number }[]
  active: T
  onChange: (key: T) => void
}) {
  return (
    <div className="flex items-center gap-1 border-b border-outline-variant/40 overflow-x-auto">
      {tabs.map((tab) => (
        <button
          key={tab.key}
          onClick={() => onChange(tab.key)}
          className={`px-3 h-9 font-headline-sm text-headline-sm whitespace-nowrap border-b-2 transition-colors ${
            active === tab.key
              ? 'border-primary-container text-primary-container'
              : 'border-transparent text-on-surface-variant hover:text-on-surface'
          }`}
        >
          {tab.label}
          {tab.count !== undefined ? (
            <span className="mono ml-1.5 px-1.5 py-0.5 rounded-full bg-surface-container text-on-surface-variant">
              {tab.count}
            </span>
          ) : null}
        </button>
      ))}
    </div>
  )
}

export function Meter({ value, max = 100, tone = 'bg-primary-container' }: { value: number; max?: number; tone?: string }) {
  const pct = Math.max(0, Math.min(100, (value / max) * 100))
  return (
    <div className="w-full h-1.5 rounded-full bg-surface-container-high overflow-hidden">
      <div className={`h-full rounded-full transition-all ${tone}`} style={{ width: `${pct}%` }} />
    </div>
  )
}

/** Small toast queue - confirmations without blocking the operator. */
type Toast = { id: number; text: string; tone: 'ok' | 'error' | 'info' }
let pushToast: ((text: string, tone?: Toast['tone']) => void) | null = null

export function toast(text: string, tone: Toast['tone'] = 'ok') {
  pushToast?.(text, tone)
}

export function Toaster() {
  const [items, setItems] = useState<Toast[]>([])
  const counter = useRef(0)

  useEffect(() => {
    pushToast = (text, tone = 'ok') => {
      const id = ++counter.current
      setItems((prev) => [...prev, { id, text, tone }])
      setTimeout(() => setItems((prev) => prev.filter((t) => t.id !== id)), 5000)
    }
    return () => {
      pushToast = null
    }
  }, [])

  const tones = {
    ok: 'bg-emerald-50 border-emerald-200 text-emerald-900',
    error: 'bg-red-50 border-red-200 text-red-900',
    info: 'bg-surface-container-lowest border-outline-variant text-on-surface',
  }
  const icons = { ok: 'check_circle', error: 'error', info: 'info' }

  return (
    <div className="fixed bottom-4 right-4 z-[200] flex flex-col gap-2 max-w-sm">
      {items.map((item) => (
        <div
          key={item.id}
          className={`flex items-start gap-2 px-space-md py-space-sm rounded-lg border shadow ${tones[item.tone]}`}
        >
          <Icon name={icons[item.tone]} className="text-[16px] shrink-0 mt-0.5" />
          <span className="font-body-sm text-body-sm">{item.text}</span>
        </div>
      ))}
    </div>
  )
}
