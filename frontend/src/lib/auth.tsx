/** Session context: login, current user, permission checks. */
import { ReactNode, createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import { ApiError, api, getToken, setToken, type SessionUser } from './api'

interface AuthState {
  user: SessionUser | null
  loading: boolean
  error: string | null
  login: (username: string, password: string) => Promise<void>
  logout: () => Promise<void>
  can: (permission: string) => boolean
  refresh: () => Promise<void>
}

const AuthContext = createContext<AuthState | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<SessionUser | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    if (!getToken()) {
      setUser(null)
      setLoading(false)
      return
    }
    try {
      setUser(await api.get<SessionUser>('/api/auth/me'))
      setError(null)
    } catch {
      setToken(null)
      setUser(null)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  const login = useCallback(async (username: string, password: string) => {
    setError(null)
    try {
      const result = await api.post<{ access_token: string; user: SessionUser }>('/api/auth/login', {
        username,
        password,
      })
      setToken(result.access_token)
      setUser(result.user)
    } catch (err) {
      const message = err instanceof ApiError ? err.message : String(err)
      setError(message)
      throw err
    }
  }, [])

  const logout = useCallback(async () => {
    try {
      await api.post('/api/auth/logout')
    } catch {
      /* logging out locally matters more than the server round-trip */
    }
    setToken(null)
    setUser(null)
  }, [])

  const can = useCallback(
    (permission: string) => {
      if (!user) return false
      if (user.permissions.includes('*')) return true
      if (user.permissions.includes(permission)) return true
      const prefix = permission.split(':')[0]
      return user.permissions.includes(`${prefix}:*`)
    },
    [user],
  )

  const value = useMemo(
    () => ({ user, loading, error, login, logout, can, refresh }),
    [user, loading, error, login, logout, can, refresh],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth(): AuthState {
  const context = useContext(AuthContext)
  if (!context) throw new Error('useAuth must be used inside an AuthProvider')
  return context
}
