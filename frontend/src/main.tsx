import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App'
import { AuthProvider } from './lib/auth'
import './styles/index.css'
import { setToken } from './lib/api'

// The portal hands the session over as #token=... after a verified sign-in.
// Consume it once, then scrub it from the address bar so it is not left in
// history or copied into a shared link.
const handover = new URLSearchParams(window.location.hash.replace(/^#/, ''))
const handedToken = handover.get('token')
if (handedToken) {
  setToken(handedToken)
  window.history.replaceState({}, '', '/app/')
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter basename="/app">
      <AuthProvider>
        <App />
      </AuthProvider>
    </BrowserRouter>
  </React.StrictMode>,
)
