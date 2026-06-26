'use client'

import { useState, useEffect, useRef, useCallback } from 'react'

const EMOJI_MAP = {
  happy: '😊', joy: '😊', amusement: '😄', love: '🥰',
  fear: '😱', surprise: '😮', neutral: '😐',
  sad: '😢', sadness: '😢', angry: '😠', anger: '😠', disgust: '🤢',
}

const QUICK_STORIES = [
  { label: '🐰 Sad bunny', text: 'I feel sad. Tell me a soft bunny story.' },
  { label: '🐉 Happy dragon', text: 'I feel happy! Tell me a playful dragon story.' },
  { label: '🌙 Safe bedtime', text: 'I feel scared. Tell me a safe forest story.' },
]

const EMOTION_COLORS = {
  happy: 'text-yellow-400', joy: 'text-yellow-400', amusement: 'text-yellow-400',
  love: 'text-pink-400', fear: 'text-purple-400', surprise: 'text-orange-400',
  neutral: 'text-slate-400', sad: 'text-blue-400', sadness: 'text-blue-400',
  angry: 'text-red-400', anger: 'text-red-400', disgust: 'text-green-400',
}

export default function Home() {
  const [status, setStatus] = useState({ last_emotion: 'neutral', user_emotion: 'neutral', voice_emotion: 'neutral' })
  const [history, setHistory] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [loadingMsg, setLoadingMsg] = useState('')
  const [recording, setRecording] = useState(false)
  const [faceLoading, setFaceLoading] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [error, setError] = useState('')
  const mediaRecorder = useRef(null)
  const audioChunks = useRef([])
  const chatEndRef = useRef(null)
  const audioRef = useRef(null)

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch('http://localhost:8000/api/status')
      const data = await res.json()
      setStatus(data)
    } catch {}
  }, [])

  useEffect(() => {
    fetchStatus()
    const interval = setInterval(fetchStatus, 3000)
    return () => clearInterval(interval)
  }, [fetchStatus])

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [history])

  const playAudio = useCallback((b64) => {
    if (audioRef.current) { audioRef.current.pause() }
    const audio = new Audio(`data:audio/mp3;base64,${b64}`)
    audioRef.current = audio
    audio.play().catch(() => {})
  }, [])

  const handleStoryResponse = useCallback((data) => {
    if (data.error) { setError(data.error); return }
    setHistory(data.history || [])
    if (data.audio_b64) playAudio(data.audio_b64)
    fetchStatus()
  }, [playAudio, fetchStatus])

  const sendText = async (text) => {
    if (!text.trim() || loading) return
    setError('')
    setLoading(true)
    setLoadingMsg('Crafting your story...')
    try {
      const res = await fetch('http://localhost:8000/api/story/text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      })
      const data = await res.json()
      handleStoryResponse(data)
    } catch {
      setError('Could not connect to the story server.')
    } finally {
      setLoading(false)
      setLoadingMsg('')
      setInput('')
    }
  }

  const startRecording = async () => {
    setError('')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mr = new MediaRecorder(stream)
      mediaRecorder.current = mr
      audioChunks.current = []
      mr.ondataavailable = (e) => audioChunks.current.push(e.data)
      mr.onstop = async () => {
        stream.getTracks().forEach(t => t.stop())
        const blob = new Blob(audioChunks.current, { type: 'audio/webm' })
        setLoading(true)
        setLoadingMsg('Processing your voice...')
        const form = new FormData()
        form.append('audio', blob, 'recording.webm')
        try {
          const res = await fetch('http://localhost:8000/api/story/voice', { method: 'POST', body: form })
          const data = await res.json()
          handleStoryResponse(data)
        } catch {
          setError('Voice processing failed.')
        } finally {
          setLoading(false)
          setLoadingMsg('')
        }
      }
      mr.start()
      setRecording(true)
    } catch {
      setError('Microphone access denied.')
    }
  }

  const stopRecording = () => {
    mediaRecorder.current?.stop()
    setRecording(false)
  }

  const detectFace = async () => {
    setFaceLoading(true)
    setError('')
    try {
      await fetch('http://localhost:8000/api/emotion/face', { method: 'POST' })
      fetchStatus()
    } catch {
      setError('Face detection failed.')
    } finally {
      setFaceLoading(false)
    }
  }

  const resetStory = async () => {
    await fetch('http://localhost:8000/api/story/reset', { method: 'POST' })
    setHistory([])
    setSidebarOpen(false)
    fetchStatus()
  }

  const faceColor = EMOTION_COLORS[status.user_emotion] || 'text-slate-400'
  const voiceColor = EMOTION_COLORS[status.voice_emotion] || 'text-slate-400'
  const faceEmoji = EMOJI_MAP[status.user_emotion] || '😐'
  const voiceEmoji = EMOJI_MAP[status.voice_emotion] || '🎙️'

  return (
    <div className="min-h-screen bg-[#0c0e13] text-slate-100 flex flex-col md:flex-row">

      {/* Mobile topbar */}
      <div className="md:hidden flex items-center justify-between px-4 py-3 bg-[#11131a] border-b border-white/5">
        <span className="font-semibold text-sm tracking-tight">🎧 Story Friend</span>
        <button
          onClick={() => setSidebarOpen(!sidebarOpen)}
          className="text-slate-400 hover:text-white transition-colors text-lg"
        >
          {sidebarOpen ? '✕' : '☰'}
        </button>
      </div>

      {/* Sidebar */}
      <aside className={`
        ${sidebarOpen ? 'flex' : 'hidden'} md:flex
        w-full md:w-56 bg-[#11131a] border-b md:border-b-0 md:border-r border-white/5
        flex-col p-4 gap-5 shrink-0
      `}>

        {/* Emotion status */}
        <div className="space-y-2">
          <p className="text-[10px] font-semibold uppercase tracking-widest text-slate-600">Current mood</p>
          <div className="flex items-center gap-2 text-sm">
            <span className={faceColor}>{faceEmoji}</span>
            <span className="text-slate-400 capitalize">{status.user_emotion}</span>
            <span className="text-slate-700 text-xs ml-auto">face</span>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <span className={voiceColor}>{status.voice_emotion !== 'neutral' ? voiceEmoji : '🎙️'}</span>
            <span className="text-slate-400 capitalize">{status.voice_emotion}</span>
            <span className="text-slate-700 text-xs ml-auto">voice</span>
          </div>
        </div>

        <div className="h-px bg-white/5" />

        {/* Wake word info */}
        <div className="space-y-1">
          <p className="text-[10px] font-semibold uppercase tracking-widest text-slate-600">Hands-free</p>
          <p className="text-xs text-slate-500">
            Say <span className="text-indigo-400 font-medium">"Hey Mycroft"</span> to record without touching your keyboard
          </p>
          <p className="text-[10px] text-slate-700 mt-1">powered by openwakeword</p>
        </div>

        <div className="h-px bg-white/5" />

        {/* Actions */}
        <div className="space-y-2">
          <p className="text-[10px] font-semibold uppercase tracking-widest text-slate-600">Actions</p>
          <button
            onClick={detectFace}
            disabled={faceLoading}
            className="w-full text-left text-sm px-3 py-2 rounded-xl bg-white/5 hover:bg-white/10 transition-colors text-slate-300 disabled:opacity-40"
          >
            {faceLoading ? '⏳ Reading...' : '📷 Scan face'}
          </button>
          <button
            onClick={resetStory}
            className="w-full text-left text-sm px-3 py-2 rounded-xl bg-white/5 hover:bg-white/10 transition-colors text-slate-300"
          >
            🔄 New story
          </button>
        </div>

        <div className="h-px bg-white/5" />

        {/* Quick stories */}
        <div className="space-y-2">
          <p className="text-[10px] font-semibold uppercase tracking-widest text-slate-600">Quick stories</p>
          {QUICK_STORIES.map((s) => (
            <button
              key={s.label}
              onClick={() => { sendText(s.text); setSidebarOpen(false) }}
              disabled={loading}
              className="w-full text-left text-sm px-3 py-2 rounded-xl bg-white/5 hover:bg-white/10 transition-colors text-slate-300 disabled:opacity-40"
            >
              {s.label}
            </button>
          ))}
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 flex flex-col h-[calc(100vh-48px)] md:h-screen min-h-0">

        {/* Header desktop only */}
        <div className="hidden md:block px-6 pt-7 pb-3">
          <h1 className="text-2xl font-bold tracking-tight text-white">
            🎧 Emotion-Aware Story Friend
          </h1>
          <p className="text-slate-500 text-sm mt-0.5">
            Speak or type — stories adapt to how you feel
          </p>
        </div>

        {/* Error */}
        {error && (
          <div className="mx-4 md:mx-6 mt-3 px-4 py-2.5 rounded-2xl bg-red-500/10 border border-red-500/20 text-red-300 text-sm">
            {error}
          </div>
        )}

        {/* Chat area */}
        <div className="flex-1 overflow-y-auto px-4 md:px-6 py-4 space-y-4 min-h-0">
          {history.length === 0 && !loading && (
            <div className="flex flex-col items-center justify-center h-full text-center gap-2 py-20">
              <span className="text-5xl">📖</span>
              <p className="text-slate-500 text-sm mt-2">Your story will appear here</p>
              <p className="text-slate-600 text-xs">Type a request or hold the mic to speak</p>
            </div>
          )}

          {history.map((msg, i) => (
            <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`
                max-w-[85%] md:max-w-[75%] px-4 py-3 text-sm leading-relaxed
                ${msg.role === 'user'
                  ? 'bg-indigo-600 text-white rounded-3xl rounded-br-md'
                  : 'bg-[#181b24] text-slate-200 border border-white/5 rounded-3xl rounded-bl-md'
                }
              `}>
                {msg.content}
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex justify-start">
              <div className="bg-[#181b24] border border-white/5 px-5 py-3.5 rounded-3xl rounded-bl-md text-sm text-slate-500 flex items-center gap-1.5">
                <span className="animate-bounce" style={{ animationDelay: '0ms' }}>·</span>
                <span className="animate-bounce" style={{ animationDelay: '150ms' }}>·</span>
                <span className="animate-bounce" style={{ animationDelay: '300ms' }}>·</span>
                <span className="ml-2 text-slate-600 text-xs">{loadingMsg}</span>
              </div>
            </div>
          )}

          <div ref={chatEndRef} />
        </div>

        {/* Input bar */}
        <div className="px-4 md:px-6 py-4 border-t border-white/5">
          <div className="flex items-center gap-2 bg-[#181b24] rounded-3xl px-3 py-3 border border-white/5">

            {/* Mic */}
            <button
              onMouseDown={startRecording}
              onMouseUp={stopRecording}
              onTouchStart={startRecording}
              onTouchEnd={stopRecording}
              disabled={loading}
              className={`
                shrink-0 w-10 h-10 rounded-2xl flex items-center justify-center text-base transition-all
                ${recording
                  ? 'bg-red-500 shadow-lg shadow-red-500/25 scale-110'
                  : 'bg-white/5 hover:bg-white/10 text-slate-400'
                }
                disabled:opacity-40
              `}
            >
              {recording ? '⏹' : '🎙️'}
            </button>

            {/* Text input */}
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  sendText(input)
                }
              }}
              placeholder="Tell me a story about a rabbit, dragon, forest..."
              rows={1}
              disabled={loading}
              className="flex-1 bg-transparent text-slate-200 placeholder-slate-600 text-sm resize-none outline-none leading-6 max-h-28 overflow-y-auto py-3 disabled:opacity-40"
            />

            {/* Send */}
            <button
              onClick={() => sendText(input)}
              disabled={!input.trim() || loading}
              className="shrink-0 w-10 h-10 rounded-2xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center text-white transition-colors text-sm font-medium"
            >
              ↑
            </button>
          </div>
          <p className="text-center text-[10px] text-slate-700 mt-2">
            Hold mic · Press Enter to send · Say "Hey Mycroft" hands-free
          </p>
        </div>
      </main>
    </div>
  )
}
