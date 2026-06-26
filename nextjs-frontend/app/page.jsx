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
  { label: '🌲 Safe bedtime', text: 'I feel scared. Tell me a safe forest story.' },
]

const EMOTION_COLORS = {
  happy: 'text-yellow-400', joy: 'text-yellow-400', amusement: 'text-yellow-400',
  love: 'text-pink-400', fear: 'text-purple-400', surprise: 'text-orange-400',
  neutral: 'text-slate-400', sad: 'text-blue-400', sadness: 'text-blue-400',
  angry: 'text-red-400', anger: 'text-red-400', disgust: 'text-green-400',
}

export default function Home() {
  const [status, setStatus] = useState({ chunk_count: 0, last_emotion: 'neutral', user_emotion: 'neutral', voice_emotion: 'neutral', indexed: false })
  const [history, setHistory] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [loadingMsg, setLoadingMsg] = useState('')
  const [recording, setRecording] = useState(false)
  const [faceLoading, setFaceLoading] = useState(false)
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
    const audio = new Audio(`data:audio/mp3;base64,${b64}`)
    audioRef.current = audio
    audio.play().catch(() => {})
  }, [])

  const handleStoryResponse = useCallback((data) => {
    if (data.error) {
      setError(data.error)
      return
    }
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
          const res = await fetch('http://localhost:8000/api/story/voice', {
            method: 'POST',
            body: form,
          })
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
      const res = await fetch('http://localhost:8000/api/emotion/face', { method: 'POST' })
      const data = await res.json()
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
    fetchStatus()
  }

  const rebuildIndex = async () => {
    setLoadingMsg('Rebuilding index...')
    setLoading(true)
    await fetch('http://localhost:8000/api/index/rebuild', { method: 'POST' })
    fetchStatus()
    setLoading(false)
    setLoadingMsg('')
  }

  const faceEmoji = EMOJI_MAP[status.user_emotion] || '😐'
  const voiceEmoji = EMOJI_MAP[status.voice_emotion] || '🎙️'
  const faceColor = EMOTION_COLORS[status.user_emotion] || 'text-slate-400'
  const voiceColor = EMOTION_COLORS[status.voice_emotion] || 'text-slate-400'

  return (
    <div className="min-h-screen bg-[#0d0f14] text-slate-100 flex">

      {/* Sidebar */}
      <aside className="w-64 bg-[#13151c] border-r border-slate-800 flex flex-col p-5 gap-4 shrink-0">
        <div>
          <h2 className="text-xs font-semibold uppercase tracking-widest text-slate-500 mb-3">Controls</h2>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-slate-500">Chunks</span>
              <span className="font-mono text-slate-300">{status.chunk_count}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Last emotion</span>
              <span className={`font-medium ${EMOTION_COLORS[status.last_emotion] || 'text-slate-300'}`}>
                {status.last_emotion}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Wake word</span>
              <span className="text-slate-300">openwakeword</span>
            </div>
          </div>
        </div>

        <div className="space-y-2">
          <button onClick={resetStory}
            className="w-full text-left text-sm px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 transition-colors text-slate-300">
            🔄 New story
          </button>
          <button onClick={rebuildIndex} disabled={loading}
            className="w-full text-left text-sm px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 transition-colors text-slate-300 disabled:opacity-50">
            🔨 Rebuild index
          </button>
        </div>

        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-slate-500 mb-3">Quick stories</p>
          <div className="space-y-2">
            {QUICK_STORIES.map((s) => (
              <button key={s.label} onClick={() => sendText(s.text)} disabled={loading}
                className="w-full text-left text-sm px-3 py-2 rounded-lg bg-[#1a1d26] hover:bg-slate-700 transition-colors text-slate-300 disabled:opacity-50">
                {s.label}
              </button>
            ))}
          </div>
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 flex flex-col max-w-3xl mx-auto w-full px-6 py-8">

        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold tracking-tight text-white flex items-center gap-3">
            <span className="text-4xl">🎧</span>
            Emotion-Aware Story Friend
          </h1>
          <p className="text-slate-500 mt-1 text-sm">Hold the mic button to speak, or type your request below</p>
        </div>

        {/* Emotion bar */}
        <div className="flex gap-4 mb-6">
          <div className="flex-1 bg-[#13151c] rounded-xl border border-slate-800 p-4 flex items-center justify-between">
            <div>
              <p className="text-xs text-slate-500 uppercase tracking-wider mb-1">Face emotion</p>
              <p className={`text-lg font-semibold ${faceColor}`}>
                {faceEmoji} {status.user_emotion}
              </p>
            </div>
            <button onClick={detectFace} disabled={faceLoading}
              className="px-3 py-2 text-sm rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white transition-colors disabled:opacity-50 disabled:cursor-wait">
              {faceLoading ? 'Reading...' : '📷 Read Face'}
            </button>
          </div>

          <div className="flex-1 bg-[#13151c] rounded-xl border border-slate-800 p-4">
            <p className="text-xs text-slate-500 uppercase tracking-wider mb-1">Voice emotion</p>
            <p className={`text-lg font-semibold ${voiceColor}`}>
              {status.voice_emotion !== 'neutral'
                ? `${voiceEmoji} ${status.voice_emotion}`
                : '🎙️ —'}
            </p>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="mb-4 px-4 py-3 rounded-lg bg-red-900/30 border border-red-800 text-red-300 text-sm">
            {error}
          </div>
        )}

        {/* Chat history */}
        <div className="flex-1 overflow-y-auto space-y-4 mb-6 min-h-[200px]">
          {history.length === 0 && !loading && (
            <div className="text-center text-slate-600 text-sm mt-16">
              <p className="text-4xl mb-3">📖</p>
              <p>Your story will appear here.</p>
              <p className="mt-1">Speak a request or type below to begin.</p>
            </div>
          )}
          {history.map((msg, i) => (
            <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[80%] px-4 py-3 rounded-2xl text-sm leading-relaxed ${
                msg.role === 'user'
                  ? 'bg-indigo-600 text-white rounded-br-sm'
                  : 'bg-[#1a1d26] text-slate-200 border border-slate-800 rounded-bl-sm'
              }`}>
                {msg.content}
              </div>
            </div>
          ))}
          {loading && (
            <div className="flex justify-start">
              <div className="bg-[#1a1d26] border border-slate-800 px-4 py-3 rounded-2xl rounded-bl-sm text-sm text-slate-400 flex items-center gap-2">
                <span className="animate-pulse">●</span>
                <span className="animate-pulse delay-75">●</span>
                <span className="animate-pulse delay-150">●</span>
                <span className="ml-2">{loadingMsg}</span>
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        {/* Input area */}
        <div className="bg-[#13151c] border border-slate-800 rounded-2xl p-3 flex items-end gap-3">

          {/* Mic button */}
          <button
            onMouseDown={startRecording}
            onMouseUp={stopRecording}
            onTouchStart={startRecording}
            onTouchEnd={stopRecording}
            disabled={loading}
            className={`shrink-0 w-11 h-11 rounded-xl flex items-center justify-center transition-all ${
              recording
                ? 'bg-red-500 scale-110 shadow-lg shadow-red-500/30'
                : 'bg-slate-700 hover:bg-slate-600'
            } disabled:opacity-50`}
            title="Hold to record"
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
            className="flex-1 bg-transparent text-slate-200 placeholder-slate-600 text-sm resize-none outline-none leading-6 max-h-32 overflow-y-auto disabled:opacity-50"
          />

          {/* Send button */}
          <button
            onClick={() => sendText(input)}
            disabled={!input.trim() || loading}
            className="shrink-0 w-11 h-11 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center transition-colors"
          >
            ↑
          </button>
        </div>

        <p className="text-center text-xs text-slate-700 mt-3">
          Hold mic to record voice • Press Enter or ↑ to send text
        </p>
      </main>
    </div>
  )
}
