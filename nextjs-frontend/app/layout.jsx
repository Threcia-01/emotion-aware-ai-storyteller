import './globals.css'

export const metadata = {
  title: 'Emotion-Aware Story Friend',
  description: 'An AI storyteller that adapts to your emotions',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
