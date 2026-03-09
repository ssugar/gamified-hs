import type { ReactNode } from 'react'
import { BottomNav } from './BottomNav'

export function Layout({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen bg-gray-950 text-white">
      <main className="pb-20 max-w-lg mx-auto px-4 pt-4">
        {children}
      </main>
      <BottomNav />
    </div>
  )
}
