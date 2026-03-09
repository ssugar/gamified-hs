import { useLocation, useNavigate } from 'react-router-dom'

const tabs = [
  { path: '/', label: 'Videos', icon: '🎥' },
  { path: '/record', label: 'Record', icon: '⏺️' },
]

export function BottomNav() {
  const location = useLocation()
  const navigate = useNavigate()

  return (
    <nav className="fixed bottom-0 left-0 right-0 bg-gray-900 border-t border-gray-700 z-50">
      <div className="flex justify-around items-center h-16 max-w-lg mx-auto">
        {tabs.map(tab => {
          const active = location.pathname === tab.path
          return (
            <button
              key={tab.path}
              onClick={() => navigate(tab.path)}
              className={`flex flex-col items-center justify-center w-full h-full min-h-[48px] transition-colors ${
                active ? 'text-cyan-400' : 'text-gray-400'
              }`}
            >
              <span className="text-2xl">{tab.icon}</span>
              <span className="text-xs mt-0.5">{tab.label}</span>
            </button>
          )
        })}
      </div>
    </nav>
  )
}
