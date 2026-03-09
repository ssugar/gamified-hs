import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { Layout } from './components/Layout'
import { Record } from './pages/Record'
import { Videos } from './pages/Videos'
import { Analysis } from './pages/Analysis'

function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Videos />} />
          <Route path="/record" element={<Record />} />
          <Route path="/analysis/:filename" element={<Analysis />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  )
}

export default App
