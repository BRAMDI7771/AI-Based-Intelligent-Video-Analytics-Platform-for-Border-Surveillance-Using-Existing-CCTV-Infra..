import React, {useState} from 'react';
import {createRoot} from 'react-dom/client';
import {Shield, Mail, Lock, Eye, EyeOff, Moon, Sun} from 'lucide-react';
import App from './App';
import './styles.css';

function Login({onLogin, theme, setTheme}) {
  const [show,setShow]=useState(false); const [email,setEmail]=useState(''); const [password,setPassword]=useState('');
  const submit=(e)=>{e.preventDefault(); onLogin();};
  return <div className="login-page">
    <button className="theme-btn login-theme" onClick={()=>setTheme(theme==='dark'?'light':'dark')}>{theme==='dark'?<Sun size={18}/>:<Moon size={18}/>}</button>
    <div className="login-card">
      <div className="brand-mark"><Shield size={34}/></div><h1>BORDER SENTINEL</h1><p>AI SURVEILLANCE COMMAND CENTER</p>
      <div className="login-line"/><h2>Admin Login</h2><p className="muted">Sign in to access the surveillance dashboard</p>
      <form onSubmit={submit}>
        <label>Admin ID / Email</label><div className="input-wrap"><Mail size={18}/><input value={email} onChange={e=>setEmail(e.target.value)} placeholder="admin@bordersentinel.ai" required/></div>
        <label>Password</label><div className="input-wrap"><Lock size={18}/><input type={show?'text':'password'} value={password} onChange={e=>setPassword(e.target.value)} placeholder="••••••••" required/><button type="button" className="icon-btn" onClick={()=>setShow(!show)}>{show?<EyeOff size={18}/>:<Eye size={18}/>}</button></div>
        <div className="login-options"><label className="check"><input type="checkbox"/> Remember me</label><span>Secure access</span></div>
        <button className="primary-btn" type="submit">Sign In</button>
      </form><div className="demo-note">Prototype access • Any valid email/password</div>
    </div>
  </div>
}

export default function Root(){const [logged,setLogged]=useState(false);const [theme,setTheme]=useState('dark');return <div className={theme==='light'?'theme-light':''}>{logged?<App theme={theme} setTheme={setTheme} onLogout={()=>setLogged(false)}/>:<Login onLogin={()=>setLogged(true)} theme={theme} setTheme={setTheme}/>}</div>}

createRoot(document.getElementById('root')).render(<Root/>);
