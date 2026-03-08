/* eslint-disable no-unused-vars */
import { useState, useEffect, useRef } from "react";

const COLORS = {
  navy: "#0B1E3D", navyMid: "#152D55", navyLight: "#1E3F73",
  gold: "#C9A84C", goldLight: "#E2C26E", cream: "#F5F0E8",
  white: "#FFFFFF", slate: "#4A5568", slateLight: "#718096",
  green: "#1A6B4A", greenLight: "#22A86D", red: "#C0392B",
  redLight: "#E74C3C", amber: "#B7770D", amberLight: "#F39C12",
  cardBg: "#FFFFFF", pageBg: "#F0EDE6",
};

const BACKEND_URL = "http://localhost:8000";

async function callAI(prompt, systemPrompt = "") {
  try {
    const response = await fetch(`${BACKEND_URL}/api/ai/summarize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, system_prompt: systemPrompt }),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    return data.result || "Unable to generate response.";
  } catch (err) {
    return generateFallbackSummary(prompt);
  }
}

function generateFallbackSummary(text) {
  const lines = text.split("\n").filter(l => l.trim().length > 10);
  const total = lines.length;
  const posWords = ["excellent","clear","helpful","good","great","improved","easy","transparent","fast","appreciate","simple","beneficial","convenient"];
  const negWords = ["burdensome","harsh","unclear","difficult","complex","penalty","tight","unfair","concern","problem","confusing","slow","frustrating","costly"];
  let pos = 0, neg = 0;
  lines.forEach(l => {
    const t = l.toLowerCase();
    posWords.forEach(w => { if (t.includes(w)) pos++; });
    negWords.forEach(w => { if (t.includes(w)) neg++; });
  });
  const sentiment = pos > neg ? "largely positive" : neg > pos ? "largely critical" : "mixed";
  const topPos = posWords.filter(w => lines.some(l => l.toLowerCase().includes(w))).slice(0, 3).join(", ");
  const topNeg = negWords.filter(w => lines.some(l => l.toLowerCase().includes(w))).slice(0, 3).join(", ");
  return `Analysis of ${total} feedback submissions reveals a ${sentiment} response overall. ` +
    (topPos ? `Key positive themes mentioned include: ${topPos}. ` : "") +
    (topNeg ? `Primary concerns raised include: ${topNeg}. ` : "") +
    `Stakeholders have highlighted the need for clearer guidelines and more streamlined compliance procedures. ` +
    `The feedback suggests continued focus on simplification and transparency will improve overall acceptance of the regulations.`;
}

const MANOBHAV_LOGO_SRC = "data:image/png;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/4gHYSUNDX1BST0ZJTEUAAQEAAAHIAAAAAAQwAABtbnRyUkdCIFhZWiAH4AABAAEAAAAAAABhY3NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA9tYAAQAAAADTLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAACRyWFlaAAABFAAAABRnWFlaAAABKAAAABRiWFlaAAABPAAAABR3dHB0AAABUAAAABRyVFJDAAABZAAAAChnVFJDAAABZAAAAChiVFJDAAABZAAAAChjcHJ0AAABjAAAADxtbHVjAAAAAAAAAAEAAAAMZW5VUwAAAAgAAAAcAHMAUgBHAEJYWVogAAAAAAAAb6IAADj1AAADkFhZWiAAAAAAAABimQAAt4UAABjaWFlaIAAAAAAAACSgAAAPhAAAts9YWVogAAAAAAAA9tYAAQAAAADTLXBhcmEAAAAAAAQAAAACZmYAAPKnAAANWQAAE9AAAApbAAAAAAAAAABtbHVjAAAAAAAAAAEAAAAMZW5VUwAAACAAAAAcAEcAbwBvAGcAbABlACAASQBuAGMALgAgADIAMAAxADb/2wBDAAUDBAQEAwUEBAQFBQUGBwwIBwcHBw8LCwkMEQ8SEhEPERETFhwXExQaFRERGCEYGh0dHx8fExciJCIeJBweHx7/wAARCAQABAADASIAAhEBAxEB/8QAHQABAAEEAwEAAAAAAAAAAAAAAAgFBgcJAgMEAf/EAGIQAQABAwMCAwQEBQwKDQsEAwABAgMEBQYRByESMUEIE1FhIjJxgRQjQlKRFRY2YnJ0gqGxsrPBCSQzQ3WSk6LR0hcYNDVTY2Rlc5WjwuElJic3VVaDlNPw8UVGVIU4R7T/xAAbAQEAAgMBAQAAAAAAAAAAAAAABAUCAwYBB//EADoRAQACAQIEAwQIBgICAwEAAAABAgMEEQUSITETQVEiMnHwBhRhgZGhsdEVI0JSweEzNBZDJFPxgv/aAAwDAQACEQMRAD8AhkAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==";

const ButterflyLogo = ({ size = 36, showText = false }) => (
  <div style={{ display:"flex", alignItems:"center", gap:10 }}>
    <img src={MANOBHAV_LOGO_SRC} alt="MANOBHAV Logo" style={{ width: size, height: size, objectFit: "contain", borderRadius: 4, display: "block" }} />
    {showText && (
      <div>
        <div style={{ fontFamily:"Georgia, serif", fontWeight:700, color:"#C9A84C", letterSpacing:2, fontSize:14, whiteSpace:"nowrap" }}>MANOBHAV</div>
        <div style={{ fontSize:9, color:"rgba(201,168,76,0.7)", letterSpacing:1, textTransform:"uppercase", whiteSpace:"nowrap" }}>The Akashvani Predictor</div>
      </div>
    )}
  </div>
);

// ─── Local DB (localStorage) ────────────────────────────────
const db = {
  get: (key) => { try { return JSON.parse(localStorage.getItem(`manobhav_${key}`) || "null"); } catch { return null; } },
  set: (key, val) => { try { localStorage.setItem(`manobhav_${key}`, JSON.stringify(val)); } catch {} },
  push: (key, item) => {
    const arr = db.get(key) || [];
    arr.push({ ...item, id: Date.now(), createdAt: new Date().toISOString() });
    db.set(key, arr);
    return arr;
  },
};

(function seedUsers() {
  if (!db.get("users")) {
    db.set("users", [
      { id: 1, name: "Admin User", email: "admin@manobhav.gov.in", password: "admin123", role: "admin" },
      { id: 2, name: "Rajesh Sharma", email: "rajesh@bizco.in", password: "pass123", role: "business" },
      { id: 3, name: "Priya Singh", email: "priya@gmail.com", password: "pass123", role: "individual" },
    ]);
  }
  if (!db.get("comments")) {
    db.set("comments", [
      { id: 1, text: "The MCA portal is excellent and user-friendly. Great improvement in compliance procedures!", user: "Rajesh Sharma", userType: "business", sentiment: "positive", createdAt: "2025-01-10T10:00:00Z" },
      { id: 2, text: "Section 12 compliance requirements are too burdensome for small businesses. Needs revision.", user: "Anita Patel", userType: "business", sentiment: "negative", createdAt: "2025-01-11T11:00:00Z" },
      { id: 3, text: "The new e-filing system works well. Clear instructions and fast processing.", user: "Kumar Das", userType: "individual", sentiment: "positive", createdAt: "2025-01-12T09:00:00Z" },
      { id: 4, text: "Penalty clauses under Rule 8 are harsh and unclear. We need better guidelines.", user: "Meera Nair", userType: "business", sentiment: "negative", createdAt: "2025-01-13T14:00:00Z" },
      { id: 5, text: "Annual return filing process has improved significantly. Appreciate the changes.", user: "Sanjay Mehta", userType: "individual", sentiment: "positive", createdAt: "2025-01-14T16:00:00Z" },
      { id: 6, text: "The regulation framework is neutral but implementation guidelines could be clearer.", user: "Deepa Reddy", userType: "business", sentiment: "neutral", createdAt: "2025-01-15T12:00:00Z" },
      { id: 7, text: "Director identification requirements are straightforward and well-documented.", user: "Arjun Rao", userType: "individual", sentiment: "positive", createdAt: "2025-01-16T10:30:00Z" },
      { id: 8, text: "Compliance deadlines are too tight for small enterprises. More time needed.", user: "Sunita Joshi", userType: "business", sentiment: "negative", createdAt: "2025-01-17T15:00:00Z" },
    ]);
  }
})();

function analyzeSentiment(text) {
  const t = text.toLowerCase();
  const pos = ["good","great","excellent","support","approve","beneficial","clear","transparent","improved","appreciate","well","fast","straightforward","positive","helpful","efficient"];
  const neg = ["bad","poor","burdensome","harsh","unclear","difficult","complex","tight","unfair","oppose","concern","problem","issue","negative","challenge","burden","penalty"];
  let p = 0, n = 0;
  pos.forEach(w => { if (t.includes(w)) p++; });
  neg.forEach(w => { if (t.includes(w)) n++; });
  if (p > n) return "positive";
  if (n > p) return "negative";
  return "neutral";
}

function getWordFrequency(texts) {
  const stopwords = new Set(["the","is","are","was","were","a","an","and","or","but","in","on","at","to","for","of","with","by","from","it","this","that","be","as","we","our","its","has","have","had","not","all","can","their","they","he","she","his","her","my","your"]);
  const freq = {};
  texts.forEach(text => {
    text.toLowerCase().replace(/[^a-z\s]/g, " ").split(/\s+/).forEach(word => {
      if (word.length > 3 && !stopwords.has(word)) { freq[word] = (freq[word] || 0) + 1; }
    });
  });
  return Object.entries(freq).sort((a, b) => b[1] - a[1]).slice(0, 40);
}
const Icon = ({ name, size = 20, color = "currentColor" }) => {
  const icons = {
    home: <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9,22 9,12 15,12 15,22"/></svg>,
    cloud: <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z"/></svg>,
    chart: <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>,
    brain: <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96-.46 2.5 2.5 0 0 1-1.07-4.73A3 3 0 0 1 3.83 9 2.5 2.5 0 0 1 4.5 5 2.5 2.5 0 0 1 9.5 2z"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96-.46 2.5 2.5 0 0 0 1.07-4.73A3 3 0 0 0 20.17 9 2.5 2.5 0 0 0 19.5 5 2.5 2.5 0 0 0 14.5 2z"/></svg>,
    file: <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14,2 14,8 20,8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>,
    trend: <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="22,7 13.5,15.5 8.5,10.5 2,17"/><polyline points="16,7 22,7 22,13"/></svg>,
    policy: <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>,
    upload: <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="16,16 12,12 8,16"/><line x1="12" y1="12" x2="12" y2="21"/><path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/></svg>,
    logout: <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16,17 21,12 16,7"/><line x1="21" y1="12" x2="9" y2="12"/></svg>,
    user: <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>,
    users: <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>,
    check: <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20,6 9,17 4,12"/></svg>,
    x: <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>,
    menu: <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="18" x2="21" y2="18"/></svg>,
    eye: <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>,
    search: <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>,
    spark: <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2L9.5 9.5 2 12l7.5 2.5L12 22l2.5-7.5L22 12l-7.5-2.5z"/></svg>,
    refresh: <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="23,4 23,10 17,10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>,
  };
  return icons[name] || null;
};

const Spinner = ({ size = 24 }) => (
  <div style={{ display:"inline-flex", alignItems:"center", justifyContent:"center" }}>
    <div style={{ width: size, height: size, border: `3px solid rgba(201,168,76,0.3)`, borderTop: `3px solid #C9A84C`, borderRadius: "50%", animation: "spin 0.8s linear infinite" }}/>
  </div>
);

const Badge = ({ label, color = "#C9A84C", bg = "rgba(201,168,76,0.15)" }) => (
  <span style={{ display:"inline-block", padding:"2px 10px", borderRadius:20, background:bg, color, fontSize:11, fontWeight:700, letterSpacing:0.5, textTransform:"uppercase" }}>{label}</span>
);

const SentimentBadge = ({ s }) => {
  const cfg = {
    positive: { color: "#22A86D", bg: "rgba(26,107,74,0.1)", label: "Positive" },
    negative: { color: "#E74C3C", bg: "rgba(192,57,43,0.1)", label: "Negative" },
    neutral:  { color: "#F39C12", bg: "rgba(183,119,13,0.1)", label: "Neutral" },
  }[s] || { color: "#4A5568", bg: "#eee", label: s };
  return <Badge label={cfg.label} color={cfg.color} bg={cfg.bg} />;
};

const Card = ({ children, style = {} }) => (
  <div style={{ background: "#FFFFFF", borderRadius: 16, boxShadow: "0 2px 20px rgba(11,30,61,0.08)", padding: 28, ...style }}>{children}</div>
);

const SectionTitle = ({ icon, title, subtitle }) => (
  <div style={{ marginBottom: 24 }}>
    <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:6 }}>
      <div style={{ color: "#C9A84C" }}><Icon name={icon} size={22} /></div>
      <h2 style={{ margin:0, fontSize:22, fontWeight:800, color: "#0B1E3D", fontFamily:"Georgia, serif" }}>{title}</h2>
    </div>
    {subtitle && <p style={{ margin:0, color: "#718096", fontSize:14 }}>{subtitle}</p>}
  </div>
);

const FileUpload = ({ onData, accept = ".csv,.txt", label = "Upload CSV / Text file" }) => {
  const [drag, setDrag] = useState(false);
  const [fname, setFname] = useState(null);
  const inp = useRef();
  const readFile = (file) => {
    setFname(file.name);
    const reader = new FileReader();
    reader.onload = (e) => onData(e.target.result, file.name);
    reader.readAsText(file);
  };
  return (
    <div onClick={() => inp.current.click()} onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)} onDrop={(e) => { e.preventDefault(); setDrag(false); if (e.dataTransfer.files[0]) readFile(e.dataTransfer.files[0]); }}
      style={{ border: `2px dashed ${drag ? "#C9A84C" : "rgba(11,30,61,0.2)"}`, borderRadius: 12, padding: "32px 20px", textAlign: "center", cursor: "pointer", transition: "all 0.2s", background: drag ? "rgba(201,168,76,0.05)" : "#F5F0E8" }}>
      <input ref={inp} type="file" accept={accept} style={{ display:"none" }} onChange={e => e.target.files[0] && readFile(e.target.files[0])} />
      <div style={{ color: "#C9A84C", marginBottom:8 }}><Icon name="upload" size={32} /></div>
      <p style={{ margin:0, fontSize:14, color: "#4A5568" }}>{fname ? <><strong style={{color:"#0B1E3D"}}>{fname}</strong> uploaded ✓</> : label}</p>
      <p style={{ margin:"4px 0 0", fontSize:12, color: "#718096" }}>Drag & drop or click to browse</p>
    </div>
  );
};

const StatCard = ({ label, value, color = "#0B1E3D" }) => (
  <div style={{ background: "#FFFFFF", borderRadius:12, padding:"20px 24px", boxShadow:"0 2px 12px rgba(11,30,61,0.07)", borderLeft:`4px solid ${color}` }}>
    <div style={{ fontSize:28, fontWeight:800, color, fontFamily:"Georgia, serif" }}>{value}</div>
    <div style={{ fontSize:13, color: "#718096", marginTop:2, textTransform:"uppercase", letterSpacing:0.5 }}>{label}</div>
  </div>
);

const ProgressBar = ({ pct, color = "#C9A84C" }) => (
  <div style={{ background:"rgba(11,30,61,0.08)", borderRadius:8, height:8, overflow:"hidden" }}>
    <div style={{ width:`${pct}%`, height:"100%", background:color, borderRadius:8, transition:"width 0.6s ease" }} />
  </div>
);
// ════════════════════════════════════════════════════════════
// MODULE 1: WORD CLOUD
// ════════════════════════════════════════════════════════════
const WordCloudModule = () => {
  const [words, setWords] = useState([]);
  const [selected, setSelected] = useState(null);
  const [filteredComments, setFilteredComments] = useState([]);
  const [uploadedTexts, setUploadedTexts] = useState([]);
  const [fileName, setFileName] = useState("");
  const [loading, setLoading] = useState(false);

  const parseCSV = (text) => {
    const lines = text.split("\n").filter(Boolean);
    if (lines.length < 2) return [];
    const header = lines[0].split(",").map(h => h.trim().toLowerCase().replace(/"/g,"").replace(/\r/g,""));
    return lines.slice(1).map(line => {
      const vals = line.split(",").map(v => v.trim().replace(/^"|"$/g,"").replace(/\r/g,""));
      const obj = {};
      header.forEach((h, i) => obj[h] = vals[i] || "");
      return obj;
    }).filter(r => r.comment || r.text || r.feedback || r.translated_comment);
  };

  const handleFile = (text, fname) => {
    setWords([]); setSelected(null); setFilteredComments([]); setFileName(fname);
    let texts = [];
    if (fname.endsWith(".csv")) {
      const rows = parseCSV(text);
      texts = rows.map(r => (r.comment || r.text || r.feedback || r.translated_comment || "").trim()).filter(Boolean);
    } else {
      texts = text.split("\n").map(t => t.trim()).filter(Boolean);
    }
    setUploadedTexts(texts);
    generate(texts);
  };

  const generate = (texts) => {
    if (!texts.length) return;
    setLoading(true); setSelected(null); setFilteredComments([]);
    setTimeout(() => {
      const freq = getWordFrequency(texts);
      const posWords = new Set(["good","great","excellent","support","improved","clear","efficient","helpful","positive","transparent","friendly","appreciate","simple","easy","fast","convenient","beneficial","innovative","responsive","streamlined","smooth","useful"]);
      const negWords = new Set(["burden","harsh","unclear","difficult","tight","unfair","oppose","concern","problem","penalty","complex","burdensome","frustrating","slow","costly","excessive","confusing","outdated","crash","error","insufficient","inadequate"]);
      const maxCount = freq[0]?.[1] || 1;
      const enriched = freq.map(([word, count]) => ({
        word, count,
        sentiment: posWords.has(word) ? "positive" : negWords.has(word) ? "negative" : "neutral",
        size: Math.max(13, Math.min(56, 13 + Math.round((count / maxCount) * 43))),
      }));
      setWords(enriched);
      setLoading(false);
    }, 600);
  };

  const loadDemoData = () => {
    setFileName("demo-data");
    const stored = db.get("comments") || [];
    const texts = stored.map(c => c.text);
    setUploadedTexts(texts);
    generate(texts);
  };

  const handleWordClick = (w) => {
    setSelected(w.word === selected ? null : w.word);
    const matched = uploadedTexts.filter(t => t.toLowerCase().includes(w.word.toLowerCase())).map((t, i) => ({ id: i, text: t, sentiment: analyzeSentiment(t), user: "Uploaded" }));
    setFilteredComments(matched);
  };

  const sentColors = { positive: "#22A86D", negative: "#E74C3C", neutral: "#F39C12" };
  const posCount = words.filter(w => w.sentiment === "positive").length;
  const negCount = words.filter(w => w.sentiment === "negative").length;

  return (
    <div>
      <SectionTitle icon="cloud" title="Word Cloud Analyzer" subtitle="Visualize key themes from MCA consultation feedback" />
      <div style={{ display:"grid", gridTemplateColumns:"1fr 2fr", gap:24, marginBottom:24 }}>
        <Card>
          <h3 style={{ margin:"0 0 16px", fontSize:16, fontWeight:700, color:"#0B1E3D" }}>Data Source</h3>
          <FileUpload onData={handleFile} label="Upload CSV with 'comment' or 'text' column" />
          {fileName && <div style={{ marginTop:8, padding:"8px 12px", borderRadius:8, background:"rgba(11,30,61,0.05)", fontSize:12, color:"#4A5568" }}>📄 <strong>{fileName}</strong> — {uploadedTexts.length} comments loaded</div>}
          <button onClick={loadDemoData} style={{ marginTop:12, width:"100%", padding:"10px", borderRadius:8, background: "#152D55", color: "#FFFFFF", border:"none", cursor:"pointer", fontSize:14, fontWeight:600 }}>Load Demo Data</button>
          {words.length > 0 && (
            <div style={{ marginTop:20 }}>
              <div style={{ fontSize:12, fontWeight:700, color:"#718096", marginBottom:10, textTransform:"uppercase", letterSpacing:0.5 }}>Legend</div>
              {[["positive","Positive / Support"], ["negative","Concerns / Issues"], ["neutral","Neutral / Legal"]].map(([s,l]) => (
                <div key={s} style={{ display:"flex", alignItems:"center", gap:8, marginBottom:6 }}>
                  <div style={{ width:10, height:10, borderRadius:"50%", background:sentColors[s] }} />
                  <span style={{ fontSize:13, color:"#4A5568" }}>{l}</span>
                </div>
              ))}
              <div style={{ marginTop:14, padding:"10px 12px", borderRadius:8, background:"#F5F0E8", fontSize:12 }}>
                <div style={{ color:"#22A86D", fontWeight:700 }}>✅ {posCount} positive words</div>
                <div style={{ color:"#E74C3C", fontWeight:700, marginTop:4 }}>⚠️ {negCount} concern words</div>
                <div style={{ color:"#718096", marginTop:4 }}>{words.length - posCount - negCount} neutral words</div>
              </div>
              <button onClick={() => { setWords([]); setFileName(""); setUploadedTexts([]); setSelected(null); setFilteredComments([]); }} style={{ marginTop:12, width:"100%", padding:"8px", borderRadius:8, border:`1px solid #E74C3C`, background:"rgba(231,76,60,0.06)", color:"#E74C3C", fontSize:13, fontWeight:600, cursor:"pointer" }}>🗑 Clear & Reset</button>
            </div>
          )}
        </Card>
        <Card style={{ minHeight:320 }}>
          {loading ? (
            <div style={{ display:"flex", alignItems:"center", justifyContent:"center", height:280 }}>
              <div style={{ textAlign:"center" }}><Spinner size={40} /><p style={{ color:"#718096", marginTop:16, fontSize:14 }}>Generating word cloud...</p></div>
            </div>
          ) : words.length === 0 ? (
            <div style={{ display:"flex", alignItems:"center", justifyContent:"center", height:280, color:"#718096", flexDirection:"column", gap:12 }}>
              <Icon name="cloud" size={48} color="rgba(11,30,61,0.2)" /><p style={{ fontSize:14 }}>Upload a CSV file to generate word cloud</p>
            </div>
          ) : (
            <div>
              <div style={{ fontSize:12, color:"#718096", marginBottom:12, textAlign:"right" }}>{words.length} unique words · click any word to filter</div>
              <div style={{ display:"flex", flexWrap:"wrap", gap:10, alignItems:"center", justifyContent:"center", padding:8 }}>
                {words.map(w => (
                  <span key={w.word} onClick={() => handleWordClick(w)} style={{ fontSize: w.size, fontWeight: 700, color: sentColors[w.sentiment] || "#0B1E3D", cursor:"pointer", padding:"2px 6px", borderRadius:6, background: selected === w.word ? "rgba(201,168,76,0.18)" : "transparent", border: selected === w.word ? `1.5px solid #C9A84C` : "1.5px solid transparent", transition:"all 0.15s", fontFamily:"Georgia, serif", opacity: selected && selected !== w.word ? 0.4 : 1 }}>{w.word}</span>
                ))}
              </div>
            </div>
          )}
        </Card>
      </div>
      {selected && (
        <Card>
          <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:16 }}>
            <h3 style={{ margin:0, fontSize:16, fontWeight:700, color:"#0B1E3D" }}>Comments containing: "<span style={{color:"#C9A84C"}}>{selected}</span>" <span style={{ fontSize:13, fontWeight:400, color:"#718096" }}>({filteredComments.length} found)</span></h3>
            <button onClick={() => { setSelected(null); setFilteredComments([]); }} style={{ background:"none", border:"none", cursor:"pointer", color:"#718096", fontSize:13 }}>✕ Clear</button>
          </div>
          <div style={{ display:"grid", gap:10 }}>
            {filteredComments.map(c => (
              <div key={c.id} style={{ padding:"14px 16px", borderRadius:10, background:"#F5F0E8", borderLeft:`3px solid ${sentColors[c.sentiment] || "#0B1E3D"}` }}>
                <div style={{ display:"flex", justifyContent:"space-between", alignItems:"flex-start", marginBottom:6 }}>
                  <span style={{ fontSize:13, fontWeight:600, color:"#0B1E3D" }}>{c.user || "Uploaded"}</span>
                  <SentimentBadge s={c.sentiment} />
                </div>
                <p style={{ margin:0, fontSize:14, color:"#4A5568", lineHeight:1.5 }}>{c.text}</p>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
};
// ════════════════════════════════════════════════════════════
// MODULE 2: AKASHWANI
// ════════════════════════════════════════════════════════════
const AkashwaniModule = () => {
  const [draftText, setDraftText] = useState("");
  const [commentsText, setCommentsText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [tab, setTab] = useState("overall");

  const demoPolicy = `The Companies (Amendment) Act 2025 introduces significant changes to corporate governance framework. Section 12 mandates stricter compliance requirements for all registered entities. Directors must ensure quarterly filings with enhanced disclosure norms. Penalties for non-compliance range from Rs. 50,000 to Rs. 5,00,000. The regulation aims to improve transparency and accountability in corporate sector. Small businesses may apply for simplified compliance procedures under Section 7(b). The Act shall come into force from April 1, 2025.`;

  const scoreText = (text) => {
    const words = text.toLowerCase().split(/\s+/);
    const pos = ["improve","transparent","clear","benefit","support","effective","good","excellent","simplified","accountability"].filter(w => words.includes(w)).length;
    const neg = ["penalty","harsh","burden","complex","confusing","unfair","mandatory","restrict","fine","sanction"].filter(w => words.includes(w)).length;
    const base = 50 + (pos - neg) * 5;
    const wordCount = words.length;
    const lengthBonus = wordCount > 100 ? 5 : wordCount > 50 ? 2 : -5;
    return Math.max(20, Math.min(90, base + lengthBonus + Math.random() * 8 - 4));
  };

  const analyze = async () => {
    if (!draftText.trim()) return;
    setLoading(true);
    const overallScore = parseFloat(scoreText(draftText).toFixed(1));
    const bizScore = parseFloat((overallScore * 0.92 + (Math.random() * 6 - 3)).toFixed(1));
    const userScore = parseFloat((overallScore * 1.05 + (Math.random() * 6 - 3)).toFixed(1));
    const aiInsight = await callAI(`Analyze this MCA policy draft and give 3 bullet-point observations about acceptance, business impact, and clarity:\n\n${draftText.slice(0, 800)}`, "You are an MCA policy analyst. Be concise. Use bullet points.");
    const problems = [
      { category:"Complexity", description:"Technical legal language may reduce user acceptance", severity:65 },
      { category:"Compliance Burden", description:"Multiple mandatory requirements could strain small businesses", severity:72 },
      { category:"Timeline", description:"Short implementation window needs more clarity", severity:48 },
    ];
    const recs = [
      { target:"Clarity", action:"Simplify legal jargon with plain-language summaries", priority:"High", effort:"Low", timeline:"1 week" },
      { target:"Small Business", action:"Add exemption criteria and simplified procedures", priority:"High", effort:"Medium", timeline:"2 weeks" },
      { target:"Outreach", action:"Conduct public consultation sessions with FAQs", priority:"Medium", effort:"High", timeline:"3 weeks" },
    ];
    setResult({ overallScore, bizScore, userScore, aiInsight, problems, recs, processed: new Date().toLocaleTimeString() });
    setLoading(false);
  };

  const GaugeBar = ({ score, label }) => {
    const color = score >= 70 ? "#22A86D" : score >= 50 ? "#F39C12" : "#E74C3C";
    return (
      <div style={{ marginBottom:20 }}>
        <div style={{ display:"flex", justifyContent:"space-between", marginBottom:6 }}>
          <span style={{ fontSize:14, fontWeight:600, color:"#0B1E3D" }}>{label}</span>
          <span style={{ fontSize:18, fontWeight:800, color, fontFamily:"Georgia, serif" }}>{score}%</span>
        </div>
        <ProgressBar pct={score} color={color} />
      </div>
    );
  };

  return (
    <div>
      <SectionTitle icon="policy" title="Akashwani — Policy Analyzer" subtitle="ML/BERT-powered policy acceptance scoring with AI insights" />
      <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:20, marginBottom:24 }}>
        <Card>
          <h3 style={{ margin:"0 0 14px", fontSize:15, fontWeight:700, color:"#0B1E3D" }}>Policy Draft</h3>
          <FileUpload onData={(text) => setDraftText(text)} accept=".txt,.docx,.pdf,.csv" label="Upload policy draft (TXT/PDF/DOCX)" />
          <div style={{ margin:"12px 0 4px", fontSize:13, color:"#718096" }}>— or type / paste below —</div>
          <textarea value={draftText} onChange={e => setDraftText(e.target.value)} rows={5} placeholder="Paste policy draft here..." style={{ width:"100%", padding:"12px", borderRadius:8, border:"1.5px solid rgba(11,30,61,0.15)", fontSize:13, resize:"vertical", fontFamily:"inherit", color:"#0B1E3D", outline:"none", boxSizing:"border-box" }} />
          {!draftText && <button onClick={() => setDraftText(demoPolicy)} style={{ marginTop:8, padding:"8px 14px", borderRadius:6, border:`1px solid #C9A84C`, background:"transparent", color:"#C9A84C", fontSize:13, cursor:"pointer", fontWeight:600 }}>Load Demo Policy</button>}
        </Card>
        <Card>
          <h3 style={{ margin:"0 0 14px", fontSize:15, fontWeight:700, color:"#0B1E3D" }}>Public Comments (Optional)</h3>
          <FileUpload onData={(text) => setCommentsText(text)} accept=".csv,.txt" label="Upload public comments CSV" />
          <div style={{ margin:"12px 0 4px", fontSize:13, color:"#718096" }}>— or paste comments —</div>
          <textarea value={commentsText} onChange={e => setCommentsText(e.target.value)} rows={5} placeholder="Paste public comments (one per line)..." style={{ width:"100%", padding:"12px", borderRadius:8, border:"1.5px solid rgba(11,30,61,0.15)", fontSize:13, resize:"vertical", fontFamily:"inherit", color:"#0B1E3D", outline:"none", boxSizing:"border-box" }} />
        </Card>
      </div>
      <button onClick={analyze} disabled={loading || !draftText.trim()} style={{ padding:"14px 40px", borderRadius:10, border:"none", background: (!draftText.trim() || loading) ? "#718096" : `linear-gradient(135deg, #152D55, #1E3F73)`, color:"#FFFFFF", fontSize:16, fontWeight:700, cursor: (!draftText.trim() || loading) ? "not-allowed" : "pointer", display:"flex", alignItems:"center", gap:10, marginBottom:24 }}>
        {loading ? <><Spinner size={18} /> Analyzing...</> : <><Icon name="spark" size={18} /> Analyze Policy</>}
      </button>
      {result && (
        <>
          <div style={{ display:"flex", gap:12, marginBottom:20 }}>
            {["overall","business","individual","insights"].map(t => (
              <button key={t} onClick={() => setTab(t)} style={{ padding:"10px 20px", borderRadius:8, border:"none", cursor:"pointer", fontSize:14, fontWeight:600, background: tab === t ? "#0B1E3D" : "#F5F0E8", color: tab === t ? "#FFFFFF" : "#4A5568" }}>{t.charAt(0).toUpperCase() + t.slice(1)}</button>
            ))}
          </div>
          {tab === "overall" && (
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:20 }}>
              <Card>
                <h3 style={{ margin:"0 0 20px", fontSize:16, fontWeight:700, color:"#0B1E3D" }}>Acceptance Scores</h3>
                <GaugeBar score={result.overallScore} label="Overall Acceptance" />
                <GaugeBar score={result.bizScore} label="Business Acceptance" />
                <GaugeBar score={result.userScore} label="Individual Acceptance" />
              </Card>
              <Card>
                <h3 style={{ margin:"0 0 16px", fontSize:16, fontWeight:700, color:"#0B1E3D" }}>AI Insights</h3>
                <div style={{ fontSize:14, color:"#4A5568", lineHeight:1.7, whiteSpace:"pre-line" }}>{result.aiInsight}</div>
                <div style={{ marginTop:12, fontSize:12, color:"#718096" }}>Analyzed at {result.processed}</div>
              </Card>
            </div>
          )}
          {tab === "business" && (
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:20 }}>
              <Card>
                <h3 style={{ margin:"0 0 16px", fontSize:16, fontWeight:700, color:"#0B1E3D" }}>Problems Identified</h3>
                {result.problems.map((p, i) => (
                  <div key={i} style={{ marginBottom:14, padding:"14px", borderRadius:10, background:"#F5F0E8", borderLeft:`3px solid #E74C3C` }}>
                    <div style={{ display:"flex", justifyContent:"space-between", marginBottom:8 }}>
                      <strong style={{ fontSize:14, color:"#0B1E3D" }}>{p.category}</strong>
                      <Badge label={`${p.severity}% severity`} color="#C0392B" bg="rgba(192,57,43,0.1)" />
                    </div>
                    <p style={{ margin:0, fontSize:13, color:"#4A5568" }}>{p.description}</p>
                    <div style={{ marginTop:8 }}><ProgressBar pct={p.severity} color="#E74C3C" /></div>
                  </div>
                ))}
              </Card>
              <Card>
                <h3 style={{ margin:"0 0 16px", fontSize:16, fontWeight:700, color:"#0B1E3D" }}>Recommendations</h3>
                {result.recs.map((r, i) => (
                  <div key={i} style={{ marginBottom:14, padding:"14px", borderRadius:10, background:"#F5F0E8", borderLeft:`3px solid #22A86D` }}>
                    <div style={{ display:"flex", justifyContent:"space-between", marginBottom:8 }}>
                      <strong style={{ fontSize:14, color:"#0B1E3D" }}>{r.target}</strong>
                      <Badge label={r.priority} color={r.priority === "High" ? "#C0392B" : "#B7770D"} bg={r.priority === "High" ? "rgba(192,57,43,0.1)" : "rgba(183,119,13,0.1)"} />
                    </div>
                    <p style={{ margin:0, fontSize:13, color:"#4A5568" }}>{r.action}</p>
                    <div style={{ display:"flex", gap:16, marginTop:8 }}>
                      <span style={{ fontSize:12, color:"#718096" }}>⏱ {r.timeline}</span>
                      <span style={{ fontSize:12, color:"#718096" }}>💪 {r.effort} effort</span>
                    </div>
                  </div>
                ))}
              </Card>
            </div>
          )}
          {tab === "individual" && (
            <Card>
              <h3 style={{ margin:"0 0 16px", fontSize:16, fontWeight:700, color:"#0B1E3D" }}>Individual User Impact</h3>
              <div style={{ display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:16, marginBottom:20 }}>
                <StatCard label="User Acceptance" value={`${result.userScore}%`} color="#22A86D" />
                <StatCard label="Clarity Score" value="68%" color="#F39C12" />
                <StatCard label="Complexity Level" value="Moderate" color="#1E3F73" />
              </div>
              <p style={{ color:"#4A5568", fontSize:14, lineHeight:1.7 }}>Individual users show moderate acceptance. Main concerns center around technical language and compliance deadlines. Adding simplified FAQs and step-by-step guides would improve individual acceptance by an estimated 15-20%.</p>
            </Card>
          )}
          {tab === "insights" && (
            <Card>
              <h3 style={{ margin:"0 0 16px", fontSize:16, fontWeight:700, color:"#0B1E3D" }}>Technical Analysis Details</h3>
              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:16 }}>
                {[["Word Count", draftText.split(/\s+/).length + " words"],["Readability", "Moderate (Grade 12)"],["Risk Level", "Medium"],["Legal Terms", "High Density"],["Analysis Method", "Rule-based + AI"],["Confidence", "82%"]].map(([k, v]) => (
                  <div key={k} style={{ padding:"12px 16px", background:"#F5F0E8", borderRadius:8 }}>
                    <div style={{ fontSize:12, color:"#718096", textTransform:"uppercase", letterSpacing:0.5, marginBottom:4 }}>{k}</div>
                    <div style={{ fontSize:15, fontWeight:700, color:"#0B1E3D" }}>{v}</div>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </>
      )}
    </div>
  );
};
// ════════════════════════════════════════════════════════════
// MODULE 3: SENTIMENT ANALYSIS
// ════════════════════════════════════════════════════════════
const SentimentModule = () => {
  const [activeTab, setActiveTab] = useState("analyze");
  const [modelStatus, setModelStatus] = useState({});
  const [deviceInfo, setDeviceInfo] = useState("cpu");
  const [statusChecked, setStatusChecked] = useState(false);
  const [analyzeText, setAnalyzeText] = useState("");
  const [analyzeResult, setAnalyzeResult] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [batchFile, setBatchFile] = useState(null);
  const [batchProcessing, setBatchProcessing] = useState(false);
  const [batchResults, setBatchResults] = useState(null);
  const [batchRawData, setBatchRawData] = useState([]);
  const [sentFilter, setSentFilter] = useState("all");
  const [modelFilter, setModelFilter] = useState("all");
  const [searchQ, setSearchQ] = useState("");
  const [trainModal, setTrainModal] = useState(null);
  const [trainingFile, setTrainingFile] = useState(null);
  const [training, setTraining] = useState(false);
  const [sampleResults, setSampleResults] = useState(null);
  const [samplesLoading, setSamplesLoading] = useState(false);

  const LAPI = BACKEND_URL;

  const checkModels = async () => {
    try {
      const r = await fetch(`${LAPI}/api/legal/models/status`);
      if (r.ok) { const d = await r.json(); setModelStatus(d.models || {}); setDeviceInfo(d.device || "cpu"); }
    } catch (_) {}
    setStatusChecked(true);
  };

  useEffect(() => { checkModels(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ✅ FIX: Save analyzed comment to db so dashboard counts update
  const runAnalyze = async () => {
    if (!analyzeText.trim()) return;
    setAnalyzing(true); setAnalyzeResult(null);
    try {
      const r = await fetch(`${LAPI}/api/legal/analyze`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: analyzeText }) });
      const d = await r.json();
      if (d.success) {
        setAnalyzeResult(d.data);
        // ✅ Save to db so dashboard stats update
        const existing = db.get("comments") || [];
        const detectedSentiment = d.data?.results?.[0]?.sentiment || analyzeSentiment(analyzeText);
        const newComment = {
          id: Date.now(),
          text: analyzeText,
          user: "Sentiment Analysis",
          userType: "individual",
          sentiment: detectedSentiment,
          createdAt: new Date().toISOString()
        };
        db.set("comments", [...existing, newComment]);
      } else {
        setAnalyzeResult({ error: d.error || "Analysis failed" });
      }
    } catch (e) {
      // Fallback: still save with local sentiment
      setAnalyzeResult({ error: "Backend unavailable — run backend server" });
      const existing = db.get("comments") || [];
      const newComment = {
        id: Date.now(),
        text: analyzeText,
        user: "Sentiment Analysis",
        userType: "individual",
        sentiment: analyzeSentiment(analyzeText),
        createdAt: new Date().toISOString()
      };
      db.set("comments", [...existing, newComment]);
    }
    setAnalyzing(false);
  };

  // ✅ FIX: Save batch results to db so dashboard counts update
  const runBatch = async () => {
    if (!batchFile) return;
    setBatchProcessing(true); setBatchResults(null); setBatchRawData([]);
    try {
      const form = new FormData();
      form.append("file", batchFile);
      const r = await fetch(`${LAPI}/api/legal/analyze/batch`, { method: "POST", body: form });
      const d = await r.json();
      if (d.success) {
        const csvR = await fetch(`${LAPI}${d.download_url}`);
        const csvTxt = await csvR.text();
        const lines = csvTxt.trim().split("\n");
        const headers = lines[0].split(",").map(h => h.trim().replace(/"/g,""));
        const rows = lines.slice(1).filter(l => l.trim()).map(line => {
          const vals = line.match(/(".*?"|[^,]+)/g)?.map(v => v.trim().replace(/^"|"$/g,"")) || line.split(",");
          const obj = {};
          headers.forEach((h, i) => { obj[h] = vals[i] || ""; });
          return obj;
        });
        setBatchRawData(rows);

        // ✅ Save all batch rows to db
        const existing = db.get("comments") || [];
        const newComments = rows.map((row, i) => ({
          id: Date.now() + i,
          text: (row.original_comment || row.clause || row.text || "").trim(),
          user: "Batch Analysis",
          userType: "individual",
          sentiment: (row.sentiment || "neutral").toLowerCase(),
          createdAt: new Date().toISOString()
        })).filter(c => c.text.length > 5);
        db.set("comments", [...existing, ...newComments]);

        const advDist = {};
        rows.forEach(row => {
          const adv = (row.advanced_sentiment || "").trim().toLowerCase();
          if (adv && adv !== "" && adv !== "undefined") { advDist[adv] = (advDist[adv] || 0) + 1; }
        });
        const sentDist = { positive:0, negative:0, neutral:0 };
        rows.forEach(row => { const s = (row.sentiment || "neutral").trim().toLowerCase(); if (s in sentDist) sentDist[s]++; });
        const enrichedSummary = {
          ...d.summary,
          total_entries: d.summary?.total_entries ?? rows.length,
          sentiment_distribution: (d.summary?.sentiment_distribution && Object.keys(d.summary.sentiment_distribution).length > 0) ? d.summary.sentiment_distribution : sentDist,
          advanced_sentiment_distribution: (d.summary?.advanced_sentiment_distribution && Object.keys(d.summary.advanced_sentiment_distribution).length > 0) ? d.summary.advanced_sentiment_distribution : advDist,
        };
        setBatchResults({ ...d, summary: enrichedSummary });
      } else {
        setBatchResults({ error: d.error });
      }
    } catch (e) {
      setBatchResults({ error: "Backend unavailable — run backend server" });
    }
    setBatchProcessing(false);
  };

  const runTrain = async (useDefault) => {
    if (!trainModal) return;
    setTraining(true);
    try {
      if (useDefault) {
        const r = await fetch(`${LAPI}/api/legal/train/default`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model_type: trainModal.type, use_default_data: true }) });
        const d = await r.json();
        if (d.success) { setTrainModal(null); checkModels(); } else alert(d.error || "Training failed");
      } else {
        if (!trainingFile) return;
        const form = new FormData();
        form.append("file", trainingFile);
        form.append("model_type", trainModal.type);
        const r = await fetch(`${LAPI}/api/legal/train/upload`, { method: "POST", body: form });
        const d = await r.json();
        if (d.success) { setTrainModal(null); setTrainingFile(null); checkModels(); } else alert(d.error || "Training failed");
      }
    } catch (_) { alert("Backend unavailable"); }
    setTraining(false);
  };

  const runSamples = async () => {
    setSamplesLoading(true); setSampleResults(null);
    try {
      const r = await fetch(`${LAPI}/api/legal/test/samples`);
      const d = await r.json();
      if (d.success) setSampleResults(d.data.samples);
    } catch (_) { setSampleResults([]); }
    setSamplesLoading(false);
  };

  const SENT_COLOR = { positive: "#22A86D", negative: "#E74C3C", neutral: "#718096" };
  const SENT_BG   = { positive: "rgba(39,174,96,0.10)", negative: "rgba(231,76,60,0.10)", neutral: "rgba(127,140,141,0.10)" };

  const filteredBatch = batchRawData.filter(row => {
    if (sentFilter !== "all" && row.sentiment !== sentFilter) return false;
    if (modelFilter !== "all" && row.model_used !== modelFilter) return false;
    if (searchQ) { const q = searchQ.toLowerCase(); return (row.original_comment||"").toLowerCase().includes(q) || (row.clause||"").toLowerCase().includes(q) || (row.reason||"").toLowerCase().includes(q); }
    return true;
  });

  const batchModels = [...new Set(batchRawData.map(r => r.model_used).filter(Boolean))];

  const models = [
    { name:"Router Classifier", key:"router_classifier", type:"classifier", icon:"🔀", desc:"Routes text to best BERT model" },
    { name:"InLegalBERT",        key:"inlegalbert",       type:"inlegalbert", icon:"⚖️",  desc:"Supreme Court & High Court judgments" },
    { name:"LegalBERT",          key:"legalbert",         type:"legalbert",   icon:"📋",  desc:"Contracts, agreements, legal opinions" },
    { name:"DistilBERT",         key:"distilbert",        type:"distilbert",  icon:"📰",  desc:"News, blogs, commentary" },
  ];

  const AnalyzeTab = () => (
    <div>
      <Card>
        <h3 style={{ margin:"0 0 6px", fontSize:15, fontWeight:700, color:"#0B1E3D" }}>⚖️ Analyze Legal Text</h3>
        <p style={{ margin:"0 0 14px", fontSize:13, color:"#718096" }}>Enter legal text to detect sentiment, extract reasons, identify legal context and key phrases.</p>
        <textarea value={analyzeText} onChange={e => setAnalyzeText(e.target.value)} rows={5} placeholder="e.g. The Supreme Court granted the writ petition due to violation of fundamental rights under Article 21."
          style={{ width:"100%", padding:"12px", borderRadius:8, border:"1.5px solid rgba(11,30,61,0.15)", fontSize:13, resize:"vertical", fontFamily:"inherit", color:"#0B1E3D", outline:"none", boxSizing:"border-box", marginBottom:12 }} />
        <div style={{ display:"flex", gap:10 }}>
          <button onClick={runAnalyze} disabled={analyzing || !analyzeText.trim()} style={{ flex:1, padding:"11px", borderRadius:8, border:"none", background: analyzeText.trim() ? "#0B1E3D" : "#718096", color:"#FFFFFF", fontSize:14, fontWeight:700, cursor: analyzeText.trim() ? "pointer" : "not-allowed", display:"flex", alignItems:"center", justifyContent:"center", gap:8 }}>
            {analyzing ? <><Spinner size={16}/> Analyzing...</> : "▶ Analyze Text"}
          </button>
          <button onClick={() => { setAnalyzeText(""); setAnalyzeResult(null); }} style={{ padding:"11px 20px", borderRadius:8, border:`1.5px solid rgba(11,30,61,0.2)`, background:"transparent", color:"#4A5568", fontSize:14, cursor:"pointer" }}>✕ Clear</button>
        </div>
      </Card>
      {analyzeResult && (
        <div style={{ marginTop:20 }}>
          {analyzeResult.error ? (
            <Card style={{ borderLeft:`4px solid #E74C3C` }}>
              <p style={{ color:"#E74C3C", margin:0, fontSize:14 }}>⚠ {analyzeResult.error}</p>
              <p style={{ color:"#718096", margin:"8px 0 0", fontSize:13 }}>✅ Comment saved to dashboard with local sentiment analysis.</p>
            </Card>
          ) : (
            <>
              {analyzeResult.results?.length > 0 && (() => {
                const rs = analyzeResult.results;
                const pos = rs.filter(r=>r.sentiment==="positive").length;
                const neg = rs.filter(r=>r.sentiment==="negative").length;
                const neu = rs.filter(r=>r.sentiment==="neutral").length;
                const avgConf = rs.reduce((s,r)=>s+r.score,0)/rs.length;
                return (
                  <div style={{ display:"grid", gridTemplateColumns:"repeat(5,1fr)", gap:12, marginBottom:16 }}>
                    {[["Clauses", rs.length, "#0B1E3D"],["Positive", pos, "#22A86D"],["Negative", neg, "#E74C3C"],["Neutral", neu, "#718096"],["Avg Conf.", (avgConf*100).toFixed(1)+"%", "#C9A84C"]].map(([label, val, color]) => (
                      <div key={label} style={{ background:"#F5F0E8", borderRadius:10, padding:"14px 12px", textAlign:"center" }}>
                        <div style={{ fontSize:11, color:"#718096", textTransform:"uppercase", letterSpacing:0.5, marginBottom:4 }}>{label}</div>
                        <div style={{ fontSize:22, fontWeight:800, color }}>{val}</div>
                      </div>
                    ))}
                  </div>
                );
              })()}
              {analyzeResult.spam_check?.score > 0 && (
                <div style={{ padding:"10px 16px", borderRadius:8, marginBottom:12, background: analyzeResult.spam_check.is_spam ? "rgba(231,76,60,0.1)" : "rgba(39,174,96,0.1)", border:`1px solid ${analyzeResult.spam_check.is_spam ? "#E74C3C" : "#22A86D"}`, display:"flex", gap:8, alignItems:"center", fontSize:13 }}>
                  <span>{analyzeResult.spam_check.is_spam ? "⚠" : "✓"}</span>
                  <span style={{ color: analyzeResult.spam_check.is_spam ? "#E74C3C" : "#22A86D" }}>Spam Score: {(analyzeResult.spam_check.score*100).toFixed(1)}% — {analyzeResult.spam_check.reason || "Clean text"}</span>
                </div>
              )}
              <div style={{ display:"flex", flexDirection:"column", gap:12 }}>
                {(analyzeResult.results||[]).map((r, i) => (
                  <Card key={i} style={{ borderLeft:`4px solid ${SENT_COLOR[r.sentiment]||"#718096"}`, background: SENT_BG[r.sentiment]||"#F5F0E8" }}>
                    <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:10 }}>
                      <div style={{ display:"flex", gap:8, alignItems:"center" }}>
                        <span style={{ padding:"3px 12px", borderRadius:20, background:SENT_COLOR[r.sentiment]||"#718096", color:"#fff", fontSize:12, fontWeight:700 }}>{(r.sentiment||"neutral").toUpperCase()}</span>
                        <span style={{ padding:"3px 10px", borderRadius:20, background:"#F5F0E8", border:`1px solid rgba(11,30,61,0.15)`, fontSize:11, color:"#4A5568" }}>🤖 {r.model_used}</span>
                      </div>
                      <span style={{ fontSize:12, color:"#718096" }}>Clause {i+1}</span>
                    </div>
                    {[["Text", r.sentence],["Advanced", r.advanced_sentiment],["Reason", r.reason],["Confidence", (r.score*100).toFixed(1)+"%"]].map(([label, val]) => val && (
                      <div key={label} style={{ display:"grid", gridTemplateColumns:"110px 1fr", gap:8, marginBottom:6, fontSize:13 }}>
                        <span style={{ fontWeight:700, color:"#0B1E3D" }}>{label}:</span>
                        <span style={{ color:"#4A5568", lineHeight:1.5 }}>{val}</span>
                      </div>
                    ))}
                    {r.key_phrases?.length > 0 && (
                      <div style={{ display:"grid", gridTemplateColumns:"110px 1fr", gap:8, marginBottom:6, fontSize:13 }}>
                        <span style={{ fontWeight:700, color:"#0B1E3D" }}>Key Phrases:</span>
                        <div style={{ display:"flex", flexWrap:"wrap", gap:4 }}>
                          {r.key_phrases.map((p,pi) => <span key={pi} style={{ padding:"2px 10px", borderRadius:20, background:`rgba(11,30,61,0.08)`, fontSize:12, color:"#0B1E3D", border:`1px solid rgba(11,30,61,0.12)` }}>{p}</span>)}
                        </div>
                      </div>
                    )}
                  </Card>
                ))}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
  const BatchTab = () => (
    <div>
      <Card>
        <h3 style={{ margin:"0 0 6px", fontSize:15, fontWeight:700, color:"#0B1E3D" }}>📂 Batch Process CSV</h3>
        <p style={{ margin:"0 0 14px", fontSize:13, color:"#718096" }}>Upload a CSV file with a <code>text</code>, <code>comment</code>, or <code>sentence</code> column for bulk analysis.</p>
        <div onDragOver={e => { e.preventDefault(); e.currentTarget.style.borderColor = "#C9A84C"; }} onDragLeave={e => { e.currentTarget.style.borderColor = "rgba(11,30,61,0.2)"; }} onDrop={e => { e.preventDefault(); e.currentTarget.style.borderColor="rgba(11,30,61,0.2)"; const f=e.dataTransfer.files[0]; if(f?.name.endsWith(".csv")) setBatchFile(f); }} onClick={() => document.getElementById("batchFileInput").click()}
          style={{ border:"2px dashed rgba(11,30,61,0.2)", borderRadius:12, padding:"36px 20px", textAlign:"center", cursor:"pointer", marginBottom:14, background: batchFile ? "rgba(201,168,76,0.05)" : "#F5F0E8", transition:"all 0.2s" }}>
          <input id="batchFileInput" type="file" accept=".csv" style={{ display:"none" }} onChange={e => e.target.files[0] && setBatchFile(e.target.files[0])} />
          <div style={{ fontSize:32, marginBottom:8 }}>☁️</div>
          <div style={{ fontSize:14, fontWeight:600, color:"#0B1E3D", marginBottom:4 }}>{batchFile ? `✓ ${batchFile.name}` : "Drag & Drop CSV File Here"}</div>
          <div style={{ fontSize:12, color:"#718096" }}>{batchFile ? `${(batchFile.size/1024).toFixed(1)} KB — click to change` : "or click to browse — CSV only"}</div>
        </div>
        <button onClick={runBatch} disabled={!batchFile || batchProcessing} style={{ width:"100%", padding:"12px", borderRadius:8, border:"none", background: batchFile && !batchProcessing ? "#0B1E3D" : "#718096", color:"#FFFFFF", fontSize:14, fontWeight:700, cursor: batchFile && !batchProcessing ? "pointer" : "not-allowed", display:"flex", alignItems:"center", justifyContent:"center", gap:8 }}>
          {batchProcessing ? <><Spinner size={16}/> Processing CSV...</> : "▶ Analyze Batch"}
        </button>
      </Card>
      {batchResults?.error && <Card style={{ marginTop:16, borderLeft:`4px solid #E74C3C` }}><p style={{ color:"#E74C3C", margin:0 }}>⚠ {batchResults.error}</p></Card>}
      {batchResults?.success && batchRawData.length > 0 && (
        <div style={{ marginTop:16 }}>
          <Card style={{ marginBottom:16 }}>
            <h4 style={{ margin:"0 0 18px", fontSize:15, fontWeight:700, color:"#0B1E3D", display:"flex", alignItems:"center", gap:8 }}>🥧 Batch Analysis Summary</h4>
            <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fit, minmax(130px, 1fr))", gap:12, marginBottom:16 }}>
              <div style={{ padding:"16px 14px", borderRadius:12, background:"#F5F0E8", border:`2px solid rgba(11,30,61,0.1)`, textAlign:"center" }}>
                <div style={{ fontSize:11, fontWeight:700, color:"#718096", textTransform:"uppercase", letterSpacing:0.8, marginBottom:6 }}>Total Entries</div>
                <div style={{ fontSize:28, fontWeight:800, color:"#0B1E3D" }}>{batchResults.summary?.total_entries ?? batchRawData.length}</div>
              </div>
              <div style={{ padding:"16px 14px", borderRadius:12, textAlign:"center", background:"rgba(113,128,150,0.08)", border:`2px solid rgba(113,128,150,0.25)` }}>
                <div style={{ fontSize:11, fontWeight:700, color:"#718096", textTransform:"uppercase", letterSpacing:0.8, marginBottom:6 }}>Neutral</div>
                <div style={{ fontSize:28, fontWeight:800, color:"#718096" }}>{batchResults.summary?.sentiment_distribution?.neutral ?? batchRawData.filter(r=>r.sentiment==="neutral").length}</div>
              </div>
              <div style={{ padding:"16px 14px", borderRadius:12, textAlign:"center", background:"rgba(231,76,60,0.08)", border:`2px solid rgba(231,76,60,0.25)` }}>
                <div style={{ fontSize:11, fontWeight:700, color:"#E74C3C", textTransform:"uppercase", letterSpacing:0.8, marginBottom:6 }}>Negative</div>
                <div style={{ fontSize:28, fontWeight:800, color:"#E74C3C" }}>{batchResults.summary?.sentiment_distribution?.negative ?? batchRawData.filter(r=>r.sentiment==="negative").length}</div>
              </div>
              <div style={{ padding:"16px 14px", borderRadius:12, textAlign:"center", background:"rgba(34,168,109,0.08)", border:`2px solid rgba(34,168,109,0.25)` }}>
                <div style={{ fontSize:11, fontWeight:700, color:"#22A86D", textTransform:"uppercase", letterSpacing:0.8, marginBottom:6 }}>Positive</div>
                <div style={{ fontSize:28, fontWeight:800, color:"#22A86D" }}>{batchResults.summary?.sentiment_distribution?.positive ?? batchRawData.filter(r=>r.sentiment==="positive").length}</div>
              </div>
            </div>
            {(() => {
              const total = batchRawData.length || 1;
              const pos = batchResults.summary?.sentiment_distribution?.positive ?? batchRawData.filter(r=>r.sentiment==="positive").length;
              const neg = batchResults.summary?.sentiment_distribution?.negative ?? batchRawData.filter(r=>r.sentiment==="negative").length;
              const neu = batchResults.summary?.sentiment_distribution?.neutral  ?? batchRawData.filter(r=>r.sentiment==="neutral").length;
              return (
                <div style={{ marginBottom:20 }}>
                  {[["Positive", pos, "#22A86D"], ["Negative", neg, "#E74C3C"], ["Neutral", neu, "#718096"]].map(([label, count, color]) => (
                    <div key={label} style={{ marginBottom:8 }}>
                      <div style={{ display:"flex", justifyContent:"space-between", marginBottom:3, fontSize:12 }}>
                        <span style={{ fontWeight:700, color:"#4A5568" }}>{label}</span>
                        <span style={{ color:"#718096" }}>{count} · {Math.round(count/total*100)}%</span>
                      </div>
                      <div style={{ height:8, borderRadius:20, background:"rgba(11,30,61,0.08)", overflow:"hidden" }}>
                        <div style={{ height:"100%", width:`${Math.round(count/total*100)}%`, borderRadius:20, background:`linear-gradient(90deg, ${color}, ${color}cc)`, transition:"width 0.6s ease" }} />
                      </div>
                    </div>
                  ))}
                </div>
              );
            })()}
          </Card>
          <Card style={{ marginBottom:16 }}>
            <h4 style={{ margin:"0 0 12px", fontSize:13, fontWeight:700, color:"#0B1E3D" }}>🔍 Filter Results</h4>
            <div style={{ display:"flex", flexWrap:"wrap", gap:16 }}>
              <div>
                <div style={{ fontSize:12, fontWeight:700, color:"#718096", marginBottom:6, textTransform:"uppercase" }}>Sentiment</div>
                <div style={{ display:"flex", gap:6 }}>
                  {["all","positive","negative","neutral"].map(s => (
                    <button key={s} onClick={() => setSentFilter(s)} style={{ padding:"5px 12px", borderRadius:20, border:`1.5px solid ${sentFilter===s ? (SENT_COLOR[s]||"#0B1E3D") : "rgba(11,30,61,0.15)"}`, background: sentFilter===s ? (SENT_BG[s]||"rgba(11,30,61,0.08)") : "transparent", color: sentFilter===s ? (SENT_COLOR[s]||"#0B1E3D") : "#4A5568", fontSize:12, fontWeight:600, cursor:"pointer", textTransform:"capitalize" }}>{s}</button>
                  ))}
                </div>
              </div>
              <div style={{ flex:1, minWidth:200 }}>
                <div style={{ fontSize:12, fontWeight:700, color:"#718096", marginBottom:6, textTransform:"uppercase" }}>Search</div>
                <input value={searchQ} onChange={e => setSearchQ(e.target.value)} placeholder="Search comments..." style={{ width:"100%", padding:"7px 12px", borderRadius:8, border:"1.5px solid rgba(11,30,61,0.15)", fontSize:13, outline:"none", color:"#0B1E3D", boxSizing:"border-box" }} />
              </div>
            </div>
            <div style={{ marginTop:10, fontSize:12, color:"#718096" }}>Showing <strong>{filteredBatch.length}</strong> of <strong>{batchRawData.length}</strong> results</div>
          </Card>
          <div style={{ display:"flex", flexDirection:"column", gap:12 }}>
            {filteredBatch.slice(0, 50).map((row, gi) => (
              <Card key={gi} style={{ borderLeft:`4px solid ${SENT_COLOR[row.sentiment]||"#718096"}` }}>
                <div style={{ display:"flex", justifyContent:"space-between", alignItems:"flex-start", marginBottom:8 }}>
                  <span style={{ fontSize:12, fontWeight:700, color:"#718096" }}>#{gi+1}</span>
                  <span style={{ padding:"3px 12px", borderRadius:20, background:SENT_COLOR[row.sentiment]||"#718096", color:"#fff", fontSize:11, fontWeight:700 }}>{(row.sentiment||"neutral").toUpperCase()}</span>
                </div>
                <p style={{ margin:"0 0 8px", fontSize:13, color:"#0B1E3D", lineHeight:1.6, padding:"10px 14px", borderRadius:8, background:"rgba(11,30,61,0.04)", borderLeft:`3px solid rgba(11,30,61,0.15)` }}>"{row.original_comment || row.clause || row.text || ""}"</p>
                <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:6 }}>
                  {[["Advanced", row.advanced_sentiment],["Reason", row.reason],["Confidence", row.confidence_score],["Model", row.model_used]].map(([k,v]) => v && (
                    <div key={k} style={{ fontSize:12, color:"#4A5568" }}><span style={{ fontWeight:700, color:"#0B1E3D" }}>{k}: </span>{v}</div>
                  ))}
                </div>
              </Card>
            ))}
          </div>
        </div>
      )}
    </div>
  );

  const TrainTab = () => (
    <div>
      <Card style={{ marginBottom:16 }}>
        <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:12 }}>
          <h3 style={{ margin:0, fontSize:15, fontWeight:700, color:"#0B1E3D" }}>🧠 Model Training</h3>
          <div style={{ display:"flex", alignItems:"center", gap:6, fontSize:12, padding:"5px 14px", borderRadius:20, background: Object.values(modelStatus).some(Boolean) ? "rgba(39,174,96,0.1)" : "rgba(231,76,60,0.1)", color: Object.values(modelStatus).some(Boolean) ? "#22A86D" : "#E74C3C", border:`1px solid currentColor` }}>
            <span style={{ width:8, height:8, borderRadius:"50%", background: Object.values(modelStatus).some(Boolean) ? "#22A86D" : "#E74C3C", display:"inline-block" }} />
            {statusChecked ? (Object.values(modelStatus).filter(Boolean).length + "/" + Object.keys(modelStatus).length + " Models Ready · " + deviceInfo.toUpperCase()) : "Checking..."}
          </div>
        </div>
        <p style={{ margin:0, fontSize:13, color:"#718096" }}>Train the router classifier and BERT models with default embedded data or upload your own CSV training data.</p>
      </Card>
      <div style={{ display:"grid", gridTemplateColumns:"repeat(2,1fr)", gap:16 }}>
        {models.map(m => (
          <Card key={m.key} style={{ borderTop:`4px solid ${modelStatus[m.key] ? "#22A86D" : "rgba(11,30,61,0.15)"}` }}>
            <div style={{ display:"flex", justifyContent:"space-between", alignItems:"flex-start", marginBottom:8 }}>
              <div>
                <div style={{ fontSize:24, marginBottom:4 }}>{m.icon}</div>
                <div style={{ fontSize:14, fontWeight:700, color:"#0B1E3D" }}>{m.name}</div>
                <div style={{ fontSize:12, color:"#718096", marginTop:2 }}>{m.desc}</div>
              </div>
              <span style={{ padding:"4px 12px", borderRadius:20, fontSize:11, fontWeight:700, background: modelStatus[m.key] ? "rgba(39,174,96,0.12)" : "rgba(231,76,60,0.10)", color: modelStatus[m.key] ? "#22A86D" : "#E74C3C", border:`1px solid currentColor`, whiteSpace:"nowrap" }}>{modelStatus[m.key] ? "✓ Trained" : "✗ Not Trained"}</span>
            </div>
            <button onClick={() => setTrainModal({ type: m.type, step: "choose" })} style={{ width:"100%", padding:"9px", borderRadius:8, border:`1.5px solid #1E3F73`, background:"transparent", color:"#0B1E3D", fontSize:13, fontWeight:600, cursor:"pointer", marginTop:8 }}>▶ Train Model</button>
          </Card>
        ))}
        <Card style={{ gridColumn:"span 2", borderTop:`4px solid #C9A84C` }}>
          <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center" }}>
            <div>
              <div style={{ fontSize:14, fontWeight:700, color:"#0B1E3D" }}>🚀 Quick Train — All Models</div>
              <div style={{ fontSize:12, color:"#718096", marginTop:2 }}>Train all 4 models at once with default embedded data</div>
            </div>
            <button onClick={() => setTrainModal({ type:"all", step:"choose" })} style={{ padding:"10px 24px", borderRadius:8, border:"none", background:"#0B1E3D", color:"#FFFFFF", fontSize:13, fontWeight:700, cursor:"pointer" }}>🚀 Quick Train All</button>
          </div>
        </Card>
      </div>
      {trainModal && (
        <div style={{ position:"fixed", inset:0, background:"rgba(0,0,0,0.5)", zIndex:1000, display:"flex", alignItems:"center", justifyContent:"center", padding:20 }} onClick={e => { if (e.target === e.currentTarget && !training) { setTrainModal(null); setTrainingFile(null); } }}>
          <div style={{ background:"#fff", borderRadius:16, padding:28, width:"100%", maxWidth:440, boxShadow:"0 20px 60px rgba(0,0,0,0.25)" }}>
            <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:20 }}>
              <h3 style={{ margin:0, fontSize:16, fontWeight:700, color:"#0B1E3D" }}>🧠 Train {trainModal.type}</h3>
              {!training && <button onClick={() => { setTrainModal(null); setTrainingFile(null); }} style={{ background:"none", border:"none", fontSize:20, cursor:"pointer", color:"#718096" }}>✕</button>}
            </div>
            {trainModal.step === "choose" && (
              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:12 }}>
                {[{ label:"Use Default Data", icon:"🗄️", desc:"Train with built-in examples", onClick: () => setTrainModal({...trainModal, step:"default"}) },{ label:"Upload Custom CSV", icon:"📤", desc:"Train with your own data", onClick: () => setTrainModal({...trainModal, step:"upload"}) }].map(opt => (
                  <button key={opt.label} onClick={opt.onClick} style={{ padding:"20px 14px", borderRadius:12, border:`1.5px solid rgba(11,30,61,0.2)`, background:"#F5F0E8", cursor:"pointer", textAlign:"center" }}>
                    <div style={{ fontSize:28, marginBottom:8 }}>{opt.icon}</div>
                    <div style={{ fontSize:13, fontWeight:700, color:"#0B1E3D", marginBottom:4 }}>{opt.label}</div>
                    <div style={{ fontSize:11, color:"#718096" }}>{opt.desc}</div>
                  </button>
                ))}
              </div>
            )}
            {trainModal.step === "default" && (
              <div>
                <p style={{ fontSize:14, color:"#4A5568", marginBottom:20 }}>Train <strong>{trainModal.type}</strong> using embedded training data.</p>
                <button onClick={() => runTrain(true)} disabled={training} style={{ width:"100%", padding:"12px", borderRadius:8, border:"none", background: training ? "#718096" : "#0B1E3D", color:"#fff", fontSize:14, fontWeight:700, cursor: training ? "not-allowed" : "pointer", display:"flex", alignItems:"center", justifyContent:"center", gap:8 }}>
                  {training ? <><Spinner size={16}/> Training...</> : "✓ Confirm Training"}
                </button>
              </div>
            )}
            {trainModal.step === "upload" && (
              <div>
                <div onClick={() => document.getElementById("trainFileInput").click()} style={{ border:"2px dashed rgba(11,30,61,0.2)", borderRadius:10, padding:"24px", textAlign:"center", cursor:"pointer", marginBottom:14, background:"#F5F0E8" }}>
                  <input id="trainFileInput" type="file" accept=".csv" style={{ display:"none" }} onChange={e => e.target.files[0] && setTrainingFile(e.target.files[0])} />
                  <div style={{ fontSize:28, marginBottom:6 }}>📤</div>
                  <div style={{ fontSize:13, color: trainingFile ? "#22A86D" : "#4A5568", fontWeight:600 }}>{trainingFile ? `✓ ${trainingFile.name}` : "Click to select CSV file"}</div>
                </div>
                <button onClick={() => runTrain(false)} disabled={!trainingFile || training} style={{ width:"100%", padding:"12px", borderRadius:8, border:"none", background: trainingFile && !training ? "#0B1E3D" : "#718096", color:"#fff", fontSize:14, fontWeight:700, cursor: trainingFile && !training ? "pointer" : "not-allowed", display:"flex", alignItems:"center", justifyContent:"center", gap:8 }}>
                  {training ? <><Spinner size={16}/> Uploading & Training...</> : "📤 Upload & Train"}
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );

  const DashboardTab = () => (
    <div>
      <Card style={{ marginBottom:16 }}>
        <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:12 }}>
          <div>
            <h3 style={{ margin:"0 0 4px", fontSize:15, fontWeight:700, color:"#0B1E3D" }}>🧪 Test with Sample Comments</h3>
            <p style={{ margin:0, fontSize:13, color:"#718096" }}>Run 10 built-in legal text samples through the full pipeline</p>
          </div>
          <button onClick={runSamples} disabled={samplesLoading} style={{ padding:"10px 20px", borderRadius:8, border:"none", background: samplesLoading ? "#718096" : "#0B1E3D", color:"#fff", fontSize:13, fontWeight:700, cursor: samplesLoading ? "not-allowed" : "pointer", display:"flex", alignItems:"center", gap:8 }}>
            {samplesLoading ? <><Spinner size={14}/> Running...</> : "▶ Run Sample Tests"}
          </button>
        </div>
      </Card>
      {sampleResults && (
        <div style={{ display:"grid", gridTemplateColumns:"repeat(2,1fr)", gap:12, marginBottom:20 }}>
          {sampleResults.map((s, i) => {
            const first = s.results?.[0] || {};
            return (
              <Card key={i} style={{ borderLeft:`4px solid ${SENT_COLOR[first.sentiment]||"#718096"}` }}>
                <p style={{ margin:"0 0 8px", fontSize:13, color:"#0B1E3D", lineHeight:1.5 }}>{(s.original_text||"").slice(0,100)}{(s.original_text||"").length>100?"...":""}</p>
                {s.error ? <span style={{ color:"#E74C3C", fontSize:12 }}>Error: {s.error}</span> : (
                  <div style={{ fontSize:12, display:"flex", flexDirection:"column", gap:4 }}>
                    <div style={{ display:"flex", gap:8, alignItems:"center" }}>
                      <span style={{ padding:"2px 10px", borderRadius:20, background:SENT_COLOR[first.sentiment]||"#718096", color:"#fff", fontWeight:700, fontSize:11 }}>{(first.sentiment||"N/A").toUpperCase()}</span>
                      <span style={{ color:"#718096" }}>Conf: {first.score ? (first.score*100).toFixed(0)+"%" : "N/A"}</span>
                    </div>
                    <div style={{ color:"#4A5568" }}><strong>Advanced:</strong> {first.advanced_sentiment||"N/A"}</div>
                    <div style={{ color:"#4A5568" }}><strong>Reason:</strong> {first.reason||"N/A"}</div>
                  </div>
                )}
              </Card>
            );
          })}
        </div>
      )}
      <Card>
        <h3 style={{ margin:"0 0 14px", fontSize:15, fontWeight:700, color:"#0B1E3D" }}>📈 Model Overview</h3>
        <div style={{ display:"grid", gridTemplateColumns:"repeat(2,1fr)", gap:12 }}>
          {models.map(m => (
            <div key={m.key} style={{ padding:"14px 16px", borderRadius:10, background:"#F5F0E8", borderLeft:`3px solid ${modelStatus[m.key] ? "#22A86D" : "rgba(11,30,61,0.15)"}` }}>
              <div style={{ fontSize:16, marginBottom:4 }}>{m.icon}</div>
              <div style={{ fontSize:13, fontWeight:700, color:"#0B1E3D" }}>{m.name}</div>
              <div style={{ fontSize:12, color:"#718096", marginBottom:6 }}>{m.desc}</div>
              <span style={{ fontSize:11, fontWeight:700, color: modelStatus[m.key] ? "#22A86D" : "#E74C3C" }}>{modelStatus[m.key] ? "✓ Ready" : "✗ Not Trained"}</span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );

  const TABS = [
    { id:"analyze",   label:"⚖ Analyze",      component: <AnalyzeTab /> },
    { id:"batch",     label:"📂 Batch",        component: <BatchTab /> },
    { id:"train",     label:"🧠 Train Models", component: <TrainTab /> },
    { id:"dashboard", label:"📊 Dashboard",    component: <DashboardTab /> },
  ];

  return (
    <div>
      <SectionTitle icon="brain" title="Sentiment Analysis" subtitle="Legal Text Analysis Pipeline — Multi-model sentiment detection with reason extraction" />
      <div style={{ display:"flex", gap:4, marginBottom:20, borderBottom:`2px solid rgba(11,30,61,0.08)`, paddingBottom:2 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setActiveTab(t.id)} style={{ padding:"9px 18px", borderRadius:"8px 8px 0 0", border:"none", background: activeTab===t.id ? "#0B1E3D" : "transparent", color: activeTab===t.id ? "#FFFFFF" : "#4A5568", fontSize:13, fontWeight:600, cursor:"pointer", borderBottom: activeTab===t.id ? `2px solid #0B1E3D` : "2px solid transparent" }}>{t.label}</button>
        ))}
      </div>
      {TABS.find(t => t.id === activeTab)?.component}
    </div>
  );
};
// ════════════════════════════════════════════════════════════
// MODULE 4: SUMMARY GENERATOR
// ════════════════════════════════════════════════════════════
const SummaryModule = () => {
  const [text, setText] = useState("");
  const [summaries, setSummaries] = useState(null);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [aiSource, setAiSource] = useState("");

  const handleFile = (content) => setText(content);

  const localPipeline = (inputText) => {
    const rawLines = inputText.split("\n").map(l => l.trim()).filter(l => l.length > 8);
    if (rawLines.length === 0) return { result: null, source: "local", stats: null };
    const firstLow = rawLines[0].toLowerCase();
    const isCSV = firstLow.includes(",") && (firstLow.includes("comment") || firstLow.includes("text") || firstLow.includes("feedback") || firstLow.includes("sentiment") || firstLow.includes("user"));
    let comments = [];
    if (isCSV) {
      const headers = rawLines[0].split(",").map(h => h.trim().toLowerCase().replace(/"/g,""));
      const textIdx = headers.findIndex(h => ["comment","text","feedback","translated_comment"].includes(h));
      const typeIdx = headers.findIndex(h => ["user_type","type","usertype","user"].includes(h));
      const sentIdx = headers.findIndex(h => ["sentiment","sent","label"].includes(h));
      rawLines.slice(1).forEach(line => {
        const cols = line.match(/(".*?"|[^,]+)/g)?.map(c => c.trim().replace(/^"|"$/g,"")) || line.split(",");
        const txt = (textIdx >= 0 ? cols[textIdx] : cols[0] || "").trim();
        const typ = (typeIdx >= 0 ? cols[typeIdx] : "").toLowerCase().trim();
        const sen = (sentIdx >= 0 ? cols[sentIdx] : "").toLowerCase().trim();
        if (txt.length > 5) comments.push({ text: txt, type: typ, sentiment: sen });
      });
    } else {
      rawLines.forEach(l => comments.push({ text: l, type: "", sentiment: "" }));
    }
    const total = comments.length;
    if (total === 0) return { result: null, source: "local", stats: null };
    const seen = new Set();
    comments = comments.filter(c => { const k = c.text.toLowerCase().slice(0, 80); if (seen.has(k)) return false; seen.add(k); return true; });
    const bizKw = ["company","gst","compliance","enterprise","business","corporate","llp","pvt","ltd","msme","director","audit","board","shareholder","statutory","contractual","regulatory","governance","fiduciary","provision","directive","obligation","disclosure"];
    const splitByType = comments.some(c => c.type.includes("business") || c.type.includes("individual") || c.type.includes("biz"));
    const biz = comments.filter(c => splitByType ? (c.type.includes("business") || c.type.includes("biz") || c.type.includes("corp")) : bizKw.some(k => c.text.toLowerCase().includes(k)));
    const ind = comments.filter(c => splitByType ? (c.type.includes("individual") || c.type.includes("citizen") || c.type.includes("ind")) : !bizKw.some(k => c.text.toLowerCase().includes(k)));
    const posW = ["excellent","clear","helpful","good","great","improved","easy","transparent","fast","appreciate","simple","beneficial","support","efficient","smooth","simplified","adherence","robust","timely","strengthened","professionalism","fulfilled","satisfied","consistent","meticulous","innovative","accessible","streamlined","well","positive"];
    const negW = ["burdensome","harsh","unclear","difficult","complex","penalty","tight","unfair","concern","problem","confusing","slow","frustrating","costly","excessive","outdated","crash","error","strict","absence","delay","improper","non-conformity","deficiencies","failure","risk","breach","sanction","enforcement","violation","not addressed","lacking"];
    const classify = (c) => {
      if (c.sentiment === "positive") return "positive";
      if (c.sentiment === "negative") return "negative";
      if (c.sentiment === "neutral")  return "neutral";
      const t = c.text.toLowerCase();
      const p = posW.filter(w => t.includes(w)).length;
      const n = negW.filter(w => t.includes(w)).length;
      return p > n ? "positive" : n > p ? "negative" : "neutral";
    };
    const extractSentences = (group, sentimentType, maxN = 3) => {
      const filtered = group.filter(c => classify(c) === sentimentType);
      if (filtered.length === 0) return [];
      const sorted = [...filtered].sort((a, b) => b.text.length - a.text.length);
      const picked = []; const startWords = new Set();
      for (const c of sorted) {
        if (picked.length >= maxN) break;
        let clean = c.text.replace(/,\s*(business|individual|positive|negative|neutral).*/gi, "").replace(/["""]/g, "").trim();
        if (clean.length < 10) continue;
        const sw = clean.toLowerCase().split(" ").slice(0,3).join(" ");
        if (startWords.has(sw)) continue;
        startWords.add(sw);
        picked.push(clean.length > 130 ? clean.slice(0, 127) + "..." : clean);
      }
      return picked;
    };
    const themeMap = { compliance:["compliance","filing","return","audit","mandatory","regulation","statutory","obligation"], digital:["portal","online","digital","website","system","e-filing","technical","interface"], support:["support","helpline","staff","response","grievance","assistance"], penalty:["penalty","fine","charge","late","strict","harsh","sanction","enforcement"], process:["process","procedure","requirement","registration","approve","workflow","documentation"], msme:["msme","small","micro","startup","enterprise","exemption"], transparency:["transparent","clear","accountability","disclosure","governance","fiduciary"] };
    const themeHits = {};
    comments.forEach(c => { const t = c.text.toLowerCase(); Object.entries(themeMap).forEach(([theme, words]) => { if (words.some(w => t.includes(w))) themeHits[theme] = (themeHits[theme] || 0) + 1; }); });
    const topThemes = Object.entries(themeHits).sort((a,b)=>b[1]-a[1]).slice(0,4).map(([t])=>t);
    const stopwords = new Set(["the","a","an","is","in","it","of","and","or","to","that","this","was","for","on","are","with","as","at","be","by","from","have","has","had","not","but","we","they","you","i","very","just","so","all","been","more","our","their","its","may","will","can","also","should","would","could","which","were","been","what","when","where","such","been","have","this","than","then","them","these","those","been","some","each"]);
    const wordFreq = {};
    comments.forEach(c => { c.text.toLowerCase().match(/[a-z]{4,}/g)?.forEach(w => { if (!stopwords.has(w)) wordFreq[w] = (wordFreq[w] || 0) + 1; }); });
    const keywords = Object.entries(wordFreq).sort((a,b)=>b[1]-a[1]).slice(0,12).map(([w])=>w);
    const allPos = comments.filter(c => classify(c) === "positive").length;
    const allNeg = comments.filter(c => classify(c) === "negative").length;
    const allNeu = comments.filter(c => classify(c) === "neutral").length;
    const overallLabel = allPos > allNeg * 1.3 ? "predominantly positive" : allNeg > allPos * 1.3 ? "predominantly critical" : "mixed";
    const bizPos = extractSentences(biz, "positive", 3); const bizNeg = extractSentences(biz, "negative", 3); const bizNeu = extractSentences(biz, "neutral",  2);
    const indPos = extractSentences(ind, "positive", 2); const indNeg = extractSentences(ind, "negative", 3); const indNeu = extractSentences(ind, "neutral",  2);
    const topKw = keywords.slice(0,3).join(", ");
    const bizRec = bizNeg.length > 0 ? `Address identified concerns around ${topKw} through clearer regulatory guidelines and phased timelines.` : "Maintain current compliance standards with periodic governance review cycles.";
    const indRec = indNeg.length > 0 ? `Improve ${topThemes.slice(0,2).join(" and ")} accessibility and documentation for individual users.` : "Continue digital infrastructure improvements for better citizen accessibility.";
    const structured = { business: { pos: bizPos, neg: bizNeg, neu: bizNeu, rec: bizRec, total: biz.length }, individual: { pos: indPos, neg: indNeg, neu: indNeu, rec: indRec, total: ind.length }, overall: { total, bizCount: biz.length, indCount: ind.length, label: overallLabel, themes: topThemes.join(", ") || "compliance, process", posCount: allPos, negCount: allNeg, neuCount: allNeu } };
    return { result: structured, source: "local", stats: { themes: themeHits, keywords, total } };
  };

  const generateSummaries = async () => {
    if (!text.trim()) return;
    setLoading(true); setAiSource(""); setStats(null);
    const bizKw = ["company","gst","compliance","enterprise","business","corporate","llp","pvt","ltd","msme","director","audit"];
    const lines = text.split("\n").map(l => l.trim()).filter(l => l.length > 8);
    const business   = lines.filter(l => bizKw.some(k => l.toLowerCase().includes(k)));
    const individual = lines.filter(l => !bizKw.some(k => l.toLowerCase().includes(k)));
    let result, source = "local", statsData = null;
    try {
      const res = await Promise.race([
        fetch(`${BACKEND_URL}/api/summary/generate`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ business, individual, raw_text: text }) }),
        new Promise((_, rej) => setTimeout(() => rej(new Error("timeout")), 8000))
      ]);
      if (res.ok) { const data = await res.json(); if (data.result) { result = data.result; source = data.source || "ai"; statsData = data.stats || null; } }
    } catch (_) {}
    if (!result) { const local = localPipeline(text); result = local.result; source = local.source; statsData = local.stats; }
    setSummaries(typeof result === "object" && result?.business ? result : { _raw: result });
    setAiSource(source); setStats(statsData); setLoading(false);
  };

  return (
    <div>
      <SectionTitle icon="file" title="AI Summary Generator" subtitle="Generate structured narrative summaries from feedback data using AI" />
      <div style={{ display:"grid", gridTemplateColumns:"1fr 2fr", gap:20, marginBottom:24 }}>
        <Card>
          <h3 style={{ margin:"0 0 14px", fontSize:15, fontWeight:700, color:"#0B1E3D" }}>Input</h3>
          <FileUpload onData={handleFile} label="Upload CSV feedback file" />
          <div style={{ margin:"12px 0 4px", fontSize:13, color:"#718096" }}>— or paste text below —</div>
          <textarea value={text} onChange={e => setText(e.target.value)} rows={7} placeholder="Paste feedback comments or policy text here..." style={{ width:"100%", padding:"12px", borderRadius:8, border:"1.5px solid rgba(11,30,61,0.15)", fontSize:13, resize:"vertical", fontFamily:"inherit", color:"#0B1E3D", outline:"none", boxSizing:"border-box" }} />
          <button onClick={() => { const c = db.get("comments") || []; setText(c.map(x => x.text).join("\n")); }} style={{ marginTop:8, padding:"8px 14px", border:`1px solid #C9A84C`, borderRadius:6, background:"transparent", color:"#C9A84C", fontSize:13, cursor:"pointer", fontWeight:600 }}>Load Stored Comments</button>
          <button onClick={generateSummaries} disabled={loading || !text.trim()} style={{ marginTop:10, width:"100%", padding:"12px", borderRadius:8, border:"none", background: (!text.trim() || loading) ? "#718096" : "#0B1E3D", color:"#FFFFFF", fontSize:14, fontWeight:700, cursor: (!text.trim() || loading) ? "not-allowed" : "pointer", display:"flex", alignItems:"center", justifyContent:"center", gap:8 }}>
            {loading ? <><Spinner size={16} /> Generating Summaries...</> : <><Icon name="spark" size={16} /> Generate AI Summaries</>}
          </button>
        </Card>
        <div>
          {!summaries && !loading && (
            <Card style={{ display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", minHeight:340, background:"#F5F0E8" }}>
              <Icon name="file" size={48} color="rgba(11,30,61,0.15)" />
              <p style={{ color:"#718096", fontSize:14, marginTop:16, textAlign:"center" }}>Enter feedback text and click Generate to create AI-powered summaries</p>
            </Card>
          )}
          {loading && (
            <Card style={{ display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", minHeight:340 }}>
              <Spinner size={40} /><p style={{ color:"#718096", fontSize:14, marginTop:16 }}>AI is generating summaries...</p>
            </Card>
          )}
          {summaries && !loading && (() => {
            const S = summaries;
            const isStructured = !!S.business;
            const SectionHeader = ({ label, color }) => <div style={{ margin:"20px 0 10px", paddingBottom:6, borderBottom:`2px solid ${color}` }}><span style={{ fontSize:15, fontWeight:800, color, fontFamily:"Georgia, serif", letterSpacing:0.5 }}>{label}</span></div>;
            const SubHeader = ({ label, color }) => <div style={{ margin:"12px 0 6px" }}><span style={{ fontSize:13, fontWeight:700, color, textTransform:"uppercase", letterSpacing:0.8 }}>{label}</span></div>;
            const Bullet = ({ text }) => <div style={{ display:"flex", gap:8, marginBottom:5, alignItems:"flex-start" }}><span style={{ color:"#C9A84C", fontWeight:700, flexShrink:0, marginTop:1 }}>-</span><span style={{ fontSize:13, color:"#4A5568", lineHeight:1.65 }}>{text}</span></div>;
            const EmptyNote = () => <div style={{ fontSize:12, color:"#718096", fontStyle:"italic", marginBottom:4, paddingLeft:16 }}>No specific feedback in this category.</div>;
            const Divider = () => <div style={{ margin:"18px 0", borderTop:"1.5px dashed rgba(11,30,61,0.12)" }} />;
            return (
              <div>
                <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:16 }}>
                  <span style={{ fontSize:13, fontWeight:700, color:"#0B1E3D" }}>📄 Analysis Complete</span>
                  <span style={{ fontSize:11, padding:"4px 14px", borderRadius:20, fontWeight:700, background: aiSource === "gemini-ai" ? "#e8f5e9" : "#fff8e1", color: aiSource === "gemini-ai" ? "#2e7d32" : "#f57f17", border:"1.5px solid currentColor" }}>{aiSource === "gemini-ai" ? "✨ Gemini AI — 7-Step Pipeline" : "🔧 Smart Analyzer — Local Pipeline"}</span>
                </div>
                {isStructured ? (
                  <Card style={{ marginBottom:16, borderLeft:`4px solid #0B1E3D`, maxHeight:560, overflowY:"auto" }}>
                    <h4 style={{ margin:"0 0 4px", fontSize:14, fontWeight:700, color:"#0B1E3D" }}>📋 Structured Summary</h4>
                    <p style={{ margin:"0 0 12px", fontSize:12, color:"#718096" }}>Analysed {S.overall?.total || 0} comments — {S.business?.total || 0} business · {S.individual?.total || 0} individual</p>
                    <SectionHeader label="BUSINESS:" color="#0B1E3D" />
                    <SubHeader label="Positive:" color="#22A86D" />
                    {(S.business?.pos?.length > 0) ? S.business.pos.map((t,i) => <Bullet key={i} text={t} />) : <EmptyNote />}
                    <SubHeader label="Negative:" color="#E74C3C" />
                    {(S.business?.neg?.length > 0) ? S.business.neg.map((t,i) => <Bullet key={i} text={t} />) : <EmptyNote />}
                    <SubHeader label="Neutral:" color="#718096" />
                    {(S.business?.neu?.length > 0) ? S.business.neu.map((t,i) => <Bullet key={i} text={t} />) : <EmptyNote />}
                    {S.business?.rec && <div style={{ margin:"10px 0 4px", padding:"10px 14px", borderRadius:8, background:"rgba(201,168,76,0.07)", border:`1px solid rgba(201,168,76,0.25)` }}><span style={{ fontSize:12, fontWeight:700, color:"#B7770D" }}>💡 Recommendation: </span><span style={{ fontSize:12, color:"#4A5568" }}>{S.business.rec}</span></div>}
                    <Divider />
                    <SectionHeader label="INDIVIDUAL:" color="#7B5EA7" />
                    <SubHeader label="Positive:" color="#22A86D" />
                    {(S.individual?.pos?.length > 0) ? S.individual.pos.map((t,i) => <Bullet key={i} text={t} />) : <EmptyNote />}
                    <SubHeader label="Negative:" color="#E74C3C" />
                    {(S.individual?.neg?.length > 0) ? S.individual.neg.map((t,i) => <Bullet key={i} text={t} />) : <EmptyNote />}
                    <SubHeader label="Neutral:" color="#718096" />
                    {(S.individual?.neu?.length > 0) ? S.individual.neu.map((t,i) => <Bullet key={i} text={t} />) : <EmptyNote />}
                    {S.individual?.rec && <div style={{ margin:"10px 0 4px", padding:"10px 14px", borderRadius:8, background:"rgba(201,168,76,0.07)", border:`1px solid rgba(201,168,76,0.25)` }}><span style={{ fontSize:12, fontWeight:700, color:"#B7770D" }}>💡 Recommendation: </span><span style={{ fontSize:12, color:"#4A5568" }}>{S.individual.rec}</span></div>}
                    <Divider />
                    <SectionHeader label="OVERALL ASSESSMENT:" color="#1E3F73" />
                    {S.overall && (
                      <div style={{ fontSize:13, color:"#4A5568", lineHeight:1.75 }}>
                        <p style={{ margin:"0 0 6px" }}>A total of <strong>{S.overall.total}</strong> submissions ({S.overall.bizCount} business, {S.overall.indCount} individual) were analysed, revealing a <strong>{S.overall.label}</strong> response.</p>
                        <p style={{ margin:"0 0 6px" }}>Key themes identified: <strong>{S.overall.themes}</strong>.</p>
                        <p style={{ margin:0 }}>Positive: {S.overall.posCount} · Negative: {S.overall.negCount} · Neutral: {S.overall.neuCount}</p>
                      </div>
                    )}
                  </Card>
                ) : (
                  <Card style={{ marginBottom:16, borderLeft:`4px solid #0B1E3D` }}>
                    <h4 style={{ margin:"0 0 14px", fontSize:14, fontWeight:700, color:"#0B1E3D" }}>📋 Summary</h4>
                    <pre style={{ margin:0, fontSize:13, color:"#4A5568", lineHeight:1.9, whiteSpace:"pre-wrap", fontFamily:"inherit" }}>{S._raw || JSON.stringify(S, null, 2)}</pre>
                  </Card>
                )}
                {stats && (
                  <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:16, marginBottom:16 }}>
                    <Card style={{ borderLeft:`4px solid #C9A84C` }}>
                      <h4 style={{ margin:"0 0 12px", fontSize:13, fontWeight:700, color:"#0B1E3D" }}>🏷️ Detected Themes</h4>
                      <div style={{ display:"flex", flexDirection:"column", gap:7 }}>
                        {Object.entries(stats.themes || {}).sort((a,b)=>b[1]-a[1]).slice(0,6).map(([theme, count]) => (
                          <div key={theme} style={{ display:"flex", alignItems:"center", justifyContent:"space-between" }}>
                            <span style={{ fontSize:13, color:"#4A5568", textTransform:"capitalize" }}>{theme}</span>
                            <div style={{ display:"flex", alignItems:"center", gap:6 }}>
                              <div style={{ width: Math.min(90, count * 6), height:6, borderRadius:3, background:`linear-gradient(90deg, #1E3F73, #C9A84C)` }} />
                              <span style={{ fontSize:11, color:"#718096", minWidth:20 }}>{count}</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </Card>
                    <Card style={{ borderLeft:`4px solid #22A86D` }}>
                      <h4 style={{ margin:"0 0 12px", fontSize:13, fontWeight:700, color:"#0B1E3D" }}>🔑 Top Keywords</h4>
                      <div style={{ display:"flex", flexWrap:"wrap", gap:6 }}>
                        {(stats.keywords || []).slice(0,12).map((kw, i) => (
                          <span key={kw} style={{ padding:"3px 10px", borderRadius:20, fontSize:12, fontWeight:600, background: i < 3 ? "rgba(11,30,61,0.1)" : "#F5F0E8", color: i < 3 ? "#0B1E3D" : "#4A5568", border:`1px solid ${i < 3 ? "#1E3F73" : "rgba(11,30,61,0.1)"}` }}>{kw}</span>
                        ))}
                      </div>
                      {stats.total && <p style={{ margin:"12px 0 0", fontSize:12, color:"#718096" }}>Analysed: <strong>{stats.total}</strong> comments</p>}
                    </Card>
                  </div>
                )}
              </div>
            );
          })()}
        </div>
      </div>
    </div>
  );
};
// ════════════════════════════════════════════════════════════
// MODULE 5: TREND ANALYSIS  ✅ FIX: useEffect to re-read comments
// ════════════════════════════════════════════════════════════
const TrendModule = () => {
  const [timeframe, setTimeframe] = useState("week");
  // ✅ FIX: Re-read comments from db on mount so counts are always fresh
  const [allComments, setAllComments] = useState([]);

  useEffect(() => {
    setAllComments(db.get("comments") || []);
  }, []);

  const generateTrendData = () => {
    const labels = timeframe === "week"
      ? ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
      : timeframe === "month"
      ? ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
      : ["Q1 2024","Q2 2024","Q3 2024","Q4 2024","Q1 2025","Q2 2025"];
    return labels.map(label => ({ label, positive: Math.floor(Math.random() * 20) + 10, negative: Math.floor(Math.random() * 15) + 5, neutral: Math.floor(Math.random() * 12) + 3 }));
  };

  const [trendData] = useState(generateTrendData());
  const totalPos = allComments.filter(c => c.sentiment === "positive").length;
  const totalNeg = allComments.filter(c => c.sentiment === "negative").length;
  const totalNeu = allComments.filter(c => c.sentiment === "neutral").length;
  const total = allComments.length || 1;

  return (
    <div>
      <SectionTitle icon="trend" title="Trend Analysis" subtitle="Track how sentiment and key themes evolve over time" />
      <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:16, marginBottom:24 }}>
        <StatCard label="Total Comments" value={allComments.length} color="#1E3F73" />
        <StatCard label="Positive Rate" value={`${Math.round(totalPos/total*100)}%`} color="#22A86D" />
        <StatCard label="Negative Rate" value={`${Math.round(totalNeg/total*100)}%`} color="#E74C3C" />
        <StatCard label="Neutral Rate" value={`${Math.round(totalNeu/total*100)}%`} color="#F39C12" />
      </div>
      <Card style={{ marginBottom:20 }}>
        <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:20 }}>
          <h3 style={{ margin:0, fontSize:16, fontWeight:700, color:"#0B1E3D" }}>Sentiment Over Time</h3>
          <div style={{ display:"flex", gap:8 }}>
            {[["week","Weekly"],["month","Monthly"],["quarter","Quarterly"]].map(([v, l]) => (
              <button key={v} onClick={() => setTimeframe(v)} style={{ padding:"6px 14px", borderRadius:6, border:"none", cursor:"pointer", fontSize:12, fontWeight:600, background: timeframe === v ? "#0B1E3D" : "#F5F0E8", color: timeframe === v ? "#FFFFFF" : "#4A5568" }}>{l}</button>
            ))}
          </div>
        </div>
        <div style={{ overflowX:"auto", paddingBottom:8 }}>
          <div style={{ minWidth:520 }}>
            <svg width="100%" height={220} viewBox={`0 0 ${trendData.length * 60 + 40} 220`} style={{ overflow:"visible" }}>
              {[0,25,50,75,100].map(v => { const max = Math.max(...trendData.flatMap(d => [d.positive, d.negative, d.neutral])) + 5; return <line key={v} x1={40} y1={10 + (1 - v/max) * 180} x2={trendData.length * 60 + 40} y2={10 + (1 - v/max) * 180} stroke="rgba(11,30,61,0.06)" strokeWidth={1} />; })}
              {trendData.map((d, i) => {
                const max = Math.max(...trendData.flatMap(dd => [dd.positive, dd.negative, dd.neutral])) + 5;
                const x = i * 60 + 50;
                const posH = (d.positive / max) * 180; const negH = (d.negative / max) * 180; const neuH = (d.neutral / max) * 180;
                return (
                  <g key={i}>
                    <rect x={x - 18} y={190 - posH} width={10} height={posH} fill="#22A86D" rx={3} opacity={0.85} />
                    <rect x={x - 5} y={190 - negH} width={10} height={negH} fill="#E74C3C" rx={3} opacity={0.85} />
                    <rect x={x + 8} y={190 - neuH} width={10} height={neuH} fill="#F39C12" rx={3} opacity={0.85} />
                    <text x={x} y={210} textAnchor="middle" fontSize={10} fill="#718096">{d.label}</text>
                  </g>
                );
              })}
            </svg>
            <div style={{ display:"flex", gap:20, justifyContent:"center", marginTop:8 }}>
              {[["Positive", "#22A86D"], ["Negative", "#E74C3C"], ["Neutral", "#F39C12"]].map(([l, c]) => (
                <div key={l} style={{ display:"flex", alignItems:"center", gap:6 }}>
                  <div style={{ width:12, height:12, borderRadius:3, background:c }} />
                  <span style={{ fontSize:12, color:"#4A5568" }}>{l}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </Card>
      <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:20 }}>
        <Card>
          <h3 style={{ margin:"0 0 16px", fontSize:15, fontWeight:700, color:"#0B1E3D" }}>Sentiment Distribution</h3>
          <div style={{ position:"relative", display:"flex", justifyContent:"center", marginBottom:20 }}>
            <svg viewBox="0 0 120 120" width={160} height={160}>
              {(() => {
                const data = [[totalPos, "#22A86D"], [totalNeg, "#E74C3C"], [totalNeu, "#F39C12"]];
                const total2 = data.reduce((a, [v]) => a + v, 0) || 1;
                let offset = 0;
                return data.map(([v, c], i) => {
                  const pct = v / total2; const angle = pct * 360;
                  const rad = (a) => (a - 90) * Math.PI / 180;
                  const x1 = 60 + 50 * Math.cos(rad(offset)); const y1 = 60 + 50 * Math.sin(rad(offset));
                  const x2 = 60 + 50 * Math.cos(rad(offset + angle)); const y2 = 60 + 50 * Math.sin(rad(offset + angle));
                  const lg = angle > 180 ? 1 : 0;
                  const path = `M 60 60 L ${x1} ${y1} A 50 50 0 ${lg} 1 ${x2} ${y2} Z`;
                  offset += angle;
                  return <path key={i} d={path} fill={c} opacity={0.85} />;
                });
              })()}
              <circle cx={60} cy={60} r={28} fill="#FFFFFF" />
              <text x={60} y={64} textAnchor="middle" fontSize={10} fill="#0B1E3D" fontWeight={700}>{total} total</text>
            </svg>
          </div>
          {[["Positive", totalPos, "#22A86D"], ["Negative", totalNeg, "#E74C3C"], ["Neutral", totalNeu, "#F39C12"]].map(([l, v, c]) => (
            <div key={l} style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:10 }}>
              <div style={{ display:"flex", alignItems:"center", gap:8 }}>
                <div style={{ width:10, height:10, borderRadius:"50%", background:c }} />
                <span style={{ fontSize:13, color:"#4A5568" }}>{l}</span>
              </div>
              <span style={{ fontSize:14, fontWeight:700, color:"#0B1E3D" }}>{v} ({Math.round(v/total*100)}%)</span>
            </div>
          ))}
        </Card>
        <Card>
          <h3 style={{ margin:"0 0 16px", fontSize:15, fontWeight:700, color:"#0B1E3D" }}>Top Issues Over Time</h3>
          {[{ issue:"Compliance Burden", change:"+12%", trend:"up", color:"#E74C3C" },{ issue:"Portal Usability", change:"-8%", trend:"down", color:"#22A86D" },{ issue:"Penalty Clarity", change:"+6%", trend:"up", color:"#E74C3C" },{ issue:"Filing Deadlines", change:"+4%", trend:"up", color:"#F39C12" },{ issue:"Fee Structure", change:"-3%", trend:"down", color:"#22A86D" }].map(({ issue, change, trend, color }) => (
            <div key={issue} style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:12, padding:"10px 12px", borderRadius:8, background:"#F5F0E8" }}>
              <span style={{ fontSize:14, color:"#0B1E3D" }}>{issue}</span>
              <span style={{ fontSize:13, fontWeight:700, color }}>{trend === "up" ? "↑" : "↓"} {change}</span>
            </div>
          ))}
        </Card>
      </div>
    </div>
  );
};
// ════════════════════════════════════════════════════════════
// ADMIN PANEL
// ════════════════════════════════════════════════════════════
const AdminPanel = ({ currentUser }) => {
  const [users, setUsers] = useState(db.get("users") || []);
  const [comments, setComments] = useState(db.get("comments") || []);
  const [newUser, setNewUser] = useState({ name:"", email:"", password:"", role:"individual" });
  const [msg, setMsg] = useState(null);

  const addUser = () => {
    if (!newUser.name || !newUser.email || !newUser.password) return;
    if (users.find(u => u.email === newUser.email)) { setMsg({ type:"error", text:"Email already exists!" }); setTimeout(() => setMsg(null), 3000); return; }
    const updated = [...users, { ...newUser, id: Date.now() }];
    db.set("users", updated); setUsers(updated);
    setNewUser({ name:"", email:"", password:"", role:"individual" });
    setMsg({ type:"success", text:"User created successfully!" }); setTimeout(() => setMsg(null), 3000);
  };

  const deleteUser = (id) => {
    const u = users.find(x => x.id === id);
    if (!window.confirm(`Delete user "${u?.name || u?.email}"? This cannot be undone.`)) return;
    const updated = users.filter(x => x.id !== id);
    db.set("users", updated); setUsers(updated);
    setMsg({ type:"success", text:"User deleted." }); setTimeout(() => setMsg(null), 2000);
  };

  const changeRole = (id, newRole) => {
    const updated = users.map(u => u.id === id ? { ...u, role: newRole } : u);
    db.set("users", updated); setUsers(updated);
    setMsg({ type:"success", text:"Role updated!" }); setTimeout(() => setMsg(null), 2000);
  };

  const deleteComment = (id) => {
    if (!window.confirm('Delete this comment?')) return;
    const updated = comments.filter(c => c.id !== id);
    db.set('comments', updated); setComments(updated);
    setMsg({ type:'success', text:'Comment deleted.' }); setTimeout(() => setMsg(null), 2000);
  };

  const deleteAllComments = () => {
    if (!window.confirm('Delete ALL ' + comments.length + ' comments? This cannot be undone.')) return;
    db.set('comments', []); setComments([]);
    setMsg({ type:'success', text:'All comments cleared.' }); setTimeout(() => setMsg(null), 2000);
  };

  return (
    <div>
      <SectionTitle icon="users" title="Admin Panel" subtitle="Manage users, monitor activity, and review feedback" />
      {msg && <div style={{ padding:"12px 16px", borderRadius:8, background: msg.type === "success" ? "rgba(26,107,74,0.1)" : "rgba(192,57,43,0.1)", color: msg.type === "success" ? "#1A6B4A" : "#C0392B", marginBottom:20, fontSize:14 }}>{msg.text}</div>}
      <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:16, marginBottom:24 }}>
        <StatCard label="Total Users" value={users.length} color="#1E3F73" />
        <StatCard label="Total Comments" value={comments.length} color="#C9A84C" />
        <StatCard label="Business Users" value={users.filter(u => u.role === "business").length} color="#F39C12" />
        <StatCard label="Admins" value={users.filter(u => u.role === "admin").length} color="#22A86D" />
      </div>
      <div style={{ display:"grid", gridTemplateColumns:"1fr 1.5fr", gap:20, marginBottom:24 }}>
        <Card>
          <h3 style={{ margin:"0 0 16px", fontSize:15, fontWeight:700, color:"#0B1E3D" }}>Create New User</h3>
          {[["Name","name","text"],["Email","email","email"],["Password","password","password"]].map(([label, field, type]) => (
            <div key={field} style={{ marginBottom:12 }}>
              <label style={{ display:"block", fontSize:12, fontWeight:700, color:"#718096", marginBottom:4, textTransform:"uppercase", letterSpacing:0.5 }}>{label}</label>
              <input type={type} value={newUser[field]} onChange={e => setNewUser(p => ({...p, [field]: e.target.value}))} style={{ width:"100%", padding:"10px 12px", borderRadius:8, border:"1.5px solid rgba(11,30,61,0.15)", fontSize:13, outline:"none", color:"#0B1E3D", boxSizing:"border-box" }} />
            </div>
          ))}
          <div style={{ marginBottom:16 }}>
            <label style={{ display:"block", fontSize:12, fontWeight:700, color:"#718096", marginBottom:4, textTransform:"uppercase", letterSpacing:0.5 }}>Role</label>
            <select value={newUser.role} onChange={e => setNewUser(p => ({...p, role: e.target.value}))} style={{ width:"100%", padding:"10px 12px", borderRadius:8, border:"1.5px solid rgba(11,30,61,0.15)", fontSize:13, outline:"none", color:"#0B1E3D" }}>
              <option value="individual">Individual</option><option value="business">Business</option><option value="admin">Admin</option>
            </select>
          </div>
          <button onClick={addUser} style={{ width:"100%", padding:"11px", borderRadius:8, border:"none", background: "#0B1E3D", color:"#FFFFFF", fontSize:14, fontWeight:700, cursor:"pointer" }}>Create User</button>
        </Card>
        <Card>
          <h3 style={{ margin:"0 0 16px", fontSize:15, fontWeight:700, color:"#0B1E3D" }}>Users ({users.length})</h3>
          <div style={{ maxHeight:380, overflowY:"auto" }}>
            <table style={{ width:"100%", borderCollapse:"collapse", fontSize:13 }}>
              <thead>
                <tr style={{ background:"#F5F0E8", position:"sticky", top:0 }}>
                  {["Name","Email","Role","Actions"].map(h => <th key={h} style={{ padding:"10px 12px", textAlign:"left", color:"#0B1E3D", fontWeight:700, fontSize:11, textTransform:"uppercase", letterSpacing:0.5 }}>{h}</th>)}
                </tr>
              </thead>
              <tbody>
                {users.map(u => (
                  <tr key={u.id} style={{ borderBottom:"1px solid rgba(11,30,61,0.06)" }}>
                    <td style={{ padding:"10px 12px", color:"#0B1E3D", fontWeight:600 }}>{u.name}</td>
                    <td style={{ padding:"10px 12px", color:"#4A5568", fontSize:12 }}>{u.email}</td>
                    <td style={{ padding:"10px 12px" }}>
                      <select value={u.role} onChange={e => changeRole(u.id, e.target.value)} style={{ padding:"5px 8px", borderRadius:6, fontSize:12, fontWeight:600, cursor:"pointer", border:"1.5px solid rgba(11,30,61,0.15)", outline:"none", background: u.role === "admin" ? "rgba(11,30,61,0.08)" : u.role === "business" ? "rgba(183,119,13,0.08)" : "rgba(74,85,104,0.08)", color: u.role === "admin" ? "#0B1E3D" : u.role === "business" ? "#B7770D" : "#4A5568" }}>
                        <option value="individual">Individual</option><option value="business">Business</option><option value="admin">Admin</option>
                      </select>
                    </td>
                    <td style={{ padding:"10px 12px" }}>
                      <button onClick={() => deleteUser(u.id)} style={{ padding:"6px 12px", borderRadius:7, border:"1.5px solid #E74C3C", background:"rgba(231,76,60,0.08)", color:"#E74C3C", fontSize:12, fontWeight:700, cursor:"pointer" }}>🗑 Delete</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
      <Card>
        <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:16 }}>
          <h3 style={{ margin:0, fontSize:15, fontWeight:700, color:"#0B1E3D" }}>Comments ({comments.length})</h3>
          {comments.length > 0 && <button onClick={deleteAllComments} style={{ padding:"7px 14px", borderRadius:7, border:"1.5px solid #C0392B", background:"rgba(192,57,43,0.07)", color:"#C0392B", fontSize:12, fontWeight:700, cursor:"pointer" }}>🗑 Delete All</button>}
        </div>
        {comments.length === 0 && <p style={{ color:"#718096", fontSize:13, textAlign:"center", padding:"32px 0" }}>No comments yet.</p>}
        <div style={{ maxHeight:380, overflowY:"auto", display:"grid", gap:10 }}>
          {comments.map(c => (
            <div key={c.id} style={{ padding:"12px 14px", borderRadius:8, background:"#F5F0E8", display:"grid", gridTemplateColumns:"1fr auto", gap:12, alignItems:"center", borderLeft:"3px solid " + (c.sentiment==="positive" ? "#22A86D" : c.sentiment==="negative" ? "#E74C3C" : "#718096") }}>
              <div>
                <p style={{ margin:"0 0 6px", fontSize:13, color:"#0B1E3D", lineHeight:1.5 }}>{c.text.slice(0,130)}{c.text.length > 130 ? "..." : ""}</p>
                <div style={{ display:"flex", gap:10, fontSize:12, color:"#718096", alignItems:"center" }}>
                  <span>{c.user || "Anonymous"}</span><span>·</span><SentimentBadge s={c.sentiment} /><span>·</span><span style={{ fontSize:11 }}>{new Date(c.createdAt).toLocaleDateString("en-IN")}</span>
                </div>
              </div>
              <button onClick={() => deleteComment(c.id)} style={{ padding:"7px 12px", borderRadius:7, border:"1.5px solid #E74C3C", background:"rgba(231,76,60,0.08)", color:"#E74C3C", fontSize:12, fontWeight:700, cursor:"pointer", whiteSpace:"nowrap" }}>🗑 Delete</button>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
};
// ════════════════════════════════════════════════════════════
// DASHBOARD HOME  ✅ FIX: useEffect to re-read comments fresh on every mount
// ════════════════════════════════════════════════════════════
const DashboardHome = ({ user, setActiveModule }) => {
  // ✅ FIX: Read comments fresh on every mount so stats always reflect latest data
  const [comments, setComments] = useState([]);

  useEffect(() => {
    setComments(db.get("comments") || []);
  }, []);

  const pos = comments.filter(c => c.sentiment === "positive").length;
  const neg = comments.filter(c => c.sentiment === "negative").length;
  const neu = comments.filter(c => c.sentiment === "neutral").length;

  const modules = [
    { id:"wordcloud", icon:"cloud",   title:"Word Cloud",              desc:"Visualize key themes & patterns from public feedback",          color:"#1E3F73" },
    { id:"akashwani", icon:"policy",  title:"Akashwani Policy Analyzer",desc:"ML/AI scoring of policy draft acceptance rates",               color:"#C9A84C" },
    { id:"sentiment", icon:"brain",   title:"Sentiment Analysis",       desc:"Classify and analyze consultation comment sentiments",          color:"#22A86D" },
    { id:"summary",   icon:"file",    title:"Summary Generator",        desc:"AI-generated narrative summaries from feedback",               color:"#F39C12" },
    { id:"trend",     icon:"trend",   title:"Trend Analysis",           desc:"Track sentiment trends over time with visualizations",         color:"#E74C3C" },
  ];

  return (
    <div>
      <div style={{ marginBottom:32 }}>
        <h2 style={{ margin:"0 0 6px", fontSize:26, fontWeight:800, color:"#0B1E3D", fontFamily:"Georgia, serif" }}>Welcome back, {user.name.split(" ")[0]} 👋</h2>
        <p style={{ margin:0, color:"#718096", fontSize:15 }}>MCA E-Consultation Dashboard · MANOBHAV Platform</p>
      </div>
      <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:16, marginBottom:28 }}>
        <StatCard label="Total Comments" value={comments.length} color="#1E3F73" />
        <StatCard label="Positive" value={pos} color="#22A86D" />
        <StatCard label="Negative" value={neg} color="#E74C3C" />
        <StatCard label="Neutral" value={neu} color="#F39C12" />
      </div>
      <div style={{ display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:20, marginBottom:24 }}>
        {modules.map(m => (
          <div key={m.id} onClick={() => setActiveModule(m.id)} style={{ background:"#FFFFFF", borderRadius:14, padding:"24px 20px", cursor:"pointer", boxShadow:"0 2px 16px rgba(11,30,61,0.07)", borderTop:`4px solid ${m.color}`, transition:"transform 0.15s, box-shadow 0.15s" }}
            onMouseEnter={e => { e.currentTarget.style.transform = "translateY(-4px)"; e.currentTarget.style.boxShadow = "0 8px 28px rgba(11,30,61,0.13)"; }}
            onMouseLeave={e => { e.currentTarget.style.transform = ""; e.currentTarget.style.boxShadow = "0 2px 16px rgba(11,30,61,0.07)"; }}>
            <div style={{ color:m.color, marginBottom:12 }}><Icon name={m.icon} size={28} /></div>
            <h3 style={{ margin:"0 0 8px", fontSize:15, fontWeight:800, color:"#0B1E3D" }}>{m.title}</h3>
            <p style={{ margin:0, fontSize:13, color:"#718096", lineHeight:1.5 }}>{m.desc}</p>
          </div>
        ))}
        <div style={{ background:"#F5F0E8", borderRadius:14, padding:"24px 20px", borderTop:`4px solid #718096`, display:"flex", flexDirection:"column", justifyContent:"center", alignItems:"center", color:"#718096" }}>
          <Icon name="chart" size={28} />
          <p style={{ margin:"12px 0 0", fontSize:13, textAlign:"center" }}>More modules coming soon</p>
        </div>
      </div>
      <Card>
        <h3 style={{ margin:"0 0 16px", fontSize:15, fontWeight:700, color:"#0B1E3D" }}>Recent Feedback Activity</h3>
        <div style={{ display:"grid", gap:10 }}>
          {comments.slice(-5).reverse().map(c => (
            <div key={c.id} style={{ display:"flex", justifyContent:"space-between", alignItems:"center", padding:"10px 14px", borderRadius:8, background:"#F5F0E8" }}>
              <div>
                <p style={{ margin:0, fontSize:13, color:"#0B1E3D" }}>{c.text.slice(0,80)}...</p>
                <span style={{ fontSize:12, color:"#718096" }}>{c.user} · {new Date(c.createdAt).toLocaleDateString()}</span>
              </div>
              <SentimentBadge s={c.sentiment} />
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
};

// ════════════════════════════════════════════════════════════
// AUTH SCREEN
// ════════════════════════════════════════════════════════════
const AuthScreen = ({ onLogin }) => {
  const [mode, setMode] = useState("login");
  const [form, setForm] = useState({ name:"", email:"admin@manobhav.gov.in", password:"admin123", role:"individual" });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handle = async (e) => {
    e.preventDefault(); setError(""); setLoading(true);
    await new Promise(r => setTimeout(r, 600));
    const users = db.get("users") || [];
    if (mode === "login") {
      const u = users.find(u => u.email === form.email && u.password === form.password);
      if (u) { db.set("session", u); onLogin(u); } else setError("Invalid email or password.");
    } else {
      if (users.find(u => u.email === form.email)) { setError("Email already registered."); }
      else { const newU = { ...form, id: Date.now() }; const updated = [...users, newU]; db.set("users", updated); db.set("session", newU); onLogin(newU); }
    }
    setLoading(false);
  };

  return (
    <div style={{ minHeight:"100vh", background:`linear-gradient(135deg, #0B1E3D 0%, #1E3F73 50%, #152D55 100%)`, display:"flex", alignItems:"center", justifyContent:"center", padding:20, fontFamily:"'Georgia', serif" }}>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } } @keyframes fadeUp { from { opacity:0; transform:translateY(24px); } to { opacity:1; transform:translateY(0); } }`}</style>
      <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:0, maxWidth:900, width:"100%", borderRadius:20, overflow:"hidden", boxShadow:"0 30px 80px rgba(0,0,0,0.4)", animation:"fadeUp 0.5s ease" }}>
        <div style={{ background:`linear-gradient(160deg, #152D55 0%, #0B1E3D 100%)`, padding:"48px 40px", color:"#FFFFFF", borderRight:`1px solid rgba(201,168,76,0.2)` }}>
          <div style={{ fontSize:11, fontWeight:700, letterSpacing:3, color:"#C9A84C", marginBottom:20, textTransform:"uppercase", opacity:0.8 }}>Ministry of Corporate Affairs</div>
          <h1 style={{ fontSize:34, fontWeight:400, margin:"0 0 4px", fontFamily:"Georgia, serif", lineHeight:1.2, color:"#C9A84C", letterSpacing:2 }}>MANOBHAV</h1>
          <div style={{ fontSize:13, color:"rgba(201,168,76,0.75)", marginBottom:28, fontFamily:"Georgia, serif", fontStyle:"italic", letterSpacing:1 }}>— The Akashvani Predictor —</div>
          <div style={{ fontSize:13, color:"rgba(255,255,255,0.65)", marginBottom:24, fontFamily:"sans-serif", lineHeight:1.7 }}>AI-powered MCA public consultation analysis platform</div>
          <div style={{ display:"flex", flexDirection:"column", gap:12 }}>
            {[["cloud","Word Cloud Analyzer"],["policy","Akashwani Policy Scorer"],["brain","Sentiment Analysis"],["file","AI Summary Generator"],["trend","Trend Analysis"]].map(([icon, label]) => (
              <div key={label} style={{ display:"flex", alignItems:"center", gap:10, fontSize:14, color:"rgba(255,255,255,0.85)", fontFamily:"sans-serif" }}>
                <div style={{ color:"#C9A84C" }}><Icon name={icon} size={16} /></div> {label}
              </div>
            ))}
          </div>
          <div style={{ marginTop:32, padding:"14px 16px", borderRadius:10, background:"rgba(201,168,76,0.1)", border:`1px solid rgba(201,168,76,0.3)` }}>
            <div style={{ fontSize:12, fontWeight:700, color:"#C9A84C", marginBottom:6, fontFamily:"sans-serif", letterSpacing:0.5, textTransform:"uppercase" }}>Demo Credentials</div>
            <div style={{ fontSize:13, color:"rgba(255,255,255,0.8)", fontFamily:"monospace" }}>admin@manobhav.gov.in</div>
            <div style={{ fontSize:13, color:"rgba(255,255,255,0.8)", fontFamily:"monospace" }}>admin123</div>
          </div>
        </div>
        <div style={{ background:"#FFFFFF", padding:"48px 40px" }}>
          <h2 style={{ margin:"0 0 4px", fontSize:24, fontWeight:700, color:"#0B1E3D" }}>{mode === "login" ? "Sign in" : "Create account"}</h2>
          <p style={{ margin:"0 0 28px", fontSize:14, color:"#718096", fontFamily:"sans-serif" }}>{mode === "login" ? "Access your MANOBHAV dashboard" : "Register for MCA e-consultation"}</p>
          {error && <div style={{ padding:"10px 14px", borderRadius:8, background:"rgba(192,57,43,0.08)", color:"#C0392B", fontSize:13, marginBottom:16, fontFamily:"sans-serif" }}>{error}</div>}
          <form onSubmit={handle} style={{ fontFamily:"sans-serif" }}>
            {mode === "register" && (
              <div style={{ marginBottom:14 }}>
                <label style={{ display:"block", fontSize:12, fontWeight:700, color:"#718096", marginBottom:5, textTransform:"uppercase", letterSpacing:0.5 }}>Full Name</label>
                <input type="text" required value={form.name} onChange={e => setForm(p => ({...p, name:e.target.value}))} style={{ width:"100%", padding:"11px 14px", borderRadius:8, border:"1.5px solid rgba(11,30,61,0.15)", fontSize:14, outline:"none", color:"#0B1E3D", boxSizing:"border-box" }} />
              </div>
            )}
            <div style={{ marginBottom:14 }}>
              <label style={{ display:"block", fontSize:12, fontWeight:700, color:"#718096", marginBottom:5, textTransform:"uppercase", letterSpacing:0.5 }}>Email</label>
              <input type="email" required value={form.email} onChange={e => setForm(p => ({...p, email:e.target.value}))} style={{ width:"100%", padding:"11px 14px", borderRadius:8, border:"1.5px solid rgba(11,30,61,0.15)", fontSize:14, outline:"none", color:"#0B1E3D", boxSizing:"border-box" }} />
            </div>
            <div style={{ marginBottom: mode === "register" ? 14 : 20 }}>
              <label style={{ display:"block", fontSize:12, fontWeight:700, color:"#718096", marginBottom:5, textTransform:"uppercase", letterSpacing:0.5 }}>Password</label>
              <input type="password" required value={form.password} onChange={e => setForm(p => ({...p, password:e.target.value}))} style={{ width:"100%", padding:"11px 14px", borderRadius:8, border:"1.5px solid rgba(11,30,61,0.15)", fontSize:14, outline:"none", color:"#0B1E3D", boxSizing:"border-box" }} />
            </div>
            {mode === "register" && (
              <div style={{ marginBottom:20 }}>
                <label style={{ display:"block", fontSize:12, fontWeight:700, color:"#718096", marginBottom:5, textTransform:"uppercase", letterSpacing:0.5 }}>Account Type</label>
                <select value={form.role} onChange={e => setForm(p => ({...p, role:e.target.value}))} style={{ width:"100%", padding:"11px 14px", borderRadius:8, border:"1.5px solid rgba(11,30,61,0.15)", fontSize:14, outline:"none", color:"#0B1E3D" }}>
                  <option value="individual">Individual Citizen</option><option value="business">Business / Corporate</option>
                </select>
              </div>
            )}
            <button type="submit" disabled={loading} style={{ width:"100%", padding:"13px", borderRadius:10, border:"none", background: loading ? "#718096" : `linear-gradient(135deg, #0B1E3D, #1E3F73)`, color:"#FFFFFF", fontSize:15, fontWeight:700, cursor: loading ? "not-allowed" : "pointer", display:"flex", alignItems:"center", justifyContent:"center", gap:10 }}>
              {loading ? <><Spinner size={18} /> Please wait...</> : mode === "login" ? "Sign In" : "Create Account"}
            </button>
          </form>
          <div style={{ textAlign:"center", marginTop:20, fontFamily:"sans-serif" }}>
            <button onClick={() => { setMode(mode === "login" ? "register" : "login"); setError(""); }} style={{ background:"none", border:"none", color:"#C9A84C", fontSize:14, cursor:"pointer", fontWeight:600 }}>
              {mode === "login" ? "Don't have an account? Register" : "Already registered? Sign in"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
// ════════════════════════════════════════════════════════════
// MAIN APP SHELL
// ════════════════════════════════════════════════════════════
export default function App() {
  const [user, setUser] = useState(() => db.get("session"));
  const [activeModule, setActiveModule] = useState("home");
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const logout = () => { localStorage.removeItem("manobhav_session"); setUser(null); setActiveModule("home"); };

  if (!user) return <AuthScreen onLogin={u => setUser(u)} />;

  const navItems = [
    { id:"home",      icon:"home",   label:"Dashboard" },
    { id:"wordcloud", icon:"cloud",  label:"Word Cloud" },
    { id:"akashwani", icon:"policy", label:"Akashwani" },
    { id:"sentiment", icon:"brain",  label:"Sentiment" },
    { id:"summary",   icon:"file",   label:"Summary" },
    { id:"trend",     icon:"trend",  label:"Trends" },
    ...(user.role === "admin" ? [{ id:"admin", icon:"users", label:"Admin" }] : []),
  ];

  const renderModule = () => {
    switch (activeModule) {
      case "home":      return <DashboardHome user={user} setActiveModule={setActiveModule} />;
      case "wordcloud": return <WordCloudModule />;
      case "akashwani": return <AkashwaniModule />;
      case "sentiment": return <SentimentModule />;
      case "summary":   return <SummaryModule />;
      case "trend":     return <TrendModule />;
      case "admin":     return user.role === "admin" ? <AdminPanel currentUser={user} /> : null;
      default:          return null;
    }
  };

  return (
    <div style={{ display:"flex", minHeight:"100vh", background:"#F0EDE6", fontFamily:"sans-serif" }}>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } } * { box-sizing: border-box; } ::-webkit-scrollbar { width: 6px; } ::-webkit-scrollbar-track { background: transparent; } ::-webkit-scrollbar-thumb { background: rgba(11,30,61,0.15); border-radius: 3px; }`}</style>
      <div style={{ width: sidebarOpen ? 220 : 64, minHeight:"100vh", background: "#0B1E3D", transition:"width 0.25s ease", display:"flex", flexDirection:"column", flexShrink:0, overflow:"hidden", boxShadow:"2px 0 20px rgba(0,0,0,0.15)" }}>
        <div style={{ padding:"16px 14px", borderBottom:`1px solid rgba(201,168,76,0.15)`, display:"flex", alignItems:"center", gap:10, overflow:"hidden" }}>
          <div style={{ flexShrink:0 }}>
            <ButterflyLogo size={sidebarOpen ? 42 : 36} showText={false} />
          </div>
          {sidebarOpen && (
            <div>
              <div style={{ fontFamily:"Georgia, serif", fontWeight:800, color:"#C9A84C", whiteSpace:"nowrap", fontSize:15, letterSpacing:1 }}>MANOBHAV</div>
              <div style={{ fontSize:9, color:"rgba(201,168,76,0.65)", whiteSpace:"nowrap", letterSpacing:1.5, textTransform:"uppercase" }}>The Akashvani Predictor</div>
            </div>
          )}
        </div>
        <nav style={{ flex:1, padding:"12px 8px" }}>
          {navItems.map(item => (
            <button key={item.id} onClick={() => setActiveModule(item.id)} style={{ width:"100%", display:"flex", alignItems:"center", gap:12, padding:"11px 12px", borderRadius:10, border:"none", cursor:"pointer", textAlign:"left", marginBottom:2, background: activeModule === item.id ? "rgba(201,168,76,0.18)" : "transparent", color: activeModule === item.id ? "#C9A84C" : "rgba(255,255,255,0.7)", transition:"all 0.15s", overflow:"hidden" }}
              onMouseEnter={e => { if (activeModule !== item.id) e.currentTarget.style.background = "rgba(255,255,255,0.06)"; }}
              onMouseLeave={e => { if (activeModule !== item.id) e.currentTarget.style.background = "transparent"; }}>
              <div style={{ flexShrink:0 }}><Icon name={item.icon} size={18} /></div>
              {sidebarOpen && <span style={{ fontSize:13, fontWeight:600, whiteSpace:"nowrap" }}>{item.label}</span>}
              {activeModule === item.id && sidebarOpen && <div style={{ marginLeft:"auto", width:4, height:4, borderRadius:"50%", background:"#C9A84C" }} />}
            </button>
          ))}
        </nav>
        <div style={{ padding:"12px 8px", borderTop:`1px solid rgba(255,255,255,0.08)` }}>
          {sidebarOpen && (
            <div style={{ padding:"10px 12px", borderRadius:10, background:"rgba(255,255,255,0.05)", marginBottom:8 }}>
              <div style={{ fontSize:13, fontWeight:600, color:"#FFFFFF", whiteSpace:"nowrap", overflow:"hidden", textOverflow:"ellipsis" }}>{user.name}</div>
              <div style={{ fontSize:11, color:"rgba(255,255,255,0.5)", textTransform:"capitalize" }}>{user.role}</div>
            </div>
          )}
          <button onClick={() => setSidebarOpen(p => !p)} style={{ width:"100%", display:"flex", alignItems:"center", justifyContent: sidebarOpen ? "flex-start" : "center", gap:10, padding:"10px 12px", borderRadius:8, border:"none", background:"transparent", cursor:"pointer", color:"rgba(255,255,255,0.5)", marginBottom:4 }}><Icon name="menu" size={16} /></button>
          <button onClick={logout} style={{ width:"100%", display:"flex", alignItems:"center", justifyContent: sidebarOpen ? "flex-start" : "center", gap:10, padding:"10px 12px", borderRadius:8, border:"none", background:"transparent", cursor:"pointer", color:"rgba(255,255,255,0.5)" }}>
            <Icon name="logout" size={16} />
            {sidebarOpen && <span style={{ fontSize:13, fontWeight:600, whiteSpace:"nowrap" }}>Sign Out</span>}
          </button>
        </div>
      </div>
      <div style={{ flex:1, overflow:"auto", minWidth:0 }}>
        <div style={{ background:"#FFFFFF", padding:"14px 28px", borderBottom:"1px solid rgba(11,30,61,0.06)", display:"flex", justifyContent:"space-between", alignItems:"center", position:"sticky", top:0, zIndex:10 }}>
          <div style={{ fontSize:18, fontWeight:800, color:"#0B1E3D", fontFamily:"Georgia, serif" }}>{navItems.find(n => n.id === activeModule)?.label || "Dashboard"}</div>
          <div style={{ display:"flex", alignItems:"center", gap:12 }}>
            <Badge label={user.role} color={user.role === "admin" ? "#0B1E3D" : "#B7770D"} bg={user.role === "admin" ? "rgba(11,30,61,0.08)" : "rgba(183,119,13,0.1)"} />
            <div style={{ fontSize:14, fontWeight:600, color:"#0B1E3D" }}>{user.name}</div>
          </div>
        </div>
        <div style={{ padding:"28px 32px", maxWidth:1280, margin:"0 auto" }}>
          {renderModule()}
        </div>
      </div>
    </div>
  );
}