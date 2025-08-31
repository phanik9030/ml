// src/App.js
import React, {
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import axios from "axios";
import {
  AppBar,
  Toolbar,
  Typography,
  Container,
  Paper,
  Box,
  TextField,
  IconButton,
  Stack,
  Table,
  TableHead,
  TableRow,
  TableCell,
  TableBody,
  Chip,
  Collapse,
  Tooltip,
  TableContainer,
  Autocomplete,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Divider,
} from "@mui/material";
import SendIcon from "@mui/icons-material/Send";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ExpandLessIcon from "@mui/icons-material/ExpandLess";
import CheckIcon from "@mui/icons-material/CheckCircle";

const API_BASE = process.env.REACT_APP_API_BASE || ""; // CRA proxy -> Flask

// ---------- Minimal CSS-in-Component ----------
const GlobalStyles = () => (
  <style>{`
  /* Layout */
  .chat-container { display:flex; flex-direction:column; height:100vh; background:#fafafa; }
  .chat-scroll { flex:1; overflow-y:auto; padding: 8px 0 12px; }
  .composer { position:sticky; bottom:0; background:transparent; padding: 6px 0 8px; }

  /* Bubbles */
  .bubble { max-width: min(720px, 100%); border-radius: 14px; padding: 10px 12px; }
  .bubble-user { background:#111; color:#fff; border:1px solid #111; }
  .bubble-assistant { background:#fff; border:1px solid #eaeaea; }
  .bubble p { margin: 0 0 6px; line-height: 1.35; font-size: 0.94rem; }
  .bubble small { color:#6a6a6a; }

  /* Section headers (dynamic) */
  .sec-title { font-weight: 700; margin: 2px 0 6px; font-size: 0.95rem; }
  .sec-block { margin: 6px 0 10px; }

  /* Code block for SQL */
  .code-block { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
                font-size: 12.5px; background:#0b0b0b; color:#eaeaea; border-radius:10px; padding:10px 12px; overflow:auto; }
  .sql-k { font-weight:700; }
  .sql-f { text-decoration: underline; }
  .sql-n { opacity:0.9; }
  .sql-s { color:#aad1ff; }
  .sql-c { color:#7e7e7e; font-style:italic; }

  /* Loader */
  .loader-bubble { display:flex; align-items:center; gap:10px; }
  .loader { display:inline-flex; gap:5px; }
  .loader-dot { width:6px; height:6px; border-radius:999px; background:#bbb; display:inline-block; animation: b 1s infinite ease-in-out; }
  .loader-dot:nth-child(2) { animation-delay: .15s; }
  .loader-dot:nth-child(3) { animation-delay: .3s; }
  @keyframes b { 0%, 80%, 100% { opacity: .25 } 40% { opacity: 1 } }

  /* Suggestion chips */
  .suggestion-bar { margin-top: 6px; }

  /* Table */
  .result-title { font-weight: 700; margin-bottom: 6px; font-size: 0.9rem; }
`}</style>
);

// --- Tiny SQL highlighter (monochrome, no extra deps) ---
function escapeHtml(str) {
  return str.replace(
    /[&<>"']/g,
    (s) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[
        s
      ])
  );
}
function highlightSQL(sql) {
  if (!sql) return "";
  let s = escapeHtml(sql);

  // Comments
  s = s.replace(/(--.*?$)/gm, '<span class="sql-c">$1</span>');
  // Strings
  s = s.replace(/'([^']*)'/g, '<span class="sql-s">&#39;$1&#39;</span>');

  // Functions
  const funcs = [
    "COUNT",
    "SUM",
    "AVG",
    "MIN",
    "MAX",
    "DATE_TRUNC",
    "CURRENT_DATE",
    "COALESCE",
    "ROUND",
    "CAST",
  ];
  const funcRe = new RegExp("\\b(" + funcs.join("|") + ")\\s*\\(", "gi");
  s = s.replace(funcRe, (m, g1) => `<span class="sql-f">${g1}</span>(`);

  // Keywords
  const keywords = [
    "SELECT",
    "FROM",
    "WHERE",
    "JOIN",
    "INNER",
    "LEFT",
    "RIGHT",
    "FULL",
    "ON",
    "AND",
    "OR",
    "NOT",
    "GROUP",
    "BY",
    "ORDER",
    "HAVING",
    "LIMIT",
    "AS",
    "DISTINCT",
    "CASE",
    "WHEN",
    "THEN",
    "ELSE",
    "END",
    "WITH",
    "UNION",
    "ALL",
    "IN",
    "IS",
    "NULL",
    "LIKE",
    "BETWEEN",
    "INTERVAL",
  ];
  const kwRe = new RegExp("\\b(" + keywords.join("|") + ")\\b", "gi");
  s = s.replace(kwRe, '<span class="sql-k">$1</span>');

  // Numbers
  s = s.replace(/\b\d+(\.\d+)?\b/g, '<span class="sql-n">$&</span>');

  return s;
}

// Loader bubble
function LoaderBubble() {
  return (
    <div className="loader-bubble">
      <div className="loader">
        <span className="loader-dot" />
        <span className="loader-dot" />
        <span className="loader-dot" />
      </div>
      <Typography variant="body2" sx={{ color: "#555", fontSize: "0.85rem" }}>
        Thinking…
      </Typography>
    </div>
  );
}

// --- Dynamic content renderer with headers ---
// Treat lines that look like headings as section titles:
// 1) Markdown "## Heading"
// 2) Bold-with-colon "**Heading:**"
// 3) Title-like "Heading:" on its own line
function renderAssistantContent(text) {
  if (!text) return null;
  const lines = text.split(/\r?\n/);

  const blocks = [];
  let cur = [];
  let pushBlock = () => {
    if (cur.length) {
      blocks.push({ type: "p", text: cur.join("\n") });
      cur = [];
    }
  };

  lines.forEach((raw) => {
    const line = raw.trimRight();

    const mdHeader = line.match(/^##\s+(.+?)\s*$/);
    const boldHeader = line.match(/^\*\*(.+?)\*\*:\s*$/);
    const colonHeader = !mdHeader && !boldHeader && line.match(/^(.+?):\s*$/);

    if (mdHeader || boldHeader || colonHeader) {
      pushBlock();
      const title = (
        mdHeader?.[1] ||
        boldHeader?.[1] ||
        colonHeader?.[1]
      ).trim();
      blocks.push({ type: "h", text: title });
    } else if (line === "---") {
      pushBlock();
      blocks.push({ type: "hr" });
    } else {
      cur.push(line);
    }
  });
  pushBlock();

  return (
    <>
      {blocks.map((b, i) => {
        if (b.type === "h") {
          return (
            <div key={`h-${i}`} className="sec-block">
              <div className="sec-title">{b.text}</div>
            </div>
          );
        }
        if (b.type === "hr") {
          return <Divider key={`hr-${i}`} sx={{ my: 0.5 }} />;
        }
        return (
          <Typography
            key={`p-${i}`}
            variant="body2"
            sx={{
              whiteSpace: "pre-wrap",
              fontSize: "0.94rem",
              color: "#1f1f1f",
              mb: 0.5,
            }}
          >
            {b.text}
          </Typography>
        );
      })}
    </>
  );
}

function MsgBubble({ role, content, table, sql, suggestions }) {
  const isUser = role === "user";
  const [showSQL, setShowSQL] = useState(false);

  return (
    <Stack
      direction="row"
      justifyContent={isUser ? "flex-end" : "flex-start"}
      sx={{ my: 0.5, px: 1.0 }}
    >
      <Paper
        elevation={0}
        className={`bubble ${isUser ? "bubble-user" : "bubble-assistant"}`}
        sx={{ p: 1.25 }}
      >
        {isUser ? (
          <Typography
            variant="body2"
            sx={{ whiteSpace: "pre-wrap", fontSize: "0.95rem" }}
          >
            {content}
          </Typography>
        ) : (
          renderAssistantContent(content)
        )}

        {!isUser && sql && (
          <Box sx={{ mt: 0.5 }}>
            <Stack direction="row" spacing={1} alignItems="center">
              <Chip
                label="SQL"
                size="small"
                sx={{
                  height: 20,
                  bgcolor: "#efefef",
                  "& .MuiChip-label": { px: 0.75, fontSize: "0.7rem" },
                }}
              />
              <IconButton size="small" onClick={() => setShowSQL((s) => !s)}>
                {showSQL ? (
                  <ExpandLessIcon fontSize="small" />
                ) : (
                  <ExpandMoreIcon fontSize="small" />
                )}
              </IconButton>
              <Tooltip title="Copy SQL">
                <IconButton
                  size="small"
                  onClick={() => navigator.clipboard.writeText(sql)}
                >
                  <ContentCopyIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </Stack>
            <Collapse in={showSQL}>
              <Box className="code-block" sx={{ mt: 0.5 }}>
                <div dangerouslySetInnerHTML={{ __html: highlightSQL(sql) }} />
              </Box>
            </Collapse>
          </Box>
        )}

        {/* Suggestions chips */}
        {/* {!isUser && Array.isArray(suggestions) && suggestions.length > 0 && (
          <Box className="suggestion-bar">
            <Typography
              variant="caption"
              sx={{ display: "block", mb: 0.25, color: "#6a6a6a" }}
            >
              Try one of these:
            </Typography>
            <Stack direction="row" spacing={0.75} sx={{ flexWrap: "wrap" }}>
              {suggestions.map((s, i) => (
                <Chip
                  key={`sugg-${i}`}
                  label={s}
                  size="small"
                  onClick={() => {
                    const ta = document.querySelector(
                      'textarea[name="chat-input"]'
                    );
                    if (ta) {
                      // Append or replace current value
                      ta.value = s;
                      ta.dispatchEvent(new Event("input", { bubbles: true }));
                      ta.focus();
                    }
                  }}
                />
              ))}
            </Stack>
          </Box>
        )} */}

        {/* Results table */}
        {!isUser && table && table.columns && table.columns.length > 0 && (
          <Box sx={{ mt: 1 }}>
            <div className="result-title">
              Results (first {table.rows?.length ?? 0} rows)
            </div>
            <TableContainer component={Box} style={{ overflow: "visible" }}>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    {table.columns.map((col) => (
                      <TableCell
                        key={col}
                        sx={{
                          fontWeight: 700,
                          background: "#fff",
                          py: 0.5,
                          fontSize: "0.82rem",
                        }}
                      >
                        {col}
                      </TableCell>
                    ))}
                  </TableRow>
                </TableHead>
                <TableBody>
                  {table.rows.map((r, i) => (
                    <TableRow key={i}>
                      {r.map((cell, j) => (
                        <TableCell
                          key={j}
                          sx={{ py: 0.4, fontSize: "0.82rem" }}
                        >
                          {String(cell)}
                        </TableCell>
                      ))}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Box>
        )}
      </Paper>
    </Stack>
  );
}

function ContextBar({
  catalog,
  selectedSchemas,
  setSelectedSchemas,
  selectedTables,
  setSelectedTables,
  onApply,
}) {
  const schemaOptions = useMemo(() => Object.keys(catalog || {}), [catalog]);

  const tableOptions = useMemo(() => {
    const out = [];
    for (const s of Object.keys(catalog || {})) {
      if (selectedSchemas.length && !selectedSchemas.includes(s)) continue;
      const tbls = catalog[s]?.tables || {};
      for (const t of Object.keys(tbls)) {
        out.push(`${s}.${t}`);
      }
    }
    return out.sort();
  }, [catalog, selectedSchemas]);

  useEffect(() => {
    setSelectedTables((prev) =>
      prev.filter((fq) => {
        const [s] = fq.split(".");
        return selectedSchemas.length === 0 || selectedSchemas.includes(s);
      })
    );
  }, [selectedSchemas, setSelectedTables]);

  return (
    <Paper elevation={0} sx={{ p: 1, mb: 1, border: "1px solid #eee" }}>
      <Stack
        direction={{ xs: "column", sm: "row" }}
        spacing={1}
        alignItems={{ xs: "stretch", sm: "flex-end" }}
      >
        <Autocomplete
          multiple
          options={schemaOptions}
          value={selectedSchemas}
          onChange={(_, v) => setSelectedSchemas(v)}
          renderInput={(params) => (
            <TextField
              {...params}
              label="Schemas"
              placeholder="Pick schemas"
              size="small"
            />
          )}
          sx={{ minWidth: 220, flex: 1 }}
        />
        <Autocomplete
          multiple
          options={tableOptions}
          value={selectedTables}
          onChange={(_, v) => setSelectedTables(v)}
          renderInput={(params) => (
            <TextField
              {...params}
              label="Tables"
              placeholder="Pick tables"
              size="small"
            />
          )}
          sx={{ minWidth: 280, flex: 2 }}
        />
        <Tooltip title="Apply context (chat will restart)">
          <span>
            <Button
              variant="contained"
              onClick={onApply}
              startIcon={<CheckIcon />}
              sx={{
                bgcolor: "#000",
                color: "#fff",
                "&:hover": { bgcolor: "#111" },
                borderRadius: 2,
                textTransform: "none",
                py: 0.5,
              }}
            >
              Apply
            </Button>
          </span>
        </Tooltip>
      </Stack>
      <Typography variant="caption" color="text.secondary">
        Tip: by default, all columns for the selected tables are considered in
        context.
      </Typography>
    </Paper>
  );
}

export default function App() {
  const [catalog, setCatalog] = useState({});
  const [conversationId, setConversationId] = useState("");
  const [messages, setMessages] = useState([]); // { role, content, table?, sql?, suggestions? }
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);

  const [selectedSchemas, setSelectedSchemas] = useState([]);
  const [selectedTables, setSelectedTables] = useState([]);

  const [confirmOpen, setConfirmOpen] = useState(false);
  const pendingApplyRef = useRef(null);

  const scrollRef = useRef(null);
  const bottomRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = (behavior = "smooth") => {
    const tryScroll = () => {
      if (bottomRef.current) {
        bottomRef.current.scrollIntoView({ behavior, block: "end" });
      } else if (scrollRef.current) {
        const el = scrollRef.current;
        el.scrollTop = el.scrollHeight;
      }
    };
    requestAnimationFrame(() => {
      tryScroll();
      setTimeout(tryScroll, 0);
      setTimeout(tryScroll, 50);
      setTimeout(tryScroll, 150);
    });
  };

  // Load catalog & set defaults, then start chat
  useEffect(() => {
    axios
      .get(`${API_BASE}/api/catalog`)
      .then((res) => {
        setCatalog(res.data || {});
        const schemas = Object.keys(res.data || {});
        const tables = [];
        schemas.forEach((s) => {
          const tbls = res.data[s]?.tables || {};
          Object.keys(tbls).forEach((t) => tables.push(`${s}.${t}`));
        });
        setSelectedSchemas(schemas);
        setSelectedTables(tables);
        return axios.post(`${API_BASE}/api/chat/start`, { schemas, tables });
      })
      .then((res) => {
        setConversationId(res.data.conversation_id);
        setMessages([{ role: "assistant", content: res.data.message }]);
        inputRef.current?.focus();
      })
      .catch(() => {
        setMessages([
          {
            role: "assistant",
            content: "Failed to initialize. Is the backend running?",
          },
        ]);
      });
  }, []);

  useLayoutEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (sending) scrollToBottom("auto");
  }, [sending]);

  const applyContext = async (schemas, tables) => {
    try {
      const res = await axios.post(`${API_BASE}/api/chat/start`, {
        schemas,
        tables,
      });
      setConversationId(res.data.conversation_id);
      setMessages([{ role: "assistant", content: res.data.message }]);
      inputRef.current?.focus();
      scrollToBottom("auto");
    } catch (e) {
      setMessages([
        {
          role: "assistant",
          content: "Could not apply context. Please try again.",
        },
      ]);
    }
  };

  const onApplyClick = () => {
    pendingApplyRef.current = {
      schemas: selectedSchemas.slice(),
      tables: selectedTables.slice(),
    };
    setConfirmOpen(true);
  };

  const confirmRestart = (proceed) => {
    setConfirmOpen(false);
    if (!proceed || !pendingApplyRef.current) return;
    const { schemas, tables } = pendingApplyRef.current;
    applyContext(schemas, tables);
    pendingApplyRef.current = null;
  };

  const sendMessage = async () => {
    const text = input.trim();
    if (!text || !conversationId) return;

    // optimistic user bubble (make sure user messages display)
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setInput("");
    scrollToBottom("auto");
    setSending(true);

    try {
      // show loader bubble (assistant)
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "", loading: true },
      ]);
      scrollToBottom("auto");

      const res = await axios.post(`${API_BASE}/api/chat/message`, {
        conversation_id: conversationId,
        message: text,
      });

      // remove loader bubble
      setMessages((prev) =>
        prev.filter((m, idx) => !(idx === prev.length - 1 && m.loading))
      );

      const { assistant, table, sql, suggestions } = res.data;
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: assistant || "",
          table: table || { columns: [], rows: [] },
          sql: sql || "",
          suggestions: suggestions || [],
        },
      ]);
    } catch (e) {
      // remove loader bubble
      setMessages((prev) =>
        prev.filter((m, idx) => !(idx === prev.length - 1 && m.loading))
      );
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Sorry — something went wrong sending your message.",
        },
      ]);
    } finally {
      setSending(false);
      inputRef.current?.focus();
      scrollToBottom();
    }
  };

  const onKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <Box className="chat-container">
      <GlobalStyles />

      <AppBar
        position="sticky"
        elevation={0}
        sx={{ bgcolor: "#fff", color: "#111", borderBottom: "1px solid #eee" }}
      >
        <Toolbar sx={{ minHeight: 52 }}>
          <Typography
            variant="h6"
            sx={{ fontWeight: 800, fontSize: "1.05rem" }}
          >
            Insight Chat
          </Typography>
        </Toolbar>
      </AppBar>

      <Container
        maxWidth="md"
        sx={{ flex: 1, display: "flex", flexDirection: "column", pt: 1 }}
      >
        <ContextBar
          catalog={catalog}
          selectedSchemas={selectedSchemas}
          setSelectedSchemas={setSelectedSchemas}
          selectedTables={selectedTables}
          setSelectedTables={setSelectedTables}
          onApply={onApplyClick}
        />

        <Box ref={scrollRef} className="chat-scroll">
          {messages.map((m, idx) => (
            <div key={idx}>
              {m.loading ? (
                <Stack
                  direction="row"
                  justifyContent="flex-start"
                  sx={{ my: 0.5, px: 1 }}
                >
                  <Paper
                    elevation={0}
                    className="bubble bubble-assistant"
                    sx={{ p: 1.25 }}
                  >
                    <LoaderBubble />
                  </Paper>
                </Stack>
              ) : (
                <MsgBubble
                  role={m.role}
                  content={m.content}
                  table={m.table}
                  sql={m.sql}
                  suggestions={m.suggestions}
                />
              )}
            </div>
          ))}
          <div ref={bottomRef} />
        </Box>

        <Box className="composer">
          <Container maxWidth="md" disableGutters>
            <Paper
              elevation={0}
              sx={{
                p: 0.75,
                border: "1px solid #eee",
                borderRadius: 2,
                background: "#fff",
              }}
            >
              <Stack direction="row" spacing={0.75} alignItems="flex-end">
                <TextField
                  inputRef={inputRef}
                  name="chat-input"
                  label="Message"
                  placeholder='e.g., "Group revenue by customer segment last 30 days"'
                  fullWidth
                  multiline
                  minRows={1}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={onKeyDown}
                  size="small"
                  InputLabelProps={{ style: { color: "#555" } }}
                />
                <Tooltip title="Send">
                  <span>
                    <IconButton
                      onClick={sendMessage}
                      disabled={sending || !input.trim()}
                      sx={{
                        bgcolor: "#000",
                        color: "#fff",
                        "&:hover": { bgcolor: "#111" },
                        borderRadius: 2,
                        width: 36,
                        height: 36,
                      }}
                    >
                      <SendIcon fontSize="small" />
                    </IconButton>
                  </span>
                </Tooltip>
              </Stack>
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{ ml: 0.5, fontSize: "0.72rem" }}
              >
                Press Enter to send, Shift+Enter for a new line
              </Typography>
            </Paper>
          </Container>
        </Box>
      </Container>

      <Dialog open={confirmOpen} onClose={() => confirmRestart(false)}>
        <DialogTitle sx={{ fontSize: "1rem" }}>Change context?</DialogTitle>
        <DialogContent>
          <Typography variant="body2" sx={{ fontSize: "0.9rem" }}>
            Changing schemas/tables will clear the current chat and start a new
            conversation with the selected context. Continue?
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => confirmRestart(false)} size="small">
            Cancel
          </Button>
          <Button
            onClick={() => confirmRestart(true)}
            size="small"
            sx={{
              bgcolor: "#000",
              color: "#fff",
              "&:hover": { bgcolor: "#111" },
            }}
          >
            Yes, restart
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
/* Full-height layout */
html,
body,
#root {
  height: 100%;
}

/* Main app container */
.chat-container {
  height: 100vh;
  display: flex;
  flex-direction: column;
}

/* Only this area scrolls */
.chat-scroll {
  flex: 1;
  overflow: auto;
  padding: 16px 0;
}

/* Composer fixed (sticky) to bottom */
.composer {
  position: sticky;
  bottom: 0;
  background: #fff;
  border-top: 1px solid #000;
  padding: 12px;
  z-index: 5; /* stays above content */
}

.bubble {
  max-width: 90%;
  font-size: 0.85rem; /* smaller text */
  line-height: 1.4;
  padding: 8px 12px; /* tighter padding */
  margin: 4px 0; /* reduced spacing */
}

.bubble-user {
  background: #e8f0fe;
  color: #000;
}

.bubble-assistant {
  background: #f9f9f9;
  color: #111;
}

/* Ensure consistent look */
.bubble-rich h3,
.bubble-rich h4,
.bubble-rich strong {
  font-weight: 600;
  font-size: 0.9rem;
  margin: 6px 0 2px; /* reduce vertical gaps */
}

.bubble-rich ul {
  margin: 4px 0 4px 18px;
  padding: 0;
}
.bubble-rich li {
  margin: 2px 0;
}

/* Loader bubble */
.loader-bubble {
  display: inline-flex;
  align-items: center;
  gap: 12px;
}
.loader {
  display: inline-flex;
  gap: 6px;
  align-items: center;
}
.loader-dot {
  width: 8px;
  height: 8px;
  background: #000;
  border-radius: 50%;
  display: inline-block;
  animation: loaderPulse 1.2s infinite ease-in-out;
}
.loader-dot:nth-child(2) {
  animation-delay: 0.15s;
}
.loader-dot:nth-child(3) {
  animation-delay: 0.3s;
}
@keyframes loaderPulse {
  0%,
  80%,
  100% {
    transform: scale(0.8);
    opacity: 0.6;
  }
  40% {
    transform: scale(1);
    opacity: 1;
  }
}

.code-block {
  background: #f4f4f4; /* subtle gray background */
  padding: 6px 8px;
  border-radius: 4px;
  font-size: 0.8rem;
  overflow-x: auto;
  white-space: pre;
}

.sql-k {
  color: #d14;
  font-weight: 600;
} /* keywords */
.sql-f {
  color: #09c;
} /* functions */
.sql-s {
  color: #690;
} /* strings */
.sql-n {
  color: #164;
} /* numbers */
.sql-c {
  color: #999;
  font-style: italic;
} /* comments */

/* Results table wrapper */
.table-wrap {
  border: 1px solid #000;
  border-radius: 10px;
  overflow: hidden;
}

.chat-container {
  height: 100vh;
  display: flex;
  flex-direction: column;
}
.chat-scroll {
  flex: 1;
  overflow: auto;
  padding: 16px 0 96px;
  scroll-behavior: smooth;
}
.composer {
  position: sticky;
  bottom: 0;
  background: #fff;
  border-top: 1px solid #000;
  padding: 12px;
  z-index: 5;
}

/* Gray bubbles (no borders) */
.bubble {
  max-width: 75%;
  padding: 16px;
  border-radius: 12px !important;
  box-shadow: none !important;
}
.bubble-user {
  background-color: #d9d9d9 !important;
  color: #000 !important;
}
.bubble-assistant {
  background-color: #f7f7f8 !important;
  color: #000 !important;
}

/* Chat bubble layout polish */
.bubble-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.25rem;
}
.bubble-title {
  font-weight: 700;
  font-size: 0.95rem;
}
.bubble-subtle {
  color: #666;
  font-size: 0.8rem;
}

/* Section shells */
.bubble-section {
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  padding: 10px 12px;
  background: #fafafa;
  margin-top: 10px;
}
.bubble-section h4 {
  margin: 0 0 6px 0;
  font-size: 0.9rem;
  font-weight: 700;
}
.bubble-kv {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas,
    "Liberation Mono", monospace;
}
.bubble-kv p {
  margin: 0.25rem 0;
}

/* Code block */
.code-block {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas,
    "Liberation Mono", monospace;
  font-size: 0.85rem;
  background: #fff;
  border: 1px solid #eee;
  border-radius: 10px;
  padding: 10px;
  overflow-x: auto;
}

/* Tiny badges */
.badge {
  border: 1px solid #ddd;
  background: #fff;
  border-radius: 999px;
  padding: 2px 8px;
  font-size: 0.72rem;
  color: #444;
}

/* SQL highlighting minimal colors */
.sql-k {
  font-weight: 700;
}
.sql-f {
  text-decoration: underline;
}
.sql-n {
  color: #1f6feb;
}
.sql-s {
  color: #0a7;
}
.sql-c {
  color: #888;
  font-style: italic;
}

/* Loader and SQL styles from previous message can stay the same */
