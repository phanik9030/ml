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
} from "@mui/material";
import SendIcon from "@mui/icons-material/Send";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ExpandLessIcon from "@mui/icons-material/ExpandLess";
import CheckIcon from "@mui/icons-material/CheckCircle";

const API_BASE = process.env.REACT_APP_API_BASE || ""; // CRA proxy -> Flask

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
      <Typography variant="body2" sx={{ color: "#000" }}>
        Thinking…
      </Typography>
    </div>
  );
}

function MsgBubble({ role, content, table, sql }) {
  const isUser = role === "user";
  const [showSQL, setShowSQL] = useState(false);

  return (
    <Stack
      direction="row"
      justifyContent={isUser ? "flex-end" : "flex-start"}
      sx={{ my: 1 }}
    >
      <Paper
        elevation={0}
        className={`bubble ${isUser ? "bubble-user" : "bubble-assistant"}`}
        sx={{ p: 2 }}
      >
        {content && (
          <Typography variant="body2" sx={{ whiteSpace: "pre-wrap" }}>
            {content}
          </Typography>
        )}

        {!isUser && sql && (
          <Box sx={{ mt: 1 }}>
            <Stack direction="row" spacing={1} alignItems="center">
              <Chip label="SQL" size="small" />
              <IconButton size="small" onClick={() => setShowSQL((s) => !s)}>
                {showSQL ? <ExpandLessIcon /> : <ExpandMoreIcon />}
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
              <Box className="code-block" sx={{ mt: 1 }}>
                <div dangerouslySetInnerHTML={{ __html: highlightSQL(sql) }} />
              </Box>
            </Collapse>
          </Box>
        )}

        {/* Results table — MUI Table with TableContainer; no inner scroll */}
        {!isUser && table && table.columns && table.columns.length > 0 && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Results (first {table.rows?.length ?? 0} rows)
            </Typography>
            <TableContainer component={Box} style={{ overflow: "visible" }}>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    {table.columns.map((col) => (
                      <TableCell
                        key={col}
                        sx={{ fontWeight: 700, background: "#fff" }}
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
                        <TableCell key={j}>{String(cell)}</TableCell>
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

  // Ensure tables remain valid if schemas change
  useEffect(() => {
    setSelectedTables((prev) =>
      prev.filter((fq) => {
        const [s] = fq.split(".");
        return selectedSchemas.length === 0 || selectedSchemas.includes(s);
      })
    );
  }, [selectedSchemas, setSelectedTables]);

  return (
    <Paper elevation={0} sx={{ p: 1.5, mb: 1 }}>
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
              label="Schemas in context"
              placeholder="Pick schemas"
            />
          )}
          sx={{ minWidth: 240, flex: 1 }}
        />
        <Autocomplete
          multiple
          options={tableOptions}
          value={selectedTables}
          onChange={(_, v) => setSelectedTables(v)}
          renderInput={(params) => (
            <TextField
              {...params}
              label="Tables in context"
              placeholder="Pick tables"
            />
          )}
          sx={{ minWidth: 320, flex: 2 }}
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
              }}
            >
              Apply Context
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
  const [messages, setMessages] = useState([]); // { role, content, table?, sql?, loading? }
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);

  // Context selections (default: all schemas & all their tables)
  const [selectedSchemas, setSelectedSchemas] = useState([]);
  const [selectedTables, setSelectedTables] = useState([]);

  // For restart confirmation dialog
  const [confirmOpen, setConfirmOpen] = useState(false);
  const pendingApplyRef = useRef(null);

  // Only the chat area scrolls
  const scrollRef = useRef(null); // scrollable container
  const bottomRef = useRef(null); // anchor
  const inputRef = useRef(null); // focus control

  // Robust autoscroll (multi-pass to beat layout shifts from tables)
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

  // Load catalog & set defaults (all schemas/tables selected)
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
        // Start default conversation with full context
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

  // After messages change, scroll to bottom (layout-aware)
  useLayoutEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (sending) scrollToBottom("auto");
  }, [sending]);

  const applyContext = async (schemas, tables) => {
    // restart conversation with chosen context
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
    // if context has changed, confirm restart
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

    // optimistic user bubble
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

      const { assistant, table, sql } = res.data;
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: assistant || "",
          table: table || { columns: [], rows: [] },
          sql: sql || "",
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

  // ENTER sends; SHIFT+ENTER = newline
  const onKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <Box className="chat-container">
      <AppBar position="sticky" elevation={0}>
        <Toolbar sx={{ minHeight: 56 }}>
          <Typography variant="h6" sx={{ fontWeight: 700 }}>
            Insight Chat
          </Typography>
        </Toolbar>
      </AppBar>

      <Container
        maxWidth="md"
        sx={{ flex: 1, display: "flex", flexDirection: "column" }}
      >
        {/* Context selectors */}
        <ContextBar
          catalog={catalog}
          selectedSchemas={selectedSchemas}
          setSelectedSchemas={setSelectedSchemas}
          selectedTables={selectedTables}
          setSelectedTables={setSelectedTables}
          onApply={onApplyClick}
        />

        {/* Chat area — the only scrollable region */}
        <Box ref={scrollRef} className="chat-scroll">
          {messages.map((m, idx) => (
            <div key={idx}>
              {m.loading ? (
                <Stack
                  direction="row"
                  justifyContent="flex-start"
                  sx={{ my: 1 }}
                >
                  <Paper elevation={0} className="bubble bubble-assistant">
                    <LoaderBubble />
                  </Paper>
                </Stack>
              ) : (
                <MsgBubble
                  role={m.role}
                  content={m.content}
                  table={m.table}
                  sql={m.sql}
                />
              )}
            </div>
          ))}
          {/* anchor for autoscroll */}
          <div ref={bottomRef} />
        </Box>

        {/* Sticky composer at the bottom */}
        <Box className="composer">
          <Container maxWidth="md" disableGutters>
            <Paper elevation={0} sx={{ p: 1 }}>
              <Stack direction="row" spacing={1} alignItems="flex-end">
                <TextField
                  inputRef={inputRef}
                  label="Message"
                  placeholder='e.g., "Group revenue by customer segment last 30 days"'
                  fullWidth
                  multiline
                  minRows={1}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={onKeyDown}
                  InputLabelProps={{ style: { color: "#000" } }}
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
                      }}
                    >
                      <SendIcon />
                    </IconButton>
                  </span>
                </Tooltip>
              </Stack>
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{ ml: 0.5 }}
              >
                Press Enter to send, Shift+Enter for a new line
              </Typography>
            </Paper>
          </Container>
        </Box>
      </Container>

      {/* Confirm dialog for context restart */}
      <Dialog open={confirmOpen} onClose={() => confirmRestart(false)}>
        <DialogTitle>Change context?</DialogTitle>
        <DialogContent>
          <Typography variant="body2">
            Changing schemas/tables will clear the current chat and start a new
            conversation with the selected context. Continue?
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => confirmRestart(false)}>Cancel</Button>
          <Button
            onClick={() => confirmRestart(true)}
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
