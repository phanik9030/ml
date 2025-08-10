import React, { useState } from "react";
import { Paper, TextField, IconButton } from "@mui/material";
import SendIcon from "@mui/icons-material/Send";

export default function InputBar({ onSend, disabled }) {
  const [val, setVal] = useState("");

  const submit = () => {
    const t = val.trim();
    if (!t) return;
    onSend(t);
    setVal("");
  };

  return (
    <Paper sx={{ p: 1, display: "flex", gap: 1 }}>
      <TextField
        fullWidth
        placeholder="Ask anything about your data... (e.g., create revenue by segment last 30 days)"
        value={val}
        onChange={(e) => setVal(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submit();
          }
        }}
        multiline
        maxRows={6}
      />
      <IconButton color="primary" onClick={submit} disabled={disabled}>
        <SendIcon />
      </IconButton>
    </Paper>
  );
}

import React, { useState } from "react";
import {
  Drawer,
  Toolbar,
  Typography,
  Box,
  List,
  ListItem,
  ListItemText,
  IconButton,
  TextField,
  Stack,
  Button,
  Divider,
} from "@mui/material";
import InfoOutlinedIcon from "@mui/icons-material/InfoOutlined";

const drawerWidth = 360;

export default function InsightDrawer({
  open,
  onClose,
  insights,
  onExplain,
  onUpdate,
}) {
  const [selected, setSelected] = useState(null);
  const [ttl, setTtl] = useState("");
  const [active, setActive] = useState("");
  const [schedule, setSchedule] = useState("");
  const [newName, setNewName] = useState("");

  const selectedInsight = insights.find((i) => i.name === selected);

  return (
    <Drawer
      variant="persistent"
      anchor="right"
      open={open}
      sx={{
        width: drawerWidth,
        flexShrink: 0,
        "& .MuiDrawer-paper": { width: drawerWidth, boxSizing: "border-box" },
      }}
    >
      <Toolbar />
      <Box p={2}>
        <Typography variant="h6">Insights</Typography>
        <List dense sx={{ maxHeight: "30vh", overflow: "auto" }}>
          {insights.map((i) => (
            <ListItem
              key={i.id}
              selected={selected === i.name}
              secondaryAction={
                <IconButton edge="end" onClick={() => onExplain(i.name)}>
                  <InfoOutlinedIcon />
                </IconButton>
              }
              onClick={() => setSelected(i.name)}
              button
            >
              <ListItemText
                primary={i.name}
                secondary={`TTL ${i.ttl}d • active ${i.active} • ${
                  i.schedule || "(no schedule)"
                }`}
              />
            </ListItem>
          ))}
        </List>

        <Divider sx={{ my: 1 }} />

        <Typography variant="subtitle1" gutterBottom>
          Edit selected
        </Typography>
        {selectedInsight ? (
          <Stack spacing={1}>
            <TextField
              size="small"
              label="Rename to"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
            />
            <TextField
              size="small"
              label="TTL (days)"
              value={ttl}
              onChange={(e) => setTtl(e.target.value)}
            />
            <TextField
              size="small"
              label="Active (true/false)"
              value={active}
              onChange={(e) => setActive(e.target.value)}
            />
            <TextField
              size="small"
              label="Schedule (cron)"
              value={schedule}
              onChange={(e) => setSchedule(e.target.value)}
            />
            <Button
              variant="contained"
              onClick={() => {
                onUpdate(selectedInsight.name, {
                  newName: newName || undefined,
                  ttl: ttl ? Number(ttl) : undefined,
                  active: active
                    ? /^(true|1|yes|on)$/i.test(active)
                    : undefined,
                  schedule: schedule || undefined,
                });
                setNewName("");
                setTtl("");
                setActive("");
                setSchedule("");
              }}
            >
              Apply
            </Button>
          </Stack>
        ) : (
          <Typography variant="body2" color="text.secondary">
            Pick an insight to edit.
          </Typography>
        )}
      </Box>
    </Drawer>
  );
}

import React from "react";
import { Box, Paper, IconButton, Tooltip } from "@mui/material";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import ReactMarkdown from "react-markdown";

export default function MessageBubble({ role, text }) {
  const isUser = role === "user";
  return (
    <Box
      className={`message ${role}`}
      display="flex"
      flexDirection="column"
      alignItems={isUser ? "flex-end" : "flex-start"}
    >
      <Paper
        elevation={isUser ? 2 : 1}
        sx={{
          p: 1.5,
          bgcolor: isUser ? "primary.main" : "background.paper",
          color: isUser ? "primary.contrastText" : "text.primary",
          borderRadius: 2,
          maxWidth: "820px",
        }}
      >
        <ReactMarkdown
          components={{
            code({ inline, className, children, ...props }) {
              if (inline) return <code {...props}>{children}</code>;
              return (
                <pre className="codeblock" {...props}>
                  <code>{children}</code>
                </pre>
              );
            },
          }}
        >
          {text}
        </ReactMarkdown>
      </Paper>
      <Box mt={0.5}>
        <Tooltip title="Copy">
          <IconButton
            size="small"
            onClick={() => navigator.clipboard.writeText(text)}
          >
            <ContentCopyIcon fontSize="inherit" />
          </IconButton>
        </Tooltip>
      </Box>
    </Box>
  );
}

import React from "react";
import {
  Card,
  CardHeader,
  CardContent,
  Table,
  TableHead,
  TableRow,
  TableCell,
  TableBody,
} from "@mui/material";

export default function PreviewCard({ title = "Preview", columns, rows }) {
  return (
    <Card sx={{ my: 1 }}>
      <CardHeader title={title} />
      <CardContent>
        {!rows || rows.length === 0 ? (
          "No rows"
        ) : (
          <Table size="small">
            <TableHead>
              <TableRow>
                {columns.map((c) => (
                  <TableCell key={c}>{c}</TableCell>
                ))}
              </TableRow>
            </TableHead>
            <TableBody>
              {rows.map((r, i) => (
                <TableRow key={i}>
                  {columns.map((c) => (
                    <TableCell key={c}>{String(r[c] ?? "")}</TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  );
}

import React from "react";
import {
  Drawer,
  Toolbar,
  Typography,
  List,
  ListItem,
  ListItemText,
  Divider,
  Box,
} from "@mui/material";

const drawerWidth = 320;

export default function SchemaDrawer({
  open,
  onClose,
  schemas,
  tablesBySchema,
  onSchemaClick,
  onTableClick,
}) {
  return (
    <Drawer
      variant="persistent"
      anchor="left"
      open={open}
      sx={{
        width: drawerWidth,
        flexShrink: 0,
        "& .MuiDrawer-paper": { width: drawerWidth, boxSizing: "border-box" },
      }}
    >
      <Toolbar />
      <Box p={2}>
        <Typography variant="h6">Schemas</Typography>
        <List dense>
          {schemas.map((s) => (
            <ListItem button key={s.name} onClick={() => onSchemaClick(s.name)}>
              <ListItemText primary={s.name} secondary={s.description} />
            </ListItem>
          ))}
        </List>
        <Divider sx={{ my: 1 }} />
        <Typography variant="subtitle1">Tables</Typography>
        <Box sx={{ maxHeight: "45vh", overflow: "auto" }}>
          {Object.entries(tablesBySchema).map(([schema, tables]) => (
            <Box key={schema} mb={1}>
              <Typography variant="subtitle2" sx={{ opacity: 0.7 }}>
                {schema}
              </Typography>
              <List dense>
                {tables.map((t) => (
                  <ListItem
                    button
                    key={`${t.schema}.${t.name}`}
                    onClick={() => onTableClick(`${t.schema}.${t.name}`)}
                  >
                    <ListItemText
                      primary={`${t.schema}.${t.name}`}
                      secondary={t.description}
                    />
                  </ListItem>
                ))}
              </List>
            </Box>
          ))}
        </Box>
      </Box>
    </Drawer>
  );
}

const API_BASE = process.env.REACT_APP_API_BASE || "http://localhost:8080";

export async function apiChat(message) {
  const r = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ message }),
  });
  if (!r.ok) throw new Error(await r.text());
  const j = await r.json();
  return j.reply;
}

export async function apiSchemas() {
  const r = await fetch(`${API_BASE}/schemas`);
  const j = await r.json();
  return j.schemas || [];
}

export async function apiTables(schema) {
  const r = await fetch(`${API_BASE}/tables/${schema}`);
  const j = await r.json();
  return j.tables || [];
}

export async function apiInsights() {
  const r = await fetch(`${API_BASE}/insights`);
  const j = await r.json();
  return j.insights || [];
}

export async function apiExplainInsight(name) {
  const r = await fetch(
    `${API_BASE}/insights/${encodeURIComponent(name)}/explain`
  );
  const j = await r.json();
  return j;
}

export async function apiUpdateInsight(name, body) {
  const r = await fetch(
    `${API_BASE}/insights/${encodeURIComponent(name)}/update`,
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    }
  );
  const j = await r.json();
  return j;
}

import React, { useEffect, useRef, useState } from "react";
import {
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Box,
  CircularProgress,
  Stack,
  Tooltip,
} from "@mui/material";
import MenuIcon from "@mui/icons-material/Menu";
import InsightsIcon from "@mui/icons-material/Insights";
import MessageBubble from "./components/MessageBubble";
import InputBar from "./components/InputBar";
import SchemaDrawer from "./components/SchemaDrawer";
import InsightDrawer from "./components/InsightDrawer";
import PreviewCard from "./components/PreviewCard";
import {
  apiChat,
  apiSchemas,
  apiTables,
  apiInsights,
  apiExplainInsight,
  apiUpdateInsight,
} from "./api";

export default function App() {
  const [leftOpen, setLeftOpen] = useState(true);
  const [rightOpen, setRightOpen] = useState(false);
  const [schemas, setSchemas] = useState([]);
  const [tablesBySchema, setTablesBySchema] = useState({});
  const [insights, setInsights] = useState([]);
  const [chat, setChat] = useState([
    {
      id: crypto.randomUUID(),
      role: "assistant",
      text: "Hi! I can describe your data and build insights.\n\nTry:\n- *what data do you have?*\n- *use schema sales and public*\n- *list tables in sales*\n- *create revenue by segment last 30 days*",
    },
  ]);
  const [loading, setLoading] = useState(false);

  const messagesEndRef = useRef(null);
  const scrollToEnd = () =>
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  useEffect(() => {
    scrollToEnd();
  }, [chat]);

  useEffect(() => {
    apiSchemas()
      .then(setSchemas)
      .catch(() => {});
    apiInsights()
      .then(setInsights)
      .catch(() => {});
  }, []);

  const handleSchemaClick = async (name) => {
    try {
      const tables = await apiTables(name);
      setTablesBySchema((s) => ({ ...s, [name]: tables }));
      pushAssistant(
        `Loaded tables for \`${name}\`. Ask me to *describe ${name}.<table>* or *show 5 rows from ${name}.<table>*`
      );
    } catch {}
  };

  const handleTableClick = (fq) => {
    pushUser(`show 5 rows from ${fq}`);
    send(`show 5 rows from ${fq}`);
  };

  const pushUser = (text) =>
    setChat((c) => [...c, { id: crypto.randomUUID(), role: "user", text }]);
  const pushAssistant = (text) =>
    setChat((c) => [
      ...c,
      { id: crypto.randomUUID(), role: "assistant", text },
    ]);

  // parse preview from our ASCII table output
  const maybeExtractPreview = (text) => {
    const lines = text.split("\n");
    const sepIdx = lines.findIndex((l) => l.includes("-+-"));
    if (sepIdx > 0) {
      const header = lines[0].split(" | ").map((s) => s.trim());
      const bodyLines = lines.slice(sepIdx + 1).filter((l) => l.trim());
      const rows = bodyLines.map((l) => {
        const cells = l.split(" | ").map((s) => s.trim());
        const obj = {};
        header.forEach((h, i) => (obj[h] = cells[i]));
        return obj;
      });
      return { columns: header, rows };
    }
    return null;
  };

  const send = async (message) => {
    setLoading(true);
    try {
      const reply = await apiChat(message);
      const preview = maybeExtractPreview(reply);
      if (preview) {
        pushAssistant("Here’s a preview:");
        pushAssistant("```text\n" + reply + "\n```");
        setChat((c) => [
          ...c,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            text: `<preview/>${JSON.stringify(preview)}`,
          },
        ]);
      } else {
        pushAssistant(reply);
      }
      if (/Saved `(.+?)`/.test(reply)) {
        const list = await apiInsights()
          .then((x) => x)
          .catch(() => null);
        if (list) setInsights(list);
      }
    } catch (e) {
      pushAssistant("Sorry, I couldn’t process that.");
    } finally {
      setLoading(false);
    }
  };

  const onSend = (text) => {
    const t = text.trim();
    if (!t) return;
    pushUser(t);
    send(t);
  };

  const renderTurn = (turn) => {
    if (turn.text.startsWith("<preview/>")) {
      try {
        const data = JSON.parse(turn.text.replace("<preview/>", ""));
        return (
          <PreviewCard key={turn.id} columns={data.columns} rows={data.rows} />
        );
      } catch {
        return (
          <MessageBubble
            key={turn.id}
            role={turn.role}
            text={"[preview parse error]"}
          />
        );
      }
    }
    return <MessageBubble key={turn.id} role={turn.role} text={turn.text} />;
  };

  const explain = async (name) => {
    try {
      const { meta, explanation, error } = await apiExplainInsight(name);
      if (error) {
        pushAssistant(error);
        return;
      }
      const md = `# Insight: ${meta.name}\n- Status: ${
        meta.active ? "Active" : "Inactive"
      } | TTL: ${meta.ttl} days | Schedule: ${
        meta.schedule || "(none)"
      }\n- Output columns: ${
        (meta.columns || []).join(", ") || "(unknown)"
      }\n\n${explanation}`;
      pushAssistant(md);
    } catch {
      pushAssistant("Could not explain that insight.");
    }
  };

  const update = async (name, updates) => {
    const body = {};
    if (updates.ttl !== undefined) body.ttl = updates.ttl;
    if (updates.active !== undefined) body.active = updates.active;
    if (updates.schedule) body.schedule = updates.schedule;
    if (updates.newName) body.name = updates.newName;
    const r = await apiUpdateInsight(name, body);
    if (r.error) {
      pushAssistant(r.error);
    } else {
      pushAssistant(`Updated \`${name}\`.`);
      apiInsights()
        .then(setInsights)
        .catch(() => {});
    }
  };

  return (
    <Box className="chat-container">
      <AppBar position="fixed">
        <Toolbar>
          <IconButton
            color="inherit"
            edge="start"
            onClick={() => setLeftOpen((o) => !o)}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" sx={{ flex: 1 }}>
            Insight Assistant
          </Typography>
          <Tooltip title="Insights">
            <IconButton color="inherit" onClick={() => setRightOpen((o) => !o)}>
              <InsightsIcon />
            </IconButton>
          </Tooltip>
        </Toolbar>
      </AppBar>

      <SchemaDrawer
        open={leftOpen}
        onClose={() => setLeftOpen(false)}
        schemas={schemas}
        tablesBySchema={tablesBySchema}
        onSchemaClick={handleSchemaClick}
        onTableClick={handleTableClick}
      />

      <Box className="chat-main">
        <Toolbar />
        <Box className="messages">
          <Stack spacing={1}>
            {chat.map(renderTurn)}
            {loading && (
              <Box>
                <CircularProgress size={18} sx={{ mr: 1 }} /> Thinking…
              </Box>
            )}
            <div ref={messagesEndRef} />
          </Stack>
        </Box>
        <Box p={1.5}>
          <InputBar onSend={onSend} disabled={loading} />
        </Box>
      </Box>

      <InsightDrawer
        open={rightOpen}
        onClose={() => setRightOpen(false)}
        insights={insights}
        onExplain={explain}
        onUpdate={update}
      />
    </Box>
  );
}

html,
body,
#root {
  height: 100%;
}
body {
  margin: 0;
}
.chat-container {
  display: flex;
  height: 100%;
}
.chat-main {
  flex: 1;
  display: flex;
  flex-direction: column;
}
.messages {
  flex: 1;
  overflow: auto;
  padding: 16px;
}
.message {
  margin-bottom: 12px;
  max-width: 820px;
}
.message.user {
  align-self: flex-end;
}
.message.assistant {
  align-self: flex-start;
}
.codeblock {
  background: #0f172a;
  color: #e2e8f0;
  padding: 12px;
  border-radius: 8px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas,
    "Liberation Mono", monospace;
}
