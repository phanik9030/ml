import React, { useEffect, useMemo, useState } from "react";
import axios from "axios";
import {
  Box,
  Container,
  TextField,
  Button,
  Typography,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Stack,
  Chip,
  Divider,
  Alert,
} from "@mui/material";

const API_BASE = process.env.REACT_APP_API_BASE || ""; // use CRA proxy if empty

function SingleSelectableTable({ title, rows, selected, onSelect }) {
  return (
    <Paper
      variant="outlined"
      sx={{ width: "100%", height: 420, overflow: "auto" }}
    >
      <Box
        sx={{
          p: 1,
          bgcolor: "background.default",
          position: "sticky",
          top: 0,
          zIndex: 1,
        }}
      >
        <Typography variant="subtitle1" fontWeight={600}>
          {title}
        </Typography>
      </Box>
      <Divider />
      <Table size="small" stickyHeader>
        <TableHead>
          <TableRow>
            <TableCell>Name</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {rows.map((name) => (
            <TableRow
              key={name}
              hover
              selected={selected === name}
              onClick={() => onSelect(name)}
              sx={{ cursor: "pointer" }}
            >
              <TableCell>{name}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Paper>
  );
}

function MultiSelectableTable({ title, rows, selected, onChange }) {
  const handleToggle = (name) => {
    const isSelected = selected.includes(name);
    const next = isSelected
      ? selected.filter((x) => x !== name)
      : [...selected, name];
    onChange(next);
  };

  return (
    <Paper
      variant="outlined"
      sx={{ width: "100%", height: 420, overflow: "auto" }}
    >
      <Box
        sx={{
          p: 1,
          bgcolor: "background.default",
          position: "sticky",
          top: 0,
          zIndex: 1,
        }}
      >
        <Typography variant="subtitle1" fontWeight={600}>
          {title}
        </Typography>
      </Box>
      <Divider />
      <Table size="small" stickyHeader>
        <TableHead>
          <TableRow>
            <TableCell>Name</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {rows.map((name) => {
            const isSelected = selected.includes(name);
            return (
              <TableRow
                key={name}
                hover
                selected={isSelected}
                onClick={() => handleToggle(name)}
                sx={{ cursor: "pointer" }}
              >
                <TableCell>{name}</TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </Paper>
  );
}

export default function App() {
  const [allSchemas, setAllSchemas] = useState([]);
  const [schemas, setSchemas] = useState([]); // MULTI
  const [tablesBySchema, setTablesBySchema] = useState({}); // {schema: [tables]}
  const [selectedTables, setSelectedTables] = useState([]); // ["schema.table", ...]

  const [columnsCache, setColumnsCache] = useState({}); // {"schema.table": [cols]}
  const [columnsByTable, setColumnsByTable] = useState({}); // {"schema.table": [selected cols]}

  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState({ english_dsl: "", sql: "" });
  const [error, setError] = useState("");
  const [revised, setRevised] = useState("");

  // Load schemas
  useEffect(() => {
    axios
      .get(`${API_BASE}/api/catalog/schemas`)
      .then((res) => setAllSchemas(res.data || []))
      .catch(() => setAllSchemas([]));
  }, []);

  // When schemas change → fetch their tables and reset downstream selections
  useEffect(() => {
    const fetchTables = async (s) => {
      const res = await axios.get(`${API_BASE}/api/catalog/tables`, {
        params: { schema: s },
      });
      return res.data || [];
    };
    (async () => {
      const map = {};
      for (const s of schemas) map[s] = await fetchTables(s);
      setTablesBySchema(map);

      // prune tables/columns that no longer belong
      const keep = new Set(
        Object.keys(map).flatMap((s) => map[s].map((t) => `${s}.${t}`))
      );
      setSelectedTables((prev) => prev.filter((st) => keep.has(st)));
      setColumnsByTable((prev) => {
        const next = {};
        for (const st of Object.keys(prev))
          if (keep.has(st)) next[st] = prev[st];
        return next;
      });
    })();
    // eslint-disable-next-line
  }, [schemas]);

  // When selected tables change → prefetch columns for each selected table
  useEffect(() => {
    const fetchColumns = async (s, t) => {
      const res = await axios.get(`${API_BASE}/api/catalog/columns`, {
        params: { schema: s, table: t },
      });
      return res.data || [];
    };
    (async () => {
      const cache = { ...columnsCache };
      for (const st of selectedTables) {
        if (typeof st !== "string" || !st.includes(".")) continue;
        if (!cache[st]) {
          const [s, t] = st.split(".", 2);
          cache[st] = await fetchColumns(s, t);
        }
      }
      setColumnsCache(cache);

      // prune column selections for tables that were unselected or whose columns changed
      const keep = new Set(selectedTables);
      setColumnsByTable((prev) => {
        const next = {};
        for (const st of Object.keys(prev)) {
          if (keep.has(st)) {
            const available = cache[st] || [];
            next[st] = (prev[st] || []).filter((c) => available.includes(c));
          }
        }
        return next;
      });
    })();
    // eslint-disable-next-line
  }, [selectedTables]);

  const flatTables = useMemo(() => {
    const rows = [];
    for (const s of schemas)
      for (const t of tablesBySchema[s] || []) rows.push(`${s}.${t}`);
    return rows;
  }, [schemas, tablesBySchema]);

  const toggleColumn = (st, col) => {
    setColumnsByTable((prev) => {
      const list = prev[st] || [];
      const next = list.includes(col)
        ? list.filter((x) => x !== col)
        : [...list, col];
      return { ...prev, [st]: next };
    });
  };

  const canGenerate = useMemo(
    () =>
      schemas.length > 0 &&
      selectedTables.length > 0 &&
      prompt.trim().length > 0,
    [schemas, selectedTables, prompt]
  );

  const handleGenerate = async () => {
    setLoading(true);
    setError("");
    setResult({ english_dsl: "", sql: "" });
    try {
      const res = await axios.post(`${API_BASE}/api/generate`, {
        schemas,
        tables: selectedTables,
        columnsByTable,
        prompt,
      });
      setResult(res.data);
      setRevised(res.data.revised_prompt || "");
    } catch (e) {
      setError(e?.response?.data?.error || "Failed to generate.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="xl" sx={{ py: 2 }}>
      <Typography variant="h5" gutterBottom>
        Simple Insight Builder (Multi-Schema)
      </Typography>

      <Stack direction="row" spacing={2}>
        <Box sx={{ width: "28%" }}>
          <MultiSelectableTable
            title="Schemas (multi-select)"
            rows={allSchemas}
            selected={schemas}
            onChange={setSchemas}
          />
        </Box>

        <Box sx={{ width: "36%" }}>
          <MultiSelectableTable
            title="Tables (multi-select, schema.table)"
            rows={flatTables}
            selected={selectedTables}
            onChange={setSelectedTables} // <<< critical: pass array setter, not a single-item toggle
          />
        </Box>

        <Box sx={{ width: "36%" }}>
          <Paper variant="outlined" sx={{ height: 420, overflow: "auto" }}>
            <Box
              sx={{
                p: 1,
                bgcolor: "background.default",
                position: "sticky",
                top: 0,
                zIndex: 1,
              }}
            >
              <Typography variant="subtitle1" fontWeight={600}>
                Columns (click to select)
              </Typography>
            </Box>
            <Divider />
            <Box sx={{ p: 2 }}>
              {selectedTables.length === 0 && (
                <Typography color="text.secondary">
                  Select one or more tables to see columns.
                </Typography>
              )}
              {selectedTables.map((st) => (
                <Box key={st} sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" sx={{ mb: 1 }}>
                    {st}
                  </Typography>
                  <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap">
                    {(columnsCache[st] || []).map((col) => {
                      const on = (columnsByTable[st] || []).includes(col);
                      return (
                        <Chip
                          key={col}
                          label={col}
                          variant={on ? "filled" : "outlined"}
                          color={on ? "primary" : "default"}
                          onClick={() => toggleColumn(st, col)}
                          sx={{ mb: 1 }}
                        />
                      );
                    })}
                  </Stack>
                  <Divider sx={{ my: 1 }} />
                </Box>
              ))}
            </Box>
          </Paper>
        </Box>
      </Stack>

      <Box sx={{ mt: 2 }}>
        <TextField
          fullWidth
          multiline
          minRows={3}
          label="Describe your insight (natural language)"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />
        {revised && (
          <Typography
            variant="caption"
            color="text.secondary"
            sx={{ display: "block", mt: 0.5 }}
          >
            Used prompt: {revised}
          </Typography>
        )}
        <Stack direction="row" spacing={2} sx={{ mt: 1 }} alignItems="center">
          <Button
            variant="contained"
            disabled={!canGenerate || loading}
            onClick={handleGenerate}
          >
            {loading ? "Generating..." : "Generate English DSL + SQL"}
          </Button>
          <Typography variant="body2" color="text.secondary">
            Only the selected schemas/tables/columns are sent to the backend. If
            names don’t match, the backend resolves them from your selection
            using embeddings as a fallback (no search UI).
          </Typography>
        </Stack>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      )}

      {result.english_dsl && (
        <Paper variant="outlined" sx={{ mt: 2, p: 2 }}>
          <Typography variant="subtitle1" fontWeight={700} sx={{ mb: 1 }}>
            English DSL
          </Typography>
          <pre
            style={{
              margin: 0,
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
            }}
          >
            {result.english_dsl}
          </pre>
        </Paper>
      )}

      {result.sql && (
        <Paper variant="outlined" sx={{ mt: 2, p: 2 }}>
          <Typography variant="subtitle1" fontWeight={700} sx={{ mb: 1 }}>
            SQL
          </Typography>
          <pre
            style={{
              margin: 0,
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
            }}
          >
            {result.sql}
          </pre>
        </Paper>
      )}
    </Container>
  );
}
