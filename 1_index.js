import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { ThemeProvider, CssBaseline, createTheme } from "@mui/material";
import "./styles.css";

// Monochrome MUI theme (no blue)
const theme = createTheme({
  palette: {
    mode: "light",
    primary: { main: "#000000" },
    secondary: { main: "#222222" },
    background: { default: "#ffffff", paper: "#ffffff" },
    text: { primary: "#000000", secondary: "#444444" },
    divider: "#000000",
  },
  components: {
    MuiAppBar: {
      styleOverrides: {
        root: { backgroundColor: "#000000", color: "#ffffff" },
      },
    },
    MuiButton: {
      styleOverrides: { root: { textTransform: "none", borderRadius: 8 } },
    },
    MuiPaper: { styleOverrides: { root: { borderRadius: 12 } } },
    MuiChip: {
      styleOverrides: { root: { background: "#f3f3f3", color: "#111" } },
    },
  },
  typography: {
    fontFamily: `'Inter', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji"`,
  },
});

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <ThemeProvider theme={theme}>
    <CssBaseline />
    <App />
  </ThemeProvider>
);
