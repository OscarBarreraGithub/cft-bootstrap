# Setup Instructions

## MCP Server Configuration

The `.mcp.json` file contains absolute paths specific to your system and should NOT be committed to version control. Follow these steps to configure it:

### 1. Copy the Example Configuration

```bash
cp .mcp.json.example .mcp.json
```

### 2. Edit with Your Paths

Open `.mcp.json` and replace the placeholder paths with your actual paths:

```json
{
  "mcpServers": {
    "wolfram": {
      "command": "/YOUR/ACTUAL/PATH/TO/Wolfram-MCP/venv/bin/python",
      "args": ["/YOUR/ACTUAL/PATH/TO/Wolfram-MCP/wolfram_mcp_server.py"],
      "env": {
        "WOLFRAM_KERNEL_PATH": "/Applications/Wolfram.app/Contents/MacOS/WolframKernel"
      }
    }
  }
}
```

### Platform-Specific Paths

**macOS (default):**
```json
"WOLFRAM_KERNEL_PATH": "/Applications/Wolfram.app/Contents/MacOS/WolframKernel"
```

**Linux:**
```json
"WOLFRAM_KERNEL_PATH": "/usr/local/Wolfram/Mathematica/14.0/Executables/WolframKernel"
```

**Windows:**
```json
"WOLFRAM_KERNEL_PATH": "C:\\Program Files\\Wolfram Research\\Mathematica\\14.0\\WolframKernel.exe"
```

### 3. For Claude Desktop

Copy your configured `.mcp.json` content to your Claude Desktop configuration file:

- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux:** `~/.config/claude/claude_desktop_config.json`

## Platform Support

> **Note:** The Wolfram MCP server is primarily tested on macOS. While it should work on other platforms, you may need to adjust paths and verify compatibility with your Wolfram installation.

## Verifying the Setup

After configuration, test the connection:

1. Start Claude (Desktop or CLI)
2. Ask: "Can you test the Wolfram connection?"
3. Claude should use `wolfram_test` and report success with version info
