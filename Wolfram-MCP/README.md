# Wolfram Language MCP Server

A sophisticated Model Context Protocol (MCP) server providing full Wolfram Language/Mathematica integration with **proof-carrying symbolic computation**. This goes far beyond simple command-line Mathematica - it turns Wolfram into a proof assistant.

## Key Features

### Core Capabilities
- **Persistent Sessions** - Variables and functions persist across calls
- **Rich Output** - Graphics as base64 images, LaTeX for symbolic expressions
- **Timeout/Abort Handling** - Long computations abort gracefully with recovery

### Proof-Carrying Computation
- **Typed Equalities** - Distinguish exact equality, equality on domain, up to order, modulo gauge
- **Assumption Provenance** - Track what assumptions were used in each computation
- **Condition Extraction** - Automatically extract conditions from `ConditionalExpression` results
- **Obligation Engine** - Register tests that must pass before accepting a derivation
- **Domain Inference** - Automatically detect validity domains, singularities, branch cuts
- **Semantic Diffs** - See what *mathematically* changed, not just textually
- **Math CI** - Build regression test suites for mathematical derivations
- **Numeric Validation** - Automatic spot-checks catch symbolic manipulation errors
- **Expression Hashing** - Equivalent expressions get the same hash for caching

## Why This Beats Command-Line Mathematica

| Feature | Command-Line | This MCP Server |
|---------|--------------|-----------------|
| Session persistence | Each call independent | Full state persistence |
| Assumption tracking | Manual, error-prone | Automatic provenance |
| Condition extraction | Often ignored | Explicit in output |
| Numeric validation | Manual | Automatic spot-checks |
| Typed equalities | Just True/False | *How* they're equal |
| Test obligations | None | Physics-style constraints |
| Regression testing | None | Full test suites |
| Semantic diffs | None | Mathematical change detection |
| Domain inference | Manual | Automatic |
| Expression caching | None | Hash-based deduplication |

## Prerequisites

- Python 3.10+
- Wolfram Mathematica 14.0+ or Wolfram Engine (free for developers)
- Claude Desktop or Claude Code

## Installation

### 1. Install Wolfram Mathematica or Wolfram Engine

**Option A: Wolfram Mathematica** (Commercial)
- Download from [Wolfram Research](https://www.wolfram.com/mathematica/)

**Option B: Wolfram Engine** (Free for developers)
- Download from [Wolfram Engine](https://www.wolfram.com/engine/)
- Requires free license activation

### 2. Install the MCP Server

```bash
# Clone the repository
git clone https://github.com/paraporoco/Wolfram-MCP.git
cd Wolfram-MCP

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Claude Desktop

Add to your Claude Desktop configuration file:

**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "wolfram": {
      "command": "/path/to/Wolfram-MCP/venv/bin/python",
      "args": ["/path/to/Wolfram-MCP/wolfram_mcp_server.py"],
      "env": {
        "WOLFRAM_KERNEL_PATH": "/Applications/Wolfram.app/Contents/MacOS/WolframKernel"
      }
    }
  }
}
```

### 4. Verify Installation

Restart Claude Desktop and ask:
```
Can you test the Wolfram connection?
```

Claude should use the `wolfram_test` tool and report success with version info.

## Available Tools

### Core Evaluation
| Tool | Description |
|------|-------------|
| `wolfram_eval` | Evaluate code with persistent state, LaTeX output, timing |
| `wolfram_define` | Define persistent functions/variables |
| `wolfram_session_info` | View session state (defined symbols, history) |
| `wolfram_clear_session` | Clear or terminate session |
| `wolfram_test` | Test connection and show version info |
| `wolfram_help` | Get documentation for any Wolfram function |

### Proof-Carrying Computation
| Tool | Description |
|------|-------------|
| `wolfram_eval_proven` | Evaluate with assumption tracking and condition extraction |
| `wolfram_typed_equality` | Verify typed equalities (exact, on domain, up to order, etc.) |
| `wolfram_semantic_diff` | Compute semantic difference between expressions |
| `wolfram_infer_domain` | Infer validity domain, singularities, branch cuts |
| `wolfram_numeric_validate` | Numerically verify two expressions are equal |
| `wolfram_canonicalize` | Canonicalize expression with various methods |
| `wolfram_expression_hash` | Compute semantic hash for caching |

### Obligation Engine (Math CI)
| Tool | Description |
|------|-------------|
| `wolfram_register_obligation` | Register a test that must pass |
| `wolfram_check_obligations` | Check registered obligations |
| `wolfram_list_obligations` | List all obligations and status |
| `wolfram_create_test_suite` | Create a test suite for derivations |
| `wolfram_add_test` | Add test case to a suite |
| `wolfram_run_test_suite` | Run all tests in a suite |

### Mathematics
| Tool | Description |
|------|-------------|
| `wolfram_solve` | Solve algebraic/differential equations |
| `wolfram_calculus` | Integration, differentiation, limits, series |
| `wolfram_symbolic` | Simplify, factor, expand with assumptions |
| `wolfram_linear_algebra` | Matrix operations, eigenvalues, decompositions |

### Visualization
| Tool | Description |
|------|-------------|
| `wolfram_plot` | 2D plots (returns base64 PNG) |
| `wolfram_plot3d` | 3D surface plots (returns base64 PNG) |

## Usage Examples

### Basic Evaluation with Persistence
```python
# Define a function
wolfram_eval("f[x_] := x^2 + 2x + 1")

# Use it later (persists!)
wolfram_eval("f[5]")  # Returns 36
wolfram_eval("D[f[x], x]")  # Returns 2 + 2x
```

### Proof-Carrying Computation
```python
# Track assumptions
wolfram_eval_proven(
    code="Simplify[Sqrt[x^2]]",
    assumptions="x > 0"
)
# Returns: x (with assumptions tracked)

# Without assumptions, correctly returns Abs[x]
wolfram_eval_proven(
    code="Simplify[Sqrt[x^2], Assumptions -> x ∈ Reals]"
)
# Returns: Abs[x]
```

### Typed Equalities
```python
# Verify equality on a domain
wolfram_typed_equality(
    lhs="(x^2 - 1)/(x - 1)",
    rhs="x + 1",
    equality_type="on_domain",
    domain="x != 1",
    variables="x"
)
# Returns: verified with type "EqualityOnDomain"
```

### Obligation Engine
```python
# Register physics constraints
wolfram_register_obligation(
    name="trig_identity",
    description="Pythagorean identity",
    test_type="identity",
    test_expression="Sin[x]^2 + Cos[x]^2",
    expected="1"
)

# Check all obligations
wolfram_check_obligations()
# Returns: passed: 1, failed: 0
```

### Test Suites
```python
# Create a test suite
wolfram_create_test_suite(
    name="Derivative Rules",
    description="Verify calculus identities"
)

# Add tests
wolfram_add_test(
    suite_name="Derivative Rules",
    test_name="power_rule",
    category="identity",
    expression="D[x^n, x]",
    expected="n*x^(n-1)"
)

# Run suite
wolfram_run_test_suite("Derivative Rules")
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WOLFRAM_KERNEL_PATH` | `/Applications/Wolfram.app/Contents/MacOS/WolframKernel` | Path to WolframKernel |
| `WOLFRAM_SESSION_TIMEOUT` | `3600` | Session timeout in seconds |

### Session Management

The server automatically cleans up Wolfram kernel sessions when:
- The MCP server shuts down (via `atexit` handler)
- A termination signal is received (SIGTERM, SIGHUP, SIGINT)
- A session is explicitly terminated via `wolfram_clear_session(terminate=True)`

## Project Structure

```
Wolfram-MCP/
├── wolfram_mcp_server.py    # Main server (2100+ lines)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── VISUALIZATION_WORKFLOW.md # Visualization guide
├── demo_notebook.ipynb       # Feature demonstration
├── stress_tests.ipynb        # Stress tests
├── examples/                 # Example code
│   ├── trig_visualization.jsx
│   ├── trig_plot.py
│   └── README.md
└── venv/                     # Virtual environment
```

## Notebooks

### demo_notebook.ipynb
Demonstrates all proof-carrying computation features with real output:
- Typed equalities
- Semantic diffs
- Domain inference
- Obligation engine
- Test suites
- Expression hashing
- The Riemann zeta integral example

### stress_tests.ipynb
Comprehensive stress tests:
- Timeout handling
- Memory pressure
- Rapid fire requests
- Error recovery
- Session state integrity
- Numeric validation edge cases
- Concurrent sessions
- Graphics under load

## Troubleshooting

### "Wolfram Kernel Not Found"
- Verify Mathematica/Wolfram Engine is installed
- Set `WOLFRAM_KERNEL_PATH` environment variable to correct path
- Try running the kernel directly in terminal

### "Session Timeout"
- Increase `WOLFRAM_SESSION_TIMEOUT` for long sessions
- Use `timeout` parameter in individual tool calls

### "Computation Aborted"
- Normal behavior for long computations exceeding timeout
- Server automatically recovers
- Increase timeout for complex calculations

### Memory Issues
- Use `wolfram_clear_session()` to clear definitions
- Large computations may need more memory
- Consider breaking into smaller steps

## Dependencies

```
fastmcp>=2.0.0
wolframclient>=1.4.0
```

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [FastMCP](https://github.com/jlowin/fastmcp)
- Powered by [Wolfram Language](https://www.wolfram.com/language/)
- Uses [wolframclient](https://github.com/WolframResearch/WolframClientForPython) for persistent sessions

## Links

- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Wolfram Language Reference](https://reference.wolfram.com/language/)
- [wolframclient Documentation](https://reference.wolfram.com/language/WolframClientForPython/)

---

**This system turns Mathematica from a calculator into a proof assistant.**
