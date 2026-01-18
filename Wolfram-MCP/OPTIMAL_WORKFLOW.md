# Optimal Wolfram + Claude Workflow

This document outlines the ideal architecture for maximizing the power of the Wolfram MCP integration with Claude. It covers skills, hooks, sub-agents, and workflow patterns that transform Claude into a rigorous mathematical reasoning system.

---

## Table of Contents

1. [Philosophy](#philosophy)
2. [Claude Skills](#claude-skills)
3. [Hooks](#hooks)
4. [Sub-Agents](#sub-agents)
5. [Workflow Patterns](#workflow-patterns)
6. [Session Management Strategy](#session-management-strategy)
7. [Error Handling Protocol](#error-handling-protocol)
8. [Verification Pipeline](#verification-pipeline)
9. [Memory and Context Optimization](#memory-and-context-optimization)
10. [Example Workflows](#example-workflows)

---

## Philosophy

The core principle: **Claude reasons, Wolfram computes, verification is mandatory.**

- Claude should never do symbolic computation in its head — always delegate to Wolfram
- Every non-trivial result should be verified (numerically, symbolically, or via obligations)
- Assumptions must be explicit and tracked
- The workflow should catch errors early and fail loudly

### The Trust Hierarchy

```
Highest Trust:  Numeric spot-checks pass + symbolic verification + obligations pass
                ↓
Medium Trust:   Symbolic verification only
                ↓
Low Trust:      Single evaluation without verification
                ↓
No Trust:       Claude's mental math (NEVER use for anything non-trivial)
```

---

## Claude Skills

Skills are reusable command patterns. Define these for common mathematical workflows:

### `/derive` — Rigorous Derivation Mode

**Purpose:** Step-by-step derivation with mandatory verification at each step.

**Behavior:**
1. Parse the goal (e.g., "derive the formula for the area of a circle")
2. Create a test suite for the derivation
3. Register obligations for known constraints (dimensional analysis, limits, symmetries)
4. Execute each step with `wolfram_eval_proven`
5. After each step, run `wolfram_check_obligations`
6. Use `wolfram_semantic_diff` to show what changed
7. Final result includes full provenance chain

**Trigger:** User says "derive", "prove", "show that", "demonstrate"

**Example invocation:**
```
/derive Show that the integral of x^n from 0 to 1 is 1/(n+1) for n > -1
```

---

### `/verify` — Verification Mode

**Purpose:** Verify a claimed identity or result.

**Behavior:**
1. Parse the claim into LHS and RHS
2. Determine the appropriate equality type (exact, on domain, up to order)
3. Run `wolfram_typed_equality` with numeric verification
4. If domain-dependent, run `wolfram_infer_domain` on both sides
5. Report with confidence level

**Trigger:** User says "verify", "check", "is this correct", "confirm"

**Example:**
```
/verify Is sin(2x) = 2sin(x)cos(x)?
```

---

### `/explore` — Mathematical Exploration Mode

**Purpose:** Explore properties of a mathematical object.

**Behavior:**
1. Define the object in Wolfram session
2. Compute basic properties (domain, singularities, symmetries)
3. Generate plots if applicable
4. Find special values and limits
5. Look for related identities
6. Build a test suite for the object

**Trigger:** User says "explore", "investigate", "what can you tell me about"

**Example:**
```
/explore the Gamma function
```

---

### `/physics` — Physics Calculation Mode

**Purpose:** Physics calculations with dimensional analysis and physical constraints.

**Behavior:**
1. Register dimensional analysis obligations
2. Register physical limit obligations (e.g., non-relativistic limit, classical limit)
3. Register symmetry obligations (gauge invariance, Lorentz invariance)
4. Execute calculation with full provenance
5. Verify all physical constraints pass
6. Express result in standard physics notation

**Trigger:** User mentions physics quantities, Lagrangians, Hamiltonians, field theory

**Example:**
```
/physics Calculate the scattering amplitude for electron-electron scattering at tree level
```

---

### `/optimize` — Optimization Mode

**Purpose:** Find and verify optima.

**Behavior:**
1. Parse the objective function and constraints
2. Use `wolfram_solve` or optimization functions
3. Verify solutions satisfy constraints (register as obligations)
4. Check second-order conditions
5. Numeric validation at found optima
6. Report with full verification

---

### `/series` — Series Expansion Mode

**Purpose:** Taylor/Laurent series with controlled error.

**Behavior:**
1. Compute series to requested order
2. Register obligation: series matches function up to that order
3. Compute and report error bounds
4. Numeric validation at multiple points
5. Use typed equality "up_to_order"

---

## Hooks

Hooks are automatic triggers that run before/after tool calls or on specific events.

### Pre-Evaluation Hook

**Trigger:** Before any `wolfram_eval` or `wolfram_eval_proven` call

**Actions:**
1. Check if expression contains potentially dangerous operations (`PowerExpand`, `ComplexExpand` without assumptions)
2. Warn if assumptions might be needed but aren't provided
3. Estimate computation complexity and suggest timeout
4. Auto-add common assumptions if context suggests them

**Implementation concept:**
```yaml
hooks:
  pre_wolfram_eval:
    - check_dangerous_operations
    - suggest_assumptions
    - estimate_timeout
```

---

### Post-Evaluation Hook

**Trigger:** After any `wolfram_eval` or `wolfram_eval_proven` call

**Actions:**
1. Check if result contains `ConditionalExpression` — extract and report conditions
2. Check if result is `Indeterminate`, `ComplexInfinity`, or `Undefined` — investigate
3. If result is numeric, sanity-check magnitude
4. Auto-register result in expression cache
5. Update session context with new definitions

---

### Obligation Failure Hook

**Trigger:** When `wolfram_check_obligations` returns any failures

**Actions:**
1. Pause the derivation
2. Analyze which obligation failed and why
3. Suggest possible fixes (wrong assumptions, algebraic error, etc.)
4. Ask user whether to continue, backtrack, or abort
5. Log the failure for debugging

---

### Session Timeout Hook

**Trigger:** When a session approaches timeout

**Actions:**
1. Warn user about impending session expiry
2. Offer to save current state (definitions, obligations, test suites)
3. Prepare session migration strategy

---

### Error Recovery Hook

**Trigger:** When Wolfram returns an error or unexpected result

**Actions:**
1. Parse error message for actionable information
2. Check for common mistakes (syntax, undefined symbols, wrong arguments)
3. Suggest corrections
4. If recoverable, auto-retry with fixes
5. If not, gracefully report with context

---

## Sub-Agents

Sub-agents handle specialized tasks autonomously.

### Verification Agent

**Purpose:** Independently verify results from the main computation.

**Behavior:**
1. Receives a claim (expression = result)
2. Uses a SEPARATE Wolfram session (isolation)
3. Verifies via independent method:
   - Different symbolic approach
   - Numeric validation
   - Special case checking
4. Reports agreement or discrepancy
5. If discrepancy, investigates root cause

**When to invoke:**
- High-stakes calculations
- Results that "look wrong"
- Before publishing or using results

---

### Literature Agent

**Purpose:** Check results against known mathematical identities.

**Behavior:**
1. Receives an expression or identity
2. Searches for matching known results (OEIS, DLMF, Wolfram Functions Site)
3. Cross-references with known special values
4. Reports matches or novelty

**Integration:**
- Use web search for DLMF (Digital Library of Mathematical Functions)
- Use Wolfram's `FunctionExpand` to find known forms
- Check OEIS for integer sequences

---

### Assumption Analyst Agent

**Purpose:** Determine minimal assumptions needed for a result.

**Behavior:**
1. Start with no assumptions
2. Incrementally add assumptions until result holds
3. Test if each assumption is necessary (try removing it)
4. Report minimal assumption set
5. Warn about hidden assumptions in operations used

**Example output:**
```
Result: √(x²) = x
Minimal assumptions: x ≥ 0 (real, non-negative)
Removing x ≥ 0: Result becomes |x|
Removing x ∈ Reals: Result becomes √(x²) (unevaluated)
```

---

### Dimensional Analysis Agent

**Purpose:** Verify dimensional consistency in physics calculations.

**Behavior:**
1. Parse expression for physical quantities
2. Assign dimensions (mass, length, time, etc.)
3. Verify dimensional homogeneity
4. Check that final result has expected dimensions
5. Register as obligations

---

### Test Generator Agent

**Purpose:** Automatically generate test cases for mathematical objects.

**Behavior:**
1. Analyze the mathematical object
2. Generate test cases:
   - Boundary values
   - Special points (0, 1, ∞, -1, i)
   - Random sampling
   - Known identities
3. Create test suite automatically
4. Run and report

---

## Workflow Patterns

### Pattern 1: The Verification Sandwich

Every computation should be sandwiched between setup and verification:

```
1. SETUP
   - Define the problem
   - Register expected constraints as obligations
   - Create test suite

2. COMPUTE
   - Execute with wolfram_eval_proven
   - Track assumptions used

3. VERIFY
   - Check all obligations
   - Numeric spot-check
   - Semantic diff from expected form
```

---

### Pattern 2: Progressive Refinement

For complex derivations, build up in verified steps:

```
1. Rough calculation (quick, may have errors)
2. Identify critical steps
3. Re-do each critical step with full verification
4. Chain verified steps together
5. Verify final result independently
```

---

### Pattern 3: Multi-Path Verification

For important results, verify via multiple independent paths:

```
1. Path A: Direct symbolic computation
2. Path B: Alternative method (e.g., contour integration vs. residues)
3. Path C: Numeric validation
4. Path D: Check against known results

All paths must agree → High confidence
```

---

### Pattern 4: Assumption Bracketing

When unsure about assumptions:

```
1. Compute with minimal assumptions
2. Compute with maximal assumptions
3. Compare results
4. Identify which assumptions actually matter
5. Report with explicit assumption dependence
```

---

### Pattern 5: The Error Budget

For numerical computations:

```
1. Estimate error at each step
2. Track cumulative error budget
3. Stop if error exceeds tolerance
4. Report final result with error bounds
```

---

## Session Management Strategy

### Session Naming Convention

```
- "default"           : General-purpose scratch session
- "derivation_XXX"    : Dedicated session for specific derivation
- "verification"      : Isolated session for independent verification
- "exploration"       : Session for mathematical exploration
```

### When to Create New Sessions

- Starting a new unrelated problem → new session
- Need independent verification → new session
- Previous session has too much state pollution → new session

### When to Reuse Sessions

- Continuing a derivation → reuse
- Building on previous definitions → reuse
- Related calculations → reuse

### Session Hygiene

- Clear scratch variables periodically
- Keep only essential definitions
- Document what's in each session
- Terminate unused sessions

---

## Error Handling Protocol

### Level 1: Automatic Recovery

- Syntax errors → suggest fix, retry
- Timeout → increase timeout, retry
- Undefined symbol → check for typo, suggest

### Level 2: User Intervention

- Multiple failures → ask user for guidance
- Ambiguous errors → present options
- Resource exhaustion → ask about simplification

### Level 3: Graceful Failure

- Unrecoverable errors → save state, report clearly
- Session corruption → terminate, start fresh
- Kernel crash → restart kernel, restore state if possible

---

## Verification Pipeline

For any significant computation, run this pipeline:

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT: Computation                    │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  1. OBLIGATION CHECK                                     │
│     - All registered obligations pass?                   │
│     - If no → STOP, investigate                          │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  2. SYMBOLIC VERIFICATION                                │
│     - Does Simplify[result - expected] == 0?             │
│     - Are there unexpected conditions?                   │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  3. NUMERIC VALIDATION                                   │
│     - Random point sampling (20+ points)                 │
│     - Edge cases (0, 1, -1, large, small)                │
│     - Near singularities (carefully)                     │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  4. DOMAIN ANALYSIS                                      │
│     - Where is result valid?                             │
│     - Any unexpected restrictions?                       │
│     - Branch cut issues?                                 │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  5. CROSS-CHECK (optional, for high-stakes)              │
│     - Independent verification agent                     │
│     - Different computational path                       │
│     - Literature check                                   │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    OUTPUT: Verified Result               │
│     - Result with confidence level                       │
│     - Full provenance chain                              │
│     - Explicit assumptions and conditions                │
└─────────────────────────────────────────────────────────┘
```

---

## Memory and Context Optimization

### What to Keep in Context

- Current problem statement
- Key definitions (keep minimal)
- Active obligations
- Current test suite results
- Error history (for debugging)

### What to Offload to Wolfram Session

- All variable definitions
- Function definitions
- Intermediate results
- Computation history

### What to Store Long-Term

- Verified results (with hash)
- Test suites (for regression)
- Useful definitions (in a library)

### Context Window Management

- Summarize long derivations
- Keep only essential steps in context
- Use Wolfram session as external memory
- Reference previous results by hash

---

## Example Workflows

### Example 1: Proving a Trig Identity

**User:** Prove that sin(3x) = 3sin(x) - 4sin³(x)

**Workflow:**

1. **Setup:**
   ```
   Create test suite "Triple Angle"
   Register obligation: identity holds for x = 0, π/6, π/4, π/3, random
   ```

2. **Compute:**
   ```
   wolfram_eval_proven("TrigExpand[Sin[3x]]")
   → Result: 3*Sin[x] - 4*Sin[x]^3
   ```

3. **Verify:**
   ```
   wolfram_typed_equality(
     lhs="Sin[3x]",
     rhs="3*Sin[x] - 4*Sin[x]^3",
     equality_type="exact",
     variables="x"
   )
   → Verified: true (symbolic + numeric)

   wolfram_run_test_suite("Triple Angle")
   → All tests pass
   ```

4. **Report:**
   ```
   ✓ Proven: sin(3x) = 3sin(x) - 4sin³(x)
   Verification: Symbolic ✓, Numeric (20 points) ✓, Test suite ✓
   Assumptions: None (identity holds ∀x ∈ ℂ)
   ```

---

### Example 2: Physics Calculation with Dimensional Analysis

**User:** Calculate the period of a simple pendulum.

**Workflow:**

1. **Setup:**
   ```
   Register obligation "dimensional_correctness":
     test_type: "dimension"
     expected: [Time]

   Register obligation "small_angle_limit":
     As θ → 0, formula should simplify

   Register obligation "g_dependence":
     Period should decrease with increasing g
   ```

2. **Compute:**
   ```
   wolfram_eval_proven(
     "DSolve[θ''[t] + (g/L)*Sin[θ[t]] == 0, θ[t], t]",
     assumptions="g > 0 && L > 0"
   )
   → Complex elliptic integral solution

   Small angle approximation:
   wolfram_eval_proven(
     "DSolve[θ''[t] + (g/L)*θ[t] == 0, θ[t], t]"
   )
   → θ[t] = C1*Cos[Sqrt[g/L]*t] + C2*Sin[Sqrt[g/L]*t]

   Period:
   wolfram_eval("2*Pi/Sqrt[g/L] // Simplify")
   → 2π√(L/g)
   ```

3. **Verify:**
   ```
   wolfram_check_obligations()
   - dimensional_correctness: ✓ (√(L/g) has dimension Time)
   - small_angle_limit: ✓
   - g_dependence: ✓ (T ∝ 1/√g)
   ```

4. **Report:**
   ```
   Period T = 2π√(L/g)

   Valid for: Small angles (θ << 1)
   Assumptions: g > 0, L > 0
   Dimensional check: [T] = √([L]/[L/T²]) = [T] ✓
   ```

---

### Example 3: Exploring a Special Function

**User:** Explore the Riemann zeta function.

**Workflow:**

1. **Basic Properties:**
   ```
   wolfram_infer_domain("Zeta[s]", "s")
   → Domain: s ≠ 1 (pole)

   wolfram_eval("FunctionSingularities[Zeta[s], s]")
   → Simple pole at s = 1
   ```

2. **Special Values:**
   ```
   Create test suite "Zeta Special Values"
   Add tests:
     Zeta[2] = π²/6
     Zeta[4] = π⁴/90
     Zeta[0] = -1/2
     Zeta[-1] = -1/12
     Zeta[-2] = 0

   wolfram_run_test_suite("Zeta Special Values")
   → All pass
   ```

3. **Functional Equation:**
   ```
   wolfram_register_obligation(
     name="functional_equation",
     test_expression="Zeta[s] - 2^s * Pi^(s-1) * Sin[Pi*s/2] * Gamma[1-s] * Zeta[1-s]",
     test_type="zero"
   )

   wolfram_check_obligations()
   → functional_equation: ✓
   ```

4. **Visualization:**
   ```
   wolfram_plot("Abs[Zeta[1/2 + I*t]]", "t", 0, 50)
   → Plot showing zeros on critical line
   ```

5. **Report:**
   ```
   Riemann Zeta Function ζ(s)

   Domain: ℂ \ {1}
   Singularities: Simple pole at s = 1 with residue 1

   Special values verified:
   - ζ(2) = π²/6 ✓
   - ζ(4) = π⁴/90 ✓
   - ζ(0) = -1/2 ✓
   - ζ(-1) = -1/12 ✓

   Functional equation verified ✓

   [Plot of |ζ(1/2 + it)| attached]
   ```

---

## Implementation Checklist

To achieve this optimal workflow, implement:

- [ ] Define Claude skills: `/derive`, `/verify`, `/explore`, `/physics`, `/optimize`, `/series`
- [ ] Create hook configurations for pre/post evaluation
- [ ] Implement verification agent as sub-agent
- [ ] Create assumption analyst agent
- [ ] Build dimensional analysis agent for physics
- [ ] Create test generator agent
- [ ] Document session naming conventions
- [ ] Build verification pipeline as reusable pattern
- [ ] Create templates for common workflows
- [ ] Set up error handling protocol
- [ ] Optimize context management

---

## Conclusion

This workflow transforms Claude + Wolfram from a simple calculator into a rigorous mathematical reasoning system. The key principles:

1. **Separation of concerns:** Claude reasons, Wolfram computes
2. **Verification is mandatory:** Never trust unverified results
3. **Assumptions are explicit:** Track and report all assumptions
4. **Errors fail loudly:** Catch problems early
5. **State is managed:** Clean sessions, clear context

With this architecture, you can trust mathematical results at a level approaching formal proof systems, while maintaining the flexibility and natural language interface of Claude.

---

*This document is a living specification. Update it as new patterns emerge and the system evolves.*
