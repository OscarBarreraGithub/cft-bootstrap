"""
Sophisticated Wolfram Language MCP Server with Proof-Carrying Computation

Provides full Mathematica/Wolfram Language integration with:
- Persistent kernel sessions (variables, functions persist across calls)
- Rich output (graphics as base64, LaTeX for symbolic expressions)
- Timeout/abort handling for long computations
- PROOF-CARRYING SYMBOLIC COMPUTATION:
  - Typed equalities (exact, on domain, up to order, modulo equivalences)
  - Assumption provenance tracking
  - Condition extraction with GenerateConditions
  - Obligation engine for registered tests
  - Domain/branch management
  - Canonicalization and semantic diffs
  - Math CI regression testing
  - Numeric spot-check validation
"""

import asyncio
import atexit
import base64
import hashlib
import json
import os
import random
import signal
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

from fastmcp import FastMCP
from wolframclient.evaluation import WolframLanguageAsyncSession
from wolframclient.language import wl, wlexpr
from wolframclient.exception import WolframKernelException

# Configuration
WOLFRAM_KERNEL_PATH = os.getenv(
    "WOLFRAM_KERNEL_PATH",
    "/Applications/Wolfram.app/Contents/MacOS/WolframKernel"
)
SESSION_TIMEOUT = int(os.getenv("WOLFRAM_SESSION_TIMEOUT", "3600"))
DEFAULT_EVAL_TIMEOUT = 30
MAX_EVAL_TIMEOUT = 600

# Initialize MCP server
mcp = FastMCP("Wolfram Language Pro")


# =============================================================================
# TYPED EQUALITY SYSTEM
# =============================================================================

class EqualityType(Enum):
    """Types of mathematical equality relations"""
    EXACT = "ExactEquality"
    ON_DOMAIN = "EqualityOnDomain"
    UP_TO_ORDER = "EqualityUpToOrder"
    MODULO_TOTAL_DERIVATIVES = "EqualityModuloTotalDerivatives"
    MODULO_GAUGE = "EqualityModuloGauge"
    MODULO_CONTACT = "EqualityModuloContact"
    AFTER_RENORMALIZATION = "EqualityAfterRenormalization"
    NUMERIC_APPROX = "NumericApproximation"
    CONDITIONAL = "ConditionalEquality"


@dataclass
class TypedEquality:
    """Represents a typed equality relation between expressions"""
    lhs: str
    rhs: str
    equality_type: EqualityType
    domain: Optional[str] = None  # e.g., "x > 0 && x < 1"
    order: Optional[str] = None   # e.g., "O[x]^5"
    modulo: Optional[str] = None  # what we're working modulo
    conditions: List[str] = field(default_factory=list)
    provenance: Optional[str] = None  # which step created this

    def to_dict(self) -> Dict:
        return {
            "lhs": self.lhs,
            "rhs": self.rhs,
            "type": self.equality_type.value,
            "domain": self.domain,
            "order": self.order,
            "modulo": self.modulo,
            "conditions": self.conditions,
            "provenance": self.provenance
        }


# =============================================================================
# ASSUMPTION PROVENANCE
# =============================================================================

@dataclass
class AssumptionContext:
    """Tracks assumptions and their provenance"""
    assumptions: str  # The $Assumptions expression
    element_constraints: List[str]  # Element[x, Reals], etc.
    positivity_constraints: List[str]  # x > 0, etc.
    integer_constraints: List[str]  # Element[n, Integers], etc.
    simplification_flags: List[str]  # PowerExpand, etc. (dangerous ones)

    def to_dict(self) -> Dict:
        return {
            "assumptions": self.assumptions,
            "element_constraints": self.element_constraints,
            "positivity_constraints": self.positivity_constraints,
            "integer_constraints": self.integer_constraints,
            "dangerous_flags": self.simplification_flags
        }


# =============================================================================
# DOMAIN/BRANCH MANAGEMENT
# =============================================================================

@dataclass
class DomainSpec:
    """Specification of a mathematical domain"""
    name: str
    base_set: str  # "Reals", "Complexes", "Integers"
    constraints: List[str]  # ["x > 0", "x < Pi"]
    excluded_points: List[str]  # Singular points
    branch_cuts: List[str]  # For complex functions
    sheet_choice: Optional[str] = None  # Branch sheet specification

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "base_set": self.base_set,
            "constraints": self.constraints,
            "excluded_points": self.excluded_points,
            "branch_cuts": self.branch_cuts,
            "sheet_choice": self.sheet_choice
        }


# =============================================================================
# OBLIGATION ENGINE
# =============================================================================

class ObligationStatus(Enum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Obligation:
    """A test obligation that must pass for a step to be accepted"""
    id: str
    name: str
    description: str
    test_type: str  # "numeric_check", "identity", "limit", "dimension", "symmetry"
    test_expression: str
    expected: Optional[str] = None
    tolerance: float = 1e-10
    status: ObligationStatus = ObligationStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "test_type": self.test_type,
            "test_expression": self.test_expression,
            "expected": self.expected,
            "tolerance": self.tolerance,
            "status": self.status.value,
            "result": self.result,
            "error": self.error
        }


# =============================================================================
# PROOF-CARRYING RESULT
# =============================================================================

@dataclass
class ProofCarryingResult:
    """A computation result with full provenance and verification"""
    # Core result
    expression: str
    result: Any
    formatted_result: str
    latex: Optional[str] = None

    # Provenance
    input_expression: str = ""
    assumptions_used: Optional[AssumptionContext] = None
    conditions_generated: List[str] = field(default_factory=list)
    transformations_applied: List[str] = field(default_factory=list)

    # Equality type
    equality_relation: Optional[TypedEquality] = None

    # Domain
    validity_domain: Optional[DomainSpec] = None

    # Verification
    obligations: List[Obligation] = field(default_factory=list)
    numeric_checks_passed: int = 0
    numeric_checks_total: int = 0

    # Metadata
    timing: Optional[float] = None
    memory_used: Optional[int] = None
    hash: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "expression": self.expression,
            "result": self.formatted_result,
            "latex": self.latex,
            "input": self.input_expression,
            "assumptions": self.assumptions_used.to_dict() if self.assumptions_used else None,
            "conditions": self.conditions_generated,
            "transformations": self.transformations_applied,
            "equality": self.equality_relation.to_dict() if self.equality_relation else None,
            "domain": self.validity_domain.to_dict() if self.validity_domain else None,
            "obligations": [o.to_dict() for o in self.obligations],
            "numeric_validation": {
                "passed": self.numeric_checks_passed,
                "total": self.numeric_checks_total
            },
            "timing": self.timing,
            "hash": self.hash
        }


# =============================================================================
# SEMANTIC DIFF
# =============================================================================

@dataclass
class SemanticDiff:
    """Represents the semantic difference between two expressions"""
    expr1: str
    expr2: str
    canonical_form1: str
    canonical_form2: str
    are_equivalent: bool
    difference_type: str  # "identical", "simplified", "terms_changed", "structure_changed"
    terms_added: List[str] = field(default_factory=list)
    terms_removed: List[str] = field(default_factory=list)
    pole_changes: List[str] = field(default_factory=list)
    degree_change: Optional[int] = None
    symmetry_changes: List[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict:
        return {
            "expr1": self.expr1,
            "expr2": self.expr2,
            "canonical1": self.canonical_form1,
            "canonical2": self.canonical_form2,
            "equivalent": self.are_equivalent,
            "difference_type": self.difference_type,
            "terms_added": self.terms_added,
            "terms_removed": self.terms_removed,
            "pole_changes": self.pole_changes,
            "degree_change": self.degree_change,
            "symmetry_changes": self.symmetry_changes,
            "summary": self.summary
        }


# =============================================================================
# TEST SUITE FOR MATH CI
# =============================================================================

@dataclass
class TestCase:
    """A single test case in a math CI suite"""
    id: str
    name: str
    category: str  # "dimension", "limit", "symmetry", "identity", "numeric"
    expression: str
    expected: str
    variables: List[str] = field(default_factory=list)
    parameters: Dict[str, str] = field(default_factory=dict)
    tolerance: float = 1e-10

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "expression": self.expression,
            "expected": self.expected,
            "variables": self.variables,
            "parameters": self.parameters,
            "tolerance": self.tolerance
        }


@dataclass
class TestSuite:
    """A collection of test cases for a mathematical derivation"""
    name: str
    description: str
    tests: List[TestCase] = field(default_factory=list)
    results: Dict[str, ObligationStatus] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "tests": [t.to_dict() for t in self.tests],
            "results": {k: v.value for k, v in self.results.items()}
        }


# =============================================================================
# LEGACY DATA CLASSES (kept for compatibility)
# =============================================================================

@dataclass
class EvaluationResult:
    """Structured result from Wolfram evaluation"""
    success: bool
    result: Any
    result_type: str
    formatted_result: str
    latex: Optional[str] = None
    messages: List[str] = field(default_factory=list)
    timing: Optional[float] = None
    aborted: bool = False
    error: Optional[str] = None


@dataclass
class SessionState:
    """Tracks the state of a Wolfram session"""
    session_id: str
    created_at: datetime
    last_activity: datetime
    kernel_session: Optional[WolframLanguageAsyncSession] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    is_active: bool = False
    # New: proof-carrying state
    obligations: Dict[str, Obligation] = field(default_factory=dict)
    test_suites: Dict[str, TestSuite] = field(default_factory=dict)
    domains: Dict[str, DomainSpec] = field(default_factory=dict)
    expression_hashes: Dict[str, str] = field(default_factory=dict)


# =============================================================================
# SESSION MANAGER
# =============================================================================

class WolframSessionManager:
    """Manages persistent Wolfram kernel sessions"""

    def __init__(self):
        self._sessions: Dict[str, SessionState] = {}
        self._default_session_id = "default"
        self._lock = asyncio.Lock()

    async def get_or_create_session(
        self,
        session_id: Optional[str] = None
    ) -> SessionState:
        """Get existing session or create new one"""
        session_id = session_id or self._default_session_id

        async with self._lock:
            if session_id in self._sessions:
                state = self._sessions[session_id]
                if state.is_active and state.kernel_session:
                    state.last_activity = datetime.now()
                    return state

            state = await self._create_session(session_id)
            self._sessions[session_id] = state
            return state

    async def _create_session(self, session_id: str) -> SessionState:
        """Create and initialize a new kernel session"""
        kernel_session = WolframLanguageAsyncSession(
            kernel=WOLFRAM_KERNEL_PATH
        )

        await kernel_session.start()

        # Initialize session with useful defaults and proof-carrying infrastructure
        await kernel_session.evaluate(wlexpr('''
            $HistoryLength = 100;
            SetOptions[$Output, PageWidth -> Infinity];

            (* Proof-carrying computation infrastructure *)
            $PCAssumptions = {};
            $PCTransformations = {};
            $PCConditions = {};

            (* Helper to track assumptions *)
            PCWithAssumptions[expr_, assumptions_] := Block[
                {$Assumptions = assumptions},
                AppendTo[$PCAssumptions, assumptions];
                expr
            ];

            (* Helper to extract conditions *)
            PCExtractConditions[expr_] := Module[{result},
                result = expr /. ConditionalExpression[e_, c_] :> (
                    AppendTo[$PCConditions, c]; e
                );
                result
            ];
        '''))

        state = SessionState(
            session_id=session_id,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            kernel_session=kernel_session,
            is_active=True
        )

        return state

    async def terminate_session(self, session_id: str) -> bool:
        """Gracefully terminate a session"""
        async with self._lock:
            if session_id not in self._sessions:
                return False

            state = self._sessions[session_id]
            if state.kernel_session:
                try:
                    await state.kernel_session.terminate()
                except Exception:
                    pass

            state.is_active = False
            del self._sessions[session_id]
            return True

    async def terminate_all(self):
        """Terminate all sessions"""
        session_ids = list(self._sessions.keys())
        for sid in session_ids:
            await self.terminate_session(sid)


# =============================================================================
# KERNEL CONTROLLER WITH PROOF-CARRYING SUPPORT
# =============================================================================

class KernelController:
    """Handles evaluation with timeout, abort, and proof-carrying features"""

    def __init__(self, session_state: SessionState):
        self.session = session_state.kernel_session
        self.state = session_state

    async def evaluate(
        self,
        expression: str,
        timeout: int = DEFAULT_EVAL_TIMEOUT,
        return_timing: bool = False
    ) -> EvaluationResult:
        """Evaluate expression with full error handling"""
        timeout = min(timeout, MAX_EVAL_TIMEOUT)

        try:
            if return_timing:
                wrapped = f'''
                Module[{{result, timing}},
                    {{timing, result}} = AbsoluteTiming[
                        TimeConstrained[{expression}, {timeout}, $Aborted]
                    ];
                    <|"result" -> result, "timing" -> timing, "aborted" -> (result === $Aborted)|>
                ]
                '''
                raw_result = await asyncio.wait_for(
                    self.session.evaluate(wlexpr(wrapped)),
                    timeout=timeout + 10
                )

                if isinstance(raw_result, dict):
                    actual_result = raw_result.get("result")
                    timing = raw_result.get("timing")
                    aborted = raw_result.get("aborted", False)
                else:
                    actual_result = raw_result
                    timing = None
                    aborted = False
            else:
                wrapped = f'TimeConstrained[{expression}, {timeout}, $Aborted]'
                actual_result = await asyncio.wait_for(
                    self.session.evaluate(wlexpr(wrapped)),
                    timeout=timeout + 10
                )
                timing = None
                aborted = str(actual_result) == "$Aborted"

            if str(actual_result) == "$Aborted":
                return EvaluationResult(
                    success=False,
                    result=None,
                    result_type="Aborted",
                    formatted_result="Computation aborted (timeout)",
                    aborted=True,
                    timing=timing
                )

            formatted = await self._format_result(actual_result)
            result_type = await self._get_type(actual_result)

            latex = None
            if result_type in ["Symbol", "Plus", "Times", "Power", "Integer", "Rational", "Real"]:
                try:
                    latex = await self.session.evaluate(wl.ToString(wl.TeXForm(actual_result)))
                except Exception:
                    pass

            return EvaluationResult(
                success=True,
                result=actual_result,
                result_type=result_type,
                formatted_result=formatted,
                latex=latex,
                timing=timing,
                aborted=False
            )

        except asyncio.TimeoutError:
            return EvaluationResult(
                success=False,
                result=None,
                result_type="Timeout",
                formatted_result=f"Python-side timeout after {timeout}s",
                aborted=True,
                error=f"Computation exceeded {timeout} seconds"
            )
        except WolframKernelException as e:
            return EvaluationResult(
                success=False,
                result=None,
                result_type="KernelError",
                formatted_result=str(e),
                error=str(e)
            )
        except Exception as e:
            return EvaluationResult(
                success=False,
                result=None,
                result_type="Error",
                formatted_result=str(e),
                error=str(e)
            )

    async def evaluate_with_provenance(
        self,
        expression: str,
        assumptions: Optional[str] = None,
        generate_conditions: bool = True,
        timeout: int = DEFAULT_EVAL_TIMEOUT
    ) -> ProofCarryingResult:
        """Evaluate with full assumption tracking and condition extraction"""
        timeout = min(timeout, MAX_EVAL_TIMEOUT)

        # Build the provenance-tracking evaluation
        assumptions_clause = f"$Assumptions = {assumptions};" if assumptions else ""
        gen_cond = "GenerateConditions -> True" if generate_conditions else "GenerateConditions -> False"

        wrapped = f'''
        Module[{{result, timing, conditions, assumptions, messages}},
            {assumptions_clause}
            assumptions = $Assumptions;
            messages = {{}};

            {{timing, result}} = AbsoluteTiming[
                TimeConstrained[
                    Quiet[
                        Check[
                            {expression},
                            AppendTo[messages, $MessageList]
                        ],
                        {{}}
                    ],
                    {timeout},
                    $Aborted
                ]
            ];

            (* Extract conditions if result is ConditionalExpression *)
            conditions = If[Head[result] === ConditionalExpression,
                {{result[[2]]}},
                {{}}
            ];

            <|
                "result" -> If[Head[result] === ConditionalExpression, result[[1]], result],
                "timing" -> timing,
                "assumptions" -> assumptions,
                "conditions" -> conditions,
                "messages" -> messages,
                "aborted" -> (result === $Aborted),
                "inputForm" -> ToString[InputForm[result]],
                "texForm" -> ToString[TeXForm[If[Head[result] === ConditionalExpression, result[[1]], result]]]
            |>
        ]
        '''

        try:
            raw_result = await asyncio.wait_for(
                self.session.evaluate(wlexpr(wrapped)),
                timeout=timeout + 10
            )

            if isinstance(raw_result, dict):
                actual_result = raw_result.get("result")
                timing = raw_result.get("timing")
                assumptions_used = str(raw_result.get("assumptions", "True"))
                conditions = [str(c) for c in raw_result.get("conditions", [])]
                latex = raw_result.get("texForm", "")
                input_form = raw_result.get("inputForm", "")
                aborted = raw_result.get("aborted", False)
            else:
                actual_result = raw_result
                timing = None
                assumptions_used = "True"
                conditions = []
                latex = ""
                input_form = str(raw_result)
                aborted = False

            if aborted:
                return ProofCarryingResult(
                    expression=expression,
                    result=None,
                    formatted_result="Computation aborted (timeout)",
                    input_expression=expression,
                    timing=timing
                )

            # Build assumption context
            assumption_context = AssumptionContext(
                assumptions=assumptions_used,
                element_constraints=[],
                positivity_constraints=[],
                integer_constraints=[],
                simplification_flags=[]
            )

            # Parse assumption details
            if assumptions:
                # Extract Element constraints
                elem_result = await self.session.evaluate(wlexpr(f'''
                    Cases[{assumptions}, Element[_, _], Infinity]
                '''))
                if elem_result:
                    assumption_context.element_constraints = [str(e) for e in elem_result] if isinstance(elem_result, list) else []

            # Compute hash for caching/deduplication
            hash_input = f"{expression}|{assumptions_used}|{conditions}"
            expr_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

            return ProofCarryingResult(
                expression=input_form,
                result=actual_result,
                formatted_result=str(actual_result),
                latex=latex if latex else None,
                input_expression=expression,
                assumptions_used=assumption_context,
                conditions_generated=conditions,
                transformations_applied=[],
                timing=timing,
                hash=expr_hash
            )

        except asyncio.TimeoutError:
            return ProofCarryingResult(
                expression=expression,
                result=None,
                formatted_result=f"Timeout after {timeout}s",
                input_expression=expression
            )
        except Exception as e:
            return ProofCarryingResult(
                expression=expression,
                result=None,
                formatted_result=f"Error: {str(e)}",
                input_expression=expression
            )

    async def numeric_spot_check(
        self,
        expr1: str,
        expr2: str,
        variables: List[str],
        domain_constraints: Optional[str] = None,
        num_points: int = 10,
        tolerance: float = 1e-10
    ) -> Tuple[bool, List[Dict]]:
        """Numerically verify two expressions are equal at random points"""
        results = []
        all_passed = True

        for i in range(num_points):
            # Generate random point in domain
            if domain_constraints:
                # Build fallback random values for each variable
                var_rules_fallback = ", ".join([f"{v} -> RandomReal[{{-10, 10}}]" for v in variables])
                point_code = f'''
                Module[{{point}},
                    point = Quiet[FindInstance[{domain_constraints}, {{{", ".join(variables)}}}, Reals, 1]];
                    If[point === {{}} || !ListQ[point],
                        {{{var_rules_fallback}}},
                        point[[1]]
                    ]
                ]
                '''
            else:
                # Generate random values for each variable
                var_rules = ", ".join([f"{v} -> RandomReal[{{-10, 10}}]" for v in variables])
                point_code = f'{{{var_rules}}}'

            point_result = await self.session.evaluate(wlexpr(point_code))

            # Convert point_result to InputForm string for safe re-evaluation
            point_str = await self.session.evaluate(wl.ToString(wl.InputForm(point_result)))

            # Evaluate both expressions at this point
            check_code = f'''
            Module[{{val1, val2, point = {point_str}}},
                val1 = N[{expr1} /. point, 20];
                val2 = N[{expr2} /. point, 20];
                <|
                    "point" -> point,
                    "val1" -> val1,
                    "val2" -> val2,
                    "diff" -> Abs[val1 - val2],
                    "passed" -> (Abs[val1 - val2] < {tolerance} ||
                                 (Abs[val1] > 0 && Abs[(val1 - val2)/val1] < {tolerance}))
                |>
            ]
            '''

            check_result = await self.session.evaluate(wlexpr(check_code))

            if isinstance(check_result, dict):
                passed = check_result.get("passed", False)
                results.append({
                    "point": str(check_result.get("point", "")),
                    "val1": str(check_result.get("val1", "")),
                    "val2": str(check_result.get("val2", "")),
                    "diff": str(check_result.get("diff", "")),
                    "passed": passed
                })
                if not passed:
                    all_passed = False

        return all_passed, results

    async def compute_semantic_diff(
        self,
        expr1: str,
        expr2: str,
        canonicalization: str = "Together"
    ) -> SemanticDiff:
        """Compute semantic difference between two expressions"""
        # Canonicalize both expressions
        canon_code = f'''
        <|
            "canon1" -> ToString[InputForm[{canonicalization}[{expr1}]]],
            "canon2" -> ToString[InputForm[{canonicalization}[{expr2}]]],
            "diff" -> ToString[InputForm[Simplify[{expr1} - {expr2}]]],
            "equivalent" -> PossibleZeroQ[Simplify[{expr1} - {expr2}]]
        |>
        '''

        result = await self.session.evaluate(wlexpr(canon_code))

        if not isinstance(result, dict):
            return SemanticDiff(
                expr1=expr1,
                expr2=expr2,
                canonical_form1=expr1,
                canonical_form2=expr2,
                are_equivalent=False,
                difference_type="error",
                summary="Could not compute diff"
            )

        canon1 = result.get("canon1", expr1)
        canon2 = result.get("canon2", expr2)
        equivalent = result.get("equivalent", False)
        diff_expr = result.get("diff", "")

        # Determine difference type
        if equivalent:
            diff_type = "identical" if canon1 == canon2 else "simplified"
        else:
            diff_type = "terms_changed"

        # Analyze pole structure changes
        pole_code = f'''
        <|
            "poles1" -> Cases[{expr1}, Power[x_, n_] /; n < 0 :> x, Infinity],
            "poles2" -> Cases[{expr2}, Power[x_, n_] /; n < 0 :> x, Infinity]
        |>
        '''
        pole_result = await self.session.evaluate(wlexpr(pole_code))

        poles1 = pole_result.get("poles1", []) if isinstance(pole_result, dict) else []
        poles2 = pole_result.get("poles2", []) if isinstance(pole_result, dict) else []
        pole_changes = []
        if poles1 != poles2:
            pole_changes = [f"Poles changed: {poles1} -> {poles2}"]

        # Build summary
        if equivalent:
            summary = "Expressions are equivalent."
            if canon1 != canon2:
                summary += f" Canonical forms differ in representation."
        else:
            summary = f"Expressions differ. Difference: {diff_expr}"
            if pole_changes:
                summary += f" {pole_changes[0]}"

        return SemanticDiff(
            expr1=expr1,
            expr2=expr2,
            canonical_form1=canon1,
            canonical_form2=canon2,
            are_equivalent=equivalent,
            difference_type=diff_type,
            pole_changes=pole_changes,
            summary=summary
        )

    async def infer_domain(self, expression: str, variables: List[str]) -> DomainSpec:
        """Infer the validity domain of an expression"""
        vars_str = ", ".join(variables)

        domain_code = f'''
        Module[{{dom, singularities, branchPts}},
            dom = Quiet[FunctionDomain[{expression}, {{{vars_str}}}]];
            singularities = Quiet[FunctionSingularities[{expression}, {{{vars_str}}}]];
            branchPts = Quiet[Cases[{expression},
                Power[_, Rational[_, n_]] /; n > 1 :> "branch point", Infinity]];
            <|
                "domain" -> ToString[InputForm[dom]],
                "singularities" -> ToString[InputForm[singularities]],
                "hasBranchPoints" -> (Length[branchPts] > 0)
            |>
        ]
        '''

        result = await self.session.evaluate(wlexpr(domain_code))

        if isinstance(result, dict):
            domain_str = result.get("domain", "True")
            singularities = result.get("singularities", "False")
            has_branches = result.get("hasBranchPoints", False)

            return DomainSpec(
                name=f"Domain_{variables[0] if variables else 'x'}",
                base_set="Reals",
                constraints=[domain_str] if domain_str != "True" else [],
                excluded_points=[singularities] if singularities != "False" else [],
                branch_cuts=["Principal branch"] if has_branches else []
            )

        return DomainSpec(
            name="Unknown",
            base_set="Reals",
            constraints=[],
            excluded_points=[],
            branch_cuts=[]
        )

    async def _format_result(self, result: Any) -> str:
        """Format result for display"""
        if result is None:
            return "Null"

        if isinstance(result, (list, tuple)):
            if len(result) > 0 and isinstance(result[0], (list, tuple)):
                return self._format_matrix(result)
            elif len(result) > 20:
                preview = ", ".join(str(x) for x in result[:10])
                end = ", ".join(str(x) for x in result[-3:])
                return f"[{preview}, ..., {end}] (length: {len(result)})"

        return str(result)

    def _format_matrix(self, matrix: List[List]) -> str:
        """Format matrix as aligned table"""
        if not matrix or not matrix[0]:
            return str(matrix)

        try:
            num_cols = len(matrix[0])
            widths = []
            for i in range(num_cols):
                max_w = max(len(str(row[i])) for row in matrix if i < len(row))
                widths.append(max_w)

            lines = []
            for row in matrix:
                formatted_row = " | ".join(
                    str(val).rjust(widths[i])
                    for i, val in enumerate(row)
                )
                lines.append(f"| {formatted_row} |")

            return "\n".join(lines)
        except Exception:
            return str(matrix)

    async def _get_type(self, result: Any) -> str:
        """Get the Wolfram type of a result"""
        try:
            head = await self.session.evaluate(wl.Head(result))
            return str(head)
        except Exception:
            return type(result).__name__


# =============================================================================
# OUTPUT PROCESSOR
# =============================================================================

class OutputProcessor:
    """Handles conversion of Wolfram outputs to various formats"""

    def __init__(self, kernel_session: WolframLanguageAsyncSession):
        self.session = kernel_session

    async def export_graphics_base64(
        self,
        graphics_expr: str,
        format: str = "PNG",
        width: int = 600
    ) -> Tuple[Optional[str], str]:
        """Export graphics to base64-encoded image"""
        try:
            code = f'''
            Module[{{img, b64}},
                img = Rasterize[{graphics_expr}, ImageSize -> {width}];
                b64 = ExportString[img, "{format}"];
                BaseEncode[b64]
            ]
            '''
            result = await self.session.evaluate(wlexpr(code))

            mime_types = {
                "PNG": "image/png",
                "JPEG": "image/jpeg",
                "GIF": "image/gif"
            }

            return str(result), mime_types.get(format, "image/png")
        except Exception as e:
            return None, f"Error: {e}"

    async def get_latex(self, expr: str) -> Optional[str]:
        """Get LaTeX representation of expression"""
        try:
            result = await self.session.evaluate(wlexpr(f'ToString[TeXForm[{expr}]]'))
            return str(result)
        except Exception:
            return None


# =============================================================================
# GLOBAL STATE
# =============================================================================

session_manager: Optional[WolframSessionManager] = None


async def get_session_manager() -> WolframSessionManager:
    """Get or create the session manager"""
    global session_manager
    if session_manager is None:
        session_manager = WolframSessionManager()
    return session_manager


# =============================================================================
# MCP TOOLS - CORE EVALUATION
# =============================================================================

@mcp.tool()
async def wolfram_eval(
    code: str,
    timeout: int = 30,
    return_latex: bool = True,
    session_id: Optional[str] = None
) -> str:
    """
    Evaluate Wolfram Language code with persistent session state.

    All definitions, variables, and functions persist between calls.
    This is the primary tool for all Wolfram computations.

    Args:
        code: Wolfram Language code to evaluate
        timeout: Maximum execution time in seconds (default 30, max 600)
        return_latex: Include LaTeX representation for symbolic results
        session_id: Optional session identifier for multiple independent sessions

    Examples:
        - "2 + 2"
        - "Integrate[Sin[x]^2, x]"
        - "f[x_] := x^2; f[5]"
        - "DSolve[y'[x] == y[x], y[x], x]"
    """
    mgr = await get_session_manager()
    state = await mgr.get_or_create_session(session_id)
    controller = KernelController(state)

    result = await controller.evaluate(code, timeout=timeout, return_timing=True)

    state.history.append({
        "input": code,
        "output": result.formatted_result,
        "success": result.success,
        "timestamp": datetime.now().isoformat()
    })

    response_parts = [f"Result: {result.formatted_result}"]

    if result.latex and return_latex:
        response_parts.append(f"\nLaTeX: ${result.latex}$")

    if result.timing:
        response_parts.append(f"\nTiming: {result.timing:.4f}s")

    if result.aborted:
        response_parts.append("\n\n[Computation was aborted - timeout or explicit $Aborted]")

    if result.error:
        response_parts.append(f"\nError: {result.error}")

    return "".join(response_parts)


# =============================================================================
# MCP TOOLS - PROOF-CARRYING COMPUTATION
# =============================================================================

@mcp.tool()
async def wolfram_eval_proven(
    code: str,
    assumptions: Optional[str] = None,
    generate_conditions: bool = True,
    run_numeric_checks: bool = True,
    variables: Optional[str] = None,
    timeout: int = 60,
    session_id: Optional[str] = None
) -> str:
    """
    Evaluate with full proof-carrying computation: assumption tracking,
    condition extraction, and numeric validation.

    Args:
        code: Wolfram Language code to evaluate
        assumptions: Assumptions to use (e.g., "x > 0 && y > 0")
        generate_conditions: Extract conditions from ConditionalExpression
        run_numeric_checks: Run numeric spot-checks to validate the result
        variables: Variables for numeric checking (e.g., "x, y")
        timeout: Maximum execution time in seconds
        session_id: Optional session identifier

    Returns:
        JSON with result, assumptions used, conditions generated, and validation status

    Example:
        code="Integrate[1/Sqrt[x], {x, 0, 1}]", assumptions="True"
        -> Returns result with condition "Re[x] >= 0" extracted
    """
    mgr = await get_session_manager()
    state = await mgr.get_or_create_session(session_id)
    controller = KernelController(state)

    result = await controller.evaluate_with_provenance(
        code,
        assumptions=assumptions,
        generate_conditions=generate_conditions,
        timeout=timeout
    )

    # Run numeric checks if requested
    if run_numeric_checks and variables and result.result is not None:
        var_list = [v.strip() for v in variables.split(",")]
        # Compare original expression to result
        passed, check_results = await controller.numeric_spot_check(
            code,
            result.expression,
            var_list,
            assumptions,
            num_points=5
        )
        result.numeric_checks_passed = sum(1 for r in check_results if r["passed"])
        result.numeric_checks_total = len(check_results)

    return json.dumps(result.to_dict(), indent=2, default=str)


@mcp.tool()
async def wolfram_typed_equality(
    lhs: str,
    rhs: str,
    equality_type: str = "exact",
    domain: Optional[str] = None,
    order: Optional[str] = None,
    verify_numerically: bool = True,
    variables: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """
    Assert and verify a typed equality between expressions.

    Args:
        lhs: Left-hand side expression
        rhs: Right-hand side expression
        equality_type: Type of equality:
            - "exact": Exact mathematical equality
            - "on_domain": Equal on specified domain
            - "up_to_order": Equal up to specified order (e.g., for series)
            - "modulo_total_derivatives": Equal up to total derivatives
            - "modulo_gauge": Equal up to gauge transformations
            - "numeric": Numerically equal within tolerance
        domain: Domain specification (required for "on_domain")
        order: Order specification (required for "up_to_order")
        verify_numerically: Run numeric spot-checks
        variables: Variables for numeric checking

    Returns:
        JSON with verification status and details

    Example:
        lhs="Sin[x]^2 + Cos[x]^2", rhs="1", equality_type="exact"
        -> Verified as exact equality
    """
    mgr = await get_session_manager()
    state = await mgr.get_or_create_session(session_id)
    controller = KernelController(state)

    # Map string to enum
    type_map = {
        "exact": EqualityType.EXACT,
        "on_domain": EqualityType.ON_DOMAIN,
        "up_to_order": EqualityType.UP_TO_ORDER,
        "modulo_total_derivatives": EqualityType.MODULO_TOTAL_DERIVATIVES,
        "modulo_gauge": EqualityType.MODULO_GAUGE,
        "numeric": EqualityType.NUMERIC_APPROX,
        "conditional": EqualityType.CONDITIONAL
    }

    eq_type = type_map.get(equality_type, EqualityType.EXACT)

    # Symbolic verification
    if eq_type == EqualityType.EXACT:
        verify_code = f'PossibleZeroQ[Simplify[{lhs} - {rhs}]]'
    elif eq_type == EqualityType.ON_DOMAIN:
        verify_code = f'Reduce[{lhs} == {rhs}, Assumptions -> {domain}]'
    elif eq_type == EqualityType.UP_TO_ORDER:
        verify_code = f'PossibleZeroQ[Normal[Series[{lhs} - {rhs}, {order}]]]'
    else:
        verify_code = f'PossibleZeroQ[Simplify[{lhs} - {rhs}]]'

    result = await controller.evaluate(verify_code)
    symbolic_verified = result.result == True or str(result.result) == "True"

    # Numeric verification
    numeric_verified = True
    check_results = []
    if verify_numerically and variables:
        var_list = [v.strip() for v in variables.split(",")]
        numeric_verified, check_results = await controller.numeric_spot_check(
            lhs, rhs, var_list, domain, num_points=10
        )

    # Build typed equality object
    typed_eq = TypedEquality(
        lhs=lhs,
        rhs=rhs,
        equality_type=eq_type,
        domain=domain,
        order=order
    )

    response = {
        "equality": typed_eq.to_dict(),
        "symbolic_verified": symbolic_verified,
        "numeric_verified": numeric_verified,
        "numeric_checks": check_results if check_results else None,
        "overall_verified": symbolic_verified and numeric_verified
    }

    return json.dumps(response, indent=2, default=str)


@mcp.tool()
async def wolfram_semantic_diff(
    expr1: str,
    expr2: str,
    canonicalization: str = "Together",
    session_id: Optional[str] = None
) -> str:
    """
    Compute semantic difference between two expressions.

    Shows what mathematically changed between expressions, after
    canonicalization. Reports added/removed terms, pole structure
    changes, degree changes, and symmetry changes.

    Args:
        expr1: First expression
        expr2: Second expression
        canonicalization: Canonicalization method:
            - "Together": Combine over common denominator
            - "Cancel": Cancel common factors
            - "Factor": Factor polynomials
            - "Simplify": Full simplification
            - "None": No canonicalization

    Returns:
        JSON with semantic diff analysis

    Example:
        expr1="(x^2 - 1)/(x - 1)", expr2="x + 1"
        -> Reports they are equivalent after cancellation
    """
    mgr = await get_session_manager()
    state = await mgr.get_or_create_session(session_id)
    controller = KernelController(state)

    diff = await controller.compute_semantic_diff(expr1, expr2, canonicalization)

    return json.dumps(diff.to_dict(), indent=2)


@mcp.tool()
async def wolfram_infer_domain(
    expression: str,
    variables: str,
    session_id: Optional[str] = None
) -> str:
    """
    Infer the validity domain of an expression.

    Detects where the expression is defined, singular points,
    and branch cuts for complex functions.

    Args:
        expression: The expression to analyze
        variables: Comma-separated list of variables (e.g., "x, y")

    Returns:
        JSON with domain specification

    Example:
        expression="Log[x] + 1/x", variables="x"
        -> Domain: x > 0, excluded: x = 0
    """
    mgr = await get_session_manager()
    state = await mgr.get_or_create_session(session_id)
    controller = KernelController(state)

    var_list = [v.strip() for v in variables.split(",")]
    domain = await controller.infer_domain(expression, var_list)

    return json.dumps(domain.to_dict(), indent=2)


@mcp.tool()
async def wolfram_numeric_validate(
    expr1: str,
    expr2: str,
    variables: str,
    domain_constraints: Optional[str] = None,
    num_points: int = 20,
    tolerance: float = 1e-10,
    session_id: Optional[str] = None
) -> str:
    """
    Numerically validate that two expressions are equal.

    Samples random points in the domain and compares values.
    Essential for catching symbolic manipulation errors.

    Args:
        expr1: First expression
        expr2: Second expression
        variables: Comma-separated variables (e.g., "x, y")
        domain_constraints: Constraints on variables (e.g., "x > 0 && y > 0")
        num_points: Number of test points
        tolerance: Numerical tolerance for equality

    Returns:
        JSON with validation results and any failing points

    Example:
        expr1="Sin[2x]", expr2="2 Sin[x] Cos[x]", variables="x"
        -> All 20 checks passed
    """
    mgr = await get_session_manager()
    state = await mgr.get_or_create_session(session_id)
    controller = KernelController(state)

    var_list = [v.strip() for v in variables.split(",")]
    passed, results = await controller.numeric_spot_check(
        expr1, expr2, var_list, domain_constraints, num_points, tolerance
    )

    response = {
        "overall_passed": passed,
        "checks_passed": sum(1 for r in results if r["passed"]),
        "checks_total": len(results),
        "failing_points": [r for r in results if not r["passed"]],
        "sample_checks": results[:5]  # Show first 5
    }

    return json.dumps(response, indent=2)


# =============================================================================
# MCP TOOLS - OBLIGATION ENGINE
# =============================================================================

@mcp.tool()
async def wolfram_register_obligation(
    name: str,
    description: str,
    test_type: str,
    test_expression: str,
    expected: Optional[str] = None,
    tolerance: float = 1e-10,
    session_id: Optional[str] = None
) -> str:
    """
    Register a test obligation that must pass before a derivation step is accepted.

    Args:
        name: Short name for the obligation
        description: What this test verifies
        test_type: Type of test:
            - "identity": Test that expression equals expected
            - "zero": Test that expression is zero
            - "positive": Test that expression is positive
            - "limit": Test a limit value
            - "dimension": Dimensional analysis check
            - "symmetry": Symmetry property check
            - "numeric": Numeric comparison
        test_expression: The expression to test
        expected: Expected value (for identity/limit tests)
        tolerance: Tolerance for numeric comparisons

    Returns:
        Obligation ID for later checking

    Example:
        name="Ward identity", test_type="zero",
        test_expression="D[J[mu], x[mu]] - source"
        -> Registers obligation, returns ID
    """
    mgr = await get_session_manager()
    state = await mgr.get_or_create_session(session_id)

    obligation_id = str(uuid.uuid4())[:8]
    obligation = Obligation(
        id=obligation_id,
        name=name,
        description=description,
        test_type=test_type,
        test_expression=test_expression,
        expected=expected,
        tolerance=tolerance
    )

    state.obligations[obligation_id] = obligation

    return json.dumps({
        "obligation_id": obligation_id,
        "name": name,
        "status": "registered",
        "message": f"Obligation '{name}' registered. Run wolfram_check_obligations to verify."
    }, indent=2)


@mcp.tool()
async def wolfram_check_obligations(
    obligation_ids: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """
    Check registered obligations.

    Args:
        obligation_ids: Comma-separated IDs to check (all if not specified)
        session_id: Session identifier

    Returns:
        JSON with results for each obligation

    Example:
        -> Checks all pending obligations, returns pass/fail for each
    """
    mgr = await get_session_manager()
    state = await mgr.get_or_create_session(session_id)
    controller = KernelController(state)

    if obligation_ids:
        ids_to_check = [id.strip() for id in obligation_ids.split(",")]
    else:
        ids_to_check = list(state.obligations.keys())

    results = []

    for ob_id in ids_to_check:
        if ob_id not in state.obligations:
            results.append({"id": ob_id, "error": "Obligation not found"})
            continue

        ob = state.obligations[ob_id]

        # Run the test based on type
        if ob.test_type == "zero":
            check_code = f'PossibleZeroQ[Simplify[{ob.test_expression}]]'
        elif ob.test_type == "identity":
            check_code = f'PossibleZeroQ[Simplify[{ob.test_expression} - ({ob.expected})]]'
        elif ob.test_type == "positive":
            check_code = f'Simplify[{ob.test_expression} > 0]'
        elif ob.test_type == "numeric":
            check_code = f'Abs[N[{ob.test_expression}] - N[{ob.expected}]] < {ob.tolerance}'
        else:
            check_code = f'{ob.test_expression}'

        result = await controller.evaluate(check_code)
        passed = result.result == True or str(result.result) == "True"

        ob.status = ObligationStatus.PASSED if passed else ObligationStatus.FAILED
        ob.result = str(result.result)
        ob.resolved_at = datetime.now()

        results.append({
            "id": ob_id,
            "name": ob.name,
            "status": ob.status.value,
            "result": ob.result,
            "test_expression": ob.test_expression
        })

    summary = {
        "total": len(results),
        "passed": sum(1 for r in results if r.get("status") == "passed"),
        "failed": sum(1 for r in results if r.get("status") == "failed"),
        "results": results
    }

    return json.dumps(summary, indent=2)


@mcp.tool()
async def wolfram_list_obligations(session_id: Optional[str] = None) -> str:
    """
    List all registered obligations and their status.

    Returns:
        JSON with all obligations

    Example:
        -> Lists all obligations with names, descriptions, and status
    """
    mgr = await get_session_manager()
    state = await mgr.get_or_create_session(session_id)

    obligations = [ob.to_dict() for ob in state.obligations.values()]

    return json.dumps({
        "total": len(obligations),
        "pending": sum(1 for o in state.obligations.values() if o.status == ObligationStatus.PENDING),
        "passed": sum(1 for o in state.obligations.values() if o.status == ObligationStatus.PASSED),
        "failed": sum(1 for o in state.obligations.values() if o.status == ObligationStatus.FAILED),
        "obligations": obligations
    }, indent=2)


# =============================================================================
# MCP TOOLS - MATH CI / TEST SUITES
# =============================================================================

@mcp.tool()
async def wolfram_create_test_suite(
    name: str,
    description: str,
    session_id: Optional[str] = None
) -> str:
    """
    Create a new test suite for mathematical derivations.

    Args:
        name: Name of the test suite
        description: What this suite tests

    Returns:
        Suite ID for adding tests

    Example:
        name="QED Ward Identities", description="Tests for gauge invariance"
    """
    mgr = await get_session_manager()
    state = await mgr.get_or_create_session(session_id)

    suite = TestSuite(name=name, description=description)
    state.test_suites[name] = suite

    return json.dumps({
        "suite_name": name,
        "status": "created",
        "message": f"Test suite '{name}' created. Use wolfram_add_test to add tests."
    }, indent=2)


@mcp.tool()
async def wolfram_add_test(
    suite_name: str,
    test_name: str,
    category: str,
    expression: str,
    expected: str,
    variables: Optional[str] = None,
    tolerance: float = 1e-10,
    session_id: Optional[str] = None
) -> str:
    """
    Add a test case to a test suite.

    Args:
        suite_name: Name of the suite to add to
        test_name: Name of this test
        category: Test category:
            - "dimension": Dimensional analysis
            - "limit": Asymptotic limit check
            - "symmetry": Symmetry verification
            - "identity": Mathematical identity
            - "numeric": Numeric value check
            - "consistency": Self-consistency check
        expression: Expression to test
        expected: Expected result
        variables: Variables involved (comma-separated)
        tolerance: Numeric tolerance

    Returns:
        Test ID

    Example:
        suite_name="QED", test_name="photon_propagator_limit",
        category="limit", expression="Limit[prop, k -> Infinity]",
        expected="0"
    """
    mgr = await get_session_manager()
    state = await mgr.get_or_create_session(session_id)

    if suite_name not in state.test_suites:
        return json.dumps({"error": f"Suite '{suite_name}' not found"})

    test_id = str(uuid.uuid4())[:8]
    test = TestCase(
        id=test_id,
        name=test_name,
        category=category,
        expression=expression,
        expected=expected,
        variables=[v.strip() for v in variables.split(",")] if variables else [],
        tolerance=tolerance
    )

    state.test_suites[suite_name].tests.append(test)

    return json.dumps({
        "test_id": test_id,
        "suite": suite_name,
        "name": test_name,
        "status": "added"
    }, indent=2)


@mcp.tool()
async def wolfram_run_test_suite(
    suite_name: str,
    session_id: Optional[str] = None
) -> str:
    """
    Run all tests in a test suite.

    Args:
        suite_name: Name of the suite to run

    Returns:
        JSON with results for each test

    Example:
        -> Runs all tests, reports pass/fail with details
    """
    mgr = await get_session_manager()
    state = await mgr.get_or_create_session(session_id)
    controller = KernelController(state)

    if suite_name not in state.test_suites:
        return json.dumps({"error": f"Suite '{suite_name}' not found"})

    suite = state.test_suites[suite_name]
    results = []

    for test in suite.tests:
        # Build check based on category
        if test.category in ["identity", "dimension", "consistency"]:
            check_code = f'PossibleZeroQ[Simplify[{test.expression} - ({test.expected})]]'
        elif test.category == "limit":
            check_code = f'Simplify[{test.expression}] === {test.expected}'
        elif test.category == "symmetry":
            check_code = f'Simplify[{test.expression}] === {test.expected}'
        elif test.category == "numeric":
            check_code = f'Abs[N[{test.expression}] - N[{test.expected}]] < {test.tolerance}'
        else:
            check_code = f'Simplify[{test.expression}] === {test.expected}'

        result = await controller.evaluate(check_code)
        passed = result.result == True or str(result.result) == "True"

        status = ObligationStatus.PASSED if passed else ObligationStatus.FAILED
        suite.results[test.id] = status

        results.append({
            "test_id": test.id,
            "name": test.name,
            "category": test.category,
            "status": status.value,
            "expression": test.expression,
            "expected": test.expected,
            "actual": result.formatted_result
        })

    summary = {
        "suite": suite_name,
        "total": len(results),
        "passed": sum(1 for r in results if r["status"] == "passed"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "results": results
    }

    return json.dumps(summary, indent=2)


# =============================================================================
# MCP TOOLS - CANONICALIZATION
# =============================================================================

@mcp.tool()
async def wolfram_canonicalize(
    expression: str,
    method: str = "auto",
    session_id: Optional[str] = None
) -> str:
    """
    Canonicalize an expression to a standard form.

    Args:
        expression: Expression to canonicalize
        method: Canonicalization method:
            - "auto": Choose best method automatically
            - "rational": Rational function canonicalization (Together + Cancel)
            - "polynomial": Polynomial canonical form (Factor or Expand)
            - "trig": Trigonometric canonical form
            - "series": Series normal form
            - "full": Full simplification

    Returns:
        JSON with canonical form and hash

    Example:
        expression="(x^2 - 1)/(x - 1)", method="rational"
        -> Returns canonical form "x + 1" with hash
    """
    mgr = await get_session_manager()
    state = await mgr.get_or_create_session(session_id)
    controller = KernelController(state)

    if method == "auto" or method == "rational":
        canon_code = f'Cancel[Together[{expression}]]'
    elif method == "polynomial":
        canon_code = f'Factor[{expression}]'
    elif method == "trig":
        canon_code = f'TrigReduce[{expression}]'
    elif method == "series":
        canon_code = f'Normal[{expression}]'
    elif method == "full":
        canon_code = f'FullSimplify[{expression}]'
    else:
        canon_code = f'Simplify[{expression}]'

    result = await controller.evaluate(canon_code)

    # Compute hash of canonical form
    hash_input = f"{result.formatted_result}"
    expr_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    # Get LaTeX
    latex = None
    try:
        latex_result = await controller.evaluate(f'ToString[TeXForm[{result.formatted_result}]]')
        latex = latex_result.formatted_result
    except Exception:
        pass

    # Store hash for deduplication
    state.expression_hashes[expr_hash] = result.formatted_result

    response = {
        "original": expression,
        "canonical": result.formatted_result,
        "method": method,
        "hash": expr_hash,
        "latex": latex
    }

    return json.dumps(response, indent=2)


@mcp.tool()
async def wolfram_expression_hash(
    expression: str,
    with_assumptions: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """
    Compute a semantic hash of an expression for caching and deduplication.

    Two expressions that are mathematically equivalent should have the same hash.

    Args:
        expression: Expression to hash
        with_assumptions: Assumptions to include in hash context

    Returns:
        JSON with hash and canonical form

    Example:
        expression="x + 1", -> hash "a1b2c3d4..."
        expression="1 + x", -> same hash (equivalent)
    """
    mgr = await get_session_manager()
    state = await mgr.get_or_create_session(session_id)
    controller = KernelController(state)

    # Get canonical form
    canon_result = await controller.evaluate(f'FullSimplify[{expression}]')
    canon_form = canon_result.formatted_result

    # Include assumptions in hash if provided
    hash_input = f"{canon_form}|{with_assumptions or 'True'}"
    expr_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    # Check if we've seen this before
    seen_before = expr_hash in state.expression_hashes

    state.expression_hashes[expr_hash] = canon_form

    return json.dumps({
        "expression": expression,
        "canonical_form": canon_form,
        "hash": expr_hash,
        "assumptions": with_assumptions,
        "previously_seen": seen_before
    }, indent=2)


# =============================================================================
# MCP TOOLS - VISUALIZATION (kept from original)
# =============================================================================

@mcp.tool()
async def wolfram_plot(
    expression: str,
    variable: str,
    range_min: float,
    range_max: float,
    plot_type: str = "Plot",
    options: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """Create a 2D plot and return it as a base64-encoded image."""
    mgr = await get_session_manager()
    state = await mgr.get_or_create_session(session_id)
    processor = OutputProcessor(state.kernel_session)

    range_spec = f"{{{variable}, {range_min}, {range_max}}}"
    opts = f", {options}" if options else ""
    plot_expr = f"{plot_type}[{expression}, {range_spec}{opts}]"

    base64_data, mime_type = await processor.export_graphics_base64(plot_expr, width=800)

    if base64_data and not mime_type.startswith("Error"):
        return f"""Plot generated successfully.

Expression: {plot_expr}

Base64 PNG (first 100 chars): {base64_data[:100]}...

Full data URI for embedding:
data:{mime_type};base64,{base64_data}
"""
    else:
        return f"Failed to generate plot: {mime_type}"


@mcp.tool()
async def wolfram_plot3d(
    expression: str,
    var1: str,
    range1_min: float,
    range1_max: float,
    var2: str,
    range2_min: float,
    range2_max: float,
    plot_type: str = "Plot3D",
    options: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """Create a 3D plot and return it as a base64-encoded image."""
    mgr = await get_session_manager()
    state = await mgr.get_or_create_session(session_id)
    processor = OutputProcessor(state.kernel_session)

    range1 = f"{{{var1}, {range1_min}, {range1_max}}}"
    range2 = f"{{{var2}, {range2_min}, {range2_max}}}"
    opts = f", {options}" if options else ""

    plot_expr = f"{plot_type}[{expression}, {range1}, {range2}{opts}]"

    base64_data, mime_type = await processor.export_graphics_base64(plot_expr, width=800)

    if base64_data and not mime_type.startswith("Error"):
        return f"""3D Plot generated.

Expression: {plot_expr}

data:{mime_type};base64,{base64_data}
"""
    else:
        return f"Failed to generate 3D plot: {mime_type}"


# =============================================================================
# MCP TOOLS - ADVANCED MATHEMATICS (kept from original)
# =============================================================================

@mcp.tool()
async def wolfram_solve(
    equations: str,
    variables: str,
    method: str = "Solve",
    domain: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """Solve equations (algebraic, transcendental, differential)."""
    mgr = await get_session_manager()
    state = await mgr.get_or_create_session(session_id)
    controller = KernelController(state)

    if domain:
        code = f"{method}[{equations}, {variables}, {domain}]"
    else:
        code = f"{method}[{equations}, {variables}]"

    result = await controller.evaluate(code, timeout=60)

    response = f"Method: {method}\nEquations: {equations}\nVariables: {variables}\n\n"
    response += f"Solution: {result.formatted_result}"

    if result.latex:
        response += f"\n\nLaTeX: ${result.latex}$"

    return response


@mcp.tool()
async def wolfram_calculus(
    operation: str,
    expression: str,
    variable: str,
    bounds: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """Perform calculus operations."""
    mgr = await get_session_manager()
    state = await mgr.get_or_create_session(session_id)
    controller = KernelController(state)

    if operation == "Integrate":
        if bounds:
            code = f"Integrate[{expression}, {{{variable}, {bounds}}}]"
        else:
            code = f"Integrate[{expression}, {variable}]"
    elif operation == "D":
        if bounds:
            code = f"D[{expression}, {{{variable}, {bounds}}}]"
        else:
            code = f"D[{expression}, {variable}]"
    elif operation == "Limit":
        direction = bounds or "0"
        code = f"Limit[{expression}, {variable} -> {direction}]"
    elif operation == "Series":
        if bounds:
            parts = bounds.split(",")
            point = parts[0].strip()
            order = parts[1].strip() if len(parts) > 1 else "5"
            code = f"Series[{expression}, {{{variable}, {point}, {order}}}]"
        else:
            code = f"Series[{expression}, {{{variable}, 0, 5}}]"
    elif operation in ["Sum", "Product"]:
        if bounds:
            code = f"{operation}[{expression}, {{{variable}, {bounds}}}]"
        else:
            code = f"{operation}[{expression}, {variable}]"
    elif operation == "NIntegrate":
        if bounds:
            code = f"NIntegrate[{expression}, {{{variable}, {bounds}}}]"
        else:
            return "NIntegrate requires bounds"
    else:
        code = f"{operation}[{expression}, {variable}]"

    result = await controller.evaluate(code, timeout=60)

    response = f"Calculus: {operation}\nExpression: {expression}\nVariable: {variable}"
    if bounds:
        response += f"\nBounds: {bounds}"
    response += f"\n\nResult: {result.formatted_result}"

    if result.latex:
        response += f"\n\nLaTeX: ${result.latex}$"

    return response


@mcp.tool()
async def wolfram_symbolic(
    operation: str,
    expression: str,
    assumptions: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """Perform symbolic manipulation."""
    mgr = await get_session_manager()
    state = await mgr.get_or_create_session(session_id)
    controller = KernelController(state)

    if assumptions:
        code = f"Assuming[{assumptions}, {operation}[{expression}]]"
    else:
        code = f"{operation}[{expression}]"

    result = await controller.evaluate(code)

    latex_result = await controller.evaluate(f"ToString[TeXForm[{code}]]")

    response = f"Symbolic: {operation}\n\nResult: {result.formatted_result}"
    if latex_result.success:
        response += f"\n\nLaTeX: ${latex_result.formatted_result}$"

    return response


@mcp.tool()
async def wolfram_linear_algebra(
    operation: str,
    matrix: str,
    extra_arg: Optional[str] = None,
    session_id: Optional[str] = None
) -> str:
    """Perform linear algebra operations."""
    mgr = await get_session_manager()
    state = await mgr.get_or_create_session(session_id)
    controller = KernelController(state)

    if extra_arg:
        code = f"{operation}[{matrix}, {extra_arg}]"
    else:
        code = f"{operation}[{matrix}]"

    result = await controller.evaluate(code)

    return f"Linear Algebra: {operation}\n\nResult:\n{result.formatted_result}"


# =============================================================================
# MCP TOOLS - UTILITIES
# =============================================================================

@mcp.tool()
async def wolfram_define(
    name: str,
    definition: str,
    session_id: Optional[str] = None
) -> str:
    """Define a function or variable that persists in the session."""
    mgr = await get_session_manager()
    state = await mgr.get_or_create_session(session_id)
    controller = KernelController(state)

    full_def = f"{name}{definition}"
    result = await controller.evaluate(full_def)

    if result.success:
        return f"Defined '{name}' successfully.\nResult: {result.formatted_result}"
    else:
        return f"Failed to define '{name}': {result.error or result.formatted_result}"


@mcp.tool()
async def wolfram_session_info(session_id: Optional[str] = None) -> str:
    """Get information about the current session state."""
    mgr = await get_session_manager()
    state = await mgr.get_or_create_session(session_id)
    controller = KernelController(state)

    symbols_result = await controller.evaluate('Names["Global`*"]')

    info = {
        "session_id": state.session_id,
        "created": state.created_at.isoformat(),
        "last_activity": state.last_activity.isoformat(),
        "defined_symbols": symbols_result.result if symbols_result.success else [],
        "history_length": len(state.history),
        "recent_inputs": [h["input"] for h in state.history[-5:]],
        "obligations_pending": sum(1 for o in state.obligations.values() if o.status == ObligationStatus.PENDING),
        "test_suites": list(state.test_suites.keys()),
        "expression_cache_size": len(state.expression_hashes)
    }

    return json.dumps(info, indent=2)


@mcp.tool()
async def wolfram_clear_session(
    session_id: Optional[str] = None,
    terminate: bool = False
) -> str:
    """Clear session state or terminate session entirely."""
    mgr = await get_session_manager()

    if terminate:
        success = await mgr.terminate_session(session_id or "default")
        return "Session terminated." if success else "Session not found."
    else:
        state = await mgr.get_or_create_session(session_id)
        controller = KernelController(state)
        await controller.evaluate('ClearAll["Global`*"]')
        state.history.clear()
        state.obligations.clear()
        state.test_suites.clear()
        state.expression_hashes.clear()
        return "Session cleared. All definitions, obligations, and test suites removed."


@mcp.tool()
async def wolfram_help(topic: str, session_id: Optional[str] = None) -> str:
    """Get documentation for a Wolfram Language function or topic."""
    mgr = await get_session_manager()
    state = await mgr.get_or_create_session(session_id)
    controller = KernelController(state)

    usage = await controller.evaluate(f'{topic}::usage')

    return f"""Documentation for {topic}:

{usage.formatted_result}

Full documentation: https://reference.wolfram.com/language/ref/{topic}.html
"""


@mcp.tool()
async def wolfram_test() -> str:
    """Test the connection to Wolfram Language/Mathematica."""
    mgr = await get_session_manager()
    state = await mgr.get_or_create_session()
    controller = KernelController(state)

    result = await controller.evaluate('''
    <|
        "Version" -> $Version,
        "VersionNumber" -> $VersionNumber,
        "SystemID" -> $SystemID,
        "Test" -> (2 + 2)
    |>
    ''')

    if result.success:
        return f"""Wolfram Language Connected Successfully

{result.formatted_result}

Session is persistent and ready for computations.
Kernel path: {WOLFRAM_KERNEL_PATH}

Proof-Carrying Computation features enabled:
- Assumption tracking
- Condition extraction
- Obligation engine
- Domain inference
- Semantic diff
- Math CI test suites
- Numeric validation
"""
    else:
        return f"Connection failed: {result.error}\n\nKernel path: {WOLFRAM_KERNEL_PATH}"


# =============================================================================
# SHUTDOWN HANDLING
# =============================================================================

def _cleanup_sessions_sync():
    """Synchronous cleanup for atexit - terminates all Wolfram kernel sessions."""
    global session_manager
    if session_manager is not None:
        # Get or create event loop for cleanup
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule cleanup in running loop
                asyncio.ensure_future(session_manager.terminate_all())
            else:
                loop.run_until_complete(session_manager.terminate_all())
        except RuntimeError:
            # No event loop available, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(session_manager.terminate_all())
            finally:
                loop.close()


def _signal_handler(signum, frame):
    """Handle termination signals gracefully."""
    print(f"\nReceived signal {signum}, cleaning up Wolfram sessions...", file=sys.stderr)
    _cleanup_sessions_sync()
    sys.exit(0)


# Register cleanup handlers
atexit.register(_cleanup_sessions_sync)

# Register signal handlers for graceful shutdown
if sys.platform != "win32":
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGHUP, _signal_handler)
# SIGINT is typically handled by the event loop, but we add a fallback
signal.signal(signal.SIGINT, _signal_handler)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    mcp.run()
