import time
import platform
from dataclasses import dataclass
from typing import Any, List, Dict, Tuple

import streamlit as st
import torch

st.set_page_config(page_title="AI Compiler Lab", page_icon="⚙️", layout="wide")
st.title("AI Compiler Lab")
st.caption("SLA-3 • Compiler Design Challenges in Programming Languages for AI Applications")
st.write(
    "An educational prototype combining a custom expression IR and optimization passes "
    "with a PyTorch tensor workload. Select a ready-made scenario and run the demonstration."
)

# --------------------------- Custom scalar IR ---------------------------

@dataclass
class IRNode:
    id: int
    op: str
    inputs: List[int]
    value: Any = None

class Graph:
    def __init__(self):
        self.nodes: List[IRNode] = []
        self.next_id = 1

    def add(self, op: str, inputs=None, value=None) -> int:
        node_id = self.next_id
        self.next_id += 1
        self.nodes.append(IRNode(node_id, op, list(inputs or []), value))
        return node_id

    def by_id(self) -> Dict[int, IRNode]:
        return {n.id: n for n in self.nodes}

    def render(self) -> str:
        lines = []
        for n in self.nodes:
            if n.op == "const":
                expr = repr(n.value)
            elif n.op == "input":
                expr = str(n.value)
            elif n.op in {"+", "-", "*", "/"}:
                expr = f"%{n.inputs[0]} {n.op} %{n.inputs[1]}"
            elif n.op == "neg":
                expr = f"-%{n.inputs[0]}"
            else:
                expr = f"{n.op}(" + ", ".join(f"%{i}" for i in n.inputs) + ")"
            lines.append(f"%{n.id} = {expr}")
        return "\n".join(lines) or "(empty IR)"

def make_demo_graph(name: str) -> Tuple[Graph, int, str]:
    g = Graph()
    if name == "Constant folding":
        x = g.add("input", value="x")
        c1 = g.add("const", value=4)
        c2 = g.add("const", value=6)
        c3 = g.add("+", [c1, c2])
        root = g.add("*", [x, c3])
        # Add intentionally unreachable node to demonstrate DCE.
        unused_a = g.add("const", value=99)
        unused_b = g.add("const", value=100)
        g.add("+", [unused_a, unused_b])
        return g, root, "y = x * (4 + 6)"
    if name == "Redundant operations":
        x = g.add("input", value="x")
        one = g.add("const", value=1)
        zero = g.add("const", value=0)
        mul = g.add("*", [x, one])
        root = g.add("+", [mul, zero])
        return g, root, "y = (x * 1) + 0"
    # Neural-network-style expression, represented at scalar IR level for optimization demo.
    x = g.add("input", value="x")
    two = g.add("const", value=2)
    one = g.add("const", value=1)
    mul = g.add("*", [x, two])
    root = g.add("+", [mul, one])
    return g, root, "y = (x * 2) + 1"

def optimize_graph(graph: Graph, root: int):
    old = graph.by_id()
    new = Graph()
    remap: Dict[int, int] = {}
    notes: List[str] = []

    def emit(op, inputs=None, value=None):
        return new.add(op, inputs, value)

    def visit(old_id):
        if old_id in remap:
            return remap[old_id]
        n = old[old_id]
        args = [visit(i) for i in n.inputs]
        if n.op in {"const", "input"}:
            result = emit(n.op, value=n.value)
            remap[old_id] = result
            return result

        current = new.by_id()
        if n.op in {"+", "-", "*", "/"} and len(args) == 2:
            a, b = current[args[0]], current[args[1]]
            # Constant folding
            if a.op == "const" and b.op == "const":
                try:
                    funcs = {
                        "+": lambda x, y: x + y,
                        "-": lambda x, y: x - y,
                        "*": lambda x, y: x * y,
                        "/": lambda x, y: x / y,
                    }
                    folded = funcs[n.op](a.value, b.value)
                    notes.append(f"Constant folding: {a.value} {n.op} {b.value} → {folded}")
                    result = emit("const", value=folded)
                    remap[old_id] = result
                    return result
                except ZeroDivisionError:
                    notes.append("Skipped constant folding because division by zero would occur.")
            # Algebraic identities / simple strength reduction
            if n.op == "*" and a.op == "const" and a.value == 1:
                notes.append("Algebraic simplification: 1 * x → x")
                remap[old_id] = args[1]
                return args[1]
            if n.op == "*" and b.op == "const" and b.value == 1:
                notes.append("Algebraic simplification: x * 1 → x")
                remap[old_id] = args[0]
                return args[0]
            if n.op == "*" and ((a.op == "const" and a.value == 0) or (b.op == "const" and b.value == 0)):
                notes.append("Algebraic simplification: x * 0 → 0")
                result = emit("const", value=0)
                remap[old_id] = result
                return result
            if n.op == "+" and a.op == "const" and a.value == 0:
                notes.append("Algebraic simplification: 0 + x → x")
                remap[old_id] = args[1]
                return args[1]
            if n.op == "+" and b.op == "const" and b.value == 0:
                notes.append("Algebraic simplification: x + 0 → x")
                remap[old_id] = args[0]
                return args[0]
        result = emit(n.op, args, n.value)
        remap[old_id] = result
        return result

    new_root = visit(root)

    # Dead-code elimination: only keep nodes reachable from output.
    new_by = new.by_id()
    live = set()
    def mark(node_id):
        if node_id in live:
            return
        live.add(node_id)
        for child in new_by[node_id].inputs:
            mark(child)
    mark(new_root)
    compact = Graph()
    id_remap = {}
    for n in new.nodes:
        if n.id in live:
            id_remap[n.id] = compact.add(n.op, [id_remap[i] for i in n.inputs], n.value)
    new_root = id_remap[new_root]
    if len(live) < len(new.nodes):
        notes.append(f"Dead-code elimination: removed {len(new.nodes) - len(live)} unreachable IR node(s).")
    return compact, new_root, notes

def eval_ir(graph: Graph, root: int, x_value: float) -> float:
    nodes = graph.by_id()
    cache = {}
    def ev(i):
        if i in cache:
            return cache[i]
        n = nodes[i]
        if n.op == "const":
            v = n.value
        elif n.op == "input":
            v = x_value if n.value == "x" else 0
        else:
            args = [ev(j) for j in n.inputs]
            if n.op == "+": v = args[0] + args[1]
            elif n.op == "-": v = args[0] - args[1]
            elif n.op == "*": v = args[0] * args[1]
            elif n.op == "/": v = args[0] / args[1]
            elif n.op == "neg": v = -args[0]
            else: raise ValueError(f"Unsupported IR operation: {n.op}")
        cache[i] = v
        return v
    return float(ev(root))

def count_live(graph: Graph, root: int) -> int:
    nodes = graph.by_id()
    live = set()
    def walk(i):
        if i in live: return
        live.add(i)
        for j in nodes[i].inputs: walk(j)
    walk(root)
    return len(live)

# --------------------------- Streamlit UI ---------------------------

with st.sidebar:
    st.header("Demonstration settings")
    scenario = st.selectbox(
        "Choose a predefined scenario",
        ["Constant folding", "Redundant operations", "Simple AI-style expression"],
    )
    x_value = st.number_input("Test value for x", value=2.0)
    st.caption("No code entry is required. Choose a scenario and run it.")

run_demo = st.button("▶ Run Demonstration", type="primary")

if run_demo or "demo_result" not in st.session_state:
    try:
        original, root, source = make_demo_graph(scenario)
        optimized, optimized_root, notes = optimize_graph(original, root)
        original_value = eval_ir(original, root, x_value)
        optimized_value = eval_ir(optimized, optimized_root, x_value)
        equivalent = abs(original_value - optimized_value) <= 1e-9 * max(1.0, abs(original_value))
        st.session_state["demo_result"] = {
            "scenario": scenario,
            "source": source,
            "original": original,
            "root": root,
            "optimized": optimized,
            "optimized_root": optimized_root,
            "notes": notes,
            "original_value": original_value,
            "optimized_value": optimized_value,
            "equivalent": equivalent,
            "x_value": x_value,
        }
    except Exception as exc:
        st.error(f"Demo failed: {exc}")

result = st.session_state.get("demo_result")
if result:
    st.subheader("1. Selected scenario")
    st.write(f"**{result['scenario']}** — `{result['source']}`")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Original custom IR")
        st.code(result["original"].render(), language="text")
        st.metric("Original IR node count", len(result["original"].nodes))
    with col2:
        st.markdown("#### Optimized custom IR")
        st.code(result["optimized"].render(), language="text")
        st.metric("Optimized IR node count", len(result["optimized"].nodes))

    st.subheader("2. Optimization log")
    if result["notes"]:
        for note in result["notes"]:
            st.write("• " + note)
    else:
        st.write("No optimization was applicable for this scenario.")

    st.subheader("3. Correctness check")
    a, b, c = st.columns(3)
    a.metric("Original output", f"{result['original_value']:.6g}")
    b.metric("Optimized output", f"{result['optimized_value']:.6g}")
    c.metric("Equivalent", "PASS" if result["equivalent"] else "FAIL")
    if result["equivalent"]:
        st.success("The original and optimized IR produce equivalent results for this test input.")
    else:
        st.error("Output mismatch detected. Do not report this run as successful.")

    before = len(result["original"].nodes)
    after = len(result["optimized"].nodes)
    reduction = (before - after) / before * 100 if before else 0
    st.subheader("4. Optimization summary")
    a, b, c = st.columns(3)
    a.metric("IR nodes before", before)
    b.metric("IR nodes after", after)
    c.metric("Node-count change", f"{reduction:.1f}%")

st.divider()
st.header("5. AI tensor workload demonstration")
st.write("This separate experiment executes a small neural-network-style tensor computation using PyTorch.")
tensor_size = st.select_slider("Tensor size", options=[8, 16, 32, 64, 128, 256], value=32)
torch.manual_seed(7)
device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(tensor_size, tensor_size, device=device)
w = torch.randn(tensor_size, tensor_size, device=device)
bias = torch.randn(tensor_size, device=device)

def tensor_model(a, weight, b):
    return torch.relu((a @ weight) + b)

def tensor_model_explicit(a, weight, b):
    product = torch.matmul(a, weight)
    shifted = torch.add(product, b)
    return torch.relu(shifted)

with torch.no_grad():
    y1 = tensor_model(x, w, bias)
    y2 = tensor_model_explicit(x, w, bias)
    max_diff = (y1 - y2).abs().max().item()
    tensor_pass = torch.allclose(y1, y2, rtol=1e-4, atol=1e-5)

a, b, c = st.columns(3)
a.metric("PyTorch version", torch.__version__)
b.metric("Device", device.upper())
c.metric("Tensor output shape", str(tuple(y1.shape)))
st.graphviz_chart("""
digraph G {
  rankdir=LR;
  x [label="Input tensor x"];
  w [label="Weight tensor w"];
  mm [label="Matrix multiplication"];
  bias [label="Bias"];
  add [label="Bias addition"];
  relu [label="ReLU"];
  y [label="Output tensor y"];
  x -> mm; w -> mm; mm -> add; bias -> add; add -> relu; relu -> y;
}
""")
st.write(f"Maximum absolute output difference: `{max_diff:.3e}`")
st.success("Tensor correctness: PASS" if tensor_pass else "Tensor correctness: FAIL")

st.subheader("6. Tensor shape challenge")
st.write("Matrix multiplication requires the inner dimensions to match. The valid demonstration uses compatible square matrices.")
shape_test = st.selectbox("Shape scenario", ["Compatible shapes", "Incompatible shapes"])
if shape_test == "Compatible shapes":
    st.code("A: (4, 3) @ B: (3, 2) → output (4, 2)", language="text")
    st.success("Valid: the inner dimensions 3 and 3 match.")
else:
    st.code("A: (4, 3) @ B: (5, 2)", language="text")
    st.error("Invalid: the inner dimensions 3 and 5 do not match, so matrix multiplication cannot be performed.")

st.subheader("7. Compiler challenges covered")
st.markdown("""
- **Implemented:** custom IR for a small expression language, constant folding, algebraic simplification, dead-code elimination, and numerical equivalence checks.
- **Demonstrated:** a real PyTorch tensor workload and a tensor-shape compatibility scenario.
- **Explained but not implemented:** dynamic-shape guards, Python graph breaks, automatic differentiation transformations, production operator fusion, and custom LLVM/CUDA backend code generation.
""")
st.caption(f"Platform: {platform.platform()} • Selected tensor device: {device}")
