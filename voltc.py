#!/usr/bin/env python3
import os, sys, io, shutil, subprocess, textwrap
import platform
import shlex
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union, Set

from ply.lex import lex
from ply.yacc import yacc
from llvmlite import ir, binding

# ============================================================
# Diagnostics
# ============================================================

@dataclass
class Source:
    path: str
    text: str
    lines: List[str]

    @staticmethod
    def from_path(path: str) -> "Source":
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        return Source(path=os.path.abspath(path), text=txt, lines=txt.splitlines())

    def line_col(self, lexpos: int) -> Tuple[int, int]:
        # compute (line, col) from absolute index
        line = self.text.count("\n", 0, lexpos) + 1
        bol = self.text.rfind("\n", 0, lexpos)
        if bol < 0: bol = -1
        col = lexpos - bol
        return line, col

@dataclass
class Diag:
    kind: str  # "error" | "note"
    msg: str
    src: Source
    lexpos: int

    def format(self) -> str:
        line, col = self.src.line_col(self.lexpos)
        caret = " " * (col - 1) + "^"
        code = self.src.lines[line - 1] if 1 <= line <= len(self.src.lines) else ""
        head = f"{self.src.path}:{line}:{col}: {self.kind}: {self.msg}"
        return f"{head}\n{code}\n{caret}"

class ErrorSink:
    def __init__(self) -> None:
        self.errors: List[Diag] = []
        self.notes: List[Diag] = []

    def error(self, msg: str, src: Source, lexpos: int):
        self.errors.append(Diag("error", msg, src, lexpos))

    def note(self, msg: str, src: Source, lexpos: int):
        self.notes.append(Diag("note", msg, src, lexpos))

    def ok(self) -> bool:
        return not self.errors

    def dump(self):
        for e in self.errors:
            print(e.format())
        for n in self.notes:
            print(n.format())

# ============================================================
# Lexer
# ============================================================

reserved = {
    "fn": "FN",
    "extern": "EXTERN",
    "return": "RETURN",
    "var": "VAR",
    "const": "CONST",
    "use": "USE",
    "namespace": "NAMESPACE",
    "true": "TRUE",
    "false": "FALSE",
    "as": "AS",

    # primitive types (keywords)
    "bool": "KW_BOOL",
    "str": "KW_STR",
    "charptr": "KW_CHARPTR",
    "void": "KW_VOID",

    # ints
    "i8": "KW_I8", "u8": "KW_U8",
    "i16": "KW_I16", "u16": "KW_U16",
    "i32": "KW_I32", "u32": "KW_U32",
    "i64": "KW_I64", "u64": "KW_U64",
    "i128": "KW_I128", "u128": "KW_U128",

    # floats
    "f16": "KW_F16", "f32": "KW_F32", "f64": "KW_F64", "f128": "KW_F128",
}

tokens = (
    # literals & ids
    "NUMBER", "NAME", "STRING",

    # punctuation
    "LPAREN", "RPAREN", "LBRACE", "RBRACE",
    "COMMA", "COLON", "SEMICOLON", "ASSIGN",
    "DCOLON", "ARROW",

    # operators
    "PLUS", "MINUS", "TIMES", "DIVIDE",
    "ANDAND", "OROR", "NOT",
    "EQ", "NE", "LT", "LE", "GT", "GE",

    # keywords
    "FN", "EXTERN", "RETURN", "VAR", "CONST", "USE", "NAMESPACE",
    "TRUE", "FALSE", "AS",

    # type keywords
    "KW_BOOL", "KW_STR", "KW_CHARPTR", "KW_VOID",
    "KW_I8", "KW_U8", "KW_I16", "KW_U16", "KW_I32", "KW_U32",
    "KW_I64", "KW_U64", "KW_I128", "KW_U128",
    "KW_F16", "KW_F32", "KW_F64", "KW_F128",
)

t_ignore = " \t\r"
t_LPAREN   = r"\("
t_RPAREN   = r"\)"
t_LBRACE   = r"\{"
t_RBRACE   = r"\}"
t_COMMA    = r","
t_COLON    = r":"
t_SEMICOLON= r";"
t_ASSIGN   = r"="
t_DCOLON   = r"::"
t_ARROW    = r"->"

t_PLUS     = r"\+"
t_MINUS    = r"-"
t_TIMES    = r"\*"
t_DIVIDE   = r"/"
t_ANDAND   = r"&&"
t_OROR     = r"\|\|"
t_NOT      = r"!"
t_EQ       = r"=="
t_NE       = r"!="
t_LE       = r"<="
t_LT       = r"<"
t_GE       = r">="
t_GT       = r">"

def t_STRING(t):
    r'"([^"\\]|\\.)*"'
    t.value = bytes(t.value[1:-1], "utf-8").decode("unicode_escape")
    return t

def t_NUMBER(t):
    r'\d+'
    t.value = int(t.value)
    return t

def t_NAME(t):
    r'[A-Za-z_][A-Za-z0-9_]*'
    t.type = reserved.get(t.value, "NAME")
    return t

def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

def t_comment(t):
    r'//[^\n]*'
    pass

def t_error(t):
    # lexer error is handled by parser with source/pos; we print minimal here
    print(f"lex error at {t.value[0]!r}")
    t.lexer.skip(1)

# ============================================================
# AST
# ============================================================

@dataclass
class Node:
    kind: str
    pos: int
    data: tuple

# Helpers to build nodes
def N(kind, p, idx=1, *data):
    return Node(kind=kind, pos=p.lexpos(idx), data=data)

# ============================================================
# Parser (PLY)
# ============================================================

# Precedence
precedence = (
    ('left', 'OROR'),
    ('left', 'ANDAND'),
    ('left', 'EQ', 'NE'),
    ('left', 'LT', 'LE', 'GT', 'GE'),
    ('left', 'PLUS', 'MINUS'),
    ('left', 'TIMES', 'DIVIDE'),
    ('right', 'NOT', 'UMINUS', 'UPLUS'),
)

def attach_parser(src: Source):
    lexer = lex()
    parser = yacc(start="compilation_unit")

    # stash on parser for diagnostics
    parser._source = src
    parser._errors: List[str] = []
    lexer.input(src.text)
    return lexer, parser

# Grammar

def p_compilation_unit(p):
    """compilation_unit : items"""
    p[0] = N("unit", p, 1, p[1])

def p_items_multi(p):
    """items : items item"""
    p[0] = p[1]
    p[0].data = (p[0].data[0] + [p[2]],)

def p_items_single(p):
    """items : item"""
    p[0] = N("items", p, 1, [p[1]])

def p_item(p):
    """item : extern_decl
            | func_def
            | const_decl
            | use_decl
            | namespace_block"""
    p[0] = p[1]

def p_use_decl(p):
    """use_decl : USE path SEMICOLON"""
    p[0] = N("use", p, 1, p[2])

def p_namespace_block(p):
    """namespace_block : NAMESPACE path LBRACE items RBRACE"""
    p[0] = N("namespace", p, 1, p[2], p[4])

def p_path(p):
    """path : NAME
            | path DCOLON NAME"""
    if len(p) == 2:
        p[0] = N("path", p, 1, [p[1]])
    else:
        p[0] = p[1]
        p[0].data = (p[0].data[0] + [p[3]],)

def p_type_name_base(p):
    """type_name : KW_BOOL
                 | KW_STR
                 | KW_CHARPTR
                 | KW_VOID
                 | KW_I8 
                 | KW_U8 
                 | KW_I16 
                 | KW_U16 
                 | KW_I32 
                 | KW_U32
                 | KW_I64 
                 | KW_U64 
                 | KW_I128 
                 | KW_U128
                 | KW_F16 
                 | KW_F32 
                 | KW_F64 
                 | KW_F128"""
    p[0] = N("type", p, 1, p[1])

def p_type_name_path(p):
    """type_name : path"""
    p[0] = N("type_path", p, 1, p[1])

def p_param(p):
    """param : NAME COLON type_name
             | NAME"""
    if len(p) == 4:
        p[0] = N("param", p, 1, p[1], p[3])
    else:
        p[0] = N("param", p, 1, p[1], None)

def p_params(p):
    """params : params COMMA param
              | param
              | """
    if len(p) == 4:
        p[0] = N("params", p, 1, p[1].data[0] + [p[3]])
    elif len(p) == 2:
        p[0] = N("params", p, 1, [p[1]])
    else:
        p[0] = N("params", p, 0, [])

def p_ret_type(p):
    """ret_type : ARROW type_name
                | """
    p[0] = p[2] if len(p) == 3 else None

def p_func_def(p):
    """func_def : FN NAME LPAREN params RPAREN ret_type LBRACE stmt_list RBRACE"""
    p[0] = N("fn", p, 1, p[2], p[4], p[6], p[8])

def p_extern_decl(p):
    """extern_decl : EXTERN NAME LPAREN extern_params RPAREN ret_type SEMICOLON"""
    p[0] = N("extern", p, 1, p[2], p[4], p[6])

def p_extern_params(p):
    """extern_params : extern_params COMMA extern_param
                     | extern_param
                     | """
    if len(p) == 4:
        p[0] = N("eparams", p, 1, p[1].data[0] + [p[3]])
    elif len(p) == 2:
        p[0] = N("eparams", p, 1, [p[1]])
    else:
        p[0] = N("eparams", p, 0, [])

def p_extern_param(p):
    """extern_param : NAME COLON type_name"""
    p[0] = N("eparam", p, 1, p[1], p[3])

def p_const_decl(p):
    """const_decl : CONST NAME COLON type_name ASSIGN expression SEMICOLON"""
    p[0] = N("const", p, 1, p[2], p[4], p[6])

def p_stmt_list(p):
    """stmt_list : stmt_list stmt
                 | stmt
                 | """
    if len(p) == 3:
        p[0] = N("stmts", p, 1, p[1].data[0] + [p[2]])
    elif len(p) == 2:
        p[0] = N("stmts", p, 1, [p[1]])
    else:
        p[0] = N("stmts", p, 0, [])

def p_stmt(p):
    """stmt : var_decl
            | assign_stmt
            | return_stmt
            | expr_stmt"""
    p[0] = p[1]

def p_var_decl(p):
    """var_decl : VAR NAME COLON type_name ASSIGN expression SEMICOLON
                | VAR NAME ASSIGN expression SEMICOLON"""
    if len(p) == 8:
        p[0] = N("vardecl", p, 1, p[2], p[4], p[6])
    else:
        p[0] = N("vardecl", p, 1, p[2], None, p[4])

def p_assign_stmt(p):
    """assign_stmt : NAME ASSIGN expression SEMICOLON"""
    p[0] = N("assign", p, 1, p[1], p[3])

def p_return_stmt(p):
    """return_stmt : RETURN expression SEMICOLON"""
    p[0] = N("ret", p, 1, p[2])

def p_expr_stmt(p):
    """expr_stmt : expression SEMICOLON"""
    p[0] = N("expr", p, 1, p[1])

def p_expression_primary(p):
    """expression : NUMBER
                  | STRING
                  | TRUE
                  | FALSE"""
    tk = p.slice[1].type
    if tk == "NUMBER": p[0] = N("num", p, 1, p[1])
    elif tk == "STRING": p[0] = N("str", p, 1, p[1])
    elif tk == "TRUE": p[0] = N("bool", p, 1, True)
    else: p[0] = N("bool", p, 1, False)

def p_expression_group(p):
    """expression : LPAREN expression RPAREN"""
    p[0] = p[2]

def p_expression_name_or_call(p):
    """expression : maybe_qualified
                  | maybe_qualified LPAREN arg_list RPAREN"""
    if len(p) == 2:
        p[0] = N("name", p, 1, p[1])
    else:
        p[0] = N("call", p, 2, p[1], p[3])

def p_maybe_qualified(p):
    """maybe_qualified : path"""
    p[0] = p[1]

def p_arg_list(p):
    """arg_list : arg_list COMMA expression
                | expression
                | """
    if len(p) == 4:
        p[0] = N("args", p, 1, p[1].data[0] + [p[3]])
    elif len(p) == 2:
        p[0] = N("args", p, 1, [p[1]])
    else:
        p[0] = N("args", p, 0, [])

def p_expression_unary(p):
    """expression : NOT expression
                  | PLUS expression %prec UPLUS
                  | MINUS expression %prec UMINUS"""
    op = p.slice[1].type
    p[0] = N("unop", p, 1, op, p[2])

def p_expression_cast(p):
    """expression : expression AS type_name"""
    p[0] = N("cast", p, 2, p[1], p[3])

def p_expression_binops(p):
    """expression : expression PLUS expression
                  | expression MINUS expression
                  | expression TIMES expression
                  | expression DIVIDE expression
                  | expression ANDAND expression
                  | expression OROR expression
                  | expression EQ expression
                  | expression NE expression
                  | expression LT expression
                  | expression LE expression
                  | expression GT expression
                  | expression GE expression"""
    p[0] = N("binop", p, 2, p.slice[2].type, p[1], p[3])

def p_error(p):
    if p is None:
        print("syntax error at EOF")
    else:
        # We’ll lift this into the analyzer to print pretty context
        print(f"syntax error at {p.value!r}")

# ============================================================
# Types
# ============================================================

@dataclass(frozen=True)
class Ty:
    name: str
    bits: Optional[int] = None
    is_unsigned: bool = False
    is_bool: bool = False
    is_float: bool = False
    is_charptr: bool = False
    is_str: bool = False
    is_void: bool = False

    def __str__(self): return self.name

def make_prim_types():
    prim: Dict[str, Ty] = {}
    prim["void"] = Ty("void", is_void=True)
    prim["bool"] = Ty("bool", 1, is_bool=True)
    prim["str"] = Ty("str", is_str=True)
    prim["charptr"] = Ty("charptr", is_charptr=True)
    for n,b in [("i8",8),("i16",16),("i32",32),("i64",64),("i128",128)]:
        prim[n] = Ty(n, b, is_unsigned=False)
    for n,b in [("u8",8),("u16",16),("u32",32),("u64",64),("u128",128)]:
        prim[n] = Ty(n, b, is_unsigned=True)
    prim["f16"] = Ty("f16", 16, is_float=True)
    prim["f32"] = Ty("f32", 32, is_float=True)
    prim["f64"] = Ty("f64", 64, is_float=True)
    prim["f128"] = Ty("f128", 128, is_float=True)
    return prim

PRIMS = make_prim_types()

# ============================================================
# Frontend IR (symbols & modules)
# ============================================================

@dataclass
class Param:
    name: str
    ty: Optional[Ty]
    pos: int

@dataclass
class FuncDecl:
    name: str
    params: List[Param]
    ret: Ty
    pos: int
    body: Optional[Node]  # None for extern
    module_qual: str

@dataclass
class ConstDecl:
    name: str
    ty: Ty
    value: Node
    pos: int
    module_qual: str

@dataclass
class UseDecl:
    path: List[str]
    pos: int

@dataclass
class Module:
    name: str
    src: Source
    uses: List[UseDecl] = field(default_factory=list)
    funcs: Dict[str, FuncDecl] = field(default_factory=dict)
    consts: Dict[str, ConstDecl] = field(default_factory=dict)
    namespaces: Dict[str, "Module"] = field(default_factory=dict) # nested

# ============================================================
# Semantic analysis
# ============================================================

class Analyzer:
    def __init__(self, modules: Dict[str, Module], es: ErrorSink):
        self.modules = modules
        self.es = es

    # ---- helper: resolve type node
    def resolve_type(self, mod: Module, tnode: Node) -> Optional[Ty]:
        if tnode.kind == "type":
            tname = tnode.data[0]
            ty = PRIMS.get(tname)
            if ty is None:
                self.es.error(f"unknown primitive type '{tname}'", mod.src, tnode.pos)
            elif tname == "f128":
                # only if llvmlite supports it
                if not hasattr(ir, "FP128Type"):
                    self.es.error("type 'f128' unsupported on this platform/llvmlite", mod.src, tnode.pos)
                    return None
            return ty
        elif tnode.kind == "type_path":
            path_node = tnode.data[0]
            path = path_node.data[0]
            # For now, only allow primitive or charptr/str or qualified alias (future)
            self.es.error("user-defined/qualified types not implemented yet", mod.src, tnode.pos)
            return None
        else:
            self.es.error("invalid type syntax", mod.src, tnode.pos)
            return None

    # ---- collect declarations, namespaces, uses
    def collect(self):
        for mod in self.modules.values():
            unit_items = self._expect(mod, mod.src, "compilation unit root", "unit", Node("unit", 0, ()), self._parse_root(mod))
            for item in unit_items:
                self._collect_item(mod, item, mod.name)

    def _parse_root(self, mod: Module) -> Node:
        # parser stored in mod? we pass the AST in Module via a temporary attribute from the driver
        return mod._ast  # type: ignore[attr-defined]

    def _expect(self, mod: Module, src: Source, what: str, kind: str, fallback: Node, root: Node) -> List[Node]:
        if root.kind != "unit":
            self.es.error(f"internal: expected {kind} for {what}", src, root.pos)
            return []
        items_node = root.data[0]
        if items_node.kind != "items":
            self.es.error("internal: items missing", src, root.pos)
            return []
        return items_node.data[0]

    def _collect_item(self, mod: Module, item: Node, qual_prefix: str):
        if item.kind == "extern":
            name, eparams_node, ret_node = item.data
            params = []
            for ep in eparams_node.data[0]:
                pname, pty_node = ep.data
                pty = self.resolve_type(mod, pty_node)
                if pty is None: continue
                params.append(Param(pname, pty, ep.pos))
            if ret_node is None:
                rty = PRIMS["i32"]
            else:
                rty = self.resolve_type(mod, ret_node) or PRIMS["i32"]
            fq = f"{qual_prefix}::{name}"
            if name in mod.funcs:
                self.es.error(f"function '{name}' redefined", mod.src, item.pos)
            mod.funcs[name] = FuncDecl(name, params, rty, item.pos, None, fq)
        elif item.kind == "fn":
            name, params_node, ret_node, body_node = item.data
            params: List[Param] = []
            seen = set()
            for p in params_node.data[0]:
                pname, pty_node = p.data
                if pname in seen:
                    self.es.error(f"duplicate parameter '{pname}'", mod.src, p.pos)
                seen.add(pname)
                pty = self.resolve_type(mod, pty_node) if pty_node is not None else PRIMS["i32"]
                if pty is None: pty = PRIMS["i32"]
                params.append(Param(pname, pty, p.pos))
            rty = self.resolve_type(mod, ret_node) if ret_node else PRIMS["i32"]
            if rty is None: rty = PRIMS["i32"]
            fq = f"{qual_prefix}::{name}"
            if name in mod.funcs:
                self.es.error(f"function '{name}' redefined", mod.src, item.pos)
            mod.funcs[name] = FuncDecl(name, params, rty, item.pos, body_node, fq)
        elif item.kind == "const":
            name, tnode, expr = item.data
            ty = self.resolve_type(mod, tnode)
            if ty is None: ty = PRIMS["i32"]
            fq = f"{qual_prefix}::{name}"
            if name in mod.consts:
                self.es.error(f"const '{name}' redefined", mod.src, item.pos)
            mod.consts[name] = ConstDecl(name, ty, expr, item.pos, fq)
        elif item.kind == "use":
            path = item.data[0].data[0]
            mod.uses.append(UseDecl(path, item.pos))
        elif item.kind == "namespace":
            path_node, items_node = item.data
            segs = path_node.data[0]
            # descend/create nested modules under mod.namespaces
            cur = mod
            qual = qual_prefix
            for s in segs:
                qual = f"{qual}::{s}"
                if s not in cur.namespaces:
                    cur.namespaces[s] = Module(name=f"{cur.name}::{s}", src=mod.src)
                cur = cur.namespaces[s]
            # collect inside nested
            for inner in items_node.data[0]:
                self._collect_item(cur, inner, qual)
        else:
            self.es.error("only extern, fn, const, use, and namespace are allowed at top-level", mod.src, item.pos)

    def _find_start_module(self, mod: Module, first_seg: str) -> Optional[Module]:
        """
        Given the first path segment, figure out what 'module-like' thing it refers to:
        - a namespace inside the current module,
        - the current module itself (if segment == current module name),
        - a top-level module from self.modules,
        - or a namespace inside any used module's root.
        """
        # 1) namespace inside current module?
        if first_seg in mod.namespaces:
            return mod.namespaces[first_seg]

        # 2) explicitly referring to current module by name?
        if first_seg == mod.name:
            return mod

        # 3) a top-level module (another file)?
        if first_seg in self.modules:
            return self.modules[first_seg]

        # 4) a namespace inside any used module's root?
        for u in mod.uses:
            root = self.modules.get(u.path[0])
            if not root:
                continue
            if first_seg == root.name:
                return root
            if first_seg in root.namespaces:
                return root.namespaces[first_seg]

        return None

    def _walk_namespaces(self, start: Module, segs: List[str]) -> Optional[Module]:
        """
        Walk nested namespaces starting at 'start' over segs (all but the final symbol).
        Returns the module representing the namespace that contains the final symbol.
        """
        cur = start
        for s in segs:
            nxt = cur.namespaces.get(s)
            if nxt is None:
                return None
            cur = nxt
        return cur

    # name resolution helpers
    def resolve_const(self, mod: Module, name_path: List[str]) -> Optional[ConstDecl]:
        if len(name_path) == 1:
            nm = name_path[0]
            # local const
            if nm in mod.consts:
                return mod.consts[nm]
            # from any used module root
            for u in mod.uses:
                root = self.modules.get(u.path[0])
                if root and nm in root.consts:
                    return root.consts[nm]
            return None

        # Qualified path
        first = name_path[0]
        rest = name_path[1:-1]
        last = name_path[-1]

        start = self._find_start_module(mod, first)
        if start is None:
            return None

        holder = self._walk_namespaces(start, rest)
        if holder is None:
            return None
        return holder.consts.get(last)

    def resolve_func(self, mod: Module, name_path: List[str]) -> Optional[FuncDecl]:
        if len(name_path) == 1:
            nm = name_path[0]
            # local
            if nm in mod.funcs:
                return mod.funcs[nm]
            # from any used module root
            for u in mod.uses:
                root = self.modules.get(u.path[0])
                if root and nm in root.funcs:
                    return root.funcs[nm]
            return None

        # Qualified path
        first = name_path[0]
        rest = name_path[1:-1]
        last = name_path[-1]

        start = self._find_start_module(mod, first)
        if start is None:
            return None

        holder = self._walk_namespaces(start, rest)
        if holder is None:
            return None
        return holder.funcs.get(last)

    # ---- type-check expressions & statements
    def check_unit(self):
        # Ensure one 'main'
        mains = []
        for m in self.modules.values():
            fn = m.funcs.get("main")
            if fn:
                mains.append(fn)
        if len(mains) == 0:
            # Not hard error—let the linker fail if you only build a lib.
            pass
        elif len(mains) > 1:
            for fn in mains:
                self.es.error("multiple definitions of 'main'", self._src_of(fn), fn.pos)

        # Check each function body
        for m in self.modules.values():
            for f in m.funcs.values():
                if f.body is None: continue
                self._check_fn(m, f)

        # const values must be constant-foldable (for now: only numbers/bools/strings allowed)
        for m in self.modules.values():
            for c in m.consts.values():
                t = self._expr_type(m, {}, c.value)
                if t is None: continue
                # permissive: allow implicit cast to declared type
                if not self._can_cast(t, c.ty):
                    self.es.error(f"cannot initialize const '{c.name}' of type {c.ty} with {t}", m.src, c.pos)

    def _src_of(self, f: FuncDecl) -> Source:
        # map module name back to owning module src (best-effort)
        root = f.module_qual.split("::")[0]
        return self.modules[root].src

    def _check_fn(self, mod: Module, fn: FuncDecl):
        env: Dict[str, Ty] = {}
        for p in fn.params:
            env[p.name] = p.ty or PRIMS["i32"]
        ret_found = self._check_stmts(mod, fn, env, fn.body)
        if fn.ret.name != "void" and not ret_found:
            self.es.error(f"missing return in function '{fn.name}' returning {fn.ret}", self._src_of(fn), fn.pos)

    def _check_stmts(self, mod: Module, fn: FuncDecl, env: Dict[str, Ty], stmts: Node) -> bool:
        did_return = False
        for s in stmts.data[0]:
            if s.kind == "vardecl":
                name, tnode, expr = s.data
                rhs_t = self._expr_type(mod, env, expr)
                if tnode is not None:
                    ty = self.resolve_type(mod, tnode) or rhs_t
                else:
                    ty = rhs_t
                if ty is None: continue
                if rhs_t and not self._can_cast(rhs_t, ty):
                    self.es.error(f"cannot assign {rhs_t} to variable '{name}' of type {ty}", mod.src, s.pos)
                env[name] = ty
            elif s.kind == "assign":
                name, expr = s.data
                if name not in env:
                    self.es.error(f"assignment to undeclared variable '{name}'", mod.src, s.pos)
                else:
                    rhs_t = self._expr_type(mod, env, expr)
                    if rhs_t and not self._can_cast(rhs_t, env[name]):
                        self.es.error(f"cannot assign {rhs_t} to variable '{name}' of type {env[name]}", mod.src, s.pos)
            elif s.kind == "ret":
                t = self._expr_type(mod, env, s.data[0])
                if t and not self._can_cast(t, fn.ret):
                    self.es.error(f"return type mismatch: expected {fn.ret}, got {t}", mod.src, s.pos)
                did_return = True
            elif s.kind == "expr":
                _ = self._expr_type(mod, env, s.data[0])
            else:
                self.es.error("unknown statement", mod.src, s.pos)
        return did_return

    def _expr_type(self, mod: Module, env: Dict[str, Ty], e: Node) -> Optional[Ty]:
        k = e.kind
        if k == "num": return PRIMS["i32"]
        if k == "bool": return PRIMS["bool"]
        if k == "str": return PRIMS["str"]
        if k == "name":
            path = e.data[0].data[0]
            if len(path) == 1 and path[0] in env:
                return env[path[0]]
            c = self.resolve_const(mod, path)
            if c: return c.ty
            f = self.resolve_func(mod, path)
            if f:
                self.es.error("function used as value (call it instead)", mod.src, e.pos)
                return None
            self.es.error(f"unknown name '{'::'.join(path)}'", mod.src, e.pos)
            return None
        if k == "call":
            callee_path = e.data[0].data[0]
            fd = self.resolve_func(mod, callee_path)
            if not fd:
                self.es.error(f"unknown function '{'::'.join(callee_path)}'", mod.src, e.pos)
                return None
            args = e.data[1].data[0]
            if fd.name != "printf" and len(args) != len(fd.params):
                self.es.error(f"function '{fd.name}' expects {len(fd.params)} arg(s), got {len(args)}", mod.src, e.pos)
            # We’ll just type-check what we have
            for i,a in enumerate(args[:len(fd.params)]):
                at = self._expr_type(mod, env, a)
                if at and not self._can_cast(at, fd.params[i].ty):
                    self.es.error(f"argument {i+1} to '{fd.name}' expects {fd.params[i].ty}, got {at}", mod.src, a.pos)
            return fd.ret
        if k == "unop":
            op, rhs = e.data
            t = self._expr_type(mod, env, rhs)
            if op == "NOT":
                if t and t.name != "bool":
                    self.es.error(f"operator '!' requires bool, got {t}", mod.src, e.pos)
                return PRIMS["bool"]
            if op in ("PLUS", "MINUS"):
                if t and not (t.is_float or (t.bits and not t.is_bool and not t.is_str and not t.is_charptr)):
                    self.es.error(f"unary {'+' if op=='PLUS' else '-'} requires numeric, got {t}", mod.src, e.pos)
                return t
        if k == "binop":
            op, lhs, rhs = e.data
            lt = self._expr_type(mod, env, lhs)
            rt = self._expr_type(mod, env, rhs)
            if op in ("ANDAND","OROR"):
                if lt and lt.name!="bool": self.es.error(f"lhs of '&&/||' must be bool, got {lt}", mod.src, lhs.pos)
                if rt and rt.name!="bool": self.es.error(f"rhs of '&&/||' must be bool, got {rt}", mod.src, rhs.pos)
                return PRIMS["bool"]
            if op in ("EQ","NE","LT","LE","GT","GE"):
                # comparison—require same numeric kind
                if lt and rt and not self._same_numeric_family(lt, rt):
                    self.es.error(f"cannot compare {lt} with {rt}", mod.src, e.pos)
                return PRIMS["bool"]
            # arithmetic
            if lt and rt and not self._same_numeric_family(lt, rt):
                self.es.error(f"type mismatch: {lt} {op} {rt}", mod.src, e.pos)
                return lt
            return lt or rt
        if k == "cast":
            src_t = self._expr_type(mod, env, e.data[0])
            dst_t = self.resolve_type(mod, e.data[1]) if e.data[1] else None
            if src_t and dst_t and not self._can_cast(src_t, dst_t):
                self.es.error(f"cannot cast {src_t} to {dst_t}", mod.src, e.pos)
            return dst_t
        # grouped handled earlier
        return None

    def _same_numeric_family(self, a: Ty, b: Ty) -> bool:
        if a.is_float and b.is_float: return True
        if a.bits and b.bits and not a.is_bool and not b.is_bool and not a.is_str and not b.is_str and not a.is_charptr and not b.is_charptr:
            return True
        if a.name=="bool" and b.name=="bool": return True
        return False

    def _can_cast(self, src: Ty, dst: Ty) -> bool:
        if src == dst: return True
        if dst.is_void: return True
        # bool to int/float okay
        if src.name=="bool" and (dst.is_float or (dst.bits and not dst.is_bool)): return True
        # ints <-> ints
        if src.bits and dst.bits and not src.is_float and not dst.is_float:
            return True
        # float <-> float
        if src.is_float and dst.is_float:
            # f128 depends on platform, but analyzer already reported if unsupported
            return True
        # int <-> float
        if (src.bits and not src.is_float and not src.is_bool and not src.is_str and not src.is_charptr) and dst.is_float:
            return True
        if src.is_float and (dst.bits and not dst.is_float and not dst.is_bool and not dst.is_str and not dst.is_charptr):
            return True
        # str to charptr (C string literal)
        if src.is_str and dst.is_charptr: return True
        return False

# ============================================================
# Code generation
# ============================================================

class CodeGen:
    def __init__(self, modules: Dict[str, Module]):
        self.modules = modules
        self.module = ir.Module(name="volt_module")
        self.funcs: Dict[str, ir.Function] = {}
        self.const_strings: Dict[str, ir.GlobalVariable] = {}

    # ----- LLVM type mapping
    def ty_to_ir(self, ty: Ty):
        if ty.is_void: return ir.VoidType()
        if ty.name=="bool": return ir.IntType(1)
        if ty.is_str: return ir.IntType(8).as_pointer()  # represent 'str' as i8*
        if ty.is_charptr: return ir.IntType(8).as_pointer()
        if ty.is_float:
            if ty.name=="f16": return ir.HalfType()
            if ty.name=="f32": return ir.FloatType()
            if ty.name=="f64": return ir.DoubleType()
            if ty.name=="f128":
                if hasattr(ir, "FP128Type"): return ir.FP128Type()
                # shouldn't reach if analyzer blocked it
                return ir.DoubleType()
        if ty.bits:
            return ir.IntType(ty.bits)
        raise RuntimeError(f"unsupported type {ty}")

    # ----- compile
    def build(self):
        # declare externs & defined functions
        for m in self.modules.values():
            for f in m.funcs.values():
                self._declare_function(f)
        # emit consts as global constants (strings are already supported via string literals)
        # const numbers we inline.
        for m in self.modules.values():
            for f in m.funcs.values():
                if f.body is not None:
                    self._define_function(m, f)

    def _declare_function(self, fd: FuncDecl):
        ret = self.ty_to_ir(fd.ret)
        args = [self.ty_to_ir(p.ty) for p in fd.params]

        # Use unmangled name for externs and "main" for the entry point.
        if fd.body is None:              # extern
            llvm_name = fd.name
        elif fd.name == "main":
            llvm_name = "main"
        else:
            llvm_name = fd.module_qual

        func_ty = ir.FunctionType(ret, args, var_arg=(fd.name == "printf"))
        func = ir.Function(self.module, func_ty, name=llvm_name)
        self.funcs[fd.module_qual] = func

        for i, p in enumerate(fd.params):
            func.args[i].name = p.name


    def _mangle(self, name_path: List[str]) -> str:
        # resolve to module-qualified name
        return "::".join(name_path)

    # ----- function body
    def _define_function(self, mod: Module, fd: FuncDecl):
        func = self.funcs[fd.module_qual]
        entry = func.append_basic_block("entry")
        builder = ir.IRBuilder(entry)
        # locals: name -> (ptr, Ty)
        env: Dict[str, Tuple[ir.AllocaInstr, Ty]] = {}

        def alloca(name: str, ty: Ty):
            with builder.goto_entry_block():
                ptr = builder.alloca(self.ty_to_ir(ty), name=name)
            return ptr

        # bind params
        for i, p in enumerate(fd.params):
            ptr = alloca(p.name, p.ty)
            builder.store(func.args[i], ptr)
            env[p.name] = (ptr, p.ty)

        # expression emitter
        def emit_expr(e: Node) -> Tuple[ir.Value, Ty]:
            k = e.kind
            if k=="num":
                v = ir.Constant(ir.IntType(32), e.data[0]); return v, PRIMS["i32"]
            if k=="bool":
                v = ir.Constant(ir.IntType(1), 1 if e.data[0] else 0); return v, PRIMS["bool"]
            if k=="str":
                s = e.data[0]
                gv = self.const_strings.get(s)
                if gv is None:
                    arr = bytearray(s.encode("utf-8")+b"\0")
                    ty_arr = ir.ArrayType(ir.IntType(8), len(arr))
                    gv = ir.GlobalVariable(self.module, ty_arr, name=f".str.{len(self.const_strings)}")
                    gv.global_constant = True
                    gv.linkage = "internal"
                    gv.initializer = ir.Constant(ty_arr, arr)
                    self.const_strings[s] = gv
                ptr = builder.bitcast(gv, ir.IntType(8).as_pointer())
                return ptr, PRIMS["str"]
            if k=="name":
                path = e.data[0].data[0]
                if len(path)==1 and path[0] in env:
                    ptr, ty = env[path[0]]
                    return builder.load(ptr, name=path[0]), ty
                c = Analyzer(self.modules, ErrorSink()).resolve_const(mod, path)
                if c:
                    # only simple constants: numbers/bools/strings
                    v, _ = emit_expr(c.value)
                    return v, c.ty
                # function name used as value is invalid (checked in analyzer)
                raise RuntimeError("unresolved name at codegen")
            if k=="call":
                callee_path = e.data[0].data[0]
                callee_decl = Analyzer(self.modules, ErrorSink()).resolve_func(mod, callee_path)
                callee = self.funcs[callee_decl.module_qual] if callee_decl.name!="main" else self.funcs[callee_decl.module_qual]
                args = []
                for i, a in enumerate(e.data[1].data[0]):
                    av, aty = emit_expr(a)
                    # cast to param type if needed
                    if callee_decl.name != "printf" and i < len(callee_decl.params):
                        av = cast_value(builder, av, aty, callee_decl.params[i].ty)
                        aty = callee_decl.params[i].ty
                    args.append(av)
                call = builder.call(callee, args, name="call")
                return call, callee_decl.ret
            if k=="unop":
                op, rhs = e.data
                rv, rt = emit_expr(rhs)
                if op=="NOT":
                    if rt.name!="bool":
                        rv = to_bool(builder, rv, rt)
                    rv = builder.icmp_unsigned("==", rv, ir.Constant(rv.type, 0))
                    return rv, PRIMS["bool"]
                if op=="PLUS": return rv, rt
                if op=="MINUS":
                    if rt.is_float:
                        return builder.fsub(ir.Constant(rv.type, 0.0), rv), rt
                    else:
                        return builder.sub(ir.Constant(rv.type, 0), rv), rt
            if k=="binop":
                op, lhs, rhs = e.data
                lv, lt = emit_expr(lhs)
                rv, rt = emit_expr(rhs)
                # logical
                if op in ("ANDAND","OROR"):
                    # short-circuit
                    if op=="ANDAND":
                        # if !lv then false else rv
                        lbool = to_bool(builder, lv, lt)
                        merge = builder.append_basic_block("land.merge")
                        rhsb = builder.append_basic_block("land.rhs")
                        res = builder.alloca(ir.IntType(1), name="landtmp")
                        builder.cbranch(lbool, rhsb, merge)
                        # rhs
                        builder.position_at_end(rhsb)
                        rbool = to_bool(builder, rv, rt)
                        builder.store(rbool, res)
                        builder.branch(merge)
                        # merge
                        builder.position_at_end(merge)
                        phi = builder.load(res)
                        return phi, PRIMS["bool"]
                    else:
                        lbool = to_bool(builder, lv, lt)
                        merge = builder.append_basic_block("lor.merge")
                        rhsb = builder.append_basic_block("lor.rhs")
                        res = builder.alloca(ir.IntType(1), name="lortmp")
                        builder.cbranch(lbool, merge, rhsb)
                        builder.position_at_end(rhsb)
                        rbool = to_bool(builder, rv, rt)
                        builder.store(rbool, res)
                        builder.branch(merge)
                        builder.position_at_end(merge)
                        phi = builder.load(res)
                        return phi, PRIMS["bool"]
                # arithmetic / comparisons
                # unify to common type (favor float > signed > unsigned width)
                target_ty = lt if lt==rt else lt
                lv, rv, target_ty = coerce_numeric(builder, lv, lt, rv, rt)
                if op=="PLUS":
                    return (builder.fadd(lv, rv) if target_ty.is_float else builder.add(lv, rv)), target_ty
                if op=="MINUS":
                    return (builder.fsub(lv, rv) if target_ty.is_float else builder.sub(lv, rv)), target_ty
                if op=="TIMES":
                    return (builder.fmul(lv, rv) if target_ty.is_float else builder.mul(lv, rv)), target_ty
                if op=="DIVIDE":
                    if target_ty.is_float: return builder.fdiv(lv, rv), target_ty
                    return (builder.udiv(lv, rv) if target_ty.is_unsigned else builder.sdiv(lv, rv)), target_ty
                if op in ("EQ","NE","LT","LE","GT","GE"):
                    if target_ty.is_float:
                        cmp = {
                            "EQ":"==", "NE":"!=", "LT":"<", "LE":"<=", "GT":">", "GE":">="
                        }[op]
                        return builder.fcmp_ordered(cmp, lv, rv), PRIMS["bool"]
                    else:
                        cmp = {
                            "EQ":"==", "NE":"!=", "LT":"<", "LE":"<=", "GT":">", "GE":">="
                        }[op]
                        if target_ty.is_unsigned:
                            return builder.icmp_unsigned(cmp, lv, rv), PRIMS["bool"]
                        else:
                            return builder.icmp_signed(cmp, lv, rv), PRIMS["bool"]
            if k=="cast":
                sv, st = emit_expr(e.data[0])
                dt = Analyzer(self.modules, ErrorSink()).resolve_type(mod, e.data[1]) or st
                return cast_value(builder, sv, st, dt), dt
            raise RuntimeError(f"unhandled expr {k}")

        # emit statements
        returned = False
        for s in fd.body.data[0]:
            if s.kind=="vardecl":
                name, tnode, expr = s.data
                if tnode is None:
                    # infer from expr
                    v, t = emit_expr(expr)
                else:
                    t = Analyzer(self.modules, ErrorSink()).resolve_type(mod, tnode) or PRIMS["i32"]
                    v, t_src = emit_expr(expr)
                    v = cast_value(builder, v, t_src, t)
                ptr = alloca(name, t)
                builder.store(v, ptr)
                env[name] = (ptr, t)
            elif s.kind=="assign":
                name, expr = s.data
                ptr, t = env[name]
                v, t_src = emit_expr(expr)
                v = cast_value(builder, v, t_src, t)
                builder.store(v, ptr)
            elif s.kind=="expr":
                _ = emit_expr(s.data[0])
            elif s.kind=="ret":
                v, t = emit_expr(s.data[0])
                v = cast_value(builder, v, t, fd.ret)
                builder.ret(v)
                returned = True
                break

        if not returned:
            if fd.ret.is_void:
                builder.ret_void()
            else:
                builder.ret(ir.Constant(self.ty_to_ir(fd.ret), 0))

# ---- helpers for numeric coercion & casts

def to_bool(builder: ir.IRBuilder, v: ir.Value, ty: Ty):
    if ty.name=="bool": return v
    if ty.is_float:
        zero = ir.Constant(v.type, 0.0)
        return builder.fcmp_ordered("!=", v, zero)
    if ty.bits:
        zero = ir.Constant(v.type, 0)
        return builder.icmp_unsigned("!=", v, zero)
    raise RuntimeError("cannot convert to bool")

def cast_value(builder: ir.IRBuilder, v: ir.Value, src: Ty, dst: Ty):
    if src == dst: return v
    if dst.is_void: return ir.Constant(ir.IntType(32), 0)  # unused
    # str -> charptr
    if src.is_str and dst.is_charptr:
        return v
    # bool -> int/float
    if src.name=="bool" and dst.bits and not dst.is_float:
        return builder.zext(v, ir.IntType(dst.bits))
    if src.name=="bool" and dst.is_float:
        return builder.uitofp(builder.zext(v, ir.IntType(32)), _float_ir(dst))
    # int->int
    if src.bits and dst.bits and not src.is_float and not dst.is_float:
        if dst.bits > src.bits:
            return builder.zext(v, ir.IntType(dst.bits)) if src.is_unsigned else builder.sext(v, ir.IntType(dst.bits))
        elif dst.bits < src.bits:
            return builder.trunc(v, ir.IntType(dst.bits))
        else:
            return v
    # float->float
    if src.is_float and dst.is_float:
        if dst.name=="f16":
            return builder.fptrunc(v, ir.HalfType())
        if src.name=="f16" and dst.name in ("f32","f64"):
            return builder.fpext(v, _float_ir(dst))
        if src.name=="f32" and dst.name=="f64":
            return builder.fpext(v, ir.DoubleType())
        if src.name=="f64" and dst.name=="f32":
            return builder.fptrunc(v, ir.FloatType())
        if src.name==dst.name:
            return v
        # f128 best-effort if available
        if hasattr(ir, "FP128Type"):
            if dst.name=="f128":
                return builder.fpext(v, ir.FP128Type())
            if src.name=="f128":
                return builder.fptrunc(v, _float_ir(dst))
    # int->float
    if src.bits and not src.is_float and dst.is_float:
        tmp = builder.zext(v, ir.IntType(max(32, src.bits))) if src.is_unsigned else builder.sext(v, ir.IntType(max(32, src.bits)))
        return builder.uitofp(tmp, _float_ir(dst)) if src.is_unsigned else builder.sitofp(tmp, _float_ir(dst))
    # float->int
    if src.is_float and dst.bits and not dst.is_float:
        return builder.fptoui(v, ir.IntType(dst.bits)) if dst.is_unsigned else builder.fptosi(v, ir.IntType(dst.bits))
    raise RuntimeError(f"unsupported cast {src} -> {dst}")

def _float_ir(dst: Ty):
    if dst.name=="f16": return ir.HalfType()
    if dst.name=="f32": return ir.FloatType()
    if dst.name=="f64": return ir.DoubleType()
    if dst.name=="f128" and hasattr(ir, "FP128Type"): return ir.FP128Type()
    return ir.DoubleType()

def coerce_numeric(builder: ir.IRBuilder, lv, lt: Ty, rv, rt: Ty):
    # prefer float; otherwise widen to max bits; signedness preserved
    if lt.is_float or rt.is_float:
        # pick widest float
        rank = {"f16":0,"f32":1,"f64":2,"f128":3}
        tgt = lt if lt.is_float else rt
        other = rt if lt.is_float else lt
        if other.is_float and rank.get(other.name,1) > rank.get(tgt.name,1):
            tgt = other
        lv = cast_value(builder, lv, lt, tgt)
        rv = cast_value(builder, rv, rt, tgt)
        return lv, rv, tgt
    # integers
    width = max(lt.bits or 32, rt.bits or 32)
    tgt = Ty(("u" if lt.is_unsigned and rt.is_unsigned else "i")+str(width), width, lt.is_unsigned and rt.is_unsigned)
    lv = cast_value(builder, lv, lt, tgt)
    rv = cast_value(builder, rv, rt, tgt)
    return lv, rv, tgt

# ============================================================
# Driver: parse many files, analyze, codegen, link
# ============================================================

def parse_file(path: str, es: ErrorSink) -> Optional[Module]:
    src = Source.from_path(path)
    lexr, parser = attach_parser(src)
    try:
        ast = parser.parse(lexer=lexr, tracking=True)
    except Exception as e:
        print(f"{path}: parse failure: {e}")
        return None
    if ast is None:
        print(f"{path}: syntax errors")
        return None
    modname = os.path.splitext(os.path.basename(path))[0]
    mod = Module(name=modname, src=src)
    mod._ast = ast  # stash AST for analyzer
    return mod

def build_modules(paths: List[str], es: ErrorSink) -> Dict[str, Module]:
    mods: Dict[str, Module] = {}
    for p in paths:
        if not p.endswith(".volt"):
            print(f"skipping non-.volt file: {p}")
            continue
        m = parse_file(p, es)
        if not m: continue
        if m.name in mods:
            es.error(f"duplicate module name '{m.name}' (file basename collision)", m.src, 0)
            continue
        mods[m.name] = m
    return mods


@dataclass
class Toolchain:
    cc: List[str]
    triple: str
    obj_ext: str
    link_args: List[str] = field(default_factory=list)
    description: str = ""


def _quote_cmd(parts: List[str]) -> str:
    return " ".join(shlex.quote(p) for p in parts)


def _is_windows_like_triple(triple: str) -> bool:
    t = (triple or "").lower()
    return any(key in t for key in ("windows", "mingw", "msvc", "cygwin"))


def _arch_from_triple(triple: str) -> str:
    if not triple:
        return platform.machine() or ""
    return triple.split("-")[0]


def _arch_tokens(arch: str) -> List[str]:
    arch = arch.lower()
    aliases = {arch}
    mapping = {
        "x86_64": {"x86_64", "amd64"},
        "amd64": {"x86_64", "amd64"},
        "aarch64": {"aarch64", "arm64"},
        "arm64": {"aarch64", "arm64"},
        "i686": {"i686", "x86"},
        "i386": {"i686", "x86"},
    }
    aliases.update(mapping.get(arch, {arch}))
    return list(aliases)


def _clang_runtime_libs(clang_path: str, triple: str) -> List[str]:
    arch = _arch_from_triple(triple) or platform.machine()
    if not arch:
        return []
    arch_tokens = _arch_tokens(arch)
    try:
        runtime_dir = subprocess.check_output(
            [clang_path, "--print-runtime-dir"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return []

    search_roots = {runtime_dir}
    # Common layouts add per-platform subdirectories
    search_roots.add(os.path.join(runtime_dir, "windows"))
    search_roots.add(os.path.join(runtime_dir, "mingw"))

    libs: List[str] = []
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            lower = name.lower()
            if "builtins" not in lower:
                continue
            if not any(tok in lower for tok in arch_tokens):
                continue
            if not (name.endswith(".a") or name.endswith(".lib")):
                continue
            libs.append(os.path.join(root, name))
        if libs:
            break
    return libs


def _windows_linker_flags(cc_path: str, triple: str) -> List[str]:
    compiler = os.path.basename(cc_path).lower()
    flags: List[str] = []
    if "clang" in compiler:
        flags.extend(_clang_runtime_libs(cc_path, triple))
    elif compiler == "gcc" or compiler.endswith("gcc.exe"):
        # mingw gcc typically handles runtime automatically
        pass
    return flags


def _probe_compiler(cc_cmd: List[str], default_triple: str) -> Optional[Toolchain]:
    exe = os.path.basename(cc_cmd[0]).lower()
    if exe in {"cl", "clang-cl", "link", "lld-link"}:
        return None

    triple = default_triple
    try:
        proc = subprocess.run(
            cc_cmd + ["-dumpmachine"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            out = proc.stdout.strip() or proc.stderr.strip()
            if out:
                triple = out.splitlines()[0]
    except FileNotFoundError:
        return None
    except OSError:
        return None

    obj_ext = ".obj" if _is_windows_like_triple(triple) else ".o"
    link_args: List[str] = []
    if _is_windows_like_triple(triple):
        link_args.extend(_windows_linker_flags(cc_cmd[0], triple))

    desc = _quote_cmd(cc_cmd)
    return Toolchain(cc=list(cc_cmd), triple=triple, obj_ext=obj_ext, link_args=link_args, description=desc)


def _gather_cc_candidates() -> List[List[str]]:
    candidates: List[List[str]] = []
    env_cc = os.environ.get("VOLTC_CC") or os.environ.get("CC")
    if env_cc:
        candidates.append(shlex.split(env_cc))

    names = ["clang", "gcc", "cc", "clang++", "g++"]
    seen: Set[str] = set()
    for name in names:
        path = shutil.which(name)
        if not path:
            continue
        key = os.path.normcase(path)
        if key in seen:
            continue
        seen.add(key)
        candidates.append([path])

    return candidates


def detect_toolchain(default_triple: str) -> Toolchain:
    candidates = _gather_cc_candidates()
    if not candidates:
        raise RuntimeError("no C compiler found; install clang/gcc or set VOLTC_CC")

    errors: List[str] = []
    for cc_cmd in candidates:
        info = _probe_compiler(cc_cmd, default_triple)
        if info:
            print(f"Using host compiler: {info.description} (triple {info.triple})")
            return info
        errors.append(_quote_cmd(cc_cmd))

    detail = ", ".join(errors) if errors else "<none>"
    raise RuntimeError(f"unable to detect usable C compiler (tried: {detail})")


def run_linker(obj_path: str, output: str, toolchain: Toolchain):
    base_cmd = list(toolchain.cc)
    base_cmd.extend([obj_path, "-o", output])

    variants: List[List[str]] = []
    seen_variants: Set[Tuple[str, ...]] = set()

    def add_variant(extra: List[str]):
        cmd = base_cmd + extra
        key = tuple(cmd)
        if key in seen_variants:
            return
        seen_variants.add(key)
        variants.append(cmd)

    add_variant(toolchain.link_args)

    crt_libs = [
        # CRT + libgcc wrappers replicated from clang --### output for MinGW
        "-lc", "-lmingw32", "-lgcc", "-lgcc_eh", "-lmoldname",
        "-lmingwex", "-lmsvcrt", "-lpthread", "-ladvapi32",
        "-lshell32", "-luser32", "-lkernel32",
    ]

    add_variant(toolchain.link_args + crt_libs)

    if _is_windows_like_triple(toolchain.triple):
        # Provide a couple of fallbacks for MinGW-based setups.
        mingw_fallback = toolchain.link_args + [
            "-lmingw32", "-lgcc", "-lgcc_eh", "-lmingwex", "-lmsvcrt",
            "-lkernel32", "-luser32",
        ]
        add_variant(mingw_fallback)

        msvc_fallback = toolchain.link_args + [
            "-lmsvcrt", "-lkernel32", "-luser32",
        ]
        add_variant(msvc_fallback)

    last_error: Optional[subprocess.CalledProcessError] = None
    for cmd in variants:
        print(f"Linking with: {_quote_cmd(cmd)}")
        try:
            subprocess.check_call(cmd)
            return
        except subprocess.CalledProcessError as e:
            last_error = e

    if last_error is not None:
        raise last_error
    raise RuntimeError("linker invocation list was empty")

def compile_and_link(paths: List[str], output="a.out"):
    es = ErrorSink()
    modules = build_modules(paths, es)
    if not modules:
        print("no modules to build")
        return 1

    # collect & analyze
    an = Analyzer(modules, es)
    an.collect()
    an.check_unit()

    if not es.ok():
        es.dump()
        return 1

    # codegen
    cg = CodeGen(modules)
    cg.build()

    print("Generated LLVM IR:")
    print(str(cg.module))

    # compile to object + link
    binding.initialize_native_target()
    binding.initialize_native_asmprinter()

    # Host/toolchain triple & target
    default_triple = binding.get_default_triple()

    try:
        toolchain = detect_toolchain(default_triple)
    except RuntimeError as e:
        print(f"error: {e}")
        return 1

    triple = toolchain.triple or default_triple
    try:
        target = binding.Target.from_triple(triple)
    except RuntimeError:
        print(f"warning: LLVM target for '{triple}' unavailable; falling back to '{default_triple}'")
        triple = default_triple
        target = binding.Target.from_triple(triple)

    is_windows = _is_windows_like_triple(triple)
    is_64bit   = any(a in triple for a in ("x86_64", "aarch64", "arm64"))

    tm_opts = {}
    # Windows COFF needs “large” to avoid 32-bit absolute refs to globals on 64-bit
    tm_opts["codemodel"] = "large" if (is_windows and is_64bit) else "default"
    # PIC on ELF/Mach-O; static/default on Windows
    tm_opts["reloc"]     = "static" if is_windows else "pic"

    tm = target.create_target_machine(**tm_opts)

    # Make the textual IR self-describing
    cg.module.triple = triple
    cg.module.data_layout = str(tm.target_data)
    llvm_ir = str(cg.module)
    mod = binding.parse_assembly(llvm_ir)
    mod.verify()
    obj = tm.emit_object(mod)

    # write a single object (combined)
    obj_ext = toolchain.obj_ext.lstrip('.')
    obj_path = f"out.{obj_ext}"
    with open(obj_path, "wb") as f:
        f.write(obj)
    print(f"Wrote {obj_path}")

    # link to executable
    try:
        run_linker(obj_path=obj_path, output=output, toolchain=toolchain)
    except subprocess.CalledProcessError as e:
        print(f"link failed: {e}")
        return e.returncode
    except RuntimeError as e:
        print(f"link failed: {e}")
        return 1

    # Make sure it’s executable on Unix
    if os.name != "nt":
        try:
            st = os.stat(output)
            os.chmod(output, st.st_mode | 0o111)
        except OSError:
            pass

    print(f"Linked {output}")
    return 0


# ============================================================
# CLI
# ============================================================

EXAMPLE = r'''
// --- stdlike extern
extern printf(fmt: charptr) -> i32;

// -- math
fn add(a: i32, b: i32) -> i32 { 
    var x = a + b;
    return x;
}

// namespace demo
namespace util::fmt {
    fn one() -> i32 { return 1; }
    const NL: charptr = "\n";
}

use util; // import util::*

fn main() -> i32 {
    var x: i32 = add(2, 3);
    var y = (x > 3) && true || false;
    printf("x=%d y=%d", x as i32, y as i32);
    printf(util::fmt::NL);
    return 0;
}
'''

def main():
    # Default output name; can be overridden with -o/--output
    out = "a.out" if os.name != "nt" else "a.exe"

    # No CLI args? Write and compile the demo.volt in the cwd.
    if len(sys.argv) <= 1:
        demo_path = os.path.abspath("demo.volt")
        with open(demo_path, "w", encoding="utf-8") as f:
            f.write(EXAMPLE)
        print("No inputs provided. Compiling demo.volt ...")
        rc = compile_and_link([demo_path], output=out)
        sys.exit(rc)

    # Parse args: <files...> [-o OUTPUT]
    files: List[str] = []
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg in ("-o", "--output"):
            if i + 1 >= len(sys.argv):
                print("error: -o/--output requires a path")
                sys.exit(2)
            out = sys.argv[i + 1]
            i += 2
            continue
        files.append(arg)
        i += 1

    if not files:
        print("error: no input files")
        sys.exit(2)

    # Normalize to absolute paths and filter to .volt
    volt_files = [os.path.abspath(p) for p in files if p.endswith(".volt")]
    if not volt_files:
        print("error: no .volt inputs")
        sys.exit(2)

    rc = compile_and_link(volt_files, output=out)
    sys.exit(rc)


if __name__ == "__main__":
    main()
