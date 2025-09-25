import subprocess
import shutil
import sys
from ply.lex import lex
from ply.yacc import yacc
from llvmlite import ir, binding

# --- Tokenizer

tokens = (
    'PLUS', 'MINUS', 'TIMES', 'DIVIDE', 'LPAREN', 'RPAREN', 'LBRACE', 'RBRACE',
    'NAME', 'NUMBER', 'ASSIGN', 'FN', 'COLON', 'COMMA', 'EXTERN', 'STRING', 'RETURN', 'SEMICOLON', 'VAR'
)

t_ignore = ' \t'
t_PLUS = r'\+'
t_MINUS = r'-'
t_TIMES = r'\*'
t_DIVIDE = r'/'
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_ASSIGN = r'='
t_SEMICOLON = r';'
t_LBRACE = r'\{'
t_RBRACE = r'\}'
t_COLON = r':'
t_COMMA = r','

def t_FN(t):
    r'fn'
    return t

def t_VAR(t):
    r'var'
    return t

def t_EXTERN(t):
    r'extern'
    return t

def t_RETURN(t):
    r'return'
    return t

def t_STRING(t):
    r'"([^"\\]|\\.)*"'
    t.value = bytes(t.value[1:-1], "utf8").decode("unicode_escape")
    return t

t_NAME = r'[a-zA-Z_][a-zA-Z0-9_]*'

def t_NUMBER(t):
    r'\d+'
    t.value = int(t.value)
    return t

def t_ignore_newline(t):
    r'\n+'
    t.lexer.lineno += t.value.count('\n')

def t_error(t):
    print(f'Illegal character {t.value[0]!r}')
    t.lexer.skip(1)

lexer = lex()

# --- Parser

# --- Precedence (keep this near the parser functions)
precedence = (
    ('left', 'PLUS', 'MINUS'),
    ('left', 'TIMES', 'DIVIDE'),
    ('right', 'UMINUS', 'UPLUS'),
)

# --- Grammar rules

def p_program(p):
    '''
    program : statements
    '''
    p[0] = ('program', p[1])

def p_statements_multi(p):
    '''
    statements : statements statement
    '''
    p[0] = p[1] + [p[2]]

def p_statements_single(p):
    '''
    statements : statement
    '''
    p[0] = [p[1]]

def p_statement_expr_stmt(p):
    '''
    statement : expression SEMICOLON
    '''
    p[0] = ('expr', p[1])

def p_statement_vardecl(p):
    '''
    statement : VAR NAME ASSIGN expression SEMICOLON
    '''
    p[0] = ('vardecl', p[2], p[4])

def p_statement_assign(p):
    '''
    statement : NAME ASSIGN expression SEMICOLON
    '''
    p[0] = ('assign', p[1], p[3])

def p_statement_funcdef(p):
    '''
    statement : function_def
    '''
    p[0] = p[1]

def p_statement_extern(p):
    '''
    statement : extern_decl
    '''
    p[0] = p[1]

def p_statement_return(p):
    '''
    statement : RETURN expression SEMICOLON
    '''
    p[0] = ('return', p[2])

def p_function_def(p):
    '''
    function_def : FN NAME LPAREN params RPAREN LBRACE statements RBRACE
    '''
    p[0] = ('funcdef', p[2], p[4], p[7])

def p_params_multi(p):
    '''
    params : params COMMA NAME
    '''
    p[0] = p[1] + [p[3]]

def p_params_single(p):
    '''
    params : NAME
    '''
    p[0] = [p[1]]

def p_params_empty(p):
    '''
    params :
    '''
    p[0] = []

def p_extern_decl(p):
    '''
    extern_decl : EXTERN NAME LPAREN extern_params RPAREN SEMICOLON
    '''
    p[0] = ('extern', p[2], p[4])

def p_extern_params_multi(p):
    '''
    extern_params : extern_params COMMA extern_type
    '''
    p[0] = p[1] + [p[3]]

def p_extern_params_single(p):
    '''
    extern_params : extern_type
    '''
    p[0] = [p[1]]

def p_extern_params_empty(p):
    '''
    extern_params :
    '''
    p[0] = []

def p_extern_type(p):
    '''
    extern_type : NAME
    '''
    p[0] = p[1]

def p_expression_binop(p):
    '''
    expression : expression PLUS expression
               | expression MINUS expression
               | expression TIMES expression
               | expression DIVIDE expression
    '''
    p[0] = ('binop', p[2], p[1], p[3])

def p_expression_grouped(p):
    '''
    expression : LPAREN expression RPAREN
    '''
    p[0] = ('grouped', p[2])

def p_expression_unary(p):
    '''
    expression : PLUS expression %prec UPLUS
               | MINUS expression %prec UMINUS
    '''
    p[0] = ('unary', p[1], p[2])

def p_expression_number(p):
    '''
    expression : NUMBER
    '''
    p[0] = ('number', p[1])

def p_expression_name(p):
    '''
    expression : NAME
    '''
    p[0] = ('name', p[1])

def p_expression_string(p):
    '''
    expression : STRING
    '''
    p[0] = ('string', p[1])

def p_expression_func_call(p):
    '''
    expression : function_call
    '''
    p[0] = p[1]

def p_function_call(p):
    '''
    function_call : NAME LPAREN args RPAREN
    '''
    p[0] = ('funccall', p[1], p[3])

def p_args_multi(p):
    '''
    args : args COMMA expression
    '''
    p[0] = p[1] + [p[3]]

def p_args_single(p):
    '''
    args : expression
    '''
    p[0] = [p[1]]

def p_args_empty(p):
    '''
    args :
    '''
    p[0] = []

def p_error(p):
    if p:
        print(f"Syntax error at {p.value!r}")
    else:
        print("Syntax error at EOF")


parser = yacc()

# --- LLVM Code Generation ---

class CodeGenContext:
    def __init__(self, module, builder):
        self.module = module
        self.builder = builder
        self.named_values = {}
        self.functions = {}
        self.string_constants = {}

def get_llvm_type(type_str):
    if type_str == 'int':
        return ir.IntType(32)
    elif type_str == 'float':
        return ir.FloatType()
    elif type_str == 'charptr':
        return ir.PointerType(ir.IntType(8))
    else:
        raise Exception(f"Unknown type: {type_str}")

def codegen(ast, ctx):
    node_type = ast[0]
    if node_type == 'program':
        last_val = None
        for stmt in ast[1]:
            last_val = codegen(stmt, ctx)
        return last_val
    elif node_type == 'expr':
        return codegen(ast[1], ctx)
    elif node_type == 'vardecl':
        name, expr = ast[1], ast[2]
        val = codegen(expr, ctx)
        ctx.named_values[name] = val
        return val

    elif node_type == 'assign':
        name, expr = ast[1], ast[2]
        val = codegen(expr, ctx)
        # (optionally: check that name already exists to forbid implicit decls)
        ctx.named_values[name] = val
        return val

    elif node_type == 'funcdef':
        name, params, body = ast[1], ast[2], ast[3]

        # i32 (i32, i32, ...)
        func_type = ir.FunctionType(ir.IntType(32), [ir.IntType(32)] * len(params))
        func = ir.Function(ctx.module, func_type, name=name)
        ctx.functions[name] = func

        entry = func.append_basic_block('entry')
        fbuilder = ir.IRBuilder(entry)

        # Function-local context that shares tables with outer one
        fctx = CodeGenContext(ctx.module, fbuilder)
        fctx.functions = ctx.functions                   # share function table
        fctx.string_constants = ctx.string_constants     # share string pool
        fctx.named_values = {}                           # fresh locals

        # Bind parameters
        for i, arg in enumerate(func.args):
            arg.name = params[i]
            fctx.named_values[arg.name] = arg

        # Emit body
        did_return = False
        for stmt in body:
            val = codegen(stmt, fctx)
            if isinstance(stmt, tuple) and stmt[0] == 'return':
                fbuilder.ret(val)
                did_return = True
                break

        if not did_return:
            fbuilder.ret(ir.Constant(ir.IntType(32), 0))

        return func

    elif node_type == 'extern':
        name, param_types = ast[1], ast[2]
        llvm_types = [get_llvm_type(t) for t in param_types]
        func_type = ir.FunctionType(ir.IntType(32), llvm_types, var_arg=True if name == "printf" else False)
        func = ir.Function(ctx.module, func_type, name=name)
        ctx.functions[name] = func
        return func
    elif node_type == 'return':
        return codegen(ast[1], ctx)
    elif node_type == 'funccall':
        func = ctx.functions.get(ast[1])
        if not func:
            raise Exception(f"Undefined function {ast[1]}")
        args = [codegen(arg, ctx) for arg in ast[2]]
        return ctx.builder.call(func, args, name='calltmp')
    elif node_type == 'number':
        return ir.Constant(ir.IntType(32), ast[1])
    elif node_type == 'name':
        v = ctx.named_values.get(ast[1])
        return v if v is not None else ir.Constant(ir.IntType(32), 0)


    elif node_type == 'string':
        s = ast[1]
        if s in ctx.string_constants:
            return ctx.string_constants[s]
        arr_ty = ir.ArrayType(ir.IntType(8), len(s) + 1)
        global_str = ir.GlobalVariable(ctx.module, arr_ty, name=f"str_{len(ctx.string_constants)}")
        global_str.linkage = 'internal'
        global_str.global_constant = True
        global_str.initializer = ir.Constant(arr_ty, bytearray(s.encode("utf8") + b'\0'))
        str_ptr = ctx.builder.bitcast(global_str, ir.PointerType(ir.IntType(8)))
        ctx.string_constants[s] = str_ptr
        return str_ptr
    elif node_type == 'binop':
        op = ast[1]
        left = codegen(ast[2], ctx)
        right = codegen(ast[3], ctx)
        if op == '+':
            return ctx.builder.add(left, right, name='addtmp')
        elif op == '-':
            return ctx.builder.sub(left, right, name='subtmp')
        elif op == '*':
            return ctx.builder.mul(left, right, name='multmp')
        elif op == '/':
            return ctx.builder.sdiv(left, right, name='divtmp')
    elif node_type == 'unary':
        op = ast[1]
        operand = codegen(ast[2], ctx)
        if op == '+':
            return operand
        elif op == '-':
            return ctx.builder.neg(operand, name='negtmp')
    elif node_type == 'grouped':
        return codegen(ast[1], ctx)
    else:
        raise ValueError(f"Unknown AST node: {node_type}")

def semantic_analyze(ast):
    errors = []
    functions = {}  # name -> arity
    externs = {}    # name -> arity (min; printf is variadic)
    globals_scope = set()

    def check_expr(expr, scope):
        t = expr[0]
        if t == 'number' or t == 'string':
            return
        if t == 'name':
            name = expr[1]
            if name not in scope:
                errors.append(f"Use of undeclared variable '{name}'.")
        elif t == 'binop':
            check_expr(expr[2], scope)
            check_expr(expr[3], scope)
        elif t == 'unary':
            check_expr(expr[2], scope)
        elif t == 'grouped':
            check_expr(expr[1], scope)
        elif t == 'funccall':
            fname, args = expr[1], expr[2]
            # known?
            if fname not in functions and fname not in externs:
                errors.append(f"Call to undefined function '{fname}'.")
            else:
                arity = functions.get(fname, externs.get(fname))
                if fname != 'printf' and len(args) != arity:
                    errors.append(f"Function '{fname}' expects {arity} arg(s), got {len(args)}.")
            for a in args:
                check_expr(a, scope)
        else:
            # expr or others
            pass

    def check_stmt(stmt, scope):
        t = stmt[0]
        if t == 'vardecl':
            name, expr = stmt[1], stmt[2]
            if name in scope:
                errors.append(f"Redeclaration of variable '{name}'.")
            scope.add(name)
            check_expr(expr, scope)
        elif t == 'assign':
            name, expr = stmt[1], stmt[2]
            if name not in scope:
                errors.append(f"Assignment to undeclared variable '{name}'.")
            check_expr(expr, scope)
        elif t == 'expr':
            check_expr(stmt[1], scope)
        elif t == 'extern':
            # record extern arity (printf handled as variadic)
            name, params = stmt[1], stmt[2]
            externs[name] = len(params)
        elif t == 'funcdef':
            pass  # handled in first pass
        elif t == 'return':
            check_expr(stmt[1], scope)

    # First pass: collect function signatures and detect duplicates
    for item in ast[1]:
        if item[0] == 'funcdef':
            fname, params = item[1], item[2]
            if fname in functions:
                errors.append(f"Redefinition of function '{fname}'.")
            else:
                functions[fname] = len(params)
        elif item[0] == 'extern':
            externs[item[1]] = len(item[2])

    # Second pass: check top-level and function bodies
    # top-level scope: allow var decls
    top_scope = set()
    for item in ast[1]:
        if item[0] == 'funcdef':
            _, fname, params, body = item
            # function-local scope seeded with params
            fscope = set(params)
            for st in body:
                check_stmt(st, fscope)
        else:
            check_stmt(item, top_scope)

    return errors


def compile_ast_to_binary(ast):
    errs = semantic_analyze(ast)
    if errs:
        print("Semantic errors:\n  - " + "\n  - ".join(errs))
        return  # don't generate IR / object files

    module = ir.Module(name="calc_module")
    func_type = ir.FunctionType(ir.IntType(32), [])
    func = ir.Function(module, func_type, name="main")
    block = func.append_basic_block(name="entry")
    builder = ir.IRBuilder(block)
    ctx = CodeGenContext(module, builder)

    # Predefine a format string for printf
    fmt_str = "%d\n"
    arr_ty = ir.ArrayType(ir.IntType(8), len(fmt_str) + 1)
    fmt_global = ir.GlobalVariable(module, arr_ty, name="fmt")
    fmt_global.linkage = 'internal'
    fmt_global.global_constant = True
    fmt_global.initializer = ir.Constant(arr_ty, bytearray(fmt_str.encode("utf8") + b'\0'))
    fmt_ptr = builder.bitcast(fmt_global, ir.PointerType(ir.IntType(8)))
    ctx.named_values["fmt"] = fmt_ptr

    retval = codegen(ast, ctx)

    builder.ret(ir.Constant(ir.IntType(32), 0))

    print("Generated LLVM IR:")
    print(module)

    binding.initialize_native_target()
    binding.initialize_native_asmprinter()
    llvm_ir = str(module)
    mod = binding.parse_assembly(llvm_ir)
    mod.verify()
    target = binding.Target.from_default_triple()
    target_machine = target.create_target_machine()
    obj_file = "output.obj" if sys.platform == "win32" else "output.o"
    exe_file = "output.exe" if sys.platform == "win32" else "output"
    with open(obj_file, "wb") as f:
        obj = target_machine.emit_object(mod)
        f.write(obj)
    print(f"Object file '{obj_file}' generated.")

    if sys.platform == "win32":
        clang = shutil.which("clang")
        if clang:
            compile_cmd = [clang, obj_file, "-o", exe_file]
            try:
                subprocess.check_call(compile_cmd)
                print(f"Executable '{exe_file}' generated.")
            except Exception as e:
                print(f"Failed to compile and link with clang: {e}")
        else:
            print("Clang not found. MSVC CL cannot link llvmlite object files. Please install clang or use JIT execution.")
    else:
        compiler = "clang" if shutil.which("clang") else "gcc"
        compile_cmd = [compiler, obj_file, "-o", exe_file]
        try:
            subprocess.check_call(compile_cmd)
            print(f"Executable '{exe_file}' generated.")
        except Exception as e:
            print(f"Failed to compile and link: {e}")

# --- Example usage ---

source_code = """
extern printf(charptr);
fn add(a, b, c, d) { return a + b + c + d; }
var x = 2 * 3;
var y = add(x, 4, 8, 10);
y = 10 / 2;
printf("%d\n", y + 1);

"""

ast = parser.parse(source_code)
print("AST:", ast)
compile_ast_to_binary(ast)
