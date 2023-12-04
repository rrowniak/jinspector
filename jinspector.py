import pdb
import argparse
import cmd
from enum import Enum, unique
import functools
import json
import os
import re
import signal
import subprocess
import sys


AUTHOR = 'Rafal Rowniak'
PROGRAM_DESC = f"jinspector - inspect and transform JSON.\nAuthor {AUTHOR}"
CMD_TIMEOUT_SEC = 10


@unique
class ELogLev(Enum):
    TRACE = 1
    VERBOSE = 2
    NONE = 8


glob_log_level = ELogLev.NONE
glob_fn_trace_indent = 0


def set_log_lev(lev):
    global glob_log_level
    glob_log_level = lev


def log(level, msg):
    if level is ELogLev.TRACE:
        print(f'TRACE: {msg}')
    elif level is ELogLev.VERBOSE:
        print(f'VERB: {msg}')


def log_trace(msg):
    if glob_log_level.value <= ELogLev.TRACE.value:
        log(ELogLev.TRACE, msg)


def log_verb(msg):
    if glob_log_level.value <= ELogLev.VERBOSE.value:
        log(ELogLev.VERBOSE, msg)


def trace_fn(fn):
    @functools.wraps(fn)
    def decorator(*args, **kwargs):
        global glob_fn_trace_indent
        ind = ' ' * glob_fn_trace_indent
        glob_fn_trace_indent += 1
        log_trace(f'{ind}Calling {fn.__name__}({args}, {kwargs})')
        ret = fn(*args, **kwargs)
        log_trace(f'{ind}Function {fn.__name__} returned {ret}')
        glob_fn_trace_indent -= 1
        return ret
    return decorator


def call_bash_cmd(cmd):
    new_session = False
    if sys.platform.startswith('linux'):
        # POSIX only
        new_session = True
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,
                         start_new_session=new_session, shell=True)

    try:
        out = p.communicate(timeout=CMD_TIMEOUT_SEC)[0].decode('utf-8')
        timeout_reached = False
    except subprocess.TimeoutExpired:
        print('Execution timeout. Aborting...')
        kill_all(p.pid)
        out = p.communicate()[0].decode('utf-8')
        timeout_reached = True
    return p.returncode, out, timeout_reached


def kill_all(pid):
    if sys.platform.startswith('linux'):
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    elif sys.platform.startswith('cygwin'):
        winpid = int(open("/proc/{pid}/winpid".format(pid=pid)).read())
        subprocess.Popen(['TASKKILL', '/F', '/PID', str(winpid), '/T'])
    elif sys.platform.startswith('win32'):
        subprocess.Popen(['TASKKILL', '/F', '/PID', str(pid), '/T'])


@unique
class JPropT(Enum):
    NORMAL = '~'
    ROOT = '$'
    RECURSIVE_DESCENT = '..'
    WILDCARD = '*'
    FILTER_EXPRESSION = '?'
    SCRIPT_EXPRESSION = '('
    INDEX = 'i'


class JProperty:
    def __init__(self, prop_type, name=''):
        self.ptype = prop_type
        self.name = name
        self.index_fn = None
        self.expr_arr = None
        self.filter_fn = None

    def __repr__(self):
        return f'{self.ptype}{self.name}'

    def __str__(self):
        if self.ptype is JPropT.NORMAL:
            if any(c.isspace() for c in self.name):
                return f"['{self.name}']"
            return self.name
        elif self.ptype in (JPropT.WILDCARD, JPropT.RECURSIVE_DESCENT):
            return self.ptype.value
        return f'[{self.name}]'

    def __eq__(self, other):
        return self.name == other.name and self.ptype == other.ptype

    @ trace_fn
    def index_match(self, indx):
        if self.index_fn:
            return self.index_fn(indx)
        return False

    def filter_match(self, obj):
        if self.filter_fn:
            return self.filter_fn(obj)
        return False


def jpath_to_string(props):
    s = JPropT.ROOT.value
    prev_rd = False
    for p in props:
        leading_dot = '.'
        if prev_rd:
            leading_dot = ''
            prev_rd = False
        p_str = str(p)
        if any(c == '[' for c in p_str):
            leading_dot = ''
        if p.ptype == JPropT.RECURSIVE_DESCENT:
            leading_dot = ''
            prev_rd = True

        s += f'{leading_dot}{p_str}'

    return s


class ParsingError(Exception):
    pass


class QueryExecError(Exception):
    pass


class JSONPathParser:
    S_BEG = 0
    S_PROP = 1
    S_INSIDE_PAR = 2
    S_IND_OR_EXPR = 3

    def __init__(self, jsonpath):
        self.jsonpath = jsonpath.strip()

    def err(self, i, msg):
        return JSONPathParser.err_gen(self.jsonpath, i, msg)

    def err_gen(e, i, msg):
        ind = ' ' * (i - 1) + '^'
        return (f"{e}\n{ind}\n"
                f"Syntax error at {i}: {msg}")

    def parse(self):
        PROP_NAME_STOP = [JPropT.WILDCARD.value, "'", '.', '[']
        props = []
        err_msg = ''
        state = JSONPathParser.S_BEG
        i = 0
        jpath = self.jsonpath
        while i < len(jpath):
            if state == JSONPathParser.S_BEG:
                if jpath[i] == JPropT.ROOT.value:
                    # this is a root property, don't need to do anything
                    i += 1
                    continue
                state = JSONPathParser.S_PROP
                state, i, _ = self._check_if_prop(jpath, i, props, state)
                continue
            elif state == JSONPathParser.S_PROP:
                state, i, ctrl_break, err = self._extract_prop(
                    jpath, i, props, state, PROP_NAME_STOP)
                err_msg = err if err else err_msg
                if ctrl_break:
                    break
                continue
            elif state == JSONPathParser.S_INSIDE_PAR:
                if jpath[i] != "'":
                    # this might be either index field or expression
                    state = JSONPathParser.S_IND_OR_EXPR
                    continue
                i += 1
                name, ind, _ = read_up_to(jpath, i, ["'"])
                if len(name) == 0:
                    err_msg = self.err(i, '(2) expected property name')
                    break
                props.append(JProperty(JPropT.NORMAL, name))
                # we need to skip ']'
                i = ind + 2
                state = JSONPathParser.S_PROP
                continue
            elif state == JSONPathParser.S_IND_OR_EXPR:
                state, i, err = self._extract_expr(jpath, i, props)
                err_msg = err if err else err_msg
                continue
            pdb.set_trace()
            i += 1
        return props, err_msg

    def _extract_prop(self, jpath, i, props, state, PROP_NAME_STOP):
        if (res := check_next(jpath, i, JPropT.WILDCARD.value))[0]:
            props.append(JProperty(JPropT.WILDCARD))
            i = res[1]
            state = JSONPathParser.S_PROP
            return state, i, False, None
        state, i, match = self._check_if_prop(
            jpath, i, props, state)
        if match:
            return state, i, False, None
        # standard property name expected
        name, i, stop = read_up_to(jpath, i, PROP_NAME_STOP)
        if len(name) == 0:
            err_msg = self.err(i, 'expected property name')
            return state, i, True, err_msg
        props.append(JProperty(JPropT.NORMAL, name))
        if stop == '[':
            state = JSONPathParser.S_IND_OR_EXPR
            i += 1
            return state, i, False, None
        elif stop == '.':
            state = JSONPathParser.S_PROP
            return state, i, False, None

        if i == len(jpath):
            return state, i, True, None
        err_msg = self.err(
            i, f'unexpected character: "{stop}"')
        return state, i, True, err_msg

    def _extract_expr(self, jpath, i, props):
        expr, ind, _ = read_up_to_ext(jpath, i, [']'], "'", ['['])
        if len(expr) == 0:
            err_msg = self.err(i, 'got empty expression')
            return JSONPathParser.S_PROP, i, err_msg
        i = ind + 1
        props.append(self.parse_expr(expr))

        return JSONPathParser.S_PROP, i, None

    def _check_if_prop(self, jpath, i, props, state):
        match = True
        if (res := check_next(jpath, i,
                              JPropT.RECURSIVE_DESCENT.value))[0]:
            props.append(JProperty(JPropT.RECURSIVE_DESCENT))
            i = res[1]
            state = JSONPathParser.S_PROP
        elif (res := check_next(jpath, i, '.'))[0]:
            i = res[1]
            state = JSONPathParser.S_PROP
        elif (res := check_next(jpath, i, '['))[0]:
            i = res[1]
            state = JSONPathParser.S_INSIDE_PAR
        else:
            match = False
        return state, i, match

    def parse_expr(self, expr):
        expr = expr.strip()
        if expr.startswith('?('):
            p, e = self.parse_filt_expr(expr)
            if e:
                raise ParsingError(e)
            p.filter_fn = self.compile_filt_expr(p.expr_arr)
            return p
        elif expr.startswith('('):
            return JProperty(JPropT.SCRIPT_EXPRESSION, expr)
        else:  # index
            return self.parse_index(expr)

    def parse_index(self, expr):
        ind_prop = JProperty(JPropT.INDEX, expr)
        # maybe [*]?
        if expr == '*':
            ind_prop.index_fn = lambda _: True
            return ind_prop

        def parse_el(s):
            try:
                return int(s)
            except ValueError:
                return None
        # maybe [n]?
        n = parse_el(expr)
        if n is not None:
            ind_prop.index_fn = lambda i: i == n
            return ind_prop
        # maybe [index1,index2,..]?
        arr = expr.split(',')
        if len(arr) > 1:
            try:
                arr = [int(i) for i in arr]
                ind_prop.index_fn = lambda i: i in arr
                return ind_prop
            except ValueError:
                pass
        # maybe [start:end]?
        arr = expr.split(':')

        if len(arr) == 2:
            start = parse_el(arr[0])
            stop = parse_el(arr[1])
            ind_prop.index_fn = gen_slice_like_fn(start, stop)
            return ind_prop
        # something is wrong, we silently skip parsing errors here
        ind_prop.index_fn = lambda _: False
        return ind_prop

    def parse_filt_expr(self, expr):
        orig_exp = expr
        curr_index = 0

        def err(i, msg):
            return JSONPathParser.err_gen(orig_exp, i, msg)

        prop = JProperty(JPropT.FILTER_EXPRESSION, expr)
        if len(expr) < 3:
            return prop, err(curr_index, 'empty expression')
        # extract from "?(expr)"
        expr = expr[2: -1]
        curr_index += 2
        expr_arr = []
        prop.expr_arr = expr_arr
        while len(expr.strip()) != 0:
            curr_index += len(expr) - len(expr.lstrip())
            expr = expr.strip()

            found = False
            for op in EOpType:
                if expr.startswith(op.value):
                    expr_arr.append(
                        Expression(EExprType.OP, Operator(op)))
                    curr_index += len(op.value)
                    expr = expr[len(op.value):]
                    found = True
                    break
            if found:
                continue

            if expr.startswith('@.'):
                # this is property name
                p = re.search(r'^\w+', expr[2:])
                if not p:
                    return prop, err(curr_index, 'empty expression')
                p = p.group()
                expr_arr.append(Expression(EExprType.PROP_NAME, p))
                adv = len('@.') + len(p)
                curr_index += adv
                expr = expr[adv:]
                continue

            if expr.startswith('['):
                # ['a', 'b'] - a list
                _, i, _ = read_up_to_ext(expr, 0, ']', "'", '')
                vals = expr[1:i].split(',')
                arr = []
                for v in vals:
                    arr.append(self.parse_val(v))
                expr_arr.append(Expression(EExprType.VAL_ARR, arr))
                curr_index += i + 1
                expr = expr[i+1:]
                continue

            json_val = (r'^\s*(true|false|null'
                        r'|-?\d+\.\d+|-?\d+'
                        r'|"[^"]*"|\'[^\']*\')')
            match_result = re.match(json_val, expr)
            if match_result:
                val = match_result.group()
                expr_arr.append(Expression(EExprType.VAL, self.parse_val(val)))
                curr_index += len(val)
                expr = expr[len(val):]
                continue

            json_reg = (r'^(.*)(?:\b\s*\b|\s*$)')
            match_result = re.match(json_reg, expr)
            if match_result:
                val = match_result.group()
                offset = len(val)
                val = val.strip()
                if len(val) == 0:
                    print(expr_arr)
                    return prop, err(curr_index, 'got empty expr')
                expr_arr.append(Expression(EExprType.VAL, val))
                curr_index += offset
                expr = expr[offset:]
                continue

            return prop, err(curr_index, f'cannot parse: "{expr}"')

        return prop, None

    def parse_val(self, s):
        s = s.strip()
        if s.startswith("'") or s.startswith('"'):
            return s[1:-1]
        elif s == 'true':
            return True
        elif s == 'false':
            return False
        try:
            return int(s)
        except ValueError:
            pass
        try:
            return float(s)
        except ValueError:
            pass
        return None

    def compile_filt_expr(self, expr):
        # any logical operator?
        for i, ex in enumerate(expr):
            is_op = (ex.expr_type == EExprType.OP)
            if is_op and EOpType.is_logical(ex.value):
                left = self.compile_filt_expr(expr[:i])
                right = self.compile_filt_expr(expr[i+1:])
                return EvalNode(ex.value, left, right)
        # there is no logical operator
        return self.compile_filt_non_logic(expr)

    def compile_filt_non_logic(self, expr):
        en = EvalNode(None, None, None)

        def left_or_right(v):
            if not en.left:
                en.left = v
            else:
                en.right = v

        for ex in expr:
            if ex.expr_type == EExprType.OP:
                en.op = ex.value
        if en.op is None:
            en.op = Operator(EOpType.EXIST)

        for ex in expr:
            prop_name = ex.expr_type == EExprType.PROP_NAME
            neg_exist = en.op.op_type in (EOpType.NEG, EOpType.EXIST)
            empty_f = en.op.op_type in (EOpType.EMTPY, EOpType.EMPTYF)
            if prop_name and neg_exist:
                left_or_right(JSONPathParser.gen_val_getter(ex))
                left_or_right(JSONPathParser.gen_pass_json_obj())
            elif prop_name and empty_f:
                left_or_right(JSONPathParser.gen_prop_dereferencer(ex))
                left_or_right(lambda _: None)
            elif prop_name:
                left_or_right(JSONPathParser.gen_prop_dereferencer(ex))
            elif ex.expr_type in (EExprType.VAL, EExprType.VAL_ARR):
                left_or_right(JSONPathParser.gen_val_getter(ex))
        return en

    def gen_val_getter(ex):
        def fn(_):
            nonlocal ex
            return ex.value
        return fn

    def gen_prop_dereferencer(ex):
        def fn(p):
            nonlocal ex
            # try:
            return p[ex.value]
            # except Exception as e:
            #     print(f'Exc: {e}, p={p}, v={v}')
            #     return False
        return fn

    def gen_pass_json_obj():
        def fn(p):
            return p
        return fn


class EvalNode:
    def __init__(self, op, left, right):
        self.op = op
        self.right = right
        self.left = left

    def __call__(self, obj):
        return self.op(self.left(obj), self.right(obj))


@unique
class EOpType(Enum):
    EQ = '=='
    NE = '!='
    GT = '>'
    GE = '>='
    LESS = '<'
    LE = '<='
    REG = '=~'
    NEG = '!'
    EXIST = '!!'
    AND = '&&'
    OR = '||'
    IN = 'in'
    NIN = 'nin'
    SUBOF = 'subsetof'
    CONT = 'contains'
    SIZE = 'size'
    EMTPY = 'empty true'
    EMPTYF = 'empty false'

    def is_logical(op):
        return op.op_type is EOpType.AND or op.op_type is EOpType.OR


class Operator:
    CALL_MAP = {
        EOpType.EQ: lambda x, y: x == y,
        EOpType.NE: lambda x, y: x != y,
        EOpType.GT: lambda x, y: x > y,
        EOpType.GE: lambda x, y: x >= y,
        EOpType.LESS: lambda x, y: x < y,
        EOpType.LE: lambda x, y: x <= y,
        EOpType.REG: lambda s, p: re.match(p, s),
        EOpType.NEG: lambda prop, properties: prop not in properties,
        EOpType.EXIST: lambda prop, properties: prop in properties,
        EOpType.AND: lambda a, b: a and b,
        EOpType.OR: lambda a, b: a or b,
        EOpType.IN: lambda v, arr: v in arr,
        EOpType.NIN: lambda v, arr: v not in arr,
        EOpType.SUBOF: lambda v, arr: set(v).issubset(set(arr)),
        EOpType.CONT: lambda a, b: a in b,
        EOpType.SIZE: lambda a, size: len(a) == size,
        EOpType.EMTPY: lambda a, _: len(a) == 0,
        EOpType.EMPTYF: lambda a, _: len(a) != 0
    }

    def __init__(self, op_type):
        self.op_type = op_type

    def __call__(self, *args):
        try:
            return Operator.CALL_MAP[self.op_type](*args)
        except TypeError as ex:
            msg = f'Operator eval error: {ex}. Op={self.op_type}, arg={args}'
            raise QueryExecError(msg)

    def __repr__(self):
        return f'{self.op_type}'

    def __eq__(self, other):
        return self.op_type == other.op_type


@unique
class EExprType(Enum):
    OP = 1
    VAL = 2
    PROP_NAME = 3
    VAL_ARR = 4


class Expression:
    def __init__(self, expr_type, value):
        self.expr_type = expr_type
        self.value = value

    def __repr__(self):
        return f'{self.expr_type}:{self.value}'

    def __eq__(self, other):
        return self.expr_type == other.expr_type and self.value == other.value


def gen_slice_like_fn(start, stop):
    if start and stop:
        def fn(i): return i >= start and i < stop
    elif start:
        def fn(i): return i >= start
    elif stop:
        def fn(i): return i < stop
    else:
        def fn(_): return True
    return fn


def read_up_to(s, i, stop):
    ret_str = ''
    stopped_by = ''
    while i < len(s):
        if s[i] in stop:
            stopped_by = s[i]
            break
        ret_str += s[i]
        i += 1
    return (ret_str, i, stopped_by)


def read_up_to_ext(s, i, stop, quotes, nest_start):
    ret_str = ''
    stopped_by = ''
    qs = []
    ps = []
    while i < len(s):
        c = s[i]
        if c in quotes:
            if len(qs) > 0:
                if qs[-1] == c:
                    qs.pop()
            else:
                qs.append(c)
        if len(qs) == 0:
            if c in nest_start:
                ps.append(c)
            elif c in stop and len(ps) != 0:
                inx = stop.index(c)
                if ps[-1] == nest_start[inx]:
                    ps.pop()
            elif c in stop and len(ps) == 0:
                stopped_by = c
                break

        ret_str += c
        i += 1
    return (ret_str, i, stopped_by)


def check_next(s, i, match):
    j = 0
    while i < len(s) and j < len(match):
        if s[i] != match[j]:
            return (False, i)
        i += 1
        j += 1
    if j == len(match):
        return (True, i)
    return (False, i)


@unique
class EComp(Enum):
    EQUAL = 0
    EQUAL_CONTINUE = 1
    NOT_YET = 2
    DIFFERENT = 3


class TravObj:
    def __init__(self, name, index, length, parent_json, json):
        self.name = name
        self.index = index
        self.len = length
        self.parent_json = parent_json
        self.json = json

    def __repr__(self):
        if self.index is not None:
            return f'{self.name}[{self.index}]'
        else:
            return f'{self.name}'


class QueryExec:
    def __init__(self, data, jpathobj):
        self.data = data
        self.jpathobj = jpathobj
        self.log = []
        self.error_msg = ''
        self.trav_prop_stack = []

    @ trace_fn
    def build_up(self):
        self.traverse(None, self.data, '', None, None)
        if self.error_msg:
            return (False, self.error_msg)
        return (True, self.log)

    def traverse(self, parent_json, json_obj, name, index, arr_len):
        if self.error_msg:
            # break recursion
            return

        self.trav_prop_stack.append(TravObj(name, index, arr_len,
                                            parent_json, json_obj))
        ret = QueryExec.cmp(self.jpathobj, self.trav_prop_stack)
        jo = jpath_to_string(self.jpathobj)
        prop = '.'.join((str(p) for p in self.trav_prop_stack))
        log_trace(f'Comparing "{jo}" with "{prop}": {ret}')

        if ret in (EComp.EQUAL, EComp.EQUAL_CONTINUE):
            self.log.append(json_obj)

        if ret in (EComp.NOT_YET, EComp.EQUAL_CONTINUE):
            if isinstance(json_obj, dict):
                for k, v in json_obj.items():
                    self.traverse(json_obj, v, k, None, None)
            elif isinstance(json_obj, list):
                for i, v in enumerate(json_obj):
                    self.traverse(json_obj, v, '', i, len(json_obj))

        self.trav_prop_stack.pop()

    def cmp(jpath, stack):
        if any(map(lambda x: x.ptype is JPropT.RECURSIVE_DESCENT, jpath)):
            return QueryExec.cmp_rec_descent(jpath, stack)
        return QueryExec.cmp_lin(jpath, stack)

    def cmp_lin(jpath, stack):
        # TODO: replace with cmp_rec_descent
        jpath_ind = 0
        stack_ind = 0
        while jpath_ind < len(jpath) and stack_ind < len(stack):
            jp = jpath[jpath_ind]
            obj = stack[stack_ind]

            if stack_ind == 0 and obj.name == '':
                # this is root object, move forward
                stack_ind += 1
                continue

            if not QueryExec.cmp_single(jp, obj):
                return EComp.DIFFERENT
            stack_ind += 1
            jpath_ind += 1

        if jpath_ind == len(jpath) and stack_ind == len(stack):
            return EComp.EQUAL
        elif jpath_ind < len(jpath) and stack_ind == len(stack):
            return EComp.NOT_YET
        return EComp.DIFFERENT

    def cmp_single(jprop, obj):
        if jprop.ptype is JPropT.INDEX and obj.index is not None:
            if jprop.index_fn(obj.index):
                return True
        if jprop.ptype is JPropT.FILTER_EXPRESSION and jprop.filter_fn:
            if jprop.filter_fn(obj.json):
                return True
        if jprop.ptype is JPropT.WILDCARD:
            # any property is allowed
            return True
        if jprop.name == obj.name:
            return True
        return False

    def cmp_rec_descent(jpath, stack):
        jpath_ind = 0
        stack_ind = 0
        while jpath_ind < len(jpath) and stack_ind < len(stack):
            jp = jpath[jpath_ind]
            obj = stack[stack_ind]

            if stack_ind == 0 and obj.name == '' and obj.index is None:
                # this is root object, move forward
                stack_ind += 1
                continue

            if jp.ptype is JPropT.RECURSIVE_DESCENT:
                res = EComp.NOT_YET
                new_jpath = jpath[jpath_ind+1:]
                log_trace('Recursive descent begin...')
                if len(new_jpath) == 0:
                    # nothing to compare
                    log_trace('jpath empty - Ecomp.EQUAL')
                    return EComp.EQUAL
                for new_stack_ind in range(stack_ind, len(stack)):
                    new_stack = stack[new_stack_ind:]
                    log_trace(f'Trying jpath={new_jpath} stack={new_stack}')
                    r = QueryExec.cmp_rec_descent(new_jpath, new_stack)
                    if r is EComp.EQUAL:
                        log_trace('Equal')
                        return EComp.EQUAL_CONTINUE
                    elif r is EComp.EQUAL_CONTINUE:
                        log_trace('Equal continue')
                        return EComp.EQUAL_CONTINUE
                log_trace(f'res={res}')
                return res

            if not QueryExec.cmp_single(jp, obj):
                return EComp.DIFFERENT
            stack_ind += 1
            jpath_ind += 1

        if jpath_ind == len(jpath) and stack_ind == len(stack):
            return EComp.EQUAL
        elif jpath_ind < len(jpath) and stack_ind == len(stack):
            return EComp.NOT_YET
        return EComp.DIFFERENT


class TreeNode:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []
        self.types = []

    def __eq__(self, other):
        if isinstance(other, str):
            return self.name == other
        elif isinstance(other, TreeNode):
            return self.name == other.name
        return False

    def get_or_create_child(self, name):
        try:
            i = self.children.index(name)
            return self.children[i]
        except ValueError:
            n = TreeNode(name, self)
            self.children.append(n)
            return n

    def add_type(self, type_name):
        if type_name not in self.types:
            self.types.append(type_name)


class SchemaTree:
    indent = '  '

    def __init__(self, data):
        self.root = TreeNode('$')
        self.inspect(data, self.root)
        self.str_repr = []

    def inspect(self, json_data, tree_node):
        if isinstance(json_data, dict):
            tree_node.add_type('Object')
            for k, v in json_data.items():
                n = tree_node.get_or_create_child(k)
                self.inspect(v, n)
        elif isinstance(json_data, list):
            tree_node.add_type('Array')
            for v in json_data:
                n = tree_node.get_or_create_child('Array of')
                self.inspect(v, n)
        else:
            tree_node.add_type(str(type(json_data).__name__))

    def to_str_rec(self, node, indent=''):
        msg = f'{indent}{node.name}: {node.types}'
        msg = msg.replace('[', '<').replace(']', '>').replace("'", '')
        msg += ' {' if SchemaTree.is_object(node) else ''
        msg += ' [' if SchemaTree.is_array(node) else ''
        self.str_repr.append(msg)
        for n in node.children:
            self.to_str_rec(n, indent + self.indent)
        if SchemaTree.is_object(node):
            self.str_repr.append(indent + '}')
        if SchemaTree.is_array(node):
            self.str_repr.append(indent + ']')

    def is_object(n):
        return len(n.types) == 1 and n.types[0] == 'Object'

    def is_array(n):
        return 'Array' in n.types

    def __str__(self):
        self.str_repr = []
        self.to_str_rec(self.root)
        return "\n".join(self.str_repr)


class InspectorCtx:
    def __init__(self):
        self.glob_data = None
        self.curr_data = None

    def check_preconditions(self):
        if self.glob_data is None:
            return (False, 'Load json first')
        else:
            return (True, '')

    def load_from_str(self, s):
        self.glob_data = json.loads(s)
        self.curr_data = self.glob_data

    def load(self, json_file):
        try:
            with open(json_file) as f:
                self.glob_data = json.load(f)
                self.curr_data = self.glob_data
                return (True, '')
        except FileNotFoundError:
            return (False, f'File "{json_file}" not found')

    def schema_tree(self):
        ok, msg = self.check_preconditions()
        if not ok:
            return (False, msg)
        return (True, str(SchemaTree(self.curr_data)))

    def filter(self, jpath):
        parser = JSONPathParser(jpath)
        props, err = parser.parse()
        if err:
            return (False, err)
        exec = QueryExec(self.curr_data, props)
        ok, msg = exec.build_up()

        if ok:
            self.curr_data = msg
            return ok, None

        return ok, msg


class JInspectorShell(cmd.Cmd):
    intro = ('Welcome to the jispector shell. '
             'Type help or ? to list commands.\n')
    prompt = '(jinspect) '
    file = None

    def __init__(self):
        super().__init__()
        self.ctx = InspectorCtx()
        self.error_flag = False
        self.txt_out = None

    # ----- commands -----
    def do_quit(self, _):
        'Exit program'
        self.close()
        return True

    def do_load(self, arg):
        'Load json file'
        ok, msg = self.ctx.load(arg)
        if not ok:
            self.error(msg)

    def do_schema(self, arg):
        'Show schema of the loaded json'
        ok, msg = self.ctx.schema_tree()
        if ok:
            self.print(msg)
        else:
            self.error(msg)

    def do_filter(self, arg):
        'Filter data using JSONPath query'
        ok, msg = self.ctx.filter(arg)
        if ok:
            self.print(msg)
        else:
            self.error(msg)

    def do_log(self, arg):
        'Set log level: none, trace, verbose'
        if arg == 'none':
            set_log_lev(ELogLev.NONE)
        elif arg == 'trace':
            set_log_lev(ELogLev.TRACE)
        elif arg == 'verbose':
            set_log_lev(ELogLev.VERBOSE)

    def do_sh(self, arg):
        'Execute any shell command'
        code, out, timeout = call_bash_cmd(arg)
        if not code and not timeout:
            self.add_txt(out)
        else:
            print(f'Code={code}, timeout={timeout}')
            print(out)

    def do_grep(self, arg):
        'Print lines that contain given pattern'
        self.txt_out = '\n'.join([line for line in self.txt_out.splitlines()
                                  if arg in line])

    def do_save(self, arg):
        'Save current json data'
        with open(arg, 'w') as json_file:
            json.dump(self.ctx.curr_data, json_file, indent=2)

    def do_show_json(self, arg):
        'Print current json data'
        self.add_txt(str(self.ctx.curr_data))

    def precmd(self, line):
        cmds = line.split('|')
        self.txt_out = ''
        if len(cmds) > 1:
            self.error_flag = False
            for c in cmds[:-1]:
                self.onecmd(c)
                if self.error_flag:
                    self.error('Aborted due to errors')
                    return ''
            self.error_flag = False
            return cmds[-1]
        return line

    def postcmd(self, stop, line):
        # print(f'Postcmd: stop={stop}, line={line}')
        self.print(self.txt_out)
        return stop

    def close(self):
        if self.file:
            self.file.close()
            self.file = None

    def add_txt(self, msg):
        self.txt_out += msg
        self.txt_out += '\n'

    def error(self, msg, *args):
        self.error_flag = True
        print(msg, *args)

    def print(self, msg, *args):
        print(msg, *args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=PROGRAM_DESC)

    # Add positional arguments
    # parser.add_argument('input_file', help='Path to the input file')
    parser.add_argument('-s', '--script', type=str,
                        help='Script to be executed. Non-interactive mode')

    args = parser.parse_args()

    # input_file = args.input_file

    if args.script:
        cmd = args.script + '|quit'
        JInspectorShell().precmd(cmd)
    else:
        # interactive mode
        JInspectorShell().cmdloop()
