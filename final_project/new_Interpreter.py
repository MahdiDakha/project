############################################
# IMPORTS
############################################
from strings_with_arrows import *


############################################
# ERROR
############################################
class Error:
    def __init__(self, pos_start, pos_end, message, details):
        self.pos_start = pos_start
        self.pos_end = pos_end
        self.message = message
        self.details = details

    def as_string(self):
        result = f'{self.message}: {self.details}\n'
        result += f'File {self.pos_start.fn},line{self.pos_start.ln + 1}'
        result += '\n\n' + string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)
        return result


class IllegalCharError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, 'Illegal Character', details)


class RuntimeError(Error):
    def __init__(self, pos_start, pos_end, details, context):
        super().__init__(pos_start, pos_end, 'RuntimeError', details)
        self.context = context

    def as_string(self):
        result = self.generate_traceback()
        result += f'{self.message}:{self.details}'
        result += '\n\n' + string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)
        return result

    def generate_traceback(self):
        result = ''
        pos = self.pos_start
        ctx = self.context
        while ctx:
            result = f' File {pos.fn},line{str(pos.ln + 1)},in{ctx.dis_name}\n' + result
            pos = ctx.par_entery_pos
            ctx = ctx.par
        return 'Traceback (most recent call last):\n' + result


class InvalidSyntaxError(Error):
    def __init__(self, pos_start, pos_end, details=''):
        super().__init__(pos_start, pos_end, 'Invalid Syntax', details)


############################################
# POSITION
############################################
class Position:
    def __init__(self, idx, ln, col, fn, ftxt):
        self.idx = idx
        self.ln = ln
        self.col = col
        self.fn = fn
        self.ftxt = ftxt

    def advance(self, current_char=None):
        self.idx += 1
        self.col += 1

        if current_char == '\n':
            self.ln += 1
            self.col = 0

        return self

    def copy(self):
        return Position(self.idx, self.ln, self.col, self.fn, self.ftxt)


############################################
# TOKENS
############################################
# Data Types
INT = 'INT'
BOOL = 'BOOL'

# Arithmetic Operations
PLUS = 'PLUS'
MINUS = 'MINUS'
MUL = 'MUL'
DIV = 'DIV'
MOD = 'MOD'

# Boolean Operations
EQUALS = "E"
AND = 'AND'
OR = 'OR'
NOT = 'NOT'
EQ = 'EQ'
NEQ = 'NEQ'
GT = 'GT'
LT = 'LT'
GTE = 'GTE'
LTE = 'LTE'

# Function-related
ID = 'ID'
FUNC = "FUNC"
LPAREN = 'LPAREN'
RPAREN = 'RPAREN'
ARROW = 'ARROW'
COMMA = 'COMMA'

# Other
COMMENT = 'COMMENT'
EOF = 'EOF'
KEYWORD = 'KEYWORD'
DIGITS = '0123456789'

KEYWORDS = [
    'and',
    'or',
    'not',
    'if',
    'do',
    'elif',
    'else',
    'func',
    'lambda'
]


############################################
class Token:
    def __init__(self, ttype, value=None, pos_start=None, pos_end=None):
        self.type = ttype
        self.value = value

        if pos_start:
            self.pos_start = pos_start.copy()
            self.pos_end = pos_start.copy()
            self.pos_end.advance()

        if pos_end:
            self.pos_end = pos_end.copy()

    def matches(self, ttype, value):
        return self.type == ttype and self.value == value

    def __repr__(self):
        if self.value is not None:
            return f'Token({self.type}, {self.value})'

        return f'Token({self.type})'


############################################
# LEXER
############################################

class Lexer:
    def __init__(self, fn, text):
        self.fn = fn
        self.text = text
        self.pos = Position(-1, 0, -1, fn, text)
        self.current_char = None
        self.advance()

    def advance(self):
        self.pos.advance(self.current_char)
        self.current_char = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None

    def tokenize(self):
        tokens = []

        while self.current_char is not None:
            if self.current_char in ' \t':
                self.advance()
            elif self.current_char == '#':
                self.make_comment()
            elif self.current_char in DIGITS:
                token, error = self.make_number()
                if error:
                    return [], IllegalCharError(self.pos.copy(), self.pos, error)
                if token:
                    tokens.append(token)
            elif self.current_char.isalpha():
                tokens.append(self.make_identifier())
            elif self.current_char == '+':
                tokens.append(Token(PLUS, pos_start=self.pos))
                self.advance()
            elif self.current_char == '-':
                self.advance()
                if self.current_char == '>':
                    tokens.append(Token(ARROW, pos_start=self.pos))
                    self.advance()
                else:
                    tokens.append(Token(MINUS, pos_start=self.pos))
                    self.advance()
            elif self.current_char == '*':
                tokens.append(Token(MUL, pos_start=self.pos))
                self.advance()
            elif self.current_char == '/':
                tokens.append(Token(DIV, pos_start=self.pos))
                self.advance()
            elif self.current_char == '%':
                tokens.append(Token(MOD, pos_start=self.pos))
                self.advance()
            elif self.current_char == ',':
                tokens.append(Token(COMMA, pos_start=self.pos))
                self.advance()
            elif self.current_char == '(':
                tokens.append(Token(LPAREN, pos_start=self.pos))
                self.advance()
            elif self.current_char == ')':
                tokens.append(Token(RPAREN, pos_start=self.pos))
                self.advance()
            elif self.current_char in ['>', '<', '!', '=']:
                tokens, error = self.handle_comparison_operators(tokens)
                if error:
                    return [], error

            else:
                char = self.current_char
                pos_start = self.pos.copy()
                self.advance()
                return [], IllegalCharError(pos_start, self.pos, f"({char})")

        tokens.append(Token(EOF, pos_start=self.pos))  # Append EOF token after the loop
        return tokens, None

    def make_comment(self):
        while self.current_char is not None and self.current_char != '\n':
            self.advance()
            if self.current_char == '\n':
                break
        self.advance()

    def make_number(self):
        number_str = ''
        pos_start = self.pos.copy()

        while self.current_char is not None and self.current_char in DIGITS:
            number_str += self.current_char
            self.advance()

        if self.current_char == '.':
            # Handle the error: floating-point numbers are not allowed
            error_message = f"Unexpected '.' in number at position {self.pos.idx}"
            return None, error_message

        return Token(INT, int(number_str), pos_start, self.pos), None

    def make_identifier(self):
        identifier_str = ''
        pos_start = self.pos.copy()

        while self.current_char is not None and (self.current_char.isalnum() or self.current_char == '_'):
            identifier_str += self.current_char
            self.advance()

        token_type = KEYWORD if identifier_str in KEYWORDS else ID
        return Token(token_type, identifier_str, pos_start, self.pos)

    def handle_comparison_operators(self, tokens):
        pos_start = self.pos.copy()
        char = self.current_char
        self.advance()

        if char == '<':
            if self.current_char == '=':
                tokens.append(Token(LTE, pos_start=pos_start, pos_end=self.pos))
                self.advance()
            elif self.current_char in ['>', '*', '/', '%', '+', '-']:
                return [], IllegalCharError(self.pos.copy(), self.pos, f'Unexpected character: {self.current_char}')
            else:
                tokens.append(Token(LT, pos_start=pos_start, pos_end=self.pos))
        elif char == '>':
            if self.current_char == '=':
                tokens.append(Token(GTE, pos_start=pos_start, pos_end=self.pos))
                self.advance()
            elif self.current_char in ['<', '*', '/', '%', '+', '-']:
                return [], IllegalCharError(self.pos.copy(), self.pos, f'Unexpected character: {self.current_char}')
            else:
                tokens.append(Token(GT, pos_start=pos_start, pos_end=self.pos))
        elif char == '!':
            if self.current_char == '=':
                tokens.append(Token(NEQ, pos_start=pos_start, pos_end=self.pos))
                self.advance()
            elif self.current_char in ['<', '>', '*', '/', '%', '+', '-']:
                return [], IllegalCharError(self.pos.copy(), self.pos, f'Unexpected character: {self.current_char}')
            else:
                tokens.append(Token(NOT, pos_start=pos_start, pos_end=self.pos))
        elif char == '=':
            if self.current_char == '=':
                tokens.append(Token(EQ, pos_start=pos_start, pos_end=self.pos))
                self.advance()
            elif self.current_char in ['<', '>', '*', '/', '%', '+', '-']:
                return [], IllegalCharError(self.pos.copy(), self.pos, f'Unexpected character: {self.current_char}')
            else:
                tokens.append(Token(EQUALS, pos_start=pos_start, pos_end=self.pos))
                self.advance()

        return tokens, None


############################################
# NODES
############################################
class NumberNode:
    def __init__(self, tok):
        self.tok = tok
        self.pos_start = self.tok.pos_start
        self.pos_end = self.tok.pos_end

    def __repr__(self):
        return f'{self.tok}'


class LambdaNode:
    def __init__(self, param_names, body):
        self.param_names = param_names
        self.body = body

    def __repr__(self):
        return f"(lambda ({', '.join([p.value for p in self.param_names])}) -> {self.body})"


class BooleanNode:
    def __init__(self, tok):
        self.tok = tok
        self.pos_start = self.tok.pos_start
        self.pos_end = self.tok.pos_end

    def __repr__(self):
        return f'{self.tok}'


class AssignNode:
    def __init__(self, var_name_tok, expr_node):
        self.var_name_tok = var_name_tok
        self.expr_node = expr_node

    def __repr__(self):
        return f"(Assign {self.var_name_tok}, {self.expr_node})"


class BinOpNode:
    def __init__(self, left_node, op_tok, right_node):
        self.left_node = left_node
        self.op_tok = op_tok
        self.right_node = right_node

        self.pos_start = self.left_node.pos_start
        self.pos_end = self.right_node.pos_end

    def __repr__(self):
        return f'{self.left_node}, {self.op_tok}, {self.right_node}'


class UnaryOpNode:
    def __init__(self, op_tok, node):
        self.op_tok = op_tok
        self.node = node
        self.pos_start = self.node.pos_start
        self.pos_end = self.node.pos_end

    def __repr__(self):
        return f'{self.op_tok} {self.node}'


class IfNode:
    def __init__(self, cases, else_case):
        self.cases = cases
        self.else_case = else_case
        self.pos_start = self.cases[0][0].pos_start
        self.pos_end = (self.else_case or self.cases[len(self.cases) - 1][0].pos_end)

    def __repr__(self):
        return f'{self.cases}, {self.else_case}'


class FunctionNode:
    def __init__(self, var_name_tok, arg_name_toks, body_node):
        self.var_name_tok = var_name_tok
        self.arg_name_toks = arg_name_toks
        self.body_node = body_node

        if self.var_name_tok:
            self.pos_start = self.var_name_tok.pos_start
        elif len(self.arg_name_toks) > 0:
            self.pos_start = self.arg_name_toks[0].pos_start
        else:
            self.pos_start = self.body_node.pos_start

        self.pos_end = self.body_node.pos_end

    def __repr__(self):
        return f'Function({self.var_name_tok}, {self.arg_name_toks}, {self.body_node})'


class LambdaNode:
    def __init__(self, arg_name_toks, body_node):
        self.arg_name_toks = arg_name_toks
        self.body_node = body_node
        if len(self.arg_name_toks) > 0:
            self.pos_start = self.arg_name_toks[0].pos_start

        else:
            self.pos_start = self.body_node.pos_start
        self.pos_end = self.body_node.pos_end

    def __repr__(self):
        arg_names = ', '.join([str(arg) for arg in self.arg_name_toks])
        return f'(lambda {arg_names}  -> {self.body_node})'


class VarAccessNode:
    def __init__(self, var_name_tok):
        self.var_name_tok = var_name_tok

        self.pos_start = self.var_name_tok.pos_start
        self.pos_end = self.var_name_tok.pos_end

    def __repr__(self):
        return f'VarAccess({self.var_name_tok})'


class FunctionCallNode:
    def __init__(self, node_to_call, arg_nodes):
        self.node_to_call = node_to_call
        self.arg_nodes = arg_nodes
        self.pos_start = self.node_to_call.pos_start

        if len(self.arg_nodes) > 0:
            self.pos_end = self.arg_nodes[len(self.arg_nodes) - 1].pos_end
        else:
            self.pos_end = self.node_to_call.pos_end

    def __repr__(self):
        return f'FunctionCall({self.node_to_call}, {self.arg_nodes})'


class LambdaCallNode:
    def __init__(self, lambda_node, arg_nodes):
        self.lambda_node = lambda_node
        self.arg_nodes = arg_nodes
        if len(self.arg_nodes) > 0:
            self.pos_end = self.arg_nodes[len(self.arg_nodes) - 1].pos_end
        else:
            self.pos_end = self.node_to_call.pos_end


############################################
# PARSER RESULT
############################################
class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None

    def register(self, res):
        if isinstance(res, ParseResult):
            if res.error: self.error = res.error
            return res.node

        return res

    def success(self, node):
        self.node = node
        return self

    def failure(self, error):
        self.error = error
        return self


############################################
# PARSER
############################################
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.token_idx = -1
        self.advance()

    def advance(self):
        self.token_idx += 1
        if self.token_idx < len(self.tokens):
            self.current_tok = self.tokens[self.token_idx]
        return self.current_tok

    def parse(self):
        res = self.expr()
        if not res.error and self.current_tok.type != EOF:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected '+', '-', '*', '/', '==', '!=', '<', '>', <=', '>=', 'and' or 'or'"

            ))
        return res

    #########################################

    def expr(self):
        res = ParseResult()

        if self.current_tok.type == ID and self.tokens[self.token_idx + 1].type == EQUALS:
            return self.assignment()

        node = res.register(self.bin_op(self.comp_expr, ((KEYWORD, 'and'), (KEYWORD, 'or'))))

        if res.error:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected , 'if', 'func', int, identifier, '+', '-', '(' or 'not'"
            ))

        return res.success(node)

    def assignment(self):
        res = ParseResult()
        var_name_tok = self.current_tok
        self.advance()
        self.advance()
        if not self.current_tok.matches(KEYWORD, 'lambda'):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected 'lambda'"
            ))

        expr_node = res.register(self.expr())
        if res.error:
            return res

        return res.success(AssignNode(var_name_tok, expr_node))

    def comp_expr(self):
        res = ParseResult()

        if self.current_tok.matches(KEYWORD, 'not'):
            op_tok = self.current_tok
            self.advance()

            node = res.register(self.comp_expr())
            if res.error:
                return res
            return res.success(UnaryOpNode(op_tok, node))
        node = res.register(self.bin_op(self.arith_expr, (EQ, NEQ, LT, GT, LTE, GTE)))

        if res.error:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end,
                                                  "Expected int, identifier, '+', '-', '(' or 'not'"))

        return res.success(node)

    def arith_expr(self):
        return self.bin_op(self.term, (PLUS, MINUS))

    def term(self):
        return self.bin_op(self.factor, (MUL, DIV, MOD))

    def factor(self):
        res = ParseResult()
        tok = self.current_tok

        if tok.type in (PLUS, MINUS):
            self.advance()

            factor = res.register(self.factor())
            if res.error:
                return res
            return res.success(UnaryOpNode(tok, factor))

        return self.call()

    def call(self):
        res = ParseResult()
        atom = res.register(self.atom())
        if res.error:
            return res

        if self.current_tok.type == LPAREN:
            self.advance()
            arg_nodes = []

            if self.current_tok.type == RPAREN:
                self.advance()
            else:
                arg_nodes.append(res.register(self.expr()))
                if res.error:
                    return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end,
                                                          "Expected ')', 'if' ,'func', int, , identifier, '+', '-', '(' or 'not'"
                                                          ))
                while self.current_tok.type == COMMA:
                    self.advance()
                    arg_nodes.append(res.register(self.expr()))
                    if res.error:
                        return res

                if self.current_tok.type != RPAREN:
                    return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end,
                                                          f"Expected ',' or ')'"
                                                          ))
                self.advance()

            return res.success(FunctionCallNode(atom, arg_nodes))

        return res.success(atom)

    def atom(self):
        res = ParseResult()
        tok = self.current_tok

        if tok.type == INT:
            self.advance()
            return res.success(NumberNode(tok))

        elif tok.type == ID:
            self.advance()
            return res.success(VarAccessNode(tok))
        elif tok.type == LPAREN:
            self.advance()
            expr = res.register(self.expr())
            if res.error:
                return res
            if self.current_tok.type == RPAREN:
                self.advance()
                return res.success(expr)
            else:
                return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end,
                                                      "Expected ')'"
                                                      ))
        elif tok.matches(KEYWORD, 'if'):
            if_expr = res.register(self.if_expr())
            if res.error:
                return res
            return res.success(if_expr)
        elif tok.matches(KEYWORD, 'func'):
            func_expr = res.register(self.func_expr())
            if res.error:
                return res
            return res.success(func_expr)
        elif tok.matches(KEYWORD, 'lambda'):
            lambda_expr = res.register(self.lambda_expr())
            if res.error:
                return res
            return res.success(lambda_expr)

        return res.failure(InvalidSyntaxError(
            tok.pos_start, tok.pos_end,
            "Expected int, identifier, '+', '-', '(', 'if','func'"
        ))

    def lambda_expr(self):
        res = ParseResult()

        if not self.current_tok.matches(KEYWORD, 'lambda'):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected 'lambda'"
            ))

        self.advance()

        param_names = []
        if self.current_tok.type == ID:
            param_names.append(self.current_tok)
            self.advance()

            while self.current_tok.type == COMMA:
                self.advance()

                if self.current_tok.type != ID:
                    return res.failure(InvalidSyntaxError(
                        self.current_tok.pos_start, self.current_tok.pos_end,
                        f"Expected identifier after ','"
                    ))

                param_names.append(self.current_tok)
                self.advance()

        if self.current_tok.type != ARROW:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected '->'"
            ))

        self.advance()

        # Parse the expression body of the lambda
        body = res.register(self.expr())
        if res.error:
            return res

        return res.success(LambdaNode(param_names, body))

    def if_expr(self):
        res = ParseResult()
        cases = []
        else_case = None

        if not self.current_tok.matches(KEYWORD, 'if'):
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end,
                                                  f"Expected 'if'"))

        self.advance()

        con = res.register(self.expr())
        if res.error:
            return res

        if not self.current_tok.matches(KEYWORD, 'do'):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected 'do'"
            ))

        self.advance()

        expr = res.register(self.expr())
        if res.error:
            return res
        cases.append((con, expr))

        while self.current_tok.matches(KEYWORD, 'elif'):
            self.advance()
            con = res.register(self.expr())
            if res.error:
                return res
            if not self.current_tok.matches(KEYWORD, 'do'):
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    f"Expected 'do'"
                ))

            self.advance()
            expr = res.register(self.expr())
            if res.error:
                return res
            cases.append((con, expr))

        if self.current_tok.matches(KEYWORD, 'else'):
            self.advance()
            else_case = res.register(self.expr())
            if res.error:
                return res

        return res.success(IfNode(cases, else_case))

    def func_expr(self):
        res = ParseResult()

        if not self.current_tok.matches(KEYWORD, 'func'):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected 'func'"
            ))

        self.advance()

        # Expect an identifier for the function name
        if self.current_tok.type != ID:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected function name identifier"
            ))
        var_name_tok = self.current_tok
        self.advance()
        if self.current_tok.type != LPAREN:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected '(' after function name"
            ))

        self.advance()
        arg_name_toks = []

        if self.current_tok.type == ID:
            arg_name_toks.append(self.current_tok)
            self.advance()
            while self.current_tok.type == COMMA:
                self.advance()

                if self.current_tok.type != ID:
                    return res.failure(InvalidSyntaxError(
                        self.current_tok.pos_start, self.current_tok.pos_end,
                        f"Expected identifier after ','"
                    ))
                arg_name_toks.append(self.current_tok)
                self.advance()
            if self.current_tok.type != RPAREN:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    f"Expected ',' or ')' after function parameters"
                ))
        else:
            if self.current_tok.type != RPAREN:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    f"Expected identifier or ')' in function parameters"
                ))

        self.advance()

        if self.current_tok.type != ARROW:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected '->' after function parameters"
            ))

        self.advance()

        # Parse the expression that the function returns
        body_node = res.register(self.expr())
        if res.error:
            return res
        return res.success(FunctionNode(var_name_tok, arg_name_toks, body_node))

    def bin_op(self, func_1, ops, func_2=None):
        if func_2 == None:
            func_2 = func_1

        res = ParseResult()
        left = res.register(func_1())
        if res.error: return res

        while self.current_tok.type in ops or (self.current_tok.type, self.current_tok.value) in ops:
            op_tok = self.current_tok
            self.advance()
            right = res.register(func_2())
            if res.error: return res
            left = BinOpNode(left, op_tok, right)

        return res.success(left)


#########################################
# VALUES
#########################################
class Value:
    def __init__(self):
        self.set_pos()
        self.set_context()

    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self

    def set_context(self, context=None):
        self.context = context
        return self

    def added_to(self, other):
        return None, self.illegal_operation(other)

    def subbed_by(self, other):
        return None, self.illegal_operation(other)

    def multed_by(self, other):
        return None, self.illegal_operation(other)

    def dived_by(self, other):
        return None, self.illegal_operation(other)

    def mod_by(self, other):
        return None, self.illegal_operation(other)

    def is_eq(self, other):
        return None, self.illegal_operation(other)

    def is_not_eq(self, other):
        return None, self.illegal_operation(other)

    def less_than(self, other):
        return None, self.illegal_operation(other)

    def greater_than(self, other):
        return None, self.illegal_operation(other)

    def less_than_eq(self, other):
        return None, self.illegal_operation(other)

    def greater_than_eq(self, other):
        return None, self.illegal_operation(other)

    def and_comparison(self, other):
        return None, self.illegal_operation(other)

    def or_comparison(self, other):
        return None, self.illegal_operation(other)

    def notted(self):
        return None, self.illegal_operation(other)

    def execute(self, args):
        return RTResult().failure(self.illegal_operation())

    def copy(self):
        raise Exception('No copy method defined')

    def is_true(self):
        return False

    def illegal_operation(self, other=None):
        if not other: other = self
        return RuntimeError(
            self.pos_start, other.pos_end,
            'Illegal operation',
            self.context
        )


#########################################
# NUMBER
#########################################
class Number(Value):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def added_to(self, other):
        if isinstance(other, Number):
            return Number(self.value + other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def subbed_by(self, other):
        if isinstance(other, Number):
            return Number(self.value - other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def multed_by(self, other):
        if isinstance(other, Number):
            return Number(self.value * other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def dived_by(self, other):
        if isinstance(other, Number):
            if other.value == 0:
                return None, RuntimeError(
                    other.pos_start, other.pos_end,
                    'Division by zero',
                    self.context
                )
            return Number(self.value / other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def mod_by(self, other):
        if isinstance(other, Number):
            return Number(self.value % other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def greater_than(self, other):
        if isinstance(other, Number):
            return Number(int(self.value > other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def greater_than_eq(self, other):
        if isinstance(other, Number):
            return Number(int(self.value >= other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def less_than(self, other):
        if isinstance(other, Number):
            return Number(int(self.value < other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def less_than_eq(self, other):
        if isinstance(other, Number):
            return Number(int(self.value <= other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def is_eq(self, other):
        if isinstance(other, Number):
            return Number(int(self.value == other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def is_not_eq(self, other):
        if isinstance(other, Number):
            return Number(int(self.value != other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def and_comparison(self, other):
        if isinstance(other, Number):
            return Number(int((self.value and other.value))).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def or_comparison(self, other):
        if isinstance(other, Number):
            return Number(int((self.value or other.value))).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def is_true(self):
        return self.value != 0

    def copy(self):
        copy = Number(self.value)
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.context)
        return copy

    def notted(self):
        return Number(1 if self.value == 0 else 0).set_context(self.context), None

    def __repr__(self):
        return str(self.value)


class Function(Value):
    def __init__(self, name, body, arg_names):
        super().__init__()
        self.name = name or "<lambda>"
        self.body = body
        self.arg_names = arg_names

    def execute(self, args):
        res = RTResult()
        inter = Interpreter()
        new_context = Context(self.name, self.context, self.pos_start)
        new_context.table = SymbolTable(new_context.par.table)

        if len(args) != len(self.arg_names):
            return res.failure(RuntimeError(
                self.pos_start, self.pos_end,
                f"{len(args)} arguments passed, but {len(self.arg_names)} expected",
                    self.context
            ))
        for n in range(len(args)):
            arg_name = self.arg_names[n]
            arg_value = args[n]
            arg_value.set_context(new_context)
            new_context.table.set(arg_name, arg_value)

        value = res.register(inter.visit(self.body, new_context))
        if res.error:
            return res

        return res.success(value)

    def copy(self):
        copy = Function(self.name, self.body, self.arg_names)
        copy.set_context(self.context)
        copy.set_pos(self.pos_start, self.pos_end)
        return copy

    def __repr__(self):
        return f"<func {self.name}>"


#######################################
# SYMBOL TABLE
#######################################
class SymbolTable:
    def __init__(self, parent=None):
        self.symbols = {}
        self.parent = parent

    def get(self, name):
        value = self.symbols.get(name, None)
        if value == None and self.parent:
            return self.parent.get(name)
        return value

    def set(self, name, value):
        self.symbols[name] = value

    def __contains__(self, name):
        return name in self.symbols


#########################################
# CONTEXT
#########################################
class Context:
    def __init__(self, dis_name, par=None, par_entery_pos=None):
        self.dis_name = dis_name
        self.par = par
        self.par_entery_pos = par_entery_pos
        self.table = None


#########################################
# INTERPRETER
#########################################
class Interpreter:
    def visit(self, node, context):
        method_name = f'visit_{type(node).__name__}'
        method = getattr(self, method_name, self.no_visit_method)
        return method(node, context)

    def no_visit_method(self, node, context):
        raise Exception(f'No visit_{type(node).__name__} method defined')

    ########################################
    def visit_NumberNode(self, node, context):
        return RTResult().success(Number(node.tok.value).set_context(context).set_pos(node.pos_start, node.pos_end))

    def visit_VarAccessNode(self, node, context):
        res = RTResult()
        var_name = node.var_name_tok.value
        value = context.table.get(var_name)

        if not value:
            return res.failure(RuntimeError(
                node.pos_start, node.pos_end,
                f"'{var_name}' is not defined",
                context
            ))

        value = value.copy().set_pos(node.pos_start, node.pos_end)
        return res.success(value)

    def visit_AssignNode(self, node, context):
        res = RTResult()
        value = res.register(self.visit(node.expr_node, context))
        if res.error:
            return res

        if isinstance(value, Error):
            return res.failure(RuntimeError(
                node.pos_start, node.pos_end,
                'Invalid assignment value',
                context
            ))

        context.table.set(node.var_name_tok.value, value)
        return res.success(value)

    def visit_IfNode(self, node, context):
        res = RTResult()

        for cond, expr in node.cases:
            cond_value = res.register(self.visit(cond, context))
            if res.error:
                return res

            if cond_value.is_true():
                expr_value = res.register(self.visit(expr, context))
                if res.error:
                    return res
                return res.success(expr_value)

        if node.else_case:
            el_value = res.register(self.visit(node.else_case, context))
            if res.error:
                return res
            return res.success(el_value)
        return res.success(None)

    def visit_FunctionNode(self, node, context):
        res = RTResult()
        func_name = node.var_name_tok.value if node.var_name_tok else None
        body = node.body_node
        arg_names = [arg_name.value for arg_name in node.arg_name_toks]
        func_value = Function(func_name, body, arg_names).set_context(context).set_pos(node.pos_start, node.pos_end)

        if node.var_name_tok:
            context.table.set(func_name, func_value)
        return res.success(func_value)

    def visit_LambdaNode(self, node, context):
        res = RTResult()
        func_name = None
        body = node.body_node
        arg_names = [arg_name.value for arg_name in node.arg_name_toks]
        func_value = Function(func_name, body, arg_names).set_context(context).set_pos(node.pos_start, node.pos_end)

        return res.success(func_value)

    def visit_FunctionCallNode(self, node, context):
        res = RTResult()
        args = []

        func_to_call = res.register(self.visit(node.node_to_call, context))
        if res.error:
            return res

        func_to_call = func_to_call.copy().set_pos(node.pos_start, node.pos_end)

        # Evaluate the arguments
        for arg_node in node.arg_nodes:
            args.append(res.register(self.visit(arg_node, context)))
            if res.error:
                return res

        if not isinstance(func_to_call, Function):
            return res.failure(RuntimeError(
                node.pos_start, node.pos_end,
                'Attempted to call a non-function',
                context
            ))
        result = res.register(func_to_call.execute(args))
        if res.error:
            return res

        return res.success(result)

    def visit_BinOpNode(self, node, context):
        res = RTResult()
        left = res.register(self.visit(node.left_node, context))
        if res.error:
            return res
        right = res.register(self.visit(node.right_node, context))
        if res.error:
            return res

        if node.op_tok.type == PLUS:
            result, error = left.added_to(right)
        elif node.op_tok.type == MINUS:
            result, error = left.subbed_by(right)
        elif node.op_tok.type == MUL:
            result, error = left.multed_by(right)
        elif node.op_tok.type == DIV:
            result, error = left.dived_by(right)
        elif node.op_tok.type == MOD:
            result, error = left.mod_by(right)
        elif node.op_tok.type == GT:
            result, error = left.greater_than(right)
        elif node.op_tok.type == LT:
            result, error = left.less_than(right)
        elif node.op_tok.type == GTE:
            result, error = left.greater_than_eq(right)
        elif node.op_tok.type == LTE:
            result, error = left.less_than_eq(right)
        elif node.op_tok.type == EQ:
            result, error = left.is_eq(right)
        elif node.op_tok.type == NEQ:
            result, error = left.is_not_eq(right)
        elif node.op_tok.type == AND:
            result, error = left.and_comparison(right)
        elif node.op_tok.type == OR:
            result, error = left.or_comparison(right)
        elif node.op_tok.matches(KEYWORD, 'and'):
            result, error = left.and_comparison(right)
        elif node.op_tok.matches(KEYWORD, 'or'):
            result, error = left.or_comparison(right)
        if error:
            return res.failure(error)
        else:
            return res.success(result.set_pos(node.pos_start, node.pos_end))

    def visit_UnaryOpNode(self, node, context):
        res = RTResult()
        number = res.register(self.visit(node.node, context))
        if res.error: return res

        error = None

        if node.op_tok.type == MINUS:
            number = number.multed_by(Number(-1))
        elif node.op_tok.matches(KEYWORD, 'not'):

            number, error = number.notted()
        if error:
            return res.failure(error)
        else:
            return res.success(number.set_pos(node.pos_start, node.pos_end))


#######################################
# RUNTIME RESULT
#######################################

class RTResult:
    def __init__(self):
        self.value = None
        self.error = None

    def register(self, res):
        self.error = res.error
        return res.value

    def success(self, value):
        self.value = value
        return self

    def failure(self, error):
        self.error = error
        return self


############################################
# RUN
############################################


global_symbol_table = SymbolTable()
global_symbol_table.set("false", Number(0))
global_symbol_table.set("true", Number(1))
def repl(file_name=None):
    print("Welcome to the functional programming interpreter!")
    print("Type 'exit' to quit.")

    if file_name:
        with open(file_name, 'r') as file:
            for line in file:
                text = line.strip()
                if not text or text.lower() == 'exit':
                    continue
                try:
                    print(f'>>> {text}')
                    # Tokenize
                    lexer = Lexer('<stdin>', text)
                    tokens, error = lexer.tokenize()
                    if error:
                        print(error.as_string())
                        continue

                    # Parse
                    parser = Parser(tokens)
                    tree = parser.parse()
                    if tree.error:
                        print(tree.error.as_string())
                        continue

                    # Interpret
                    inter = Interpreter()
                    context = Context('<program>')
                    context.table = global_symbol_table
                    res = inter.visit(tree.node, context)
                    if res.error:
                        print(res.error.as_string())
                        continue

                    # Print result
                    print(res.value)

                except Exception as e:
                    print(f"An error occurred: {e}")

def run_repl_with_file():
    repl(file_name='test_cases')

# Run the REPL with the test cases from the file
def make_test():
    run_repl_with_file()

while True:
    print("Enter 1 to run the repl and 2 for the test and to stop type exit")
    x = input()
    if x == '1':
        repl()
    elif x == '2':
        make_test()
    elif x == 'exit':
        break
    else:
        continue


