"""Grammars for graph embedding symbolic programs."""


GRAMMAR_BASE = r"""
<embed>  ::= x=<expr>;y=<expr>
<expr>   ::= <term> | <term> + <expr> | <term> - <expr>
<term>   ::= <factor> | <factor> * <term>
<factor> ::= <atom>
           | tanh(<expr>)
           | log1p(abs(<expr>))
           | sqrtabs(<expr>)
           | safe_div(<expr>,<expr>)
           | (<expr>)
           | -<factor>
<atom>   ::= <var> | <const>
<var>    ::= f0 | f1 | f2 | f3 | f4 | m0 | m1 | m2 | m3 | m4
<const>  ::= 0.25 | 0.5 | 1.0 | 2.0 | 3.0
"""


GRAMMAR_LITE = r"""
<embed>  ::= x=<expr>;y=<expr>
<expr>   ::= <term> | <term> + <expr> | <term> - <expr>
<term>   ::= <atom> | <atom> * <term> | tanh(<expr>)
<atom>   ::= <var> | <const> | safe_div(<var>,<atom>)
<var>    ::= f0 | f1 | f2 | m0 | m1 | m2
<const>  ::= 0.25 | 0.5 | 1.0 | 2.0
"""


GRAMMARS = {
    "base": GRAMMAR_BASE,
    "lite": GRAMMAR_LITE,
}

DEFAULT_GRAMMAR = "base"
