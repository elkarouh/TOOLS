from __future__ import annotations

def hello(e:list[str]):
    print("eee")

def print_it(e: list[str]):
    if e >= 4:
        import os
        if os.path.exists("dsd"):
            print_it(e)

class DFS_Parser: # one class per production of the grammar
    def get_next_state(self, current_state, remaining_tokens):
        print("please override 1")
        if TERMINAL:
            yield AST_TERMINAL
        else: # NON-TERMINAL
            ast,remaining=parse_non_terminal(tokens)
            if ast:
                yield ast,remaining
    def is_final_state(self, state):
        print("please override 3")
        return False
    def parse(self, tokens):
        """return (ast, remaining tokens) in case of success and (None, original tokens) in case of failure"""
        fringe = [[]]
        while fringe:
            path = fringe.pop()  # LIFO
            current_state = path[-1][0] if path else None
            remaining_tokens = path[-1][2] if path else tokens
            if self.is_final_state(current_state):
                return path # a list of (state,token, remaining_tokens)
            next_token=remaining_tokens[0]
            for next_state in self.get_next_state(current_state, next_token):
                fringe.append(path+[(next_state,next_token, remaining_tokens[1:])])
        return None, tokens

Grammar="""Exp -> Exp [ + | - | * | / ] Exp | ( Exp ) | number"""
SUBJECT, VERB, COMPLEMENT= range(3)
class My_Parser(DFS_Parser):
    def get_next_state(self, current_state, next_token):
        if current_state is None:
            if next_token in ("I","you","he","we","they"):
                yield SUBJECT
        elif current_state == SUBJECT:
            if next_token in ("want","eat","drink"):
                yield VERB
        elif current_state == VERB:
            if next_token in ("water","bread"):
                yield COMPLEMENT
        elif current_state == COMPLEMENT:
            yield None
    def is_final_state(self, state):
        return state==COMPLEMENT

a=My_Parser()
path=a.parse(["I","want","water"])
print(f"{path=}")
