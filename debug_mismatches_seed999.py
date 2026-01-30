import ast
import re
import numpy as np
from pymilvus import connections, Collection
from milvus_fuzz_oracle import DataManager, HOST, PORT, COLLECTION_NAME

# Helpers for safe access
class NullObj:
    def __getitem__(self, key):
        return self
    def __getattr__(self, item):
        return self
    def __bool__(self):
        return False
    def __repr__(self):
        return "NULL"

class SafeDict(dict):
    def __getitem__(self, key):
        val = super().get(key, NullObj())
        if isinstance(val, dict):
            return SafeDict(val)
        if isinstance(val, list):
            return SafeList(val)
        if val is None:
            return NullObj()
        return val

class SafeList(list):
    def __getitem__(self, idx):
        try:
            val = super().__getitem__(idx)
        except Exception:
            return NullObj()
        if isinstance(val, dict):
            return SafeDict(val)
        if isinstance(val, list):
            return SafeList(val)
        if val is None:
            return NullObj()
        return val

def like(val, pattern):
    if val is None:
        return False
    val = str(val)
    # simple % wildcard matching
    regex = '^' + re.escape(pattern).replace('%', '.*').replace('_', '.') + '$'
    return re.match(regex, val) is not None

def array_contains(arr, target):
    if arr is None:
        return False
    if not isinstance(arr, (list, tuple)):
        return False
    return target in arr

def exists(val):
    return val is not None

def json_contains(val, target):
    if isinstance(val, list):
        return target in val
    return False

def safe_cmp(a, b, op):
    if isinstance(a, NullObj) or isinstance(b, NullObj):
        if op in ('eq','ne'):
            return False if op == 'eq' else True
        return False
    if op in ('eq','ne'):
        if op == 'eq':
            return a == b
        return a != b
    # For ordering comparisons, if any is None -> False
    if a is None or b is None:
        return False
    try:
        if op == 'lt':
            return a < b
        if op == 'le':
            return a <= b
        if op == 'gt':
            return a > b
        if op == 'ge':
            return a >= b
    except Exception:
        return False
    return False

class SafeCompareTransformer(ast.NodeTransformer):
    def visit_Compare(self, node: ast.Compare):
        self.generic_visit(node)
        left = node.left
        values = []
        cur_left = left
        for op, comp in zip(node.ops, node.comparators):
            if isinstance(op, ast.Lt): op_name = 'lt'
            elif isinstance(op, ast.LtE): op_name = 'le'
            elif isinstance(op, ast.Gt): op_name = 'gt'
            elif isinstance(op, ast.GtE): op_name = 'ge'
            elif isinstance(op, ast.NotEq): op_name = 'ne'
            else: op_name = 'eq'
            call = ast.Call(func=ast.Name(id='safe_cmp', ctx=ast.Load()), args=[cur_left, comp, ast.Constant(op_name)], keywords=[])
            values.append(call)
            cur_left = comp
        if len(values) == 1:
            return values[0]
        return ast.BoolOp(op=ast.And(), values=values)

# Preprocess expression into Python-friendly form

def to_python_expr(expr: str) -> str:
    expr = expr.replace('true', 'True').replace('false', 'False')
    expr = re.sub(r'\bis null\b', ' is None', expr)
    expr = re.sub(r'\bis not null\b', ' is not None', expr)
    expr = re.sub(r'(?P<var>[A-Za-z0-9_]+) like "([^"]*)"', r'like(\g<var>, "\2")', expr)
    return expr

def compile_expr(expr: str):
    py_expr = to_python_expr(expr)
    tree = ast.parse(py_expr, mode='eval')
    tree = SafeCompareTransformer().visit(tree)
    ast.fix_missing_locations(tree)
    code = compile(tree, '<expr>', 'eval')
    return code

def eval_row(expr_code, row_dict):
    ctx = {k: row_dict.get(k) for k in row_dict.keys()}
    # Wrap json and array fields to avoid KeyError/None
    if 'meta_json' in ctx:
        val = ctx['meta_json']
        if isinstance(val, dict):
            ctx['meta_json'] = SafeDict(val)
        elif val is None:
            ctx['meta_json'] = SafeDict()
    if 'tags_array' in ctx:
        val = ctx['tags_array']
        if isinstance(val, list):
            ctx['tags_array'] = SafeList(val)
        elif val is None:
            ctx['tags_array'] = SafeList()
    ctx.update({
        'array_contains': array_contains,
        'exists': exists,
        'json_contains': json_contains,
        'like': like,
        'safe_cmp': safe_cmp,
        'True': True,
        'False': False,
        'None': None,
    })
    return bool(eval(expr_code, {'__builtins__': None}, ctx))


def fetch_rows(ids):
    connections.connect("default", host=HOST, port=PORT)
    col = Collection(COLLECTION_NAME)
    return col.query(expr=f"id in {ids}", output_fields=["*"])

# Rebuild dataframe with seed 999
import random
import pandas as pd

random.seed(999)
np.random.seed(999)

DM = DataManager()
DM.generate_schema()
DM.generate_data()
DF = DM.df

cases = [
    {
        'name': 'Case1975',
        'expr': '(((((((((meta_json["price"] > 463 and meta_json["price"] < 625) and (meta_json["active"] == true and meta_json["color"] == "Red")) and (c15 > 1556.8050193874226 and c3 != -99998.46875)) or ((meta_json["history"][0] > 23 or c5 == true) and (c9 > 503.4998272499878 or c9 > 194.4876672970281))) or (c9 is null or ((c16 > 909.9641280072691 or c16 is not null) and meta_json["history"][0] > 34))) or ((((array_contains(tags_array, 90) or c4 > 5256.112660002518) or (meta_json["price"] == 203 and meta_json["config"]["version"] == 7)) and ((meta_json["price"] > 127 and meta_json["price"] < 252) and (c17 == true and c10 != "2XYozvhgTnPbYykNNSxleO"))) or ((c7 == true and (c5 == true and meta_json["config"]["version"] == 9)) or (c5 == true and (c0 < 1966.2588381906974 and meta_json["config"]["version"] == 6))))) or (((((c0 > 4129.654973903646 or c15 <= 3.3076436461084415) and (c13 > -180175 or c5 == true)) or ((c15 == 564.6191103698479 and c16 == 3354.885359219524) or ((meta_json["active"] == true and meta_json["color"] == "Blue") or array_contains(tags_array, 72)))) and (((c7 == true or c11 >= -62527) and (c4 < 4288.414515163085 or c14 == 4988.054496278829)) or c8 like "4%")) and ((((c14 != 4888.954894598184 or c0 < 5187.787592036658) and (c10 > "j" or c3 >= 3698.6741686416713)) or ((meta_json["active"] == true and meta_json["color"] == "Blue") and (c2 >= 46635 or c3 >= 4924.124373049356))) or (((meta_json is null or c1 == false) or (c4 is null or c5 == false)) or ((c13 == -21131 or meta_json["config"]["version"] == 6) and (c3 > 3827.306124490623 and c10 like "j%")))))) or ((((c7 == false and (((meta_json["active"] == true and meta_json["color"] == "Blue") or meta_json["config"]["version"] == 2) or (meta_json["active"] == true and meta_json["color"] == "Blue"))) and (((c2 != -9423 and c16 < 5351.945732881858) or c4 > 2082.7298672426687) and ((c14 <= 4721.14537113771 and (meta_json["price"] > 155 and meta_json["price"] < 298)) and (c0 == 3098.676090293314 and (meta_json["active"] == true and meta_json["color"] == "Blue"))))) and ((((meta_json["config"]["version"] == 7 or (meta_json["price"] > 121 and meta_json["price"] < 277)) or (meta_json["config"]["version"] == 1 and c17 == false)) and (meta_json["history"][0] > 68 and (meta_json["history"][0] > 38 or c6 == 64847))) or (((c0 == 4617.993714803526 or meta_json["config"]["version"] == 4) or (c10 != "e" or (meta_json["price"] > 346 and meta_json["price"] < 408))) and ((meta_json["history"][0] > 74 or meta_json["history"][0] > 80) or (c4 <= 3665.194649233507 and c13 < 73597))))) and ((((c14 is not null and (c0 == 5362.092494258532 and c8 like "V%")) or (((meta_json["active"] == true and meta_json["color"] == "Red") and tags_array is not null) and (c6 != 180150 or c2 >= -40983))) and (((c15 >= 912.1039030666655 and meta_json["history"][0] > 78) and (c11 is not null or c14 is not null)) or ((c7 == true or c12 == true) or (tags_array is null and c12 == false)))) and ((((array_contains(tags_array, 9) and meta_json["history"][0] > 48) and c16 > 1177.6205306555007) and ((meta_json["config"]["version"] == 7 and c11 >= 65534) and (c0 is null or c14 == 3492.8015446493064))) and (c9 != 4187.526573748907 or ((c5 == false or c13 <= 378) or (c9 >= 2321.102311719105 and c12 == false))))))) and (c5 == false or (meta_json["active"] == true and meta_json["color"] == "Blue")))',
        'extra': [1784,1452,3332]
    },
    {
        'name': 'Case4663',
        'expr': '(((meta_json["active"] == true and meta_json["color"] == "Blue") or ((c16 >= 2538.641880850278 or ((meta_json["active"] == true and meta_json["color"] == "Blue") and (meta_json["price"] > 107 and meta_json["price"] < 261))) and ((c0 > 2146.802199098024 and c2 >= 36295) and (c12 == false and (meta_json["active"] == true and meta_json["color"] == "Blue"))))) and (c17 == false or (((c10 < "IHSVh" and c5 == false) or (c0 <= 4.787315444874056 or (meta_json["price"] > 293 and meta_json["price"] < 453))) and ((meta_json["config"]["version"] == 3 or c17 is null) and (c11 > -51460 and c9 != 105382.9375)))))',
        'extra': [4526]
    },
    {
        'name': 'Case5328',
        'expr': '(((((((meta_json["active"] == true and meta_json["color"] == "Red") and c0 != 1957.3169282528372) and (((array_contains(tags_array, 66285) or c1 == false) and (c10 like "U%" or (meta_json["price"] > 266 and meta_json["price"] < 451))) or ((c3 is null or meta_json["history"][0] > 46) and ((meta_json["price"] > 387 and meta_json["price"] < 539) or (meta_json["price"] > 261 and meta_json["price"] < 424))))) and ((((c4 == 219.92917147022987 and c10 == "z") or (c11 is null or array_contains(tags_array, 38))) or ((c2 >= 64254 and c0 == -99998.4296875) or ((meta_json["active"] == true and meta_json["color"] == "Blue") and (meta_json["active"] == true and meta_json["color"] == "Blue")))) or (((meta_json["history"][0] > 21 and c14 != 1473.0224856780214) and (c10 == "cwjDB%@qghW3%U$U" and c0 is null)) and ((meta_json is null and c5 == true) and (c3 > 4426.567247916293 and c0 <= 3321.0857528722418))))) or (((((c1 == false or c4 >= 3433.630964065678) or (c2 == 71184 or c15 == 5032.783607095196)) or meta_json["config"]["version"] == 9) and (c3 < 4655.258895799657 or ((c9 <= 1973.6866725585976 or (meta_json["active"] == true and meta_json["color"] == "Red")) and ((meta_json["active"] == true and meta_json["color"] == "Blue") or meta_json["config"]["version"] == 5)))) and c16 < 2722.2390468467383)) and ((((exists(meta_json["price"]) or ((c17 == false or meta_json["config"]["version"] == 4) and (meta_json["history"][0] > 59 or meta_json["config"]["version"] == 3))) and array_contains(tags_array, 1)) and (((c11 > 180200 or c5 == true) and ((c5 == false and c1 is null) and (c11 >= -180073 or (meta_json["active"] == true and meta_json["color"] == "Blue"))))) and c17 == true)) and c0 < 5138.408252055341)) or (c1 == false or (((c4 is not null or ((((meta_json["price"] > 349 and meta_json["price"] < 464) or c13 >= 6544) and c7 == false) or (c2 > 56203 and c17 == true))) and ((((meta_json["config"]["version"] == 7 or c6 != 54118) and meta_json["history"][0] > 60) and c15 < 2857.5424844550816) or (((c5 == true or (meta_json["active"] == true and meta_json["color"] == "Blue")) or (c7 == true and (meta_json["price"] > 484 and meta_json["price"] < 628))) and (c3 >= 2953.9840606239923 or (meta_json["history"][0] > 57 and c0 <= 4.787315444874056))))) or meta_json["config"]["version"] == 4)))',
        'extra': [4204,3553,2236]
    },
]


def main():
    for case in cases:
        expr = case['expr']
        code = compile_expr(expr)
        df_mask = DF.apply(lambda r: eval_row(code, r.to_dict()), axis=1)
        expected = int(df_mask.sum())
        print(f"\n=== {case['name']} ===")
        print(f"Expected (recomputed) count: {expected}")
        # Check extras
        rows = DF[DF['id'].isin(case['extra'])]
        for _, row in rows.iterrows():
            res = eval_row(code, row.to_dict())
            print(f"ID {row['id']} satisfies? {res}")
    
if __name__ == "__main__":
    main()
