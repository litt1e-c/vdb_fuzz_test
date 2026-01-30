import time
from pymilvus import (
    connections, utility, FieldSchema, CollectionSchema, DataType, Collection
)

# --- 配置 ---
HOST = "127.0.0.1"
PORT = "19531"
COLLECTION_NAME = "debug_round_27_recursive"

# --- 数据 (Target ID: 4490) ---
ROW_DATA = {
    'id': 4490, 
    'vector': [0.1]*128,
    'c0': 'ggrJlo97PrbYJax3mNnp', 
    'c1': 1833.261158056107, 
    'c2': True, 
    'c3': -43819, 
    'c4': 'mwWTAzpX1r9cVXd', 
    'c5': None, 
    'c6': 1072.9129892268643, 
    'c7': None, 
    'c8': False, 
    'c9': 5512, 
    'c10': 'wPr4T', 
    'meta_json': {'price': 939, 'color': 'Blue', 'active': False, 'random_payload': True}, 
    'tags_array': None
}

# 原始巨大表达式 (去掉了换行符，保证单行)
FULL_EXPR = """((not (not (((((((not ((((not ((c7 > 261.246527660411 and (c4 < "sjf&zZCtyceoeHN56$O97" and tags_array is null))))) or (((not (c1 is null)) and id > -1))))) and ((((not ((c5 == true and c9 >= 153246))) or (((c0 is not null or c10 is not null) or (exists(meta_json["non_exist"]) and c7 > 5732.420786193687))))))))) and ((not ((not (((id > -1 or c7 > 7600.987365071607) or (not (c3 == -21503))))))) and (((((((not ((not (id > -1))))))))) and ((c1 < 5815.101848945394 and c9 < 153246) or (((c7 > 8902.154073446341 and c10 != "1Mr"))))))) or ((not (not (((((((((id > -1 or c9 is not null) or (not (c4 like "b%"))))) or ((((id > -1 or c7 is null) and (c8 == false and (c9 != 153246 and c2 is not null)))))))) or ((((c7 is null or c6 >= 3633.441382777904) and ((not (meta_json["random_payload"] <= "xs9Sq")))) or (((c5 == false or (id > -1)) and (((not ((c1 < 2043.6755316194606)))))))) or (c1 <= 1959.3153573647178 and not (c6 < 1072.9129892268643))))))) or (((((not (((((not (c6 > 2585.5757323910407)))) or ((((not (c10 == "vaSX4IzJvivzpaz")) or meta_json["random_payload"] >= 86746) and c3 == -34705)))))))) or (((not (c5 == true)) or (((not (c7 > 1632.064371131486))))) or ((not (c7 < 5647.9684720442665)) and (c10 != "QB6" and c2 == true))))))))) or ((not (((((((not (((id > -1 and (c0 like "f%")) or (not (id > -1))))) or ((((((not (c3 < 9962))) and (c1 <= 6013.495668362449 or meta_json["_non_exist_key"] == "E5sRJTD5Y25JH"))) or ((c10 like "q%" or c1 <= 1637.5833391954334) or (exists(meta_json["non_exist"]) and (c5 == false))))))) or (((((((not (meta_json["random_payload"] < 1)) or (c4 < "GYhcpZcmTS")) or (not ((c3 is not null)))) and ((not (c4 == "LIMK9M")) or (not ((c2 == true))))))) or (((not (((c3 != 31133 or meta_json["k_15"][0] < 1) or (c9 == -153265 or (c10 is not null and c2 == false))))))))) or (((((not (c9 != -17272)) and ((c3 > -23212 and (c1 > 9078.90830984525)))) and (((not (c6 < 299.50607944929885))) and (((not (c3 != 50611)))))) and (not ((((c8 == true and c5 == false)) and ((not (c9 >= 29465))))))) or (not ((((((not (meta_json["k_10"]["k_2"]["log"] != 1)))) and (not (c5 == true))) and ((((c1 >= 3084.623013290577 and c3 < 153249) or ((not (c10 == "4zrvvBaa"))))))))))) and ((not (((not (((not (meta_json["price"] >= 95)) and (c2 == false and (c1 > 213.5180078442501 and (not (c8 == false))))))) and (((((c0 != "SBpVh4" and c2 == true))) and ((not (c2 == true)) or (c7 >= 8363.196319083747))) and (not ((((c9 > -10387 or c0 != "&fP&GNR1EZ^RVIiqbj6^2^#*") and c9 == -27955)))))))) and (((((not ((c9 != 5734 or c2 == true)))) or ((c7 <= 8446.288383178919 or c6 < 5132.611014319633) and ((((not (meta_json is not null))))))) or (((c7 <= 2619.345920609738 or c8 == true) or ((c7 > 9138.324894956826) or c6 < 2584.235220650769)) or (((not (c8 == true)) and (not ((c0 == "iz9iiTPX5YcbdX2t" and c1 >= 7459.402507926338)))) and (c2 == true and c0 < "hRt9SU68tS")))) and (((not (((c8 == false or id > -1)))) or (not (((c5 == true and c9 <= -49296))))) or (((not (json_contains(meta_json["history"], 48))) or (not (((c5 == true and c5 is not null) and c9 > -153265)))) or ((meta_json["active"] == false and c9 is null) or (c9 > 4194 or meta_json["active"] != false))))))))) or (((not ((((not ((not (((((c3 >= -153279 or c8 == true)) or id > -1)))))) and ((((not (((not (c8 == false)) and (c7 is not null and c2 == false)))))))) and ((not ((((id > -1 and (c8 == true and c1 < 306.5940145765486)) and (c8 == true or (c5 == true or exists(meta_json["non_exist"])))) or ((not (((c7 <= 6702.194627909424 and c2 == true)))))))))))) and (((((((c2 == true or c3 is null)) and (not ((c10 is null or (not (c5 == false)))))) and (not (((c5 is not null or meta_json is null))))) and (((not (c0 is null)) and (c1 < 8425.717260805732 and c9 <= -47709)) or (((not (c10 is null))) and (c4 == "KIJm" and (c10 is not null))))) or (((((c4 == "wP" or c9 == 37591) and (not ((meta_json is not null or c0 > "7MlesI1Y5VFw")))) and ((((not (c5 == false)))) and (((not (c5 is not null))))))) and ((((((((not (c2 is null))))) or (((c2 is null or c4 != "") or c7 >= 1145.9959248011444)))) or ((((c1 <= 5943.36525758282 or id > -1) or c6 is not null)) and (c8 == true and c0 != "p6kNc0V5XHx60X9CNFFMlwnBRHIODVF")))))) or (((not ((((c1 > 8278.247064415575 or c8 == false) and (c4 != "sYX7kzoHL3njjoA*B" and c9 == -46555)) or (not ((not (c8 == true)))))))) and ((((((not (c0 is not null)) or c9 <= 6058) or (c10 == "ifE1l" or c9 >= 23244))) or ((not (c3 < 13964)) or (c2 == false or (meta_json["config"]["version"] != 5)))) and (((not (c6 < 4180.479950522304)) or (not (meta_json["color"] < "Red"))) or (((c10 < "XyBMAahc" or c9 >= -153265)) or (not ((not (c5 == false)))))))))) and ((((((((not (c1 >= 7023.360241967227)) and ((not (c3 is not null)))) and ((not (c4 > "sNVnTRPis2v5cFUuixYRzf")) or (not (c2 is null)))) or (((((not (c2 == true)) and id > -1) or c3 > -153279) or ((c6 <= 1479.7996243734935 or c5 == false))) or ((c9 is not null and c2 == false) and (((c0 < "tf7WUrVmTpUNNVTLG" or (not (c2 == true))) and (not ((not (id > -1))))) or c5 == false)))) or ((((not ((c9 != -232 or c5 == false))) and ((c1 <= 5983.487744003746 or c9 != 2660) or (not (meta_json["price"] == 714))))) or (((id > -1 or meta_json is not null) and ((c5 == false and (not (tags_array is null))))) or ((((not (id > -1)))) and (not ((exists(meta_json["non_exist"]) and c7 > 7811.191870166165))))))) and (((((not ((c5 == false and ((c6 < -99995.6875) or (not (id > -1)))))) or (((not ((c0 < "c")))) and ((not (meta_json["history"][0] <= 65)) and c1 < 5070.793735320897))) and ((not ((((c7 <= 1107.0027213555152 or (c0 < "&ab&KETsEYj$Fwh*u4Z%RN!")) or (not (c9 < -153265))) and (not (c8 == true)))))))) or ((not ((not (((c5 == false or (not (c4 != "8vMJ1FQ8oubJcPQYKOUzH1l7gVrOzi0jCgl"))) and (not (c1 >= 6711.710282703723)))))))))) and (((not ((((not (c7 >= 76.45015533665186)) or (c8 == true and ((c10 like "o%") and c4 > "W@b2IUj!RKRXIv@20qHl!J"))) or ((not ((not (meta_json is null)))))))) or ((((((c7 <= -99999.2265625 and c3 >= 43406))) and ((((meta_json["k_15"]["k_2"]["k_9"] >= 852.215489950153 or c6 <= -99995.6875))))) or (((not ((meta_json["price"] >= 475)))) or (c5 == true or c7 < 4996.667460553808))) or (((c6 is not null or c6 is null) and ((c4 is not null and c4 != "6d"))) and ((not ((((not ((c3 != 10920 or meta_json["_non_exist_key"] != "o5j7TynDmn3"))))))))))) or (((((((c8 is not null or c3 > 15506)) or (not ((c9 == 27233)))) or (not ((c7 <= 5613.574971242804 and (c10 > "fpXLf4Mg0u75mu")))))) and (((c9 > -16861 or c3 >= 44265) or ((not (c2 == true)))) or ((((id > -1 or c3 < 37957)) or (c4 is not null and c1 >= 862.248082229557))))) or ((((((((c0 is null and c8 == false) or ((not (meta_json is null)))))) or (((not (c1 >= 5617.825947400379)) or ((c9 <= -179 and c4 > "z7OMy6knbRl8VGDliLw") or c7 > 2336.7991253782575)))))) or (((c6 >= 6834.725783509132 and tags_array is null) or (((c0 == "X!Za7iseiyzq4tkrrJe5@L" or c0 < "P1ac5ktFJYF")))) and (((c1 >= 8554.416089856808 and c8 is null)) or ((c1 >= 7031.508756871085 and c9 is not null) and c9 is null)))))))))))"""

FULL_EXPR = FULL_EXPR.replace("\n", "").replace("  ", "")

def setup_collection():
    try:
        connections.connect("default", host=HOST, port=PORT)
    except Exception as e:
        print(f"Connection failed: {e}")
        return None
    
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128),
        FieldSchema(name="c0", dtype=DataType.VARCHAR, max_length=512, nullable=True),
        FieldSchema(name="c1", dtype=DataType.DOUBLE, nullable=True),
        FieldSchema(name="c2", dtype=DataType.BOOL, nullable=True),
        FieldSchema(name="c3", dtype=DataType.INT64, nullable=True),
        FieldSchema(name="c4", dtype=DataType.VARCHAR, max_length=512, nullable=True),
        FieldSchema(name="c5", dtype=DataType.BOOL, nullable=True),
        FieldSchema(name="c6", dtype=DataType.DOUBLE, nullable=True),
        FieldSchema(name="c7", dtype=DataType.DOUBLE, nullable=True),
        FieldSchema(name="c8", dtype=DataType.BOOL, nullable=True),
        FieldSchema(name="c9", dtype=DataType.INT64, nullable=True),
        FieldSchema(name="c10", dtype=DataType.VARCHAR, max_length=512, nullable=True),
        FieldSchema(name="meta_json", dtype=DataType.JSON, nullable=True),
        FieldSchema(name="tags_array", dtype=DataType.ARRAY, element_type=DataType.INT64, max_capacity=50, nullable=True)
    ]
    
    schema = CollectionSchema(fields, enable_dynamic_field=True)
    col = Collection(COLLECTION_NAME, schema)
    
    col.insert([ROW_DATA])
    col.flush()
    col.create_index("vector", {"metric_type": "L2", "index_type": "FLAT", "params": {}})
    col.load()
    return col

# --- 核心：递归逻辑探测器 ---

def split_expr(expr):
    """
    根据括号平衡原则，找到最顶层的 OR 或 AND，将表达式拆分为 Left 和 Right。
    如果没有顶层操作符，尝试剥离外层括号 (A) -> A。
    """
    expr = expr.strip()
    
    # 1. 尝试剥离外层括号
    while expr.startswith('(') and expr.endswith(')'):
        # 必须确保剥离后依然平衡
        # 简单检查：去掉首尾后，中间的括号是否平衡？
        inner = expr[1:-1].strip()
        balance = 0
        is_balanced = True
        for char in inner:
            if char == '(': balance += 1
            elif char == ')': balance -= 1
            if balance < 0: 
                is_balanced = False
                break
        
        if balance == 0 and is_balanced:
            expr = inner
        else:
            break

    # 2. 扫描寻找顶层 operator
    balance = 0
    split_index = -1
    op_str = None
    
    # 我们只找最顶层的 ' or ' 或 ' and ' (注意空格)
    # 从左到右扫描
    i = 0
    while i < len(expr):
        char = expr[i]
        if char == '(':
            balance += 1
        elif char == ')':
            balance -= 1
        
        if balance == 0:
            # 检查是否是 operator
            rest = expr[i:]
            if rest.startswith(" or ") or rest.startswith(" OR "):
                split_index = i
                op_str = "OR"
                break # 找到第一个顶层分割点即可
            elif rest.startswith(" and ") or rest.startswith(" AND "):
                split_index = i
                op_str = "AND"
                break
        i += 1
        
    if split_index != -1:
        left = expr[:split_index].strip()
        right_start = split_index + len(op_str) + 2 # +2 for spaces
        right = expr[right_start:].strip()
        return op_str, left, right
    
    return None, None, None

def check_expr(col, expr):
    """ 执行查询，返回 hits """
    try:
        res = col.query(expr, output_fields=["id"])
        return len(res)
    except Exception as e:
        print(f"⚠️ Query Error for expr: {expr} : {e}")
        return -1

def recursive_probe(col, expr, depth=0):
    """
    递归探测：
    1. 执行当前 expr。
    2. 如果 expr 返回 1 (True)，说明这个分支是好的，或者是被 PQS 选中的分支。
    3. 如果 expr 返回 0 (False)，但我们预期它可能是 True（因为整个 PQS 应该是 True），
       我们需要找出是哪个子部分导致了 False。
    """
    indent = "  " * depth
    hits = check_expr(col, expr)
    
    status = "✅ TRUE" if hits > 0 else "❌ FALSE"
    print(f"{indent}[D{depth}] Hits:{hits} | Expr: {expr}" if len(expr)>60 else f"{indent}[D{depth}] Hits:{hits} | Expr: {expr}")
    
    # 拆分
    op, left, right = split_expr(expr)
    
    if op:
        # 如果是组合逻辑
        print(f"{indent}  Op: {op}")
        
        # 递归检查子节点
        h_left = check_expr(col, left)
        h_right = check_expr(col, right)
        
        # 逻辑诊断
        if op == "OR":
            if hits == 0 and (h_left > 0 or h_right > 0):
                print(f"{indent}  🚨 BUG FOUND: Left({h_left}) OR Right({h_right}) => Parent(0)!")
                print(f"{indent}  👉 Left: {left}")
                print(f"{indent}  👉 Right: {right}")
                return # 找到问题了，停止
            elif hits == 0 and h_left == 0 and h_right == 0:
                print(f"{indent}  Both children 0. Diving deeper to find which one *should* be 1...")
                recursive_probe(col, left, depth+1)
                recursive_probe(col, right, depth+1)
        
        elif op == "AND":
             if hits == 0:
                 if h_left > 0 and h_right > 0:
                     print(f"{indent}  🚨 BUG FOUND: Left({h_left}) AND Right({h_right}) => Parent(0)!")
                 else:
                     # 如果是 AND，只要有一个为 0，结果就是 0。
                     # 我们需要找出哪个为 0，然后看看它是否应该是 1。
                     # (这需要人工判断逻辑，或者继续下钻看看有没有 False Positive)
                     if h_left == 0: recursive_probe(col, left, depth+1)
                     if h_right == 0: recursive_probe(col, right, depth+1)
    
    elif expr.strip().startswith("not ") or expr.strip().startswith("NOT "):
        # 处理 NOT
        inner = expr.strip()[4:].strip()
        # 去掉可能的括号
        if inner.startswith('(') and inner.endswith(')'):
             # 简单去除，不够严谨但对于自动生成的 usually ok
             pass 
        
        # 对于 NOT，如果当前是 False (0)，说明 inner 是 True。
        # 如果 inner 实际上是 False，那 not inner 应该是 True。
        # 我们可以探测一下 inner
        h_inner = check_expr(col, inner)
        print(f"{indent}  (NOT Logic Check): Inner Hits: {h_inner}")
        
        if hits == 0 and h_inner == 0:
             print(f"{indent}  🚨 BUG FOUND: Inner(0) -> NOT Inner -> Parent(0)? (Should be 1)")
             recursive_probe(col, inner, depth+1)


def diagnose_recursive(col):
    print("\n🔍 Starting Recursive Diagnosis...")
    recursive_probe(col, FULL_EXPR)

if __name__ == "__main__":
    col = setup_collection()
    if col:
        diagnose_recursive(col)