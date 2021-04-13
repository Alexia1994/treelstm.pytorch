#coding: utf-8
# 运算符优先
ops_rule = {
    '=': 0,
    '+': 1,
    '-': 1,
    '*': 2,
    '/': 2,
    '^': 3
}

# tree object from stanfordnlp/treelstm
class Tree(object):
    def __init__(self, item):
        self.parent = None
        self.left = None
        self.right = None
        self.input = item
        self.children = list()
        self.gold_label = None # node label for SST
        self.output = None # output node for SST
# 后缀表达式转树的过程
def build_tree(expression):
    stack  = []
    for item in expression:
        if item in ops_rule:
            right = stack.pop()
            left = stack.pop()
            node = Tree(item)
            node.left = left
            node.right = right
            stack.append(node)
        else:
            node = Tree(item)
            stack.append(node)

    if len(stack) != 1:
        print("bug")
    root = stack[0]
    return root

def inorder(root):
    if root == None:
        return 
    inorder(root.left)
    print(root.input)
    inorder(root.right)

def preorder(root):
    if root == None:
        return
    print(root.input)
    preorder(root.left)
    preorder(root.right)

def middle_to_after(s):
    #中缀表达式变为后缀表达式
    operator = ['+', '-', '*', '/',"^", "="]
    #数字栈
    expression = []
    # 运算符栈
    ops = []
    # need fix: why item[0]?
    for item in s:
        # 当遇到运算符
        if item in operator:
            while len(ops) >= 0:
                # 如果栈中没有运算符，直接将运算符添加到后缀表达式
                if len(ops) == 0:
                    ops.append(item)
                    break
                # 如果栈中有运算符
                op = ops.pop()
                # 如果栈顶的运算符比当前运算符级别低，当前运算符入栈
                if op == '(' or ops_rule[item] > ops_rule[op]:
                    ops.append(op)
                    ops.append(item)
                    break
                else:
                    # 如果栈顶的运算符比当前运算符级别高，将栈顶运算符加入到表达式
                    # 当前运算符与栈中后面的运算符比较
                    expression.append(op)
        # 遇到左括号入栈
        elif item == '(':
            ops.append(item)
        # 遇到右括号，将栈中运算符加入到表达式直到遇到左括号
        elif item == ')':
            while len(ops) > 0:
                op = ops.pop()
                if op == '(':
                    break
                else:
                    expression.append(op)
        # 遇到运算数，添加到表达式
        else:
            expression.append(item)
    # 最后将栈中全部运算符加到后缀表达式中
    while len(ops) > 0:
        expression.append(ops.pop())

    return expression


def expression_to_value(expression):
    """后缀表达式计算"""
    stack_value = []
    for item in expression:
        if item in ['+', '-', '*', '/','^']:
            n2 = stack_value.pop()
            n1 = stack_value.pop()
            result = cal(n1, n2, item)
            stack_value.append(result)
        else:
            stack_value.append(int(item))
    return stack_value[0]


# 计算函数
def cal(n1, n2, op):
    if op == '+':
        return n1 + n2
    if op == '-':
        return n1 - n2
    if op == '*':
        return n1 * n2
    if op == '/':
        return n1 // n2
    if op == '^':
        return n1**n2


if __name__ == '__main__':
    exp = "( 3 + 4 ) * 5 + 7 = 23"
    exp = exp.split()
    expression = middle_to_after(exp)
    print(expression)
    root = build_tree(expression)
    # 注意此时输出的是字符类型
    inorder(root)
