# 作者：liuyankai
# 时间：2025年08月03日17时42分12秒
# liuyankai23@mails.ucas.ac.cn
def J(x):
    """目标函数"""
    return (x**2 - 2) ** 2

def gradient(x):
    """梯度"""
    return 4 * x**3 - 8 * x

