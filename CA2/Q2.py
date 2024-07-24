import sympy
import math
import random
import numpy
s = sympy.symbols('s')
#1
def coupon_collector(n , k):
    results = []
    for i in range(k):
        sum = 0
        numbers = []
        while(len(numbers) != n):
            v = random.randint(1,n)
            if v not in numbers:
                numbers.append(v)
            sum += 1
        results.append(sum)
    return numpy.mean(results)
#2
print(coupon_collector(10,10))
print(coupon_collector(10,100))
print(coupon_collector(10,1000))

#3
def moment_generating_Xi(i):
    p = (10-i+1)/10
    return (p*sympy.exp(s))/(1-(1-p)*sympy.exp(s))
#4
def moment_generating_X(n):
    fi_X = moment_generating_Xi(1)
    for i in range(2,n+1):
        fi_X *= moment_generating_Xi(i)
    return fi_X
#5
moment_generating_function_X = moment_generating_X(10)
expectation_function = sympy.diff(moment_generating_function_X, s)
print("expectation = ", expectation_function.subs({s : 0}))

