a = int(input('Число а = '))
b = int(input('Число b = '))
try:
    c1 = a / b
except ZeroDivisionError:
    c1 = 0
try:
    c2 = b / a
except ZeroDivisionError:
    c2 = 0
try:
    c = c1/c2
except ZeroDivisionError:
    c = 0

print(c)