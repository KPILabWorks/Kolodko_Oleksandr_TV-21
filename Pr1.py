from functools import reduce

# Функція для обчислення добутку двох чисел
def multiply(x, y):
    return x * y

# Початковий список чисел
numbers = [2, 3, 5, 7, 11]

# Використання reduce для обчислення добутку всіх елементів
product = reduce(multiply, numbers)

print("Добуток елементів списку:", product)
