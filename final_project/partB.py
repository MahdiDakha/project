#######################################
#IMPORTS
#######################################
from functools import reduce

#######################################
#1Generates the first n Fibonacci numbers and returns them as a list.
def func1():
    fibonacci = lambda n: (lambda f: [f(i) for i in range(n)])(lambda i, a=[0, 1]: a.append(a[-1] + a[-2]) or a[i])
    n = 10
    result = fibonacci(n)
    return result
#2Concatenates a list of strings into a single string, with a space between each string.
def func2():
    concat_with_space = lambda lst: reduce(lambda acc, s: acc + (' ' if acc else '') + s, lst, '')

    # Example usage:
    strings = ["best", "code", "in", "the","world"]
    result = concat_with_space(strings)
    return result
#3Takes a list of lists of numbers and returns a new list,
# where each element is the cumulative sum of squares of the even numbers in the corresponding sublist.
def func3():
    def cumulative_sum_of_squares(lists):
        return list(map(
            lambda sublist: reduce(
                lambda acc, num: acc + num ** 2,
                filter(
                    lambda x: x % 2 == 0,
                    sublist
                ),
                0
            ),
            lists
        ))

    input_lists = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    result = cumulative_sum_of_squares(input_lists)
    return result
#4Implements a higher-order function that applies a binary operation cumulatively to a sequence
#Uses this higher-order function to calculate the factorial of a number and to perform exponentiation.
def func4():

    def cumulative_operation(operation):
        return lambda seq: reduce(operation, seq)

    factorial = lambda n: cumulative_operation(lambda x, y: x * y)(range(1, n + 1))
    exponentiation = lambda base, exp: cumulative_operation(lambda x, y: x * y)([base] * exp)

    return factorial(5), exponentiation(2, 3)
#5Rewrites a program that computes the sum of the squares of even numbers in a list,
# using nested filter, map, and reduce functions.
def func5():
    nums = [1, 2, 3, 4, 5, 6]
    evens = []
    for num in nums:
        if num % 2 == 0:
            evens.append(num)
    squared = []
    for even in evens:
        squared.append(even ** 2)

    sum_squared = 0
    print("Befor")
    for x in squared:
        sum_squared += x

    print(sum_squared)

    print("After")
    sum_squared = reduce(lambda acc, x: acc + x,
                  map(lambda x: x ** 2, filter(lambda num: num % 2 == 0, [1, 2, 3, 4, 5, 6])))
    return sum_squared
#6
#Counts the number of palindromes in each sublist of a list of lists of strings.
def func6():
    list_ = [['momo', 'rtoto', 'lolo'], ['level', 'ther', 'is'], ['you', 'for']]
    count_palindromes = lambda lst: list(
        map(lambda sublist: reduce(lambda count, s: count + (s == s[::-1]), sublist, 0), lst))
    return count_palindromes(list_)
# 7
def func7():

    str_ = "lazyEvaluation:\nValues are generated and squared one at a time. The generator only produces a value when needed, delaying computation until absolutely necessary."
    return str_
#8
#Filters out prime numbers from a list and sorts them in descending order.
def func8():
    sorted_primes = lambda lst: sorted([x for x in lst if x > 1 and all(x % i != 0 for i in range(2, int(x ** 0.5) + 1))], reverse=True)
    lst = [10, 7, 2, 3, 11, 4, 9]
    sorted_lst = sorted_primes(lst)
    return sorted_lst

def switch_case(case_value):
    switcher = {
        1: func1,
        2: func2,
        3: func3,
        4: func4,
        5: func5,
        6: func6,
        7: func7,
        8: func8
    }
    # Get the function from the dictionary, defaulting to a lambda function returning "Default" if the case_value is not found
    func = switcher.get(case_value, lambda: "Default")
    return func()


ok = True
while ok:
    print("Question 1 press 1")
    print("Question 2 press 2")
    print("Question 3 press 3")
    print("Question 4 press 4")
    print("Question 5 press 5")
    print("Question 6 press 6")
    print("Question 7 press 7")
    print("Question 8 press 8")
    print("Enter -1 to exit")

    user_input = int(input())

    if user_input == -1:
        ok = False
    else:
        result = switch_case(user_input)
        if callable(result):
            output = result() 
            print(output)
        else:
            print(result)
