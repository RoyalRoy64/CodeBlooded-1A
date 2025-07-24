import timeit
from itertools import zip_longest

# Original approach (string +=)
def merge_original(word1, word2):
    len1 = len(word1)
    len2 = len(word2)
    ret = ''
    if len1 == len2:
        for i in range(len1):
            ret += word1[i] + word2[i]
    elif len1 > len2:
        for i in range(len2):
            ret += word1[i] + word2[i]
        ret += word1[len2:len1]
    else:
        for i in range(len1):
            ret += word1[i] + word2[i]
        ret += word2[len1:len2]
    return ret

# Optimized list-append + join
def merge_optimized(word1, word2):
    len1, len2 = len(word1), len(word2)
    result = []

    for i in range(min(len1, len2)):
        result.append(word1[i])
        result.append(word2[i])

    if len1 > len2:
        result.extend(word1[len2:])
    else:
        result.extend(word2[len1:])

    return ''.join(result)

# Pythonic zip_longest approach
def merge_pythonic(word1, word2):
    return ''.join(a + b for a, b in zip_longest(word1, word2, fillvalue=''))

# Test data (long enough to see real performance difference)
word1 = 'abcde' * 1000
word2 = '12345' * 1000

# Time comparison
repeat = 5
number = 1000

print("Timing for {} runs each (average of {} repeats):".format(number, repeat))
print()

t1 = timeit.repeat(lambda: merge_original(word1, word2), repeat=repeat, number=number)
print("Original (+=):       {:.5f} sec".format(sum(t1) / repeat))

t2 = timeit.repeat(lambda: merge_optimized(word1, word2), repeat=repeat, number=number)
print("Optimized (list+join): {:.5f} sec".format(sum(t2) / repeat))

t3 = timeit.repeat(lambda: merge_pythonic(word1, word2), repeat=repeat, number=number)
print("Pythonic (zip_longest): {:.5f} sec".format(sum(t3) / repeat))
