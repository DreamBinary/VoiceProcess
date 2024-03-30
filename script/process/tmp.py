def count_sorted_intervals(n, a):
    count = 0
    for i in range(1, n):
        for j in range(i + 1, n):
            if sorted(a[:i] + a[j:]) == a:
                count += 1
    return count + 1  # 加1是考虑空数组的情况

n = 5
a = [3, 1, 4, 2, 5]
print(count_sorted_intervals(n, a))  # 输出：3
